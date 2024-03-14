import sys
import math

import numpy as np
import Box2D
from Box2D.b2 import (
    edgeShape,
    vec2,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

import gym
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
from gym.envs.registration import register
import keras




# This is simple 4-joints walker robot environment.
#
# There are two versions:
#
# - Normal, with slightly uneven terrain.
#
# - Hardcore with ladders, stumps, pitfalls.
#
# Reward is given for moving forward, total 300+ points up to the far end. If the robot falls,
# it gets -100. Applying motor torque costs a small amount of points, more optimal agent
# will get better score.
#
# Heuristic is provided for testing, it's also useful to get demonstrations to
# learn from. To run heuristic:
#
# python gym/envs/box2d/bipedal_walker.py
#
# State consists of hull angle speed, angular velocity, horizontal speed, vertical speed,
# position of joints and joints angular speed, legs contact with ground, and 10 lidar
# rangefinder measurements to help to deal with the hardcore version. There's no coordinates
# in the state vector. Lidar is less useful in normal version, but it works.
#
# To solve the game you need to get 300 points in 1600 time steps.
#
# To solve hardcore version you need 300 points in 2000 time steps.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.hull == contact.fixtureA.body
            or self.env.hull == contact.fixtureB.body
        ):
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class ModeledBipedalWalker(gym.Env, EzPickle):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}

    hardcore = False

    def __init__(self, model_path):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None
        self.hull = None
        self.state = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        self.model = keras.models.load_model(model_path)

        def run_nn(obs, action):
            latent_shape = self.model.input["latent"].shape
            latent = np.array([[]]) if latent_shape[1] == 0 else np.random.normal(latent_shape[1:])
            return self.model({"state": np.array([obs]), "action": np.array([action]), "latent": latent}, training=False)[0]

        self.nn = run_nn

        self.reset()

        high = np.array([10.0, 10.0, 10.0, 10.0, 3.0, 1.0, 1.0] * 5).astype(np.float32)
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(-high, high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.scroll = 0.0
        self.state = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateStaticBody(
            position=(init_x, init_y), fixtures=HULL_FD
        )
        self.hull.color1 = (0.9, 0.4, 0.5)
        self.hull.color2 = (0.5, 0.3, 0.3)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateStaticBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (0.6 - i / 10.0, 0.3 - i / 10.0, 0.5 - i / 10.0)
            leg.color2 = (0.4 - i / 10.0, 0.2 - i / 10.0, 0.3 - i / 10.0)

            self.legs.append(leg)

            lower = self.world.CreateStaticBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (0.6 - i / 10.0, 0.3 - i / 10.0, 0.5 - i / 10.0)
            lower.color2 = (0.4 - i / 10.0, 0.2 - i / 10.0, 0.3 - i / 10.0)
            lower.ground_contact = False
            self.legs.append(lower)

        self.drawlist = self.terrain + self.legs + [self.hull]

        return self.step(np.array([0, 0, 0, 0]))[0]

    def step(self, action):
        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        bodies = [ self.hull ] + self.legs
        if self.state is None:
            body_states = map(lambda body: (
                body.position.x,
                body.position.y,
                body.linearVelocity.x,
                body.linearVelocity.y,
                body.angularVelocity,
                math.sin(body.angle),
                math.cos(body.angle)), bodies)
            self.state = np.array([item for sublist in body_states for item in sublist])
        
        self.state = self.nn(self.state, action)

        for body, state in zip(bodies, chunks(self.state, 7)):
            body.position = vec2(float(state[0]), float(state[1]))
            # body.linearVelocity = vec2(float(state[2]), float(state[3]))
            # body.angularVelocity = float(state[4])
            body.angle = math.atan2(state[5], state[6])
        # self.hull.position.x = float(self.state[0])
        # self.hull.position.y = float(self.state[1])

        self.scroll = self.hull.position.x - VIEWPORT_W / SCALE / 5

        return self.state, 0, False, {}

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(
            self.scroll, VIEWPORT_W / SCALE + self.scroll, 0, VIEWPORT_H / SCALE
        )

        self.viewer.draw_polygon(
            [
                (self.scroll, 0),
                (self.scroll + VIEWPORT_W / SCALE, 0),
                (self.scroll + VIEWPORT_W / SCALE, VIEWPORT_H / SCALE),
                (self.scroll, VIEWPORT_H / SCALE),
            ],
            color=(1.0, 0.9, 1.0),
        )
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(
                [(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1, 1)
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            self.viewer.draw_polygon(poly, color=color)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color1
                    ).add_attr(t)
                    self.viewer.draw_circle(
                        f.shape.radius, 30, color=obj.color2, filled=False, linewidth=2
                    ).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline(
            [(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10 / SCALE),
            (x + 25 / SCALE, flagy2 - 5 / SCALE),
        ]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class ModeledBipedalWalkerHardcore(ModeledBipedalWalker):
    hardcore = True


if __name__ == "__main__":
    # Heurisic: suboptimal, have no notion of balance.
    # import src.utils as utils
    env = ModeledBipedalWalker(utils.latest_model())
    env.reset()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0, 0.0, 0.0])
    STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1, 2, 3
    SPEED = 0.29  # Will fall forward on higher speed
    state = STAY_ON_ONE_LEG
    moving_leg = 0
    supporting_leg = 1 - moving_leg
    SUPPORT_KNEE_ANGLE = +0.1
    supporting_knee_angle = SUPPORT_KNEE_ANGLE
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 5 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4]]))
            print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:8]]))
            print("leg1 " + str(["{:+0.2f}".format(x) for x in s[8:12]]))
            print("lower0 " + str(["{:+0.2f}".format(x) for x in s[12:16]]))
            print("lower1 " + str(["{:+0.2f}".format(x) for x in s[16:20]]))
        steps += 1
        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5 * moving_leg
        supporting_s_base = 4 + 5 * supporting_leg

        hip_targ = [None, None]  # -0.8 .. +1.1
        knee_targ = [None, None]  # -0.6 .. +0.9
        hip_todo = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if state == STAY_ON_ONE_LEG:
            hip_targ[moving_leg] = 1.1
            knee_targ[moving_leg] = -0.6
            supporting_knee_angle += 0.03
            if s[2] > SPEED:
                supporting_knee_angle += 0.03
            supporting_knee_angle = min(supporting_knee_angle, SUPPORT_KNEE_ANGLE)
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base + 0] < 0.10:  # supporting leg is behind
                state = PUT_OTHER_DOWN
        if state == PUT_OTHER_DOWN:
            hip_targ[moving_leg] = +0.1
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base + 4]:
                state = PUSH_OFF
                supporting_knee_angle = min(s[moving_s_base + 2], SUPPORT_KNEE_ANGLE)
        if state == PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = +1.0
            if s[supporting_s_base + 2] > 0.88 or s[2] > 1.2 * SPEED:
                state = STAY_ON_ONE_LEG
                moving_leg = 1 - moving_leg
                supporting_leg = 1 - moving_leg

        if hip_targ[0]:
            hip_todo[0] = 0.9 * (hip_targ[0] - s[4]) - 0.25 * s[5]
        if hip_targ[1]:
            hip_todo[1] = 0.9 * (hip_targ[1] - s[9]) - 0.25 * s[10]
        if knee_targ[0]:
            knee_todo[0] = 4.0 * (knee_targ[0] - s[6]) - 0.25 * s[7]
        if knee_targ[1]:
            knee_todo[1] = 4.0 * (knee_targ[1] - s[11]) - 0.25 * s[12]

        hip_todo[0] -= 0.9 * (0 - s[0]) - 1.5 * s[1]  # PID to keep head strait
        hip_todo[1] -= 0.9 * (0 - s[0]) - 1.5 * s[1]
        knee_todo[0] -= 15.0 * s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= 15.0 * s[3]

        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5 * a, -1.0, 1.0)

        env.render()
        if done:
            break