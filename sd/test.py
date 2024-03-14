from pathlib import Path
import sd.envs
import gymnasium as gym
import numpy as np
import tensorflow as tf
import keras
from keras import layers
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib
from sd.envs.amazingball.constant import constants
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import utils 
import argparse
import pygame
import time
def angle_to_setpoint(angle):
    return np.array([np.cos(angle),np.sin(angle),0.0])  


def plot_lyapunov(lyapunov, actor, dynamics, set_point, fname, interactive=False):
    # def calculate_lyapunov(set_point):
    #     pts = 200
    #     theta = np.linspace(-np.pi, np.pi, pts).reshape(-1,1)
    #     theta_dot = np.linspace(-7.0, 7.0,pts).reshape(-1,1)

    #     thetav, theta_dotv = np.meshgrid(theta, theta_dot)
    #     inputs = np.array([np.cos(thetav), np.sin(thetav), theta_dotv]).T.reshape(-1,3)
    #     set_points = inputs*0 + set_point
    #     z = lyapunov({"state": inputs, "setpoint": set_points}, training=False)
    #     acts = actor({"state":inputs, "setpoint": set_points}, training=False)
    #     after = dynamics({"state": inputs, "action": acts, "latent": np.random.normal(size=(inputs.shape[0],)+dynamics.input["latent"].shape[1:])}, training=False)
    #     next_z = lyapunov({"state": after, "setpoint": set_points}, training=False)
    #     after = after.numpy().reshape(pts,pts,3)
    #     z = z.numpy().reshape(pts,pts, 1)
    #     next_z = next_z.numpy().reshape(pts,pts,1)
    #     acts = acts.numpy().reshape(pts,pts,1)
    #     return thetav, theta_dotv, z
    
    def calculate_lyapunov(set_point):
        pts = 200
        x = np.linspace(-constants["max_ball_pos_x"], constants["max_ball_pos_x"], pts).reshape(-1,1)
        y = np.linspace(-constants["max_ball_pos_y"], constants["max_ball_pos_y"],pts).reshape(-1,1)

        xv, yv = np.meshgrid(x, y)
        # pl_rot_x, pl_rot_y,  pl_vel_x, pl_vel_y, ba_pos_x, ba_pos_y, ba_v_x, ba_v_y
        inputs = np.array([0*xv, 0*xv, 0*xv, 0*xv, xv, yv, 0*xv,0*xv]).reshape(-1, 8)
        set_points = np.zeros((inputs.shape[0], 4))
        z = lyapunov({"state": inputs, "setpoint": set_points}, training=False)
        # acts = actor({"state": inputs, "setpoint": set_points}, training=False)
        # after = dynamics({"state": inputs, "action": acts, "latent": np.random.normal(size=(inputs.shape[0],) + dynamics.input["latent"].shape[1:])}, training=False)
        # next_z = lyapunov({"state": after, "setpoint": set_points}, training=False)
        z = z.numpy().reshape(pts, pts, 1)
        # next_z = next_z.numpy().reshape(pts,pts,1)
        # acts = acts.numpy().reshape(pts,pts,1)
        return xv, yv, z
        

    init=True

    def draw_lyapunov(set_point):
        nonlocal init
            
        thetav, theta_dotv, z = calculate_lyapunov(set_point)
        plt.pcolormesh(thetav, theta_dotv, z.T[0][:-1, :-1], vmin=0.0, vmax=1.0)
        if init:
            plt.colorbar()
            init = False
        plt.xlabel('$\\theta$')
        plt.ylabel('$\\dot{\\theta}$')
        if interactive:
            plt.show()
        else:
            plt.savefig(f'{fname}.png')

    def distance_to_setpoint(state, setpoint):
        return np.sqrt((state[0,0] - setpoint[0])**2 + (state[0,1] - setpoint[1])**2 + (state[0,2] - setpoint[2])**2)
    
    MAX_UPDATES = 100  
    CLOSE_ENOUGH_THRESHOLD = 0.05  
    cur_setpoint = set_point
    scatter_collections = []

    def mouse_event(event):
        nonlocal scatter_collections
        nonlocal cur_setpoint

        if event.button == 1:  # Left mouse button
            set_point = np.array([np.cos(event.xdata), np.sin(event.xdata), event.ydata])
            cur_setpoint = set_point
            draw_lyapunov(set_point)
            print(f'theta: {event.xdata} and theta_dot: {event.ydata}')

        elif event.button == 3:  # Right mouse button
            for scatter_plot in scatter_collections:
                scatter_plot.remove()
                scatter_collections = []

            state = np.array([np.cos(event.xdata), np.sin(event.xdata), event.ydata]).reshape(1,3)
            print(f'Init state set to theta: {event.xdata} and theta_dot: {event.ydata}')

            update_count = 0
            colors = cm.Reds(np.linspace(0, 1, MAX_UPDATES))
            while distance_to_setpoint(state, cur_setpoint) > CLOSE_ENOUGH_THRESHOLD and update_count < MAX_UPDATES:
                pt = (np.arctan2(state[0,1], state[0,0]), state[0,2])
                if -np.pi < pt[0] < np.pi and -7 < pt[1] < 7:
                    scatter_plot = plt.scatter(pt[0], pt[1], c=[colors[update_count]], s=10)
                    scatter_collections.append(scatter_plot)
                    fig.canvas.draw_idle()
                    fig.canvas.start_event_loop(0.1)
                    
                    
                    # plt.draw()
                    # plt.pause(0.0)
                act = actor({"state":state, "setpoint": np.array([cur_setpoint])}, training=False)
                state = dynamics({"state": state, "action": act, "latent": np.random.normal(size=(1,)+dynamics.input["latent"].shape[1:])}, training=False)
                update_count += 1

    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', mouse_event)
    draw_lyapunov(set_point)

    # plt.colorbar()
    # plt.pcolormesh(thetav, theta_dotv, (next_z - z).T[0][:-1,:-1])
    # plt.colorbar()
    # plt.show()

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, help="model_path", default=None)
    parser.add_argument('--random_actor', action="store_true")
    parser.add_argument('--actor_path', type=Path, help="actor path", default=None)
    parser.add_argument('--lyapunov_path', type=Path, help="actor path", default=None)
    parser.add_argument('--low_actor', action="store_true")
    parser.add_argument('--no_lyapunov', action="store_true")
    parser.add_argument('--no_test', action="store_true")
    parser.add_argument('--interactive', action="store_true")
    parser.add_argument('--angle', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=np.random.randint(100000))
    args = parser.parse_args()

    try:
        checkpoint_path = args.model if args.model else utils.latest_model()
        print("using checkpoint:", checkpoint_path)
    except:
        print("there are no trained models")
        exit()


    dynamics = utils.load_checkpoint(checkpoint_path) 
    env_name = utils.extract_env_name(checkpoint_path)

    if args.random_actor:
        action_space = gym.make(env_name).action_space
        actor = lambda x,**ignored: action_space.sample()
    elif args.low_actor:
        actor = lambda x,**ignored: np.array([0])
    else:
        try:
            actor = keras.models.load_model(args.actor_path if args.actor_path else checkpoint_path.parent / "actor.keras")
            actor.summary()
        except:
            print(f"there is no actor trained for the model {checkpoint_path}")


    setpoint = angle_to_setpoint(args.angle)
    lyapunov = None
    if not args.no_lyapunov:
        lyapunov = keras.models.load_model(args.lyapunov_path if args.lyapunov_path else checkpoint_path.parent / "lyapunov.keras")
        plot_lyapunov(lyapunov, actor, dynamics, setpoint, f'V_{args.angle}', interactive=args.interactive)

    if args.no_test:
        exit()

    def run_test(num_steps=2000):
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((500*2, 500))
        pygame.display.set_caption("test")
        surface1 = pygame.Surface((500, 500))
        surface2 = pygame.Surface((500, 500))
        
        
        modeled_env = gym.make('Modeled' + env_name,
            model_path=checkpoint_path, render_mode="human", screen=surface1)#, test=True, gui=True)

        orig_env = gym.make(env_name, render_mode="human", screen=surface2)
        print("seed:", args.seed)
        env_obs, _ = modeled_env.reset(seed=args.seed)
        orig_env_obs, _ = orig_env.reset(seed=args.seed)
        def feed_obs(obs):
            global setpoint
            if pygame.mouse.get_pressed()[0]:
                cur_pos = np.array(pygame.mouse.get_pos())
                setpoint = angle_to_setpoint(np.arctan2(*(np.array(surface1.get_size())/2.0 - cur_pos)))
            return {"state": np.array([obs]), "setpoint": np.array([setpoint])}
        
        for i in range(num_steps):
            window.blits(((surface1, (0, 0)), (surface2, (500, 0))))
            # random_act = np.random.uniform(2,size=(1,))
            act = actor(feed_obs(env_obs), training=False)
            # print(orig_env_obs)
            # print(env_obs)
            if lyapunov:
                print("lyapunov", lyapunov(feed_obs(env_obs)))
            orig_act = actor(feed_obs(orig_env_obs), training=False)
            env_obs, env_reward, env_done, env_term, env_info = modeled_env.step(act)
            orig_env_obs, orig_env_reward, orig_env_done, orig_env_term, orig_env_info = orig_env.step(orig_act)

    run_test()