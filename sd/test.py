from pathlib import Path
import sd.envs
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import utils 
import argparse
import pygame


def angle_to_setpoint(angle):
    return np.array([np.cos(angle),np.sin(angle),0.0])  


def plot_lyapunov(lyapunov, actor, dynamics, set_point, fname, interactive=False):
    def calculate_lyapunov(set_point):
        pts = 200
        theta = np.linspace(-np.pi, np.pi, pts).reshape(-1,1)
        theta_dot = np.linspace(-7.0, 7.0,pts).reshape(-1,1)

        thetav, theta_dotv = np.meshgrid(theta, theta_dot)
        inputs = np.array([np.cos(thetav), np.sin(thetav), theta_dotv]).T.reshape(-1,3)
        set_points = inputs*0 + set_point
        z = lyapunov({"state": inputs, "setpoint": set_points}, training=False)
        acts = actor({"state":inputs, "setpoint": set_points}, training=False)
        after = dynamics({"state": inputs, "action": acts, "latent": np.random.normal(size=(inputs.shape[0],)+dynamics.input["latent"].shape[1:])}, training=False)
        next_z = lyapunov({"state": after, "setpoint": set_points}, training=False)
        after = after.numpy().reshape(pts,pts,3)
        z = z.numpy().reshape(pts,pts, 1)
        next_z = next_z.numpy().reshape(pts,pts,1)
        acts = acts.numpy().reshape(pts,pts,1)
        return thetav, theta_dotv, z
    
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


    def mouse_event(event):
        set_point = np.array([np.cos(event.xdata), np.sin(event.xdata), event.ydata])
        draw_lyapunov(set_point)
        print(f'theta: {event.xdata} and theta_dot: {event.ydata}')

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
    parser.add_argument('--low_actor', action="store_true")
    parser.add_argument('--no_lyapunov', action="store_true")
    parser.add_argument('--no_test', action="store_true")
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
            actor = keras.models.load_model(checkpoint_path.parent / "actor.keras")
            actor.summary()
        except:
            print(f"there is no actor trained for the model {checkpoint_path}")


    setpoint = angle_to_setpoint(args.angle)
    lyapunov = None
    if not args.no_lyapunov:
        try:
            lyapunov = keras.models.load_model(checkpoint_path.parent / "lyapunov.keras")
            plot_lyapunov(lyapunov, actor, dynamics, setpoint, fname = f'V_{args.angle}')
        except:
            pass

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
        # seed = 632732 #bottom almost
        # seed = 154911 # almost rotate
        # seed = 47039 # almost rotate, then rotate
        # seed = 364366 # rotate
        print("seed:", args.seed)
        env_obs, _ = modeled_env.reset(seed=args.seed)
        # env_obs = env.env.init_with_state(np.array([0.9474508 , 0.31990144, 1.06079]))
        orig_env_obs, _ = orig_env.reset(seed=args.seed)
        prev_pos = None
        def feed_obs(obs):
            global setpoint
            #print("state_shape", np.array([obs]).shape)
            #print("setpoint_shape", np.array([set_point]).shape)
            if pygame.mouse.get_pressed()[0]:
                cur_pos = np.array(pygame.mouse.get_pos())
                # if prev_pos is not None:

                setpoint = angle_to_setpoint(np.arctan2(*(np.array(surface1.get_size())/2.0 - cur_pos)))
                # print(setpoint)
                # prev_pos = cur_pos
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