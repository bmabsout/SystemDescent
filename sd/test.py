import sd.envs
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from . import utils 
import argparse

def plot_lyapunov(lyapunov, actor, dynamics, set_point):
    pts = 200
    theta = np.linspace(-np.pi, np.pi, pts).reshape(-1,1)
    theta_dot = np.linspace(-7.0, 7.0,pts).reshape(-1,1)

    thetav, theta_dotv = np.meshgrid(theta, theta_dot)
    inputs = np.array([np.cos(thetav), np.sin(thetav), theta_dotv]).T.reshape(-1,3)
    set_points = inputs*0 + set_point
    print(inputs.shape)
    print(set_points.shape)
    z = lyapunov({"state": inputs, "setpoint": set_points}, training=False)
    acts = actor({"state":inputs, "setpoint": set_points}, training=False)
    print(actor([inputs, set_points]).shape)
    after = dynamics({"state": inputs, "action": acts, "latent": np.random.normal(size=(inputs.shape[0],)+dynamics.input["latent"].shape[1:])}, training=False)
    next_z = lyapunov({"state": after, "setpoint": set_points}, training=False)
    after = after.numpy().reshape(pts,pts,3)
    z = z.numpy().reshape(pts,pts, 1)
    next_z = next_z.numpy().reshape(pts,pts,1)
    acts = acts.numpy().reshape(pts,pts,1)
    # actor = lambda x, **kwargs: np.array([0.0])
    # res = dynamics([states,acts], training=False)

    # z = lyapunov([states, set_points], training=False)

    # plt.plot(x[:,1], lyapunov([res, set_points]) - y)
    plt.pcolormesh(thetav, theta_dotv, z.T[0][:-1, :-1], vmin=0.0, vmax=1.0)
    plt.colorbar()
    # plt.savefig(checkpoint_path + "/lyapunov_tf/lyapunov.png")


    # plt.pcolormesh(thetav, theta_dotv, acts.T[0][:-1, :-1])
    # plt.colorbar()
    plt.show()


    # plt.pcolormesh(thetav, theta_dotv, (next_z - z).T[0][:-1,:-1])
    # plt.colorbar()
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model_path", default=None)
    parser.add_argument('--random_actor', action="store_true")
    parser.add_argument('--low_actor', action="store_true")
    parser.add_argument('--seed', type=int, default=np.random.randint(100000))
    args = parser.parse_args()

    try:
        checkpoint_path = args.model if args.model else utils.latest_model()
        print("using checkpoint:", checkpoint_path)
    except:
        print("there are no trained models")
        exit()


    env_name = utils.extract_env_name(checkpoint_path)

    env = gym.make('Modeled' + env_name,
        model_path=checkpoint_path, render_mode="human")#, test=True, gui=True)

    dynamics = keras.models.load_model(checkpoint_path)

    if args.random_actor:
        actor = lambda x,**ignored: env.action_space.sample()
    elif args.low_actor:
        actor = lambda x,**ignored: env.action_space.low
    else:
        try:
            actor = keras.models.load_model(checkpoint_path + "/actor_tf")
            actor.summary()
        except:
            print(f"there is no actor trained for the model {checkpoint_path}")

    set_point_angle = 0.0
    set_point = np.array([np.cos(set_point_angle),np.sin(set_point_angle),0.0])

    try:
        lyapunov = keras.models.load_model(checkpoint_path + "/lyapunov_tf")
        plot_lyapunov(lyapunov, actor, dynamics, set_point)
    except:
        lyapunov = None


    orig_env = gym.make(env_name, render_mode="human")
    # seed = 632732 #bottom almost
    # seed = 154911 # almost rotate
    # seed = 47039 # almost rotate, then rotate
    # seed = 364366 # rotate
    print("seed:", args.seed)
    env_obs, _ = env.reset(seed=args.seed)
    # env_obs = env.env.init_with_state(np.array([0.9474508 , 0.31990144, 1.06079]))
    orig_env_obs, _ = orig_env.reset(seed=args.seed)
    def feed_obs(obs):
        return {"state": np.array([obs]), "setpoint": np.array([set_point])}

    for i in range(20000):
        # random_act = np.random.uniform(2,size=(1,))
        act = actor(feed_obs(env_obs), training=False)
        # print(orig_env_obs)
        # print(env_obs)
        if lyapunov:
            print("lyapunov", lyapunov(feed_obs(env_obs)))
        orig_act = actor(feed_obs(orig_env_obs), training=False)
        env_obs, env_reward, env_done, env_term, env_info = env.step(act)
        orig_env_obs, orig_env_reward, orig_env_done, orig_env_term, orig_env_info = orig_env.step(orig_act)
        print("error", np.mean(np.abs(env_obs-orig_env_obs)))
        env.render()
        orig_env.render()
