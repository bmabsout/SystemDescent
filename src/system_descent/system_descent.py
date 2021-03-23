import numpy as np
import tensorflow as tf
import gym
import time
from rl_smoothness.utils.logx import EpochLogger

def load_dynamics():
    local_path = "saved/ff44f9/checkpoints/checkpoint50.tf"
    dynamics = keras.models.load_model("/home/bmabsout/Documents/gymfc-nf1/training_code/neuroflight_trainer/dynamics_learning/"+local_path)
    return dynamics

def policy_model(obs_space, act_space, hidden_sizes=[32,32]):
    if(not (isinstance(act_space, gym.spaces.Box) and isinstance(obs_space, gym.spaces.Box))):
        raise NotImplementedError
    state_input = keras.Input(shape=(observation_space.shape[0],))
    dense = state_input
    for hidden_size in hidden_sizes:
        dense = layers.Dense(hidden_size, activation="relu")(dense)
    dense = layers.Dense(act_space.shape[0], activation="sigmoid")(dense)
    output = layers.Lambda(lambda x: x * act_space.shape[0])(dense)
    model = keras.Model(inputs=state_input, outputs=output, name="policy")
    model.summary()
    return model


def system_descent(dynamics, policy, reward, state_space, act_space, seed=0, 
        ep_len=200, epochs=50, batch_size=512, policy_lr=0.01
        logger_kwargs=dict(), save_freq=1, num_test_episodes=10
        ):
    """
    System Descent


    Args:
        dynamics : A keras model with the following signature state x action -> state

        state_space: A gym.Space representing the set of all states of the system

        act_space: A gym.Space representing the set of all actions an actor can take

        policy: A keras model with the following signature: state -> action

        reward: A tensorflow function with the following signature: [[state x action]] -> R

        seed (int): Seed for random number generators.

        steps_per_episode (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each episode.

        epochs (int): Number of epochs to run and train agent.

        batch_size (int): Minibatch size for Gradient Descent.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy.
    """

    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)


    start_time = time.time()

    o, ep_ret, ep_len = env.reset(), 0, 0
    total_steps = steps_per_epoch * epochs
    def run_full_model():
        initial_state = keras.Input(shape=state_space.shape)
        states = [initial_state]
        for i in range(steps_per_episode):
            states.append(dynamics([states[i],policy(states[i])]))
        full_loop = keras.Model(inputs=initial_state, outputs=states)

        full_loop.summary()
        return full_loop

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        # if t % (200*10) < 40:
        #     env.render()
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             xnxt_ph: batch['obs2'],
                             xnxt2_dummy_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }

                if lam_s > 0:
                    feed_dict[xbar_ph] = np.random.normal(feed_dict[x_ph], eps_s)
                
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='Pendulum-v0')
    # parser.add_argument('--hid', type=int, default=256)
    # parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--a_hid_size', nargs='+', type=int, default=[32,32])
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_ep_len', type=int, default=200)
    parser.add_argument('--exp_name', type=str, default='system_descent')
    args = parser.parse_args()

    from rl_smoothness.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    env = gym.make("Pendulum-v0")
    system_descent(load_nn(), a_hidden_sizes=args.a_hid_size
        , logger_kwargs=logger_kwargs, **dict(args))
