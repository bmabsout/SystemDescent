import itertools
import numpy as np
import gym
import time
import sd.rl.sac_core as core
import tensorflow as tf

from sd.rl.log_utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, num_opt_steps=30, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, on_save=lambda *_:(), anchor_q=None):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    with tf.name_scope('main'):
        ac = actor_critic(obs_dim, env.action_space, **ac_kwargs)

    with tf.name_scope('target'):
        ac_targ = actor_critic(obs_dim, env.action_space, **ac_kwargs)

    # make sure network and target network is using the same weights
    ac_targ.pi.set_weights(ac.pi.get_weights())
    ac_targ.q1.set_weights(ac.q1.get_weights())
    ac_targ.q2.set_weights(ac.q2.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    @tf.function
    def q_update(o, a, r, o2, d):
        # Bellman backup for Q functions
        a2, logp_a2 = ac.pi.get_pi_logpi(o2)

        # Target Q-values
        q1_pi_targ = tf.squeeze(ac_targ.q1(tf.concat([o2, a2], axis=-1)), axis=-1)
        q2_pi_targ = tf.squeeze(ac_targ.q2(tf.concat([o2, a2], axis=-1)), axis=-1)
        q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        with tf.GradientTape() as tape:
            q1 = tf.squeeze(ac.q1(tf.concat([o, a], axis=-1)), axis=-1)
            q2 = tf.squeeze(ac.q2(tf.concat([o, a], axis=-1)), axis=-1)

            # MSE loss against Bellman backup
            loss_q1 = tf.reduce_mean((q1 - backup)**2)
            loss_q2 = tf.reduce_mean((q2 - backup)**2)
            loss_q = loss_q1 + loss_q2

        qs_trainable_vars = ac.q1.trainable_variables + ac.q2.trainable_variables
        grads = tape.gradient(loss_q, qs_trainable_vars)
        q_optimizer.apply_gradients(zip(grads, qs_trainable_vars))

        return loss_q


    # Set up optimizers for policy and q-function
    pi_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # Set up function for computing SAC pi loss
    @tf.function
    def pi_update(obs):
        with tf.GradientTape() as tape:
            pi, logp_pi = ac.pi.get_pi_logpi(obs)
            q1_pi = tf.squeeze(ac.q1(tf.concat([obs, pi], axis=-1)), axis=-1)
            q2_pi = tf.squeeze(ac.q2(tf.concat([obs, pi], axis=-1)), axis=-1)
            q_pi = tf.minimum(q1_pi, q2_pi)
            if anchor_q:
                anchor_pi = tf.squeeze(anchor_q(tf.concat([obs, pi], axis=-1)), axis=-1)
            else:
                anchor_pi = q_pi*0

            # Entropy-regularized policy loss
            loss_pi = tf.reduce_mean(alpha * logp_pi - q_pi - anchor_pi)

        grads = tape.gradient(loss_pi, ac.pi.trainable_variables)
        pi_optimizer.apply_gradients(zip(grads, ac.pi.trainable_variables))

        return loss_pi

    @tf.function
    def target_update(ac, ac_targ):
        for v_main, v_targ in zip(ac.pi.variables, ac_targ.pi.variables):
            v_targ.assign(polyak*v_targ + (1-polyak)*v_main)
        for v_main, v_targ in zip(ac.q1.variables, ac_targ.q1.variables):
            v_targ.assign(polyak*v_targ + (1-polyak)*v_main)
        for v_main, v_targ in zip(ac.q2.variables, ac_targ.q2.variables):
            v_targ.assign(polyak*v_targ + (1-polyak)*v_main)

    def update(data):
        o, a, r, o2, d = data['obs1'], data['acts'], data['rews'], data['obs2'], data['done']
        # First run one gradient descent step for Q1 and Q2
        loss_q = q_update(o, a, r, o2, d)
        # print(loss_q)

        # Record things
        logger.store(LossQ=loss_q.numpy())

        # Next run one gradient descent step for pi.
        loss_pi = pi_update(o)

        # Record things
        logger.store(LossPi=loss_pi.numpy())

        # Finally, update target networks by polyak averaging.
        target_update(ac, ac_targ)

    def get_action(o, deterministic=False):
        obs = tf.constant(o.reshape(1,-1))
        if deterministic:
            return ac.pi.deterministic_actor(obs).numpy()[0]
        else:
            pi_action, _ = ac.pi.get_pi_logpi(obs)
            return pi_action.numpy()[0]

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = env.step(get_action(o, deterministic=True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
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
            for j in range(num_opt_steps):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                on_save(ac, epoch//save_freq, replay_buffer)

            # Test the performance of the deterministic version of the agent.
            # test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

    return ac.pi.deterministic_actor

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from sd.rl.log_utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    sac(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(a_hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
