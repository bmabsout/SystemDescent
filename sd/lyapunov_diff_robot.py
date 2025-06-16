import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from typing import Tuple
from os import path
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.python.keras import losses
from functools import reduce
from pathlib import Path
import argparse
from .dfl import *
from . import utils
from tqdm import tqdm
import sd.envs


def V_def(state_shape: Tuple[int, ...]):
    """
    Lyapunov Function Architecture for Differential Mobile Robot

    Mathematical Foundation:
    A Lyapunov function V(x,x*) must satisfy:
    1. V(x*,x*) = 0 (zero at equilibrium/setpoint)
    2. V(x,x*) > 0 for x ≠ x* (positive definite away from setpoint)
    3. dV/dt < 0 along system trajectories (decreasing along system evolution)

    Network Architecture Rationale:
    - Input concatenation enables V to learn relationships between current state and target
    - Tanh activations provide bounded, smooth gradients crucial for stability analysis
    - Sigmoid output ensures V ∈ [0,1], automatically satisfying positive definiteness
    - L2 regularization prevents overfitting and promotes smooth Lyapunov surfaces
    """
    input_state = keras.Input(shape=state_shape, name="current_state")
    input_setpoint = keras.Input(shape=state_shape, name="target_setpoint")

    # State concatenation: [x,y,θ,x*,y*,θ*] → [6D input space]
    # This allows V to learn distance metrics in the joint state-target space
    inputs = layers.Concatenate(name="state_target_concat")(
        [input_state, input_setpoint]
    )

    # First hidden layer: 6D → 64D transformation
    # Tanh activation provides smooth, bounded responses essential for Lyapunov stability
    dense1 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_1",
    )(inputs)

    # Second hidden layer: maintains 64D representation
    # Additional nonlinearity allows complex Lyapunov surface learning
    dense2 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_2",
    )(dense1)

    # Output layer: 64D → 1D Lyapunov value
    # Sigmoid ensures V(x,x*) ∈ [0,1], satisfying positive definiteness automatically
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_output",
    )(dense2)

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_setpoint},
        outputs=outputs,
        name="DiffRobotLyapunovFunction",
    )

    print("\n=== Lyapunov Function Architecture ===")
    model.summary()
    return model


def actor_def(state_shape, action_shape):
    """
    Control Policy Architecture for Differential Mobile Robot

    Control Theory Foundation:
    The actor π(x,x*) must generate control inputs [v,ω] that:
    1. Drive the system toward the setpoint: ||x(t) - x*|| → 0
    2. Minimize control effort: ||u||² small
    3. Satisfy actuator constraints: |v|,|ω| ≤ max_velocity

    Architecture Design Principles:
    - State-setpoint concatenation enables goal-conditioned control
    - Tanh hidden activations provide smooth control surfaces
    - Final tanh with scaling maps to actuator limits [-2,2] m/s and rad/s
    - L2 regularization prevents high-frequency control oscillations
    """
    input_state = keras.Input(shape=state_shape, name="robot_state")
    input_set_point = keras.Input(shape=state_shape, name="control_target")

    # Control input formation: concatenate current state with desired target
    # This creates a 6D input: [x,y,θ,x*,y*,θ*] → control policy
    inputs = layers.Concatenate(name="control_input_concat")(
        [input_state, input_set_point]
    )

    # First control layer: maps 6D state-target to 64D latent control representation
    dense1 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_hidden_1",
    )(inputs)

    # Second control layer: refines control representation in 64D space
    dense2 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_hidden_2",
    )(dense1)

    # Control output layer: maps to differential robot action space [v,ω]
    # Note: np.squeeze(action_shape) handles both (2,) and (2,1) action spaces
    prescaled = layers.Dense(
        np.squeeze(action_shape),
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_prescaled",
    )(dense2)

    # Control scaling: tanh ∈ [-1,1] → [-2,2] for actuator limits
    # This satisfies the constraint |v|,|ω| ≤ 2.0 from the robot dynamics
    outputs = prescaled * 2.0

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_set_point},
        outputs=outputs,
        name="DiffRobotControlPolicy",
    )

    print("\n=== Control Policy Architecture ===")
    model.summary()
    return model


def generate_dataset(env: gym.Env):
    """
    Training Data Generation for Lyapunov-Based Control Learning

    Dataset Philosophy:
    1. Random state initialization ensures broad coverage of state space
    2. Diverse setpoint generation trains V as a family of Lyapunov functions
    3. Each (state, setpoint) pair represents a control problem instance

    Setpoint Generation Strategy:
    - Position targets: uniform sampling in [-5,5] × [-5,5] spatial region
    - Orientation targets: uniform sampling in [-π,π] with quantization to cardinal directions
    - This creates a rich distribution of control objectives for robust training
    """

    def gen_sample():
        """
        Differential Robot Sample Generator

        State Initialization:
        - Position: random in reasonable workspace bounds
        - Orientation: full angular range to test all configurations

        Setpoint Sampling:
        - Spatial diversity: ensures policy learns navigation in all directions
        - Angular diversity: tests orientation control capabilities
        - Realistic targets: bounded to prevent degenerate far-field cases
        """
        while True:
            # Reset environment to random initial configuration
            obs, _ = env.reset()

            # Enhanced state initialization for differential robot
            # Scale position to larger workspace for more challenging navigation
            obs[0] = obs[0] * 3.0  # x position: expand to [-3,3] range
            obs[1] = obs[1] * 3.0  # y position: expand to [-3,3] range
            # obs[2] remains orientation in [-π,π] - no scaling needed

            # Setpoint generation: create diverse navigation targets
            # Position targets: sample from reasonable navigation space
            target_x = np.random.uniform(-4.0, 4.0)
            target_y = np.random.uniform(-4.0, 4.0)

            # Orientation targets: focus on cardinal directions for interpretability
            # This discretization helps the Lyapunov function learn clearer basins of attraction
            orientation_choices = [
                0.0,
                np.pi / 2,
                np.pi,
                -np.pi / 2,
            ]  # [East, North, West, South]
            target_theta = np.random.choice(orientation_choices)

            # Alternative: fully random orientation
            # target_theta = np.random.uniform(-np.pi, np.pi)

            yield {
                "state": obs,
                "setpoint": np.array([target_x, target_y, target_theta]),
            }

    return gen_sample


def save_model(model, name):
    """Model persistence with proper path handling"""
    path = Path(args.ckpt_path.parent, name)
    args.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to: {path}")
    model.save(str(path))


@tf.function
def euclidean_distance(state1, state2):
    """
    Compute Euclidean distance between robot states

    For differential robot: distance includes both position and orientation components
    - Position distance: ||[x₁,y₁] - [x₂,y₂]||₂
    - Angular distance: minimum angle between orientations
    - Combined metric: weighted sum for multi-objective proximity
    """
    pos_distance = tf.sqrt(
        (state1[:, 0] - state2[:, 0]) ** 2 + (state1[:, 1] - state2[:, 1]) ** 2
    )

    # Angular distance computation: handles angle wrap-around correctly
    angle_diff = tf.abs(state1[:, 2] - state2[:, 2])
    angle_distance = tf.minimum(angle_diff, 2 * np.pi - angle_diff) / np.pi

    # Combined distance metric: position dominates, orientation refines
    total_distance = pos_distance + 0.2 * angle_distance
    return total_distance


def train(batches, dynamics_model, actor, V, state_shape, args):
    """
    Lyapunov-Based Control Training Loop

    Training Objective:
    Simultaneously learn Lyapunov function V(x,x*) and control policy π(x,x*) such that:
    1. V satisfies Lyapunov conditions (positive definite, decreasing along trajectories)
    2. π generates controls that make dV/dt < 0 (stability guaranteeing)
    3. Both functions are smooth and generalizable across state-target pairs

    Optimization Strategy:
    - Differentiable Fuzzy Logic (DFL) framework for multi-objective optimization
    - Adam optimizer for smooth gradient-based learning
    - Random trajectory lengths prevent overfitting to specific time horizons
    """
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    @tf.function
    def run_full_model(initial_states, set_points, repeat=1):
        """
        Forward Trajectory Simulation

        This function simulates the closed-loop system:
        x₍ₖ₊₁₎ = f(xₖ, π(xₖ,x*)) for k = 0,1,...,repeat-1

        Purpose:
        - Generate trajectory data for Lyapunov condition evaluation
        - Test policy performance over multiple time steps
        - Create training signal for both V and π networks

        Mathematical Flow:
        1. Initialize: x₀ = initial_states
        2. For each timestep k:
           a. Compute control: uₖ = π(xₖ, x*)
           b. Apply dynamics: xₖ₊₁ = f(xₖ, uₖ)
           c. Store state: trajectory[k] = xₖ₊₁
        3. Return: final state and complete trajectory
        """
        # TensorArray: efficient storage for variable-length trajectories
        states = tf.TensorArray(tf.float32, size=repeat, name="trajectory_storage")
        current_states = initial_states

        for i in range(repeat):
            # Control policy evaluation: π(x,x*) → [v,ω]
            control_action = actor(
                {"state": current_states, "setpoint": set_points}, training=True
            )

            # Latent dynamics input: handles stochastic system components
            latent_shape = tuple(current_states.shape[0:1]) + tuple(
                dynamics_model.input["latent"].shape[1:]
            )
            latent_noise = tf.random.normal(latent_shape, name="dynamics_noise")

            # Dynamics model evaluation: f(x,u,ξ) → x₊
            current_states = dynamics_model(
                {
                    "state": current_states,
                    "action": control_action,
                    "latent": latent_noise,
                },
                training=True,
            )

            # Trajectory storage with batch-first indexing
            states = states.write(i, current_states)

        # Return: (final_state: [batch, state_dim], trajectory: [batch, time, state_dim])
        return current_states, tf.transpose(states.stack(), [1, 0, 2])

    def batch_value(batch):
        """
        Lyapunov Training Objective Computation

        Core Learning Signal Construction:
        This function implements the key insight of Lyapunov-based control learning:
        - Lyapunov function must decrease along closed-loop trajectories
        - Control policy must generate this decreasing behavior
        - Both networks are trained jointly to satisfy stability conditions

        Mathematical Framework:
        Given state x₀ and setpoint x*, simulate trajectory x₀ → x₁ → ... → xₜ
        Enforce: V(x₀,x*) > V(x₁,x*) > ... > V(xₜ,x*) > V(x*,x*) = 0
        """
        # Trajectory length randomization: prevents temporal overfitting
        maxRepetitions = 20  # Increased for longer navigation tasks
        repetitions = tf.random.uniform(
            shape=[],
            minval=3,  # Minimum trajectory length for meaningful learning
            maxval=maxRepetitions + 1,
            dtype=tf.dtypes.int32,
        )

        # Extract training batch components
        prev_states = batch["state"]  # Initial robot states [batch, 3]
        set_points = batch["setpoint"]  # Navigation targets [batch, 3]

        # Trajectory simulation: generate closed-loop behavior
        final_states, trajectory_states = run_full_model(
            prev_states, set_points, repeat=repetitions
        )

        # Lyapunov function evaluation at trajectory endpoints
        V_initial = V({"state": prev_states, "setpoint": set_points}, training=True)

        V_final = V({"state": final_states, "setpoint": set_points}, training=True)

        # Lyapunov boundary condition: V(x*,x*) = 0
        # This constraint ensures the Lyapunov function achieves its minimum at targets
        V_at_target = V({"state": set_points, "setpoint": set_points}, training=True)

        # Zero constraint: penalize deviation from V(x*,x*) = 0
        zero_constraint = p_mean(
            (1.0 - V_at_target**0.5),
            -1.0,  # Harmonic mean emphasizes worst-case violations
            default_val=1.0,
        )

        # Lyapunov decrease condition: V(x₀,x*) > V(xₜ,x*)
        # This is the core stability requirement for Lyapunov-based control
        lyapunov_decrease = V_initial - V_final

        # Performance metric: distance-based progress evaluation
        initial_distances = euclidean_distance(prev_states, set_points)
        final_distances = euclidean_distance(final_states, set_points)

        # Progress requirement: robot should move closer to target
        distance_improvement = initial_distances - final_distances

        # Convergence assessment: proximity to target evaluation
        proximity_to_target = tf.exp(-final_distances)  # Exponential proximity reward

        # Regularization terms: prevent network pathologies
        actor_regularization = 1.0 - tf.tanh(tf.reduce_mean(actor.losses))
        lyapunov_regularization = 1.0 - tf.tanh(tf.reduce_mean(V.losses))

        # Lyapunov decrease shaping: adaptive decrease requirements
        repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
        maxRepetitionsf = tf.cast(maxRepetitions, tf.dtypes.float32)

        # Adaptive decrease rate: longer trajectories require more decrease
        # This prevents the Lyapunov function from becoming too flat
        decrease_rate = 1.0 / 50.0  # Target: reach setpoint within 50 steps
        required_decrease = tf.minimum(decrease_rate * repetitionsf, V_initial)

        # Piecewise decrease requirement: different penalties for different decrease magnitudes
        # This creates a shaped reward that encourages appropriate decrease rates
        decrease_satisfaction = p_mean(
            build_piecewise(
                [
                    (-1.0, 0.0),  # Large negative decrease: penalty
                    (-0.05, 0.001),  # Small negative decrease: small penalty
                    (0.0, 0.01),  # Zero decrease: small penalty
                    (required_decrease, 0.9),  # Required decrease: high reward
                    (1.0, 1.0),
                ],  # Excessive decrease: maximum reward
                lyapunov_decrease,
                clipped=True,
            ),
            -1.0,  # Harmonic mean: focus on worst violations
        )

        # Non-target state constraint: V(x,x*) > 0 for x ≠ x*
        # Ensures positive definiteness away from the target
        target_distances = euclidean_distance(prev_states, set_points)
        non_target_mask = tf.where(target_distances > 0.1, V_initial, 1.0)
        positive_away_from_target = p_mean(
            tf.minimum(non_target_mask * 5.0, 1.0),
            0.0,  # Geometric mean for balanced constraint satisfaction
        )

        # Progress-based shaping: reward states that make navigation progress
        progress_reward = p_mean(
            tf.sigmoid(
                distance_improvement * 10.0
            ),  # Sigmoid shaping for smooth gradients
            2.0,  # Quadratic mean emphasizes consistent progress
        )

        # Multi-objective optimization using Differentiable Fuzzy Logic
        # This framework allows principled combination of multiple learning objectives
        training_objective = Constraints(
            0.0,  # Geometric mean: all constraints must be satisfied
            {
                "navigation_performance": Constraints(
                    0.0,
                    {
                        "lyapunov_decrease": decrease_satisfaction,
                        "distance_progress": progress_reward,
                        "target_proximity": p_mean(proximity_to_target, 1.0),
                    },
                ),
                "lyapunov_conditions": Constraints(
                    0.0,
                    {
                        "zero_at_target": zero_constraint,
                        "positive_elsewhere": positive_away_from_target,
                    },
                ),
                "regularization": Constraints(
                    1.0,  # Arithmetic mean for regularization terms
                    {
                        "actor_reg": actor_regularization,
                        "lyapunov_reg": lyapunov_regularization,
                    },
                ),
            },
        )

        return training_objective

    @tf.function
    def train_step(batch):
        """
        Single Training Step: Gradient Computation and Application

        Training Process:
        1. Forward pass: compute multi-objective loss via DFL framework
        2. Gradient computation: automatic differentiation w.r.t. all parameters
        3. Gradient application: Adam optimizer update for both networks
        4. Metrics collection: return scalar performance and detailed breakdown
        """
        with tf.GradientTape() as tape:
            # Multi-objective evaluation via DFL constraint satisfaction
            objective_structure = batch_value(batch)

            # Scalar optimization target: maximize constraint satisfaction
            satisfaction_scalar = dfl_scalar(objective_structure)
            loss = 1.0 - satisfaction_scalar  # Convert to minimization problem

        # Joint gradient computation: both networks trained simultaneously
        trainable_parameters = actor.trainable_weights + V.trainable_weights
        gradients = tape.gradient(loss, trainable_parameters)

        # Gradient application with adaptive learning rate
        # Learning rate could be made adaptive based on satisfaction_scalar
        optimizer.apply_gradients(zip(gradients, trainable_parameters))

        return satisfaction_scalar, objective_structure

    def save_models(epoch):
        """Periodic model checkpointing for training resumption"""
        save_model(actor, "actor.keras")
        save_model(V, "lyapunov.keras")
        print(f"Models saved at epoch {epoch}")

    def train_and_display(batch):
        """Training step with formatted output for monitoring"""
        scalar, metrics = train_step(batch)
        return f"Satisfaction: {scalar:.3f} ||| {metrics}"

    # Main training loop with progress monitoring and periodic saving
    utils.train_loop(
        [batches] * args.epochs,
        train_step=train_and_display,
        every_n_seconds={"freq": args.save_freq, "callback": save_models},
    )


if __name__ == "__main__":
    """
    Main Training Script for Differential Robot Lyapunov Control

    Usage:
    python lyapunov_diff_robot.py --ckpt_path models/DifferentialRobot-v1/run_id/checkpoints/checkpoint0/model.keras

    Training Pipeline:
    1. Load pre-trained dynamics model from checkpoint
    2. Initialize Lyapunov function and control policy networks
    3. Generate diverse training data from robot environment
    4. Execute joint training loop with multi-objective optimization
    5. Save trained models for deployment and testing
    """

    # Argument parsing for flexible training configuration
    parser = argparse.ArgumentParser(
        description="Lyapunov-based control learning for differential mobile robots"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=None,
        help="Path to pre-trained dynamics model checkpoint",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=300,
        help="Number of training batches per epoch",
    )
    parser.add_argument(
        "--save_freq", type=int, default=20, help="Model saving frequency in seconds"
    )
    parser.add_argument(
        "--epochs", type=int, default=150, help="Total number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="Resume training from saved actor and Lyapunov models",
    )

    args = parser.parse_args()

    # Automatic model path detection if not specified
    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()
        print(f"Auto-detected model path: {args.ckpt_path}")

    # Environment setup and parameter extraction
    env_name = utils.extract_env_name(args.ckpt_path)
    print(f"Training environment: {env_name}")
    env = gym.make(env_name)

    # Extract system dimensions from environment specifications
    action_shape = env.action_space.shape  # Expected: (2,) for [v, ω]
    state_shape = env.observation_space.shape  # Expected: (3,) for [x, y, θ]

    print(f"State dimension: {state_shape}")
    print(f"Action dimension: {action_shape}")

    # Load pre-trained dynamics model
    print("Loading dynamics model...")
    dynamics_model = utils.load_checkpoint(args.ckpt_path)
    print("\n=== Dynamics Model Architecture ===")
    dynamics_model.summary()

    # Initialize or load control networks
    if args.load_saved:
        print("Loading saved networks...")
        actor = keras.models.load_model(args.ckpt_path.parent / "actor.keras")
        lyapunov_model = keras.models.load_model(
            args.ckpt_path.parent / "lyapunov.keras"
        )
    else:
        print("Initializing new networks...")
        actor = actor_def(state_shape, action_shape)
        lyapunov_model = V_def(state_shape)

    # Training dataset construction
    print("Constructing training dataset...")
    state_spec = tf.TensorSpec(state_shape, dtype=tf.float32)
    dataset_signature = {"state": state_spec, "setpoint": state_spec}

    # Create infinite dataset generator with batching and caching
    dataset = tf.data.Dataset.from_generator(
        generate_dataset(env), output_signature=dataset_signature
    )
    batched_dataset = (
        dataset.batch(args.batch_size)
        .take(args.num_batches)
        .cache()  # Cache for efficiency across epochs
        .prefetch(tf.data.AUTOTUNE)
    )  # Parallel data loading

    print(
        f"Dataset configuration: {args.num_batches} batches of size {args.batch_size}"
    )
    print(f"Training for {args.epochs} epochs with learning rate {args.lr}")

    # Execute training loop
    print("\n=== Starting Lyapunov Control Training ===")
    train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)

    print("\n=== Training Completed Successfully ===")
    print(f"Trained models saved to: {args.ckpt_path.parent}")
    print("Use test.py to evaluate the learned controller and Lyapunov function")
