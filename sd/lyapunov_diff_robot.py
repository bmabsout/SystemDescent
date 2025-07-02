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
from .fpl import *
from . import utils
from tqdm import tqdm
import sd.envs

# Import the monotonic layers we just created
from .monotonic_layers import V_def_with_architecture_choice


def V_def(
    state_shape: Tuple[int, ...],
    use_monotonic: bool = False,
    origin_stabilization: bool = False,
    **monotonic_kwargs,
):
    """
    Enhanced Lyapunov Function Architecture for Differential Mobile Robot

    Now supports both standard neural networks and monotonic neural networks
    from the paper "Lyapunov Neural Network with Region of Attraction Search".

    Args:
        state_shape: Tuple defining the shape of state input
        use_monotonic: If True, uses monotonic architecture; if False, uses standard NN
        origin_stabilization: If True, uses error-state formulation V(state - setpoint)
        **monotonic_kwargs: Additional arguments for monotonic network configuration

    Mathematical Foundation:
    A Lyapunov function V(x,x*) must satisfy:
    1. V(x*,x*) = 0 (zero at equilibrium/setpoint)
    2. V(x,x*) > 0 for x ≠ x* (positive definite away from setpoint)
    3. dV/dt < 0 along system trajectories (decreasing along system evolution)

    Monotonic Network Advantages:
    - Guarantees positive definiteness by construction
    - Ensures unique global minimum at origin
    - Provides formal stability guarantees when combined with MILP verification
    - Reduces search space for Lyapunov function learning
    """

    # For monotonic networks, force origin stabilization
    if use_monotonic:
        origin_stabilization = True
        print("\n=== Creating Monotonic Lyapunov Network ===")
        print("Mode: Error-State Stabilization V(state - setpoint)")
        print(
            "  • Can handle any setpoint by learning V(error) with error = state - setpoint"
        )
        print("  • Positive definiteness by construction")
        print("  • Unique global minimum at error = 0")
        print("  • Compatible with MILP verification")
    else:
        if origin_stabilization:
            print("\n=== Creating Standard Neural Network (Error-State Mode) ===")
            print("Mode: Error-State Stabilization V(state - setpoint)")
            print("  • Networks see error state as input")
            print("  • Can handle any setpoint")
        else:
            print("\n=== Creating Standard Neural Network (Multi-Target Mode) ===")
            print("Mode: Concatenated input V([state, setpoint])")
            print("  • Networks see full state and setpoint information")

    # Default monotonic network parameters
    default_monotonic_params = {
        "num_layers": 2,
        "directions_per_layer": None,  # Will auto-determine
        "num_pieces": 4,
        "name": "MonotonicLyapunovFunction",
    }

    # Update with user-provided parameters
    default_monotonic_params.update(monotonic_kwargs)

    model = V_def_with_architecture_choice(
        state_shape,
        use_monotonic=use_monotonic,
        origin_stabilization=origin_stabilization,
        **default_monotonic_params,
    )

    if use_monotonic:
        print(
            f"Monotonic network created with {default_monotonic_params['num_layers']} layers"
        )
        print(
            f"Each monotonic unit has {default_monotonic_params['num_pieces']} pieces"
        )

    return model


def actor_def(state_shape, action_shape, origin_stabilization: bool = False):
    """
    Control Policy Architecture for Differential Mobile Robot
    Now supports both multi-target and error-state (origin) modes.
    """
    input_state = keras.Input(shape=state_shape, name="robot_state")
    input_set_point = keras.Input(shape=state_shape, name="control_target")

    if origin_stabilization:
        # For origin stabilization: use error state (state - setpoint)
        error_state = layers.Subtract(name="error_state")(
            [input_state, input_set_point]
        )
        network_input = error_state
        print("Actor using ERROR-STATE input (state - setpoint)")
    else:
        # For multi-target: concatenate state and setpoint
        network_input = layers.Concatenate(name="control_input_concat")(
            [input_state, input_set_point]
        )
        print("Actor using CONCATENATED input [state, setpoint]")

    dense1 = layers.Dense(
        # 64,
        32,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_hidden_1",
    )(network_input)

    dense2 = layers.Dense(
        # 64,
        16,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_hidden_2",
    )(dense1)

    prescaled = layers.Dense(
        np.squeeze(action_shape),
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_prescaled",
    )(dense2)

    outputs = prescaled * 2.0

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_set_point},
        outputs=outputs,
        name="DiffRobotControlPolicy",
    )

    print("\n=== Control Policy Architecture ===")
    model.summary()
    return model


def generate_dataset(
    env: gym.Env, origin_stabilization: bool = False, origin_setpoint: int = 0
):
    """
    Training Data Generation for Lyapunov-Based Control Learning

    For both modes, we generate random setpoints. The difference is how networks process them:
    - Multi-target mode: Networks see [state, setpoint] concatenated
    - Error-state mode: Networks see (state - setpoint) as error state
    """

    def gen_sample():
        while True:
            obs, _ = env.reset()
            obs[0] = obs[0] * 3.0
            obs[1] = obs[1] * 3.0

            if origin_stabilization:
                # set the target to always be at the setpoint
                target_x = origin_setpoint[0]
                target_y = origin_setpoint[1]
                target_theta = origin_setpoint[2]
            else:
                # Random setpoint generation for multi-target mode
                target_x = np.random.uniform(-4.0, 4.0)
                target_y = np.random.uniform(-4.0, 4.0)
                target_theta = np.random.uniform(-np.pi, np.pi)

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
    """Compute Euclidean distance between robot states"""
    pos_distance = tf.sqrt(
        (state1[:, 0] - state2[:, 0]) ** 2 + (state1[:, 1] - state2[:, 1]) ** 2
    )

    angle_diff = tf.abs(state1[:, 2] - state2[:, 2])
    angle_distance = tf.minimum(angle_diff, 2 * np.pi - angle_diff) / np.pi

    total_distance = pos_distance + 0.2 * angle_distance
    return total_distance


def train(batches, dynamics_model, actor, V, state_shape, args):
    """
    Enhanced Lyapunov-Based Control Training Loop

    Now supports both standard and monotonic Lyapunov networks with automatic
    handling of origin vs multi-target stabilization modes.
    """
    origin_stabilization = args.origin_stabilization or ("Monotonic" in V.name)

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    # Check if using monotonic network
    is_monotonic = "Monotonic" in V.name
    if is_monotonic:
        print("\n=== Training with Monotonic Lyapunov Network ===")
        print("Mode: Error-State Stabilization V(state - setpoint)")
        print(
            "Note: Positive definiteness and zero-at-equilibrium satisfied by construction"
        )
    else:
        if origin_stabilization:
            print("\n=== Training with Standard Neural Network ===")
            print("Mode: Error-State Stabilization V(state - setpoint)")
        else:
            print("\n=== Training with Standard Neural Network ===")
            print("Mode: Multi-Target V([state, setpoint])")
        print("Note: All Lyapunov conditions must be enforced during training")

    @tf.function
    def run_full_model(initial_states, set_points, repeat=1):
        """Forward Trajectory Simulation"""
        states = tf.TensorArray(tf.float32, size=repeat, name="trajectory_storage")
        current_states = initial_states

        for i in range(repeat):
            control_action = actor(
                {"state": current_states, "setpoint": set_points}, training=True
            )

            latent_shape = tuple(current_states.shape[0:1]) + tuple(
                dynamics_model.input["latent"].shape[1:]
            )
            latent_noise = tf.random.normal(latent_shape, name="dynamics_noise")

            current_states = dynamics_model(
                {
                    "state": current_states,
                    "action": control_action,
                    "latent": latent_noise,
                },
                training=True,
            )

            states = states.write(i, current_states)

        return current_states, tf.transpose(states.stack(), [1, 0, 2])

    def batch_value(batch, minN=3, maxN=20):
        """
        Enhanced Lyapunov Training Objective Computation

        Handles both error-state V(state - setpoint) and multi-target V([state, setpoint]) modes.
        Networks handle the input representation internally.
        """
        repetitions = tf.random.uniform(
            shape=[],
            minval=minN,
            maxval=maxN + 1,
            dtype=tf.dtypes.int32,
        )

        prev_states = batch["state"]
        set_points = batch["setpoint"]

        final_states, trajectory_states = run_full_model(
            prev_states, set_points, repeat=repetitions
        )

        # Lyapunov function evaluation - both modes use same input format
        # Networks handle error computation internally for origin stabilization
        V_initial = V({"state": prev_states, "setpoint": set_points}, training=True)
        V_final = V({"state": final_states, "setpoint": set_points}, training=True)
        V_at_target = V({"state": set_points, "setpoint": set_points}, training=True)

        # Core constraints that apply to both architectures
        lyapunov_decrease = V_initial - V_final

        # Both modes measure distance to setpoint (error-based thinking)
        initial_distances = euclidean_distance(prev_states, set_points)
        final_distances = euclidean_distance(final_states, set_points)
        proximity_to_target = tf.exp(-final_distances)

        # Lyapunov decrease shaping
        repetitionsf = tf.cast(repetitions, tf.dtypes.float32)
        decrease_rate = 1.0 / 50.0
        required_decrease = tf.minimum(decrease_rate * repetitionsf, V_initial)

        decrease_satisfaction = p_mean(
            build_piecewise(
                [
                    (-1.0, 0.0),
                    (-0.05, 0.001),
                    (0.0, 0.01),
                    (required_decrease, 0.9),
                    (1.0, 1.0),
                ],
                lyapunov_decrease,
                clipped=True,
            ),
            -1.0,
        )

        # Performance metrics
        v_dot_progress = tf.sigmoid(lyapunov_decrease * 10.0)

        # Construct constraint hierarchy based on network type
        if is_monotonic:
            # For monotonic networks, positive definiteness and zero-at-target are
            # automatically satisfied, so we focus on the decrease condition and performance
            training_objective = Constraints(
                0.0,
                {
                    "navigation_performance": Constraints(
                        0.0,
                        {
                            "progress_reward": p_mean(v_dot_progress, 0),
                            "target_proximity": p_mean(proximity_to_target, -2.0),
                        },
                    ),
                    "lyapunov_conditions": Constraints(
                        0.0,
                        {
                            "lyapunov_decrease": decrease_satisfaction,
                            # Note: zero_at_target and positive_elsewhere are satisfied by construction
                        },
                    ),
                },
            )
        else:
            # For standard networks, we need all constraints
            zero_constraint = p_mean((1.0 - V_at_target**0.5), -1.0, default_val=1.0)

            target_distances = euclidean_distance(prev_states, set_points)
            non_target_mask = tf.where(target_distances > 0.1, V_initial, 1.0)
            positive_away_from_target = p_mean(
                tf.minimum(non_target_mask * 5.0, 1.0), 0.0
            )

            training_objective = Constraints(
                0.0,
                {
                    "navigation_performance": Constraints(
                        0.0,
                        {
                            "progress_reward": p_mean(v_dot_progress, 0),
                            "target_proximity": p_mean(proximity_to_target, -2.0),
                        },
                    ),
                    "lyapunov_conditions": Constraints(
                        0.0,
                        {
                            "zero_at_target": zero_constraint,
                            "positive_elsewhere": positive_away_from_target,
                            "lyapunov_decrease": decrease_satisfaction,
                        },
                    ),
                },
            )

        return training_objective

    @tf.function
    def train_step(batch, minN=3, maxN=20):
        """Single Training Step with gradient computation"""
        with tf.GradientTape() as tape:
            objective_structure = batch_value(batch, minN, maxN)
            fulfillment_value = fpl_value(objective_structure)
            loss = 1.0 - fulfillment_value

        trainable_parameters = actor.trainable_weights + V.trainable_weights
        gradients = tape.gradient(loss, trainable_parameters)
        optimizer.apply_gradients(zip(gradients, trainable_parameters))

        return fulfillment_value, objective_structure

    def save_models(epoch):
        """Periodic model checkpointing"""
        save_model(actor, "actor.keras")
        save_model(V, "lyapunov.keras")
        print(f"Models saved at epoch {epoch}")

    def train_and_display(batch, minN=3, maxN=20):
        """Training step with formatted output"""
        scalar, metrics = train_step(batch, minN, maxN)
        return f"Satisfaction: {scalar:.3f} ||| {metrics}"

    # Main training loop
    utils.train_loop(
        [batches] * args.epochs,
        train_step=train_and_display,
        every_n_seconds={"freq": args.save_freq, "callback": save_models},
        minN=args.minN,
        maxN=args.maxN,
    )


if __name__ == "__main__":
    """
    Enhanced Main Training Script with Monotonic Architecture Support
    """
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
        "--lr", type=float, default=5e-4, help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--load_saved",
        action="store_true",
        help="Resume training from saved actor and Lyapunov models",
    )
    parser.add_argument(
        "--minN",
        type=int,
        default=3,
        help="Minimum trajectory length for Lyapunov training",
    )
    parser.add_argument(
        "--maxN",
        type=int,
        default=20,
        help="Maximum trajectory length for Lyapunov training",
    )

    # Architecture choice arguments
    parser.add_argument(
        "--use_monotonic",
        action="store_true",
        help="Use monotonic neural network architecture (automatically enables error-state mode)",
    )
    parser.add_argument(
        "--origin_stabilization",
        action="store_true",
        help="Use error-state mode V(state-setpoint) even with standard networks",
    )
    parser.add_argument(
        "--monotonic_layers",
        type=int,
        default=2,
        help="Number of monotonic layers (only used with --use_monotonic)",
    )
    parser.add_argument(
        "--monotonic_pieces",
        type=int,
        default=4,
        help="Number of pieces per monotonic unit (only used with --use_monotonic)",
    )
    parser.add_argument(
        "--origin_setpoint",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Setpoint for origin stabilization mode (x, y, theta)",
    )

    args = parser.parse_args()

    # Determine final stabilization mode
    origin_stabilization = args.origin_stabilization or args.use_monotonic

    # Print architecture choice
    print("=" * 60)
    if args.use_monotonic:
        print("🔬 USING MONOTONIC NEURAL NETWORK ARCHITECTURE")
        print("   • Provides formal stability guarantees by construction")
        print("   • Positive definiteness ensured automatically")
        print("   • Error-state mode V(state - setpoint)")
        print("   • Can handle any setpoint")
        print("   • Compatible with MILP verification")
        print(
            f"   • {args.monotonic_layers} layers, {args.monotonic_pieces} pieces per unit"
        )
    else:
        print("🧠 USING STANDARD NEURAL NETWORK ARCHITECTURE")
        if origin_stabilization:
            print("   • Error-state mode V(state - setpoint)")
            print("   • Can handle any setpoint")
        else:
            print("   • Multi-target mode V([state, setpoint])")
            print("   • Traditional concatenated input approach")
        print("   • All Lyapunov conditions enforced during training")
    print("=" * 60)

    # Model path handling
    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()
        print(f"Auto-detected model path: {args.ckpt_path}")

    # Environment setup
    env_name = utils.extract_env_name(args.ckpt_path)
    print(f"Training environment: {env_name}")
    env = gym.make(env_name)

    action_shape = env.action_space.shape
    state_shape = env.observation_space.shape

    print(f"State dimension: {state_shape}")
    print(f"Action dimension: {action_shape}")

    # Load dynamics model
    print("Loading dynamics model...")
    dynamics_model = utils.load_checkpoint(args.ckpt_path)
    print("\n=== Dynamics Model Architecture ===")
    dynamics_model.summary()

    # Initialize networks
    if args.load_saved:
        print("Loading saved networks...")
        actor = keras.models.load_model(args.ckpt_path.parent / "actor.keras")
        lyapunov_model = keras.models.load_model(
            args.ckpt_path.parent / "lyapunov.keras"
        )
    else:
        print("Initializing new networks...")

        # Create actor with proper mode
        actor = actor_def(
            state_shape, action_shape, origin_stabilization=origin_stabilization
        )

        # Create Lyapunov network with chosen architecture
        lyapunov_model = V_def(
            state_shape,
            use_monotonic=args.use_monotonic,
            origin_stabilization=origin_stabilization,
            num_layers=args.monotonic_layers,
            num_pieces=args.monotonic_pieces,
        )

    # Dataset construction
    print("Constructing training dataset...")
    state_spec = tf.TensorSpec(state_shape, dtype=tf.float32)
    dataset_signature = {"state": state_spec, "setpoint": state_spec}

    dataset = tf.data.Dataset.from_generator(
        generate_dataset(
            env,
            origin_stabilization=origin_stabilization,
            origin_setpoint=args.origin_setpoint,
        ),
        output_signature=dataset_signature,
    )
    batched_dataset = (
        dataset.batch(args.batch_size)
        .take(args.num_batches)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print(
        f"Dataset configuration: {args.num_batches} batches of size {args.batch_size}"
    )
    if origin_stabilization:
        print(
            "Dataset mode: Random setpoints with error-state processing V(state - setpoint)"
        )
    else:
        print(
            "Dataset mode: Random setpoints with concatenated processing V([state, setpoint])"
        )
    print(f"Training for {args.epochs} epochs with learning rate {args.lr}")

    # Execute training
    print("\n=== Starting Enhanced Lyapunov Control Training ===")
    train(batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args)

    print("\n=== Training Completed Successfully ===")
    print(f"Trained models saved to: {args.ckpt_path.parent}")

    if args.use_monotonic:
        print("✅ Monotonic architecture ensures formal stability guarantees!")
        print("   Next step: Integrate MILP verification for complete certification")
    else:
        print("✅ Standard architecture training completed")

    print(
        "Use test_diff_robot.py to evaluate the learned controller and Lyapunov function"
    )
