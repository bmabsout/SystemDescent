import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import math
from typing import Tuple, List, Dict, Optional
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
import time
import json

# Import the monotonic layers we just created
from .monotonic_layers import V_def_with_architecture_choice

# SMT Solver imports
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not available. Install with: pip install z3-solver")

try:
    import cvc5
    from cvc5 import Kind

    CVC5_AVAILABLE = True
except ImportError:
    CVC5_AVAILABLE = False
    print("Warning: CVC5 not available. Install with: pip install cvc5")

try:
    import dreal

    DREAL_AVAILABLE = True
except ImportError:
    DREAL_AVAILABLE = False
    print("Warning: dReal not available. Install with: pip install dreal")


# === SMT Verification Classes ===
class LyapunovVerifier:
    """SMT-based verification for Lyapunov conditions with support for multiple solvers"""

    def __init__(self, solver="z3", timeout=30, precision=1e-3):
        self.solver_name = solver
        self.timeout = timeout
        self.precision = precision

        # Initialize solver
        if solver == "z3" and Z3_AVAILABLE:
            self.solver = z3.Solver()
            self.solver.set("timeout", timeout * 1000)  # Z3 uses milliseconds
        elif solver == "cvc5" and CVC5_AVAILABLE:
            self.solver = cvc5.Solver()
            self.solver.setOption("produce-models", "true")
            self.solver.setLogic("QF_NRA")
        elif solver == "dreal" and DREAL_AVAILABLE:
            self.precision = precision
        else:
            available = []
            if Z3_AVAILABLE:
                available.append("z3")
            if CVC5_AVAILABLE:
                available.append("cvc5")
            if DREAL_AVAILABLE:
                available.append("dreal")
            raise ValueError(f"Solver {solver} not available. Available: {available}")

    def verify_lyapunov_conditions(
        self,
        V_model,
        actor_model,
        dynamics_model,
        state_bounds,
        is_monotonic=False,
        num_samples=100,
    ):
        """Unified verification for both standard and monotonic networks"""
        counterexamples = []
        verification_results = {}

        # Determine which conditions need verification
        conditions_to_check = self._get_verification_conditions(is_monotonic)

        print(
            f"\n=== Verifying {len(conditions_to_check)} Lyapunov Conditions with {self.solver_name.upper()} ==="
        )

        for condition in conditions_to_check:
            print(f"🔍 Checking: {condition}")
            start_time = time.time()

            violations = self._verify_condition(
                V_model,
                actor_model,
                dynamics_model,
                condition,
                state_bounds,
                num_samples,
            )

            verification_time = time.time() - start_time
            verification_results[condition] = {
                "violations": len(violations),
                "time": verification_time,
                "status": "PASS" if len(violations) == 0 else "FAIL",
            }

            if violations:
                counterexamples.extend(violations)
                print(
                    f"❌ Found {len(violations)} violations for {condition} ({verification_time:.2f}s)"
                )
            else:
                print(
                    f"✅ {condition} verified successfully ({verification_time:.2f}s)"
                )

        # Summary
        total_violations = len(counterexamples)
        passed = sum(1 for r in verification_results.values() if r["status"] == "PASS")
        total_conditions = len(conditions_to_check)

        print(f"\n📊 Verification Summary:")
        print(f"   Conditions passed: {passed}/{total_conditions}")
        print(f"   Total violations: {total_violations}")
        print(f"   Network type: {'Monotonic' if is_monotonic else 'Standard'}")

        return counterexamples, verification_results

    def _get_verification_conditions(self, is_monotonic):
        """Determine which conditions need verification based on architecture"""
        if is_monotonic:
            # Monotonic networks satisfy positive definiteness and zero-at-target by construction
            return ["lyapunov_decrease", "trajectory_convergence"]
        else:
            # Standard networks need all conditions verified
            return [
                "zero_at_target",
                "positive_definite",
                "lyapunov_decrease",
                "trajectory_convergence",
            ]

    def _verify_condition(
        self, V_model, actor_model, dynamics_model, condition, state_bounds, num_samples
    ):
        """Verify specific Lyapunov condition using SMT solver"""

        if condition == "zero_at_target":
            return self._verify_zero_at_target(V_model, state_bounds, num_samples)
        elif condition == "positive_definite":
            return self._verify_positive_definite(V_model, state_bounds, num_samples)
        elif condition == "lyapunov_decrease":
            return self._verify_decrease_condition(
                V_model, actor_model, dynamics_model, state_bounds, num_samples
            )
        elif condition == "trajectory_convergence":
            return self._verify_trajectory_convergence(
                V_model, actor_model, dynamics_model, state_bounds, num_samples
            )
        else:
            return []

    def _verify_zero_at_target(self, V_model, state_bounds, num_samples):
        """Verify V(x*, x*) ≈ 0 for standard networks"""
        violations = []

        # Sample random setpoints
        setpoints = self._sample_states(state_bounds, num_samples // 2)

        for setpoint in setpoints:
            # Evaluate V(setpoint, setpoint) - should be ≈ 0
            V_val = float(
                V_model(
                    {
                        "state": tf.expand_dims(setpoint, 0),
                        "setpoint": tf.expand_dims(setpoint, 0),
                    }
                )[0]
            )

            # Check if significantly different from zero
            if abs(V_val) > 0.1:  # Tolerance for "approximately zero"
                violations.append(
                    {
                        "type": "zero_at_target",
                        "state": setpoint,
                        "setpoint": setpoint,
                        "V_value": V_val,
                        "violation_magnitude": abs(V_val),
                    }
                )

        return violations

    def _verify_positive_definite(self, V_model, state_bounds, num_samples):
        """Verify V(x, x*) > 0 for x ≠ x* for standard networks"""
        violations = []

        # Sample state-setpoint pairs where state ≠ setpoint
        for _ in range(num_samples):
            state = self._sample_states(state_bounds, 1)[0]
            setpoint = self._sample_states(state_bounds, 1)[0]

            # Ensure they're different
            distance = np.linalg.norm(state - setpoint)
            if distance < 0.1:  # Too close, skip
                continue

            V_val = float(
                V_model(
                    {
                        "state": tf.expand_dims(state, 0),
                        "setpoint": tf.expand_dims(setpoint, 0),
                    }
                )[0]
            )

            # Should be positive
            if V_val <= 0.01:  # Small positive threshold
                violations.append(
                    {
                        "type": "positive_definite",
                        "state": state,
                        "setpoint": setpoint,
                        "V_value": V_val,
                        "distance_to_target": distance,
                    }
                )

        return violations

    def _verify_decrease_condition(
        self, V_model, actor_model, dynamics_model, state_bounds, num_samples
    ):
        """Verify V̇ < 0 (Lyapunov decrease condition)"""
        violations = []

        for _ in range(num_samples):
            state = self._sample_states(state_bounds, 1)[0]
            setpoint = self._sample_states(state_bounds, 1)[0]

            # Skip if already at target
            if np.linalg.norm(state - setpoint) < 0.1:
                continue

            # Current Lyapunov value
            V_current = float(
                V_model(
                    {
                        "state": tf.expand_dims(state, 0),
                        "setpoint": tf.expand_dims(setpoint, 0),
                    }
                )[0]
            )

            # Get control action
            action = actor_model(
                {
                    "state": tf.expand_dims(state, 0),
                    "setpoint": tf.expand_dims(setpoint, 0),
                }
            )

            # Simulate next state
            latent_noise = tf.zeros(
                (1,) + tuple(dynamics_model.input["latent"].shape[1:])
            )
            next_state = dynamics_model(
                {
                    "state": tf.expand_dims(state, 0),
                    "action": action,
                    "latent": latent_noise,
                }
            )[0]

            # Next Lyapunov value
            V_next = float(
                V_model(
                    {
                        "state": tf.expand_dims(next_state, 0),
                        "setpoint": tf.expand_dims(setpoint, 0),
                    }
                )[0]
            )

            # Check decrease condition
            V_decrease = V_current - V_next
            if V_decrease <= 0:  # Should decrease
                violations.append(
                    {
                        "type": "lyapunov_decrease",
                        "state": state,
                        "next_state": next_state.numpy(),
                        "setpoint": setpoint,
                        "V_current": V_current,
                        "V_next": V_next,
                        "V_decrease": V_decrease,
                        "action": action.numpy().flatten(),
                    }
                )

        return violations

    def _verify_trajectory_convergence(
        self, V_model, actor_model, dynamics_model, state_bounds, num_samples
    ):
        """Verify long-term trajectory convergence"""
        violations = []
        trajectory_length = 20

        for _ in range(num_samples // 4):  # Fewer samples due to computational cost
            initial_state = self._sample_states(state_bounds, 1)[0]
            setpoint = self._sample_states(state_bounds, 1)[0]

            # Skip if already at target
            if np.linalg.norm(initial_state - setpoint) < 0.1:
                continue

            # Simulate trajectory
            current_state = initial_state
            V_values = []
            states = [current_state]

            for step in range(trajectory_length):
                # Current Lyapunov value
                V_current = float(
                    V_model(
                        {
                            "state": tf.expand_dims(current_state, 0),
                            "setpoint": tf.expand_dims(setpoint, 0),
                        }
                    )[0]
                )
                V_values.append(V_current)

                # Control action
                action = actor_model(
                    {
                        "state": tf.expand_dims(current_state, 0),
                        "setpoint": tf.expand_dims(setpoint, 0),
                    }
                )

                # Next state
                latent_noise = tf.zeros(
                    (1,) + tuple(dynamics_model.input["latent"].shape[1:])
                )
                current_state = dynamics_model(
                    {
                        "state": tf.expand_dims(current_state, 0),
                        "action": action,
                        "latent": latent_noise,
                    }
                )[0].numpy()

                states.append(current_state)

            # Check if trajectory is converging (Lyapunov decreasing overall)
            initial_V = V_values[0]
            final_V = V_values[-1]
            final_distance = np.linalg.norm(current_state - setpoint)

            # Trajectory should show progress (lower V or closer to target)
            if final_V >= initial_V and final_distance > np.linalg.norm(
                initial_state - setpoint
            ):
                violations.append(
                    {
                        "type": "trajectory_convergence",
                        "initial_state": initial_state,
                        "final_state": current_state,
                        "setpoint": setpoint,
                        "initial_V": initial_V,
                        "final_V": final_V,
                        "initial_distance": np.linalg.norm(initial_state - setpoint),
                        "final_distance": final_distance,
                        "V_trajectory": V_values,
                        "state_trajectory": states,
                    }
                )

        return violations

    def _sample_states(self, bounds, num_samples):
        """Sample random states within bounds"""
        low, high = bounds
        # Assuming 3D state space (x, y, theta) for differential robot
        states = []
        for _ in range(num_samples):
            x = np.random.uniform(low, high)
            y = np.random.uniform(low, high)
            theta = np.random.uniform(-np.pi, np.pi)
            states.append(np.array([x, y, theta], dtype=np.float32))
        return states


# === Helper Functions for SMT Integration ===
def augment_batch_with_counterexamples(batch, counterexamples, max_ce_ratio=0.3):
    """Add counter examples to training batch with ratio control"""
    if not counterexamples:
        return batch

    # Use fixed ratio instead of dynamic calculation to avoid tensor conversion issues
    max_ce_count = min(len(counterexamples), 20)  # Max 20 counter examples per batch

    # Sample recent counter examples
    selected_ce = (
        counterexamples[-max_ce_count:]
        if len(counterexamples) > max_ce_count
        else counterexamples
    )

    if not selected_ce:
        return batch

    # Convert counter examples to batch format - handle different violation types
    ce_states = []
    ce_setpoints = []

    for ce in selected_ce:
        # Extract state - different violation types use different keys
        if "state" in ce:
            state = ce["state"]
        elif "initial_state" in ce:
            state = ce["initial_state"]
        else:
            continue  # Skip if no recognizable state key

        # Extract setpoint
        setpoint = ce.get("setpoint", np.array([0.0, 0.0, 0.0], dtype=np.float32))

        ce_states.append(tf.constant(state, dtype=tf.float32))
        ce_setpoints.append(tf.constant(setpoint, dtype=tf.float32))

    if not ce_states:  # No valid counter examples found
        return batch

    # Stack and combine with original batch
    ce_states_tensor = tf.stack(ce_states)
    ce_setpoints_tensor = tf.stack(ce_setpoints)

    # Combine with original batch
    return {
        "state": tf.concat([batch["state"], ce_states_tensor], axis=0),
        "setpoint": tf.concat([batch["setpoint"], ce_setpoints_tensor], axis=0),
    }


def compute_counterexample_penalty(
    V_model, actor_model, counterexamples, penalty_weight=5.0
):
    """Compute penalty term for known violations"""
    if not counterexamples:
        return 0.0

    penalties = []

    for ce in counterexamples[-50:]:  # Use recent counter examples
        # Extract state - handle different violation types
        if "state" in ce:
            state = tf.constant(ce["state"], dtype=tf.float32)
        elif "initial_state" in ce:
            state = tf.constant(ce["initial_state"], dtype=tf.float32)
        else:
            continue  # Skip if no recognizable state key

        setpoint = tf.constant(ce.get("setpoint", [0.0, 0.0, 0.0]), dtype=tf.float32)
        violation_type = ce["type"]

        state_batch = tf.expand_dims(state, 0)
        setpoint_batch = tf.expand_dims(setpoint, 0)

        V_val = V_model({"state": state_batch, "setpoint": setpoint_batch})[0]

        if violation_type == "positive_definite":
            # Force V > 0.05 when away from target
            penalty = tf.maximum(0.0, 0.05 - V_val)
        elif violation_type == "zero_at_target":
            # Force V ≈ 0 at target
            penalty = tf.abs(V_val)
        elif violation_type == "lyapunov_decrease":
            # Encourage decrease - use stored violation info
            required_decrease = 0.01  # Minimum required decrease
            penalty = tf.maximum(0.0, required_decrease - tf.abs(V_val))
        elif violation_type == "trajectory_convergence":
            # Penalty for trajectory convergence issues
            penalty = tf.maximum(0.0, 0.1 - V_val)  # Should be small for convergence
        else:
            penalty = tf.abs(V_val) * 0.1  # Generic penalty

        penalties.append(penalty)

    if penalties:
        return penalty_weight * tf.reduce_mean(penalties)
    return 0.0


# === Import your existing functions from lyapunov_diff_robot.py ===


def V_def(
    state_shape: Tuple[int, ...],
    use_monotonic: bool = False,
    origin_stabilization: bool = False,
    **monotonic_kwargs,
):
    """Enhanced Lyapunov Function Architecture for Differential Mobile Robot"""

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
    """Control Policy Architecture for Differential Mobile Robot"""
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
        32,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="control_hidden_1",
    )(network_input)

    dense2 = layers.Dense(
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
    """Training Data Generation for Lyapunov-Based Control Learning"""

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


# === Enhanced Training Function with SMT Verification ===


def train_with_smt_verification(batches, dynamics_model, actor, V, state_shape, args):
    """Enhanced training loop with SMT-based verification and counter-example guided refinement"""

    # Detect network architecture
    is_monotonic = "Monotonic" in V.name
    origin_stabilization = args.origin_stabilization or is_monotonic

    # Initialize SMT verifier
    verifier = LyapunovVerifier(
        solver=args.solver, timeout=args.verify_timeout, precision=args.verify_precision
    )

    # Training state
    counterexamples = []
    verification_history = []
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    print(f"\n{'='*80}")
    print(f"🚀 ENHANCED LYAPUNOV TRAINING WITH SMT VERIFICATION")
    print(f"{'='*80}")
    print(f"Network type: {'Monotonic' if is_monotonic else 'Standard'}")
    print(f"Mode: {'Error-state' if origin_stabilization else 'Multi-target'}")
    print(f"SMT Solver: {args.solver.upper()}")
    print(f"Verification: Every {args.verify_every} epochs")
    print(f"State bounds: {args.state_bounds}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}")

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
        """Enhanced Lyapunov Training Objective Computation with SMT integration"""
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

        # Lyapunov function evaluation
        V_initial = V({"state": prev_states, "setpoint": set_points}, training=True)
        V_final = V({"state": final_states, "setpoint": set_points}, training=True)
        V_at_target = V({"state": set_points, "setpoint": set_points}, training=True)

        # Core constraints that apply to both architectures
        lyapunov_decrease = V_initial - V_final

        # Both modes measure distance to setpoint
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

        # Counter example penalty
        ce_penalty = 0.0
        if counterexamples:
            ce_penalty = compute_counterexample_penalty(V, actor, counterexamples)

        # Construct constraint hierarchy based on network type
        if is_monotonic:
            base_constraints = Constraints(
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
                        },
                    ),
                },
            )
        else:
            # Standard networks need all constraints
            zero_constraint = p_mean((1.0 - V_at_target**0.5), -1.0, default_val=1.0)

            target_distances = euclidean_distance(prev_states, set_points)
            non_target_mask = tf.where(target_distances > 0.1, V_initial, 1.0)
            positive_away_from_target = p_mean(
                tf.minimum(non_target_mask * 5.0, 1.0), 0.0
            )

            base_constraints = Constraints(
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

        # Add counter example penalty to the objective
        base_fulfillment = fpl_value(base_constraints)
        if ce_penalty > 0:
            # Penalty reduces the fulfillment value
            enhanced_fulfillment = base_fulfillment - ce_penalty * 0.1  # Scale penalty
            return enhanced_fulfillment, base_constraints

        return base_fulfillment, base_constraints

    @tf.function
    def train_step_with_ce(batch):
        """Training step with counter example integration"""
        with tf.GradientTape() as tape:
            fulfillment_value, objective_structure = batch_value(
                batch, args.minN, args.maxN
            )
            loss = 1.0 - fulfillment_value

        trainable_parameters = actor.trainable_weights + V.trainable_weights
        gradients = tape.gradient(loss, trainable_parameters)
        optimizer.apply_gradients(zip(gradients, trainable_parameters))

        return fulfillment_value, objective_structure

    def save_models(epoch):
        """Save models and verification history"""
        save_model(actor, "actor.keras")
        save_model(V, "lyapunov.keras")

        # Save verification history
        history_path = args.ckpt_path.parent / "verification_history.json"
        with open(history_path, "w") as f:
            json.dump(verification_history, f, indent=2, default=str)

        print(f"📁 Models and history saved at epoch {epoch}")

    # Main training loop with periodic verification
    best_verification_score = -1
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_losses = []

        print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")

        # Training phase
        batch_count = 0
        for batch in batches:
            # Augment batch with counter examples
            if counterexamples:
                batch = augment_batch_with_counterexamples(batch, counterexamples)

            fulfillment, metrics = train_step_with_ce(batch)
            epoch_losses.append(float(fulfillment))
            batch_count += 1

        avg_fulfillment = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start

        print(
            f"   📊 Avg Fulfillment: {avg_fulfillment:.4f} | Time: {epoch_time:.1f}s | Batches: {batch_count}"
        )
        if counterexamples:
            print(f"   🎯 Active Counter Examples: {len(counterexamples)}")

        # Periodic verification
        if (epoch + 1) % args.verify_every == 0 or epoch == args.epochs - 1:
            print(f"\n🔍 Running SMT Verification at epoch {epoch+1}")
            verification_start = time.time()

            # Run verification
            new_violations, verification_results = verifier.verify_lyapunov_conditions(
                V,
                actor,
                dynamics_model,
                state_bounds=tuple(args.state_bounds),
                is_monotonic=is_monotonic,
                num_samples=args.verify_samples,
            )

            verification_time = time.time() - verification_start

            # Process results
            total_violations = len(new_violations)
            passed_conditions = sum(
                1 for r in verification_results.values() if r["status"] == "PASS"
            )
            total_conditions = len(verification_results)
            verification_score = (
                passed_conditions / total_conditions if total_conditions > 0 else 0
            )

            # Update counter examples
            if new_violations:
                counterexamples.extend(new_violations)
                # Keep only recent counter examples to avoid memory issues
                if len(counterexamples) > args.max_counterexamples:
                    counterexamples = counterexamples[-args.max_counterexamples :]

                print(f"   ❌ Found {total_violations} new violations")
                print(f"   📝 Total counter examples: {len(counterexamples)}")

                # Adaptive learning rate on many violations
                if total_violations > 20:
                    new_lr = optimizer.learning_rate * 0.8
                    optimizer.learning_rate.assign(new_lr)
                    print(f"   🔧 Reduced learning rate to {float(new_lr):.2e}")
            else:
                print(f"   ✅ No new violations found!")

                # Slight learning rate increase on clean verification
                if len(counterexamples) > 0:
                    new_lr = optimizer.learning_rate * 1.02
                    optimizer.learning_rate.assign(new_lr)
                    print(f"   🚀 Increased learning rate to {float(new_lr):.2e}")

            # Track verification progress
            verification_entry = {
                "epoch": epoch + 1,
                "avg_fulfillment": avg_fulfillment,
                "verification_score": verification_score,
                "passed_conditions": passed_conditions,
                "total_conditions": total_conditions,
                "new_violations": total_violations,
                "total_counterexamples": len(counterexamples),
                "verification_time": verification_time,
                "learning_rate": float(optimizer.learning_rate),
                "results": verification_results,
            }
            verification_history.append(verification_entry)

            print(
                f"   📈 Verification Score: {verification_score:.2%} ({passed_conditions}/{total_conditions})"
            )
            print(f"   ⏱️  Verification Time: {verification_time:.1f}s")

            # Save best model
            if verification_score > best_verification_score:
                best_verification_score = verification_score
                save_model(actor, "best_actor.keras")
                save_model(V, "best_lyapunov.keras")
                print(f"   🏆 New best verification score: {verification_score:.2%}")

        # Regular model saving
        if (epoch + 1) % args.save_freq == 0:
            save_models(epoch + 1)

    # Final verification and summary
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING COMPLETED")
    print(f"{'='*80}")

    if verification_history:
        final_verification = verification_history[-1]
        print(
            f"Final verification score: {final_verification['verification_score']:.2%}"
        )
        print(f"Best verification score: {best_verification_score:.2%}")
        print(f"Total counter examples collected: {len(counterexamples)}")

        # Show improvement over time
        if len(verification_history) > 1:
            initial_score = verification_history[0]["verification_score"]
            final_score = verification_history[-1]["verification_score"]
            improvement = final_score - initial_score
            print(f"Verification improvement: {improvement:+.2%}")

    save_models(args.epochs)

    return verification_history


if __name__ == "__main__":
    """Enhanced Main Training Script with SMT Verification"""
    parser = argparse.ArgumentParser(
        description="Lyapunov-based control learning with SMT verification"
    )

    # Existing arguments
    parser.add_argument(
        "--ckpt_path", type=Path, default=None, help="Path to dynamics model checkpoint"
    )
    parser.add_argument(
        "--num_batches", type=int, default=300, help="Training batches per epoch"
    )
    parser.add_argument(
        "--save_freq", type=int, default=20, help="Model saving frequency (epochs)"
    )
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--load_saved", action="store_true", help="Resume from saved models"
    )
    parser.add_argument("--minN", type=int, default=3, help="Min trajectory length")
    parser.add_argument("--maxN", type=int, default=20, help="Max trajectory length")

    # Architecture arguments
    parser.add_argument(
        "--use_monotonic", action="store_true", help="Use monotonic architecture"
    )
    parser.add_argument(
        "--origin_stabilization", action="store_true", help="Use error-state mode"
    )
    parser.add_argument(
        "--monotonic_layers", type=int, default=2, help="Monotonic layers"
    )
    parser.add_argument(
        "--monotonic_pieces", type=int, default=4, help="Pieces per monotonic unit"
    )
    parser.add_argument(
        "--origin_setpoint",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Origin setpoint",
    )

    # SMT Verification arguments
    parser.add_argument(
        "--solver", choices=["z3", "cvc5", "dreal"], default="z3", help="SMT solver"
    )
    parser.add_argument(
        "--verify_every", type=int, default=10, help="Verification frequency (epochs)"
    )
    parser.add_argument(
        "--verify_timeout", type=int, default=30, help="SMT solver timeout (seconds)"
    )
    parser.add_argument(
        "--verify_precision", type=float, default=1e-3, help="Verification precision"
    )
    parser.add_argument(
        "--verify_samples", type=int, default=100, help="Verification sample count"
    )
    parser.add_argument(
        "--state_bounds",
        type=float,
        nargs=2,
        default=[-5.0, 5.0],
        help="State space bounds",
    )
    parser.add_argument(
        "--max_counterexamples",
        type=int,
        default=500,
        help="Max stored counter examples",
    )

    args = parser.parse_args()

    # Setup and initialization
    if args.ckpt_path is None:
        args.ckpt_path = utils.latest_model()
        print(f"Auto-detected model path: {args.ckpt_path}")

    env_name = utils.extract_env_name(args.ckpt_path)
    env = gym.make(env_name)
    action_shape = env.action_space.shape
    state_shape = env.observation_space.shape

    print(f"Environment: {env_name}")
    print(f"State shape: {state_shape}, Action shape: {action_shape}")

    # Load or create models
    dynamics_model = utils.load_checkpoint(args.ckpt_path)

    if args.load_saved:
        actor = keras.models.load_model(args.ckpt_path.parent / "actor.keras")
        lyapunov_model = keras.models.load_model(
            args.ckpt_path.parent / "lyapunov.keras"
        )
    else:
        origin_stabilization = args.origin_stabilization or args.use_monotonic
        actor = actor_def(
            state_shape, action_shape, origin_stabilization=origin_stabilization
        )
        lyapunov_model = V_def(
            state_shape,
            use_monotonic=args.use_monotonic,
            origin_stabilization=origin_stabilization,
            num_layers=args.monotonic_layers,
            num_pieces=args.monotonic_pieces,
        )

    # Create dataset
    state_spec = tf.TensorSpec(state_shape, dtype=tf.float32)
    dataset_signature = {"state": state_spec, "setpoint": state_spec}

    dataset = tf.data.Dataset.from_generator(
        generate_dataset(
            env,
            origin_stabilization=args.origin_stabilization or args.use_monotonic,
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

    # Run enhanced training with SMT verification
    verification_history = train_with_smt_verification(
        batched_dataset, dynamics_model, actor, lyapunov_model, state_shape, args
    )

    print(
        f"\n🎉 Training completed! Check {args.ckpt_path.parent} for saved models and verification history."
    )
