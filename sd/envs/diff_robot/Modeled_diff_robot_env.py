"""
Complete ModeledDiffRobotEnv Implementation with Learned Dynamics

This is the corrected version that actually uses the learned neural network model
for dynamics prediction instead of analytical equations.
"""

__credits__ = ["Adapted from Pendulum ModeledEnv for Differential Robot Systems"]

from os import path
from typing import Optional
from pathlib import Path

import numpy as np
import tensorflow as tf

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from sd.envs.modelable_env import ModelableEnv
from sd import utils as sd_utils

DEFAULT_X = 0.0  # Default initial x position
DEFAULT_Y = 0.0  # Default initial y position
DEFAULT_THETA = 0.0  # Default initial orientation


def ensure_numpy_action(action):
    """
    Robust action conversion to NumPy array with proper shape validation

    This function handles the tensor-to-array conversion that was causing
    the indexing errors in your original implementation.
    """
    # Convert TensorFlow tensor to NumPy if needed
    if hasattr(action, "numpy"):
        action = action.numpy()

    # Ensure NumPy array format
    action = np.asarray(action, dtype=np.float32)

    # Handle batch dimension if present: (1,2) → (2,)
    if action.ndim == 2 and action.shape[0] == 1:
        action = action.squeeze(0)

    # Validate final shape for differential robot
    if action.shape != (2,):
        raise ValueError(
            f"Action must have shape (2,), got {action.shape}. Action: {action}"
        )

    return action


class ModeledDiffRobotEnv(ModelableEnv):
    """
    Model-Based Differential Mobile Robot Environment

    This environment uses a learned neural network model for dynamics prediction
    instead of analytical kinematic equations. This enables:

    1. **Model Validation**: Compare learned vs analytical dynamics
    2. **Robustness Testing**: Evaluate controller performance with model uncertainty
    3. **Sim-to-Real Transfer**: Test how well learned models generalize
    4. **Ablation Studies**: Isolate the effect of dynamics modeling errors

    ## Key Implementation Differences from Analytical Environment:

    - **Dynamics**: Neural network prediction replaces kinematic integration
    - **Stochasticity**: Supports latent noise injection for uncertainty modeling
    - **Model Loading**: Requires trained dynamics model checkpoint path
    - **Interface**: Identical observation/action spaces for seamless comparison

    ## Usage Pattern:
    ```python
    # Create both environments for comparison
    analytical_env = gym.make('DiffRobot-v1')
    modeled_env = gym.make('ModeledDiffRobot-v1',
                          model_path='path/to/trained/model.keras')

    # Test controller on both environments
    obs_a, _ = analytical_env.reset(seed=42)
    obs_m, _ = modeled_env.reset(seed=42)

    action = controller(obs_a)
    next_obs_a, _, _, _, _ = analytical_env.step(action)
    next_obs_m, _, _, _, _ = modeled_env.step(action)

    # Compare dynamics accuracy
    dynamics_error = np.linalg.norm(next_obs_a - next_obs_m)
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        model_path: Optional[
            str
        ] = None,  # ← CRITICAL: model_path must be first parameter
        render_mode: Optional[str] = None,
        max_vel=2.0,
        dt=0.05,
        screen=None,
    ):
        """
        Initialize Model-Based Environment with Learned Dynamics

        Parameter Order Critical for Gymnasium Compatibility:
        gymnasium passes model_path as the first positional argument,
        so it must be the first parameter in the constructor signature.

        Initialization Pipeline:
        1. Store model path and validate existence
        2. Initialize environment parameters (identical to analytical version)
        3. Load and validate learned dynamics model
        4. Set up rendering infrastructure
        """

        # Step 1: Model Path Validation and Storage
        if model_path is not None:
            self.model_path = Path(model_path)
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
        else:
            # Fallback: attempt to find latest trained model
            try:
                self.model_path = sd_utils.latest_model()
                print(f"No model_path provided, using latest: {self.model_path}")
            except Exception as e:
                raise ValueError(
                    f"No model_path provided and could not find latest model: {e}"
                )

        # Step 2: Environment Parameter Initialization
        # These parameters must match those used during dynamics model training
        self.max_linear_vel = max_vel
        self.max_angular_vel = max_vel
        self.dt = dt

        # Environment bounds: consistent with training environment
        self.max_position = 10.0

        # Target configuration: can be modified during operation
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0

        # Step 3: Rendering Infrastructure Setup
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = screen
        self.clock = None
        self.isopen = True

        # Step 4: Action and Observation Space Definition
        # Must exactly match the spaces used during model training
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_vel, -self.max_angular_vel]),
            high=np.array([self.max_linear_vel, self.max_angular_vel]),
            shape=(2,),  # [linear_velocity, angular_velocity]
            dtype=np.float32,
        )

        high = np.array([self.max_position, self.max_position, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, shape=(3,), dtype=np.float32  # [x, y, theta]
        )

        # Step 5: Load and Validate Learned Dynamics Model
        self._load_dynamics_model()

    def _load_dynamics_model(self):
        """
        Load and Validate Learned Dynamics Model

        Validation Strategy:
        1. Load model using project's custom checkpoint loader
        2. Verify input/output dimensions match environment specifications
        3. Test model with dummy inputs to ensure functionality
        4. Store model for efficient repeated evaluation

        Error Handling:
        Provides detailed diagnostics for common model loading failures
        """
        try:
            print(f"Loading learned dynamics model from: {self.model_path}")
            self.dynamics_model = sd_utils.load_checkpoint(self.model_path)

            # Dimension validation: ensure model compatibility
            expected_state_shape = self.observation_space.shape  # (3,) for [x,y,theta]
            expected_action_shape = self.action_space.shape  # (2,) for [v,w]

            model_state_shape = self.dynamics_model.input["state"].shape[1:]
            model_action_shape = self.dynamics_model.input["action"].shape[1:]
            model_latent_shape = self.dynamics_model.input["latent"].shape[1:]

            # Validate state dimension compatibility
            if model_state_shape != expected_state_shape:
                raise ValueError(
                    f"Model state dimension mismatch. "
                    f"Expected: {expected_state_shape}, Got: {model_state_shape}"
                )

            # Validate action dimension compatibility
            if model_action_shape != expected_action_shape:
                raise ValueError(
                    f"Model action dimension mismatch. "
                    f"Expected: {expected_action_shape}, Got: {model_action_shape}"
                )

            print("✓ Learned dynamics model loaded and validated successfully")
            print(f"  State shape: {model_state_shape}")
            print(f"  Action shape: {model_action_shape}")
            print(f"  Latent shape: {model_latent_shape}")

            # Functional test: verify model can process dummy inputs
            self._test_model_functionality()

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dynamics model from {self.model_path}. "
                f"Error details: {e}\n"
                f"Ensure the model was trained with compatible dimensions and "
                f"the checkpoint file is not corrupted."
            )

    def _test_model_functionality(self):
        """
        Test Model with Dummy Inputs

        Verification Process:
        1. Create dummy inputs with correct shapes and types
        2. Execute forward pass through learned model
        3. Validate output shape and numerical properties
        4. Ensure no runtime errors in model execution
        """
        try:
            # Create dummy inputs matching expected formats
            dummy_state = np.random.uniform(-1, 1, size=(1, 3)).astype(np.float32)
            dummy_action = np.random.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            dummy_latent = np.random.normal(
                0, 0.1, size=(1,) + self.dynamics_model.input["latent"].shape[1:]
            ).astype(np.float32)

            # Execute model forward pass
            dummy_inputs = {
                "state": dummy_state,
                "action": dummy_action,
                "latent": dummy_latent,
            }

            dummy_output = self.dynamics_model(dummy_inputs, training=False)

            # Validate output properties
            expected_output_shape = (1, 3)  # Batch size 1, state dimension 3
            if dummy_output.shape != expected_output_shape:
                raise ValueError(
                    f"Model output shape incorrect. "
                    f"Expected: {expected_output_shape}, Got: {dummy_output.shape}"
                )

            # Check for numerical stability (no NaN or infinite values)
            if np.any(np.isnan(dummy_output.numpy())) or np.any(
                np.isinf(dummy_output.numpy())
            ):
                raise ValueError("Model produces NaN or infinite outputs")

            print("✓ Model functionality test passed")

        except Exception as e:
            raise RuntimeError(f"Model functionality test failed: {e}")

    def step(self, action):
        """
        Model-Based Environment Step with Learned Dynamics

        Step Execution Pipeline:
        1. Validate current environment state
        2. Convert and validate action format
        3. Prepare inputs for neural network model
        4. Execute learned dynamics prediction
        5. Post-process predicted state
        6. Compute reward using analytical function
        7. Check termination conditions
        8. Return standard gymnasium step tuple

        Key Architectural Decision: Hybrid Approach
        - Dynamics: Use learned neural network model
        - Rewards: Use analytical reward function
        - Termination: Use analytical success criteria

        This hybrid approach provides the best of both worlds:
        reliable training signals with accurate dynamics modeling.
        """

        # Step 1: Environment State Validation
        if not hasattr(self, "state"):
            raise RuntimeError(
                "Environment not properly reset. Call reset() before step()."
            )

        current_state = self.state.copy()  # Defensive copy to prevent mutation

        # Step 2: Action Conversion and Validation
        # This is where the original indexing error was occurring
        try:
            action = ensure_numpy_action(action)
            print(
                f"[ModeledEnv Debug] Action after conversion: {action}, type: {type(action)}, shape: {action.shape}"
            )
        except Exception as e:
            raise ValueError(
                f"Action conversion failed in ModeledDiffRobotEnv.step(). "
                f"Raw action: {action}, Type: {type(action)}, Error: {e}"
            )

        # Action clipping for safety and model stability
        v = np.clip(action[0], -self.max_linear_vel, self.max_linear_vel)
        w = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)
        clipped_action = np.array([v, w], dtype=np.float32)

        self.last_action = clipped_action  # Store for rendering

        # Step 3: Model Input Preparation
        # Transform single samples to batch format for neural network processing
        model_state = current_state.reshape(1, -1).astype(np.float32)  # (3,) → (1,3)
        model_action = clipped_action.reshape(1, -1).astype(np.float32)  # (2,) → (1,2)

        # Latent noise injection for stochastic dynamics modeling
        latent_shape = (1,) + self.dynamics_model.input["latent"].shape[1:]
        latent_noise = np.random.normal(0, 0.01, latent_shape).astype(np.float32)

        # Step 4: Learned Dynamics Prediction
        try:
            model_inputs = {
                "state": model_state,
                "action": model_action,
                "latent": latent_noise,
            }

            print(
                f"[ModeledEnv Debug] Model inputs - state: {model_state.shape}, action: {model_action.shape}"
            )

            # Neural network forward pass: current state + action → next state
            next_state_batch = self.dynamics_model(model_inputs, training=False)
            next_state = next_state_batch.numpy().squeeze()  # (1,3) → (3,)

            print(
                f"[ModeledEnv Debug] Model output: {next_state}, shape: {next_state.shape}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Learned dynamics model prediction failed. "
                f"Current state: {current_state}, Action: {clipped_action}, Error: {e}"
            )

        # Step 5: State Post-Processing and Validation
        next_state = self._postprocess_predicted_state(next_state)

        # Step 6: State Update
        self.state = next_state

        # Step 7: Analytical Reward Computation
        # Use the same reward function as analytical environment for consistency
        reward = self._compute_analytical_reward(next_state, clipped_action)

        # Step 8: Termination Logic
        # Use same success criteria as analytical environment
        distance_error = np.sqrt(
            (next_state[0] - self.target_x) ** 2 + (next_state[1] - self.target_y) ** 2
        )
        angle_error = np.abs(
            np.arctan2(
                np.sin(next_state[2] - self.target_theta),
                np.cos(next_state[2] - self.target_theta),
            )
        )

        terminated = distance_error < 0.01 and angle_error < 0.01
        truncated = False  # Handled by time limit wrapper

        # Step 9: Optional Rendering
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _postprocess_predicted_state(self, predicted_state):
        """
        Post-Process Neural Network State Predictions

        Post-Processing Pipeline:
        1. Angle normalization: ensure theta ∈ [-π, π]
        2. Position bounds enforcement: prevent unrealistic states
        3. Type and shape validation: ensure numpy array consistency

        Why Post-Processing is Essential:
        Neural networks can produce outputs outside physically valid ranges,
        especially with distribution shift or during early training phases.
        Post-processing ensures environment constraints are maintained.
        """
        if predicted_state.shape != (3,):
            raise ValueError(
                f"Predicted state must have shape (3,), got {predicted_state.shape}"
            )

        x, y, theta = predicted_state

        # Position bounds: prevent escape from workspace
        x = np.clip(x, -self.max_position, self.max_position)
        y = np.clip(y, -self.max_position, self.max_position)

        # Angle normalization: handle wrap-around correctly
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        return np.array([x, y, theta], dtype=np.float32)

    def _compute_analytical_reward(self, state, action):
        """
        Analytical Reward Function (Identical to Base Environment)

        Reward Components:
        1. Position Error: Euclidean distance to target position
        2. Orientation Error: Angular difference from target orientation
        3. Control Cost: Quadratic penalty on control effort

        Mathematical Form:
        r = -(||pos_error||² + 0.1 * |angle_error|² + 0.001 * ||control||²)

        Design Rationale:
        Using the same reward function for both analytical and learned environments
        ensures fair comparison and consistent training signals.
        """
        x, y, theta = state
        v, w = action

        # Position error: Euclidean distance to target
        distance_error = np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)

        # Orientation error: minimum angle between current and target orientations
        angle_error = np.abs(
            np.arctan2(
                np.sin(theta - self.target_theta), np.cos(theta - self.target_theta)
            )
        )

        # Control effort: quadratic cost on velocities
        control_cost = v**2 + w**2

        # Composite cost function (negated for reward)
        cost = distance_error**2 + 0.1 * angle_error**2 + 0.001 * control_cost
        return -cost

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Environment Reset (Identical to Analytical Environment)

        Reset Process:
        1. Initialize random number generator with seed
        2. Sample initial state from specified distribution
        3. Configure target position if provided in options
        4. Return initial observation

        Critical Requirement:
        Reset logic must be identical to the analytical environment
        to ensure fair comparison between dynamics models.
        """
        super().reset(seed=seed)

        # Parse reset options
        if options is None:
            x_bound = 4.0
            y_bound = 4.0
            theta_bound = np.pi
        else:
            x_bound = options.get("x_bound", 1.0)
            y_bound = options.get("y_bound", 1.0)
            theta_bound = options.get("theta_bound", np.pi)

            # Optional target reconfiguration
            self.target_x = options.get("target_x", 0.0)
            self.target_y = options.get("target_y", 0.0)
            self.target_theta = options.get("target_theta", 0.0)

        # Sample initial state uniformly within bounds
        init_x = self.np_random.uniform(-x_bound, x_bound)
        init_y = self.np_random.uniform(-y_bound, y_bound)
        init_theta = self.np_random.uniform(-theta_bound, theta_bound)

        self.state = np.array([init_x, init_y, init_theta], dtype=np.float32)
        self.last_action = None

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Observation Formatting (Identical to Analytical Environment)

        For differential robot: direct state observation
        No encoding transformations needed (unlike pendulum with trigonometric encoding)
        """
        return np.array(self.state, dtype=np.float32)

    def set_target(self, x, y, theta=0.0):
        """
        Dynamic Target Setting for Interactive Control Testing
        """
        self.target_x = x
        self.target_y = y
        self.target_theta = theta

    def render(self):
        """
        Rendering (Identical to Analytical Environment)

        Visual Comparison Strategy:
        Identical rendering enables side-by-side comparison between
        analytical and learned dynamics, revealing model accuracy visually.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic-control]`"
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Create drawing surface
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))  # White background

        # Coordinate transformation: robot coordinates to screen pixels
        bound = self.max_position
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        # Convert robot state to screen coordinates
        x, y, theta = self.state
        screen_x = int(x * scale + offset)
        screen_y = int(-y * scale + offset)  # Flip y-axis for screen coordinates

        # Draw target position
        target_screen_x = int(self.target_x * scale + offset)
        target_screen_y = int(-self.target_y * scale + offset)
        gfxdraw.aacircle(self.surf, target_screen_x, target_screen_y, 5, (0, 255, 0))
        gfxdraw.filled_circle(
            self.surf, target_screen_x, target_screen_y, 5, (0, 255, 0)
        )

        # Draw robot as oriented triangle
        robot_size = 15
        triangle_points = [
            (robot_size, 0),  # Front point
            (-robot_size // 2, robot_size // 2),  # Left rear
            (-robot_size // 2, -robot_size // 2),  # Right rear
        ]

        # Rotate triangle points by robot orientation
        rotated_points = []
        for px, py in triangle_points:
            rotated_x = px * np.cos(theta) - py * np.sin(theta)
            rotated_y = px * np.sin(theta) + py * np.cos(theta)
            rotated_points.append((screen_x + rotated_x, screen_y + rotated_y))

        gfxdraw.aapolygon(self.surf, rotated_points, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, rotated_points, (255, 0, 0))

        # Draw velocity vector if action exists
        if self.last_action is not None:
            v, w = self.last_action
            if abs(v) > 0.01:  # Only draw if significant velocity
                vel_length = abs(v) * 30  # Scale for visibility
                vel_end_x = screen_x + vel_length * np.cos(theta)
                vel_end_y = screen_y + vel_length * np.sin(theta)
                pygame.draw.line(
                    self.surf,
                    (0, 0, 255),
                    (screen_x, screen_y),
                    (vel_end_x, vel_end_y),
                    3,
                )

        # Flip surface for correct orientation
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Resource cleanup"""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def normalize_angle(x):
    """Utility function to normalize angles to [-π, π] range"""
    return np.arctan2(np.sin(x), np.cos(x))
