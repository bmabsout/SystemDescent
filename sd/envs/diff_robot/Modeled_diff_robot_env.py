"""
Corrected ModeledDiffRobotEnv that properly handles model_path parameter

Key Architectural Insights:
1. ModeledEnv classes must accept model_path in constructor
2. They should delegate dynamics to the learned neural network model
3. They inherit the environment interface while replacing dynamics
"""

__credits__ = ["Adapted from Pendulum ModeledEnv for Differential Robot Systems"]

from os import path
from typing import Optional
from pathlib import Path

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from sd.envs.modelable_env import ModelableEnv
from sd import utils as sd_utils

DEFAULT_X = 0.0  # Default initial x position
DEFAULT_Y = 0.0  # Default initial y position
DEFAULT_THETA = 0.0  # Default initial orientation


class ModeledDiffRobotEnv(ModelableEnv):
    """
    ## Description

    A differential mobile robot environment that uses a learned neural network model
    for dynamics instead of analytical equations. This allows testing of learned
    dynamics models and comparison with ground truth physics.

    The learned model replaces the kinematic equations:
    - Analytical: [dx/dt, dy/dt, dtheta/dt] = f_analytical(x, y, theta, v, w)
    - Learned: [x_next, y_next, theta_next] = f_neural(x, y, theta, v, w)

    ## Key Architectural Differences from Base DiffRobotEnv:

    1. **Model-Based Dynamics**: Uses learned TensorFlow model instead of analytical equations
    2. **Model Loading**: Accepts model_path parameter to load pre-trained dynamics
    3. **Stochastic Dynamics**: Supports latent noise injection for uncertainty modeling
    4. **Identical Interface**: Maintains same observation/action spaces as analytical version

    ## Constructor Parameters

    - `model_path`: Path to trained dynamics model checkpoint
    - `max_vel`: maximum linear/angular velocity (inherited from analytical model)
    - `dt`: integration time step (should match training configuration)
    - `render_mode`: visualization mode ("human" or "rgb_array")
    - `screen`: pygame surface for rendering (used in testing)

    ## Usage Example

    ```python
    import gymnasium as gym
    modeled_env = gym.make('ModeledDiffRobot-v1',
                          model_path='models/DiffRobot-v1/run_id/checkpoints/checkpoint0/model.keras')
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        max_vel=2.0,
        dt=0.05,
        screen=None,
    ):
        """
        Constructor for Model-Based Differential Robot Environment

        Parameter Processing Pipeline:
        1. Store model path for dynamics loading
        2. Initialize identical parameters as analytical environment
        3. Load learned dynamics model using sd.utils infrastructure
        4. Preserve all interface compatibility with base environment

        Critical Design Decision: model_path as first parameter
        This ensures gymnasium can pass it correctly during environment creation
        """

        # Model path handling: convert to Path object for robust path operations
        if model_path is not None:
            if isinstance(model_path, (str, Path)):
                self.model_path = Path(model_path)
            else:
                self.model_path = model_path
        else:
            # Fallback: attempt to use latest model if none specified
            try:
                self.model_path = sd_utils.latest_model()
                print(f"No model_path specified, using latest: {self.model_path}")
            except Exception as e:
                raise ValueError(
                    f"No model_path provided and could not find latest model: {e}"
                )

        # Robot physical parameters: identical to analytical environment
        # These should match the parameters used during dynamics model training
        self.max_linear_vel = max_vel
        self.max_angular_vel = max_vel
        self.dt = dt

        # Environment bounds: maintain consistency with training environment
        self.max_position = 10.0

        # Target position: can be modified for different control objectives
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0

        # Rendering infrastructure: identical to base environment
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = screen
        self.clock = None
        self.isopen = True

        # Action space definition: [linear_velocity, angular_velocity]
        # Must exactly match the action space used during model training
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_vel, -self.max_angular_vel]),
            high=np.array([self.max_linear_vel, self.max_angular_vel]),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space definition: [x, y, theta]
        # Must exactly match the state space used during model training
        high = np.array([self.max_position, self.max_position, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Load learned dynamics model using project's utilities
        # This is the critical difference from the analytical environment
        self._load_dynamics_model()

    def _load_dynamics_model(self):
        """
        Dynamics Model Loading and Validation

        Loading Process:
        1. Use sd.utils.load_checkpoint to handle custom_objects correctly
        2. Validate model input/output dimensions against environment specs
        3. Cache model for efficient repeated evaluation during episodes

        Error Handling Strategy:
        - Comprehensive validation of model compatibility
        - Clear error messages for debugging model loading issues
        - Graceful fallback suggestions for common failure modes
        """
        try:
            print(f"Loading dynamics model from: {self.model_path}")
            self.dynamics_model = sd_utils.load_checkpoint(self.model_path)

            # Model validation: ensure compatibility with environment specifications
            expected_state_shape = self.observation_space.shape  # (3,) for [x,y,theta]
            expected_action_shape = self.action_space.shape  # (2,) for [v,w]

            # Validate input dimensions
            model_state_shape = self.dynamics_model.input["state"].shape[1:]
            model_action_shape = self.dynamics_model.input["action"].shape[1:]

            if model_state_shape != expected_state_shape:
                raise ValueError(
                    f"Model state dimension {model_state_shape} doesn't match "
                    f"environment state dimension {expected_state_shape}"
                )

            if model_action_shape != expected_action_shape:
                raise ValueError(
                    f"Model action dimension {model_action_shape} doesn't match "
                    f"environment action dimension {expected_action_shape}"
                )

            print("✓ Dynamics model loaded and validated successfully")
            print(f"  State shape: {model_state_shape}")
            print(f"  Action shape: {model_action_shape}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dynamics model from {self.model_path}. "
                f"Error: {e}\n"
                f"Ensure the model path is correct and the model was trained "
                f"with compatible state/action dimensions."
            )

    def step(self, action):
        """
        Model-Based Environment Step

        Step Execution Pipeline:
        1. Validate and clip action to environment bounds
        2. Prepare model inputs (state, action, latent noise)
        3. Execute learned dynamics model forward pass
        4. Extract next state from model output
        5. Compute reward using analytical reward function
        6. Handle termination conditions and return standard gym tuple

        Key Architectural Decision: Reward Function Separation
        - Use learned model for dynamics prediction
        - Keep analytical reward function for consistent training signal
        - This hybrid approach provides best of both worlds
        """

        # Current state extraction and validation
        if not hasattr(self, "state"):
            raise RuntimeError("Environment not reset. Call reset() before step().")

        current_state = self.state.copy()  # Defensive copying to prevent mutations

        # Action validation and clipping: ensure model receives valid inputs
        action = np.array(action, dtype=np.float32)

        # Handle both single actions (2,) and batched actions (1,2) or (batch_size,2)
        # This accommodates different input formats from various sources
        if action.ndim == 2:
            if action.shape[0] == 1:
                # Single sample in batch format: (1,2) → (2,)
                action = action.squeeze(0)
            else:
                raise ValueError(
                    f"Batch size must be 1 for environment step, got batch size {action.shape[0]}"
                )
        elif action.ndim == 1:
            # Already in correct single-sample format: (2,)
            pass
        else:
            raise ValueError(
                f"Action must be 1D or 2D array, got {action.ndim}D with shape {action.shape}"
            )

        # Final validation: ensure we have exactly 2 action components
        if action.shape != (2,):
            raise ValueError(f"Action must have shape (2,), got {action.shape}")

        v = np.clip(action[0], -self.max_linear_vel, self.max_linear_vel)
        w = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)
        clipped_action = np.array([v, w], dtype=np.float32)

        self.last_action = clipped_action  # Store for rendering

        # Model input preparation: format for neural network evaluation
        # Shape transformations: (3,) -> (1,3) for batch processing
        model_state = current_state.reshape(1, -1).astype(np.float32)
        model_action = clipped_action.reshape(1, -1).astype(np.float32)

        # Latent noise generation: supports stochastic dynamics modeling
        # The noise dimension comes from model architecture during training
        latent_shape = (1,) + self.dynamics_model.input["latent"].shape[1:]
        latent_noise = np.random.normal(0, 0.1, latent_shape).astype(np.float32)

        # Learned dynamics evaluation: neural network forward pass
        try:
            model_inputs = {
                "state": model_state,
                "action": model_action,
                "latent": latent_noise,
            }

            # Execute model prediction: [x,y,theta] -> [x_next,y_next,theta_next]
            next_state_batch = self.dynamics_model(model_inputs, training=False)
            next_state = next_state_batch.numpy().squeeze()  # (1,3) -> (3,)

            # Post-processing: ensure state remains in valid ranges
            next_state = self._postprocess_state(next_state)

        except Exception as e:
            raise RuntimeError(
                f"Dynamics model evaluation failed. "
                f"Current state: {current_state}, Action: {clipped_action}, "
                f"Error: {e}"
            )

        # State update: replace analytical integration with learned prediction
        self.state = next_state

        # Reward computation: use analytical reward function for consistency
        # This ensures training compatibility between analytical and learned environments
        reward = self._compute_reward(next_state, clipped_action)

        # Termination logic: same success criteria as analytical environment
        distance_error = np.sqrt(
            (next_state[0] - self.target_x) ** 2 + (next_state[1] - self.target_y) ** 2
        )
        angle_error = np.abs(
            np.arctan2(
                np.sin(next_state[2] - self.target_theta),
                np.cos(next_state[2] - self.target_theta),
            )
        )

        terminated = distance_error < 0.1 and angle_error < 0.1
        truncated = False  # Handled by time limit wrapper

        # Rendering: optional visualization for monitoring
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _postprocess_state(self, state):
        """
        State Post-Processing for Learned Dynamics

        Post-processing Pipeline:
        1. Angle normalization: ensure theta ∈ [-π, π]
        2. Position clipping: prevent infinite exploration
        3. Type conversion: maintain numpy float32 consistency

        Why Post-Processing is Necessary:
        Neural networks can produce outputs outside valid ranges,
        especially during early training or with distribution shift.
        Post-processing ensures environment constraints are maintained.
        """
        x, y, theta = state

        # Position bounds enforcement: prevent unrealistic states
        x = np.clip(x, -self.max_position, self.max_position)
        y = np.clip(y, -self.max_position, self.max_position)

        # Angle normalization: handle wrap-around correctly
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        return np.array([x, y, theta], dtype=np.float32)

    def _compute_reward(self, state, action):
        """
        Analytical Reward Function

        Reward Structure:
        - Distance penalty: encourages reaching target position
        - Orientation penalty: encourages correct final orientation
        - Control penalty: discourages excessive control effort

        Mathematical Form:
        r = -(||position_error||² + 0.1 * |angle_error|² + 0.001 * ||control||²)
        """
        x, y, theta = state
        v, w = action

        # Distance to target computation
        distance_error = np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)

        # Angular error computation with wrap-around handling
        angle_error = np.abs(
            np.arctan2(
                np.sin(theta - self.target_theta), np.cos(theta - self.target_theta)
            )
        )

        # Control effort penalty
        control_cost = v**2 + w**2

        # Composite cost function
        cost = distance_error**2 + 0.1 * angle_error**2 + 0.001 * control_cost
        return -cost

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Environment Reset: identical to analytical environment

        Reset Process:
        1. Call parent reset for seed handling
        2. Sample initial state from specified distribution
        3. Update target if specified in options
        4. Return initial observation

        Critical: Reset logic must match training environment exactly
        """
        super().reset(seed=seed)

        if options is None:
            # Default initialization bounds
            x_bound = 1.0
            y_bound = 1.0
            theta_bound = np.pi
        else:
            # Custom bounds from options dictionary
            x_bound = options.get("x_bound", 1.0)
            y_bound = options.get("y_bound", 1.0)
            theta_bound = options.get("theta_bound", np.pi)

            # Target position can also be customized
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
        Observation transformation: identical to analytical environment

        For differential robot: observation equals state directly
        This maintains interface compatibility with training environment
        """
        return np.array(self.state, dtype=np.float32)

    def set_target(self, x, y, theta=0.0):
        """
        Dynamic target setting: useful for interactive control testing
        """
        self.target_x = x
        self.target_y = y
        self.target_theta = theta

    def render(self):
        """
        Visualization using pygame for real-time monitoring.

        Rendering pipeline:
        1. Initialize pygame surfaces if needed
        2. Clear background and set coordinate transformations
        3. Draw robot as oriented triangle
        4. Draw target as circle
        5. Add trajectory trail if desired
        6. Update display buffer
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
        gfxdraw.aacircle(self.surf, target_screen_x, target_screen_y, 20, (0, 255, 0))
        gfxdraw.filled_circle(
            self.surf, target_screen_x, target_screen_y, 20, (0, 255, 0)
        )

        # Draw robot as oriented triangle
        robot_size = 15
        # Triangle vertices in robot frame
        triangle_points = [
            (robot_size, 0),  # Front point
            (-robot_size // 2, robot_size // 2),  # Left rear
            (-robot_size // 2, -robot_size // 2),  # Right rear
        ]

        # Rotate triangle points by robot orientation
        rotated_points = []
        for px, py in triangle_points:
            # 2D rotation matrix application
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
        """Resource cleanup: identical to analytical environment"""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def normalize_angle(x):
    """Utility function to normalize angles to [-π, π] range"""
    return np.arctan2(np.sin(x), np.cos(x))
