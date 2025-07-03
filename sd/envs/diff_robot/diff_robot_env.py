__credits__ = ["Adapted from Pendulum for Differential Robot Systems"]

from os import path
from typing import Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from sd.envs.modelable_env import ModelableEnv

DEFAULT_X = 0.0  # Default initial x position
DEFAULT_Y = 0.0  # Default initial y position
DEFAULT_THETA = 0.0  # Default initial orientation


def ensure_numpy_action(action):
    """
    Convert TensorFlow tensors to NumPy arrays with proper shape validation

    This function handles the tensor-to-array conversion that was causing
    the indexing errors in the environment processing.
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


class DiffRobotEnv(ModelableEnv):
    """
    ## Description

    A differential mobile robot environment based on standard wheeled robot kinematics.
    The robot consists of two independently controlled wheels that can drive and steer
    the robot through differential wheel velocities. The goal is to control the robot
    to reach desired positions and orientations.

    The coordinate system follows standard robotics conventions:
    - x-axis: forward direction in global frame
    - y-axis: left direction in global frame
    - theta: counter-clockwise rotation from positive x-axis

    ## Action Space

    The action is a `ndarray` with shape `(2,)` representing [linear_velocity, angular_velocity].

    | Num | Action           | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | Linear Velocity  | -2.0 | 2.0 |
    | 1   | Angular Velocity | -2.0 | 2.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(3,)` representing [x, y, theta].

    | Num | Observation | Min    | Max   |
    |-----|-------------|--------|-------|
    | 0   | x position  | -10.0  | 10.0  |
    | 1   | y position  | -10.0  | 10.0  |
    | 2   | orientation | -π     | π     |

    ## Rewards

    The reward function encourages reaching target positions with proper orientation:

    *r = -(distance² + 0.1 * angle_error² + 0.001 * (v² + w²))*

    where distance is Euclidean distance to target, angle_error is angular difference
    from target orientation, v is linear velocity, and w is angular velocity.

    ## Starting State

    The starting state is a random position in [-1,1] x [-1,1] and random orientation in [-π,π].

    ## Episode Truncation

    The episode truncates at 500 time steps.

    ## Arguments

    - `max_vel`: maximum linear/angular velocity in m/s and rad/s respectively.
    - `dt`: integration time step for dynamics

    ```python
    import gymnasium as gym
    gym.make('DiffRobot-v1', max_vel=2.0, dt=0.05)
    ```
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self, render_mode: Optional[str] = None, max_vel=2.0, dt=0.05, screen=None
    ):
        # Robot physical parameters
        self.max_linear_vel = max_vel
        self.max_angular_vel = max_vel
        self.dt = dt

        # Environment bounds
        self.max_position = 10.0

        # Target position (can be modified for different goals)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0

        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = screen
        self.clock = None
        self.isopen = True

        # Define action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_vel, -self.max_angular_vel]),
            high=np.array([self.max_linear_vel, self.max_angular_vel]),
            shape=(2,),
            dtype=np.float32,
        )

        # Define observation space: [x, y, theta]
        high = np.array([self.max_position, self.max_position, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def step(self, action):
        """
        Execute one time step within the environment.

        Action transformation pipeline:
        1. Extract current state [x, y, theta]
        2. Clip control inputs to physical limits
        3. Apply kinematic model with Euler integration
        4. Compute reward based on distance to target
        5. Update internal state and return observation
        """
        x, y, theta = self.state

        # ADD THIS LINE - Convert action to proper NumPy format
        action = ensure_numpy_action(action)

        # Clip actions to valid ranges - critical for numerical stability
        v = np.clip(action[0], -self.max_linear_vel, self.max_linear_vel)
        w = np.clip(action[1], -self.max_angular_vel, self.max_angular_vel)

        self.last_action = action  # Store for rendering

        # Kinematic model: differential mobile robot equations
        # These equations transform control inputs into state derivatives
        dx_dt = v * np.cos(theta)  # Velocity projected onto global x-axis
        dy_dt = v * np.sin(theta)  # Velocity projected onto global y-axis
        dtheta_dt = w  # Direct angular velocity

        # Euler integration: transform derivatives into discrete state update
        new_x = x + dx_dt * self.dt
        new_y = y + dy_dt * self.dt
        new_theta = theta + dtheta_dt * self.dt

        # Normalize theta to [-π, π] to prevent angle wrap-around issues
        new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

        # Bound position to prevent infinite exploration
        new_x = np.clip(new_x, -self.max_position, self.max_position)
        new_y = np.clip(new_y, -self.max_position, self.max_position)

        self.state = np.array([new_x, new_y, new_theta])

        # Reward computation: negative cost function encourages target reaching
        distance_error = np.sqrt(
            (new_x - self.target_x) ** 2 + (new_y - self.target_y) ** 2
        )
        angle_error = np.abs(
            np.arctan2(
                np.sin(new_theta - self.target_theta),
                np.cos(new_theta - self.target_theta),
            )
        )
        control_cost = v**2 + w**2

        # Composite cost: position error + orientation error + control effort
        cost = distance_error**2 + 0.1 * angle_error**2 + 0.001 * control_cost
        reward = -cost

        # Episode termination conditions
        terminated = distance_error < 0.01 and angle_error < 0.01  # Success condition
        truncated = False  # Handled by time limit wrapper

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state with optional parameter customization.

        State initialization strategy:
        1. Extract bounds from options or use defaults
        2. Sample uniformly within specified ranges
        3. Normalize orientation to canonical range
        4. Store as internal state vector
        """
        super().reset(seed=seed)

        if options is None:
            # Default initialization bounds
            x_bound = 4.0
            y_bound = 4.0
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

        self.state = np.array([init_x, init_y, init_theta])
        self.last_action = None

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Transform internal state to observation format.

        For differential robot: observation equals state directly
        This differs from pendulum which used trigonometric encoding
        """
        return np.array(self.state, dtype=np.float32)

    def set_target(self, x, y, theta=0.0):
        """
        Utility method to dynamically change target during operation.
        Useful for sequential navigation tasks or interactive control.
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
        gfxdraw.aacircle(self.surf, target_screen_x, target_screen_y, 5, (0, 255, 0))
        gfxdraw.filled_circle(
            self.surf, target_screen_x, target_screen_y, 5, (0, 255, 0)
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

        # Render text AFTER the flip to avoid upside-down text
        font = pygame.font.Font(None, 24)
        text = font.render("Differential Robot Gym Environment", True, (0, 0, 0))
        self.surf.blit(text, (10, 10))

        # Add a black border around the screen
        pygame.draw.rect(self.surf, (0, 0, 0), self.surf.get_rect(), 2)

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
        """Clean up pygame resources"""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def normalize_angle(x):
    """Utility function to normalize angles to [-π, π] range"""
    return np.arctan2(np.sin(x), np.cos(x))
