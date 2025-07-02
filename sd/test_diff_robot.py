from pathlib import Path
import sd.envs
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from . import utils
import argparse
import pygame
import time

from sd.enhanced_lyapunov_viz import plot_diff_robot_lyapunov_enhanced


def ensure_numpy_action(action):
    """
    Convert TensorFlow tensors to NumPy arrays with proper shape validation

    CRITICAL: This function prevents the tensor indexing errors you encountered.
    It handles the conversion between neural network outputs and environment inputs.
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


def position_to_setpoint(x, y, theta=0.0):
    """
    Create setpoint for differential robot navigation

    Differential Robot Setpoint Format:
    [x_target, y_target, theta_target] - desired position and orientation

    This replaces the pendulum's angle_to_setpoint function with proper
    2D navigation targets for the differential robot.
    """
    return np.array([x, y, theta], dtype=np.float32)


def plot_diff_robot_lyapunov_original(
    lyapunov, actor, dynamics, set_point, fname, interactive=False
):
    """
    Lyapunov Function Visualization for Differential Robot

    Visualization Strategy:
    Since differential robots have 3D state space [x, y, theta], we create
    2D slices of the Lyapunov function by fixing theta and varying x, y.

    This provides insight into the navigation behavior and basin of attraction
    for different target positions and orientations.
    """

    def calculate_lyapunov_2d(set_point, fixed_theta=0.0):
        """
        Calculate 2D slice of Lyapunov function at fixed orientation

        Parameters:
        - set_point: target [x*, y*, theta*]
        - fixed_theta: orientation at which to evaluate slice

        Returns:
        - x_grid, y_grid: coordinate meshes for plotting
        - lyapunov_values: Lyapunov function values at each point
        """
        pts = 100  # Resolution for visualization

        # Create 2D grid centered around target position
        x_range = np.linspace(set_point[0] - 2.0, set_point[0] + 2.0, pts)
        y_range = np.linspace(set_point[1] - 2.0, set_point[1] + 2.0, pts)
        x_grid, y_grid = np.meshgrid(x_range, y_range)

        # Create state inputs: [x, y, fixed_theta] for each grid point
        theta_grid = np.full_like(x_grid, fixed_theta)
        states = np.stack(
            [x_grid.flatten(), y_grid.flatten(), theta_grid.flatten()], axis=1
        )

        # Create setpoint array matching state batch size
        setpoints = np.tile(set_point.reshape(1, -1), (states.shape[0], 1))

        # Evaluate Lyapunov function across the grid
        lyapunov_inputs = {"state": states, "setpoint": setpoints}
        lyapunov_values = lyapunov(lyapunov_inputs, training=False)

        # Reshape for plotting
        lyapunov_2d = lyapunov_values.numpy().reshape(x_grid.shape)

        return x_grid, y_grid, lyapunov_2d

    def draw_lyapunov_contours(set_point):
        """
        Draw Lyapunov function contours and target position

        Visualization Components:
        1. Lyapunov contour plot showing function landscape
        2. Target position marked with star
        3. Level curves indicating basins of attraction
        4. Colorbar for value interpretation
        """
        # Calculate Lyapunov values on 2D grid
        x_grid, y_grid, lyapunov_2d = calculate_lyapunov_2d(
            set_point, fixed_theta=set_point[2]
        )

        # Create contour plot
        plt.figure(figsize=(10, 8))

        # Filled contour plot with smooth gradients
        contour_filled = plt.contourf(
            x_grid, y_grid, lyapunov_2d, levels=20, cmap="viridis", alpha=0.8
        )
        plt.colorbar(contour_filled, label="Lyapunov Function Value V(x,y,θ)")

        # Contour lines for better visualization
        contour_lines = plt.contour(
            x_grid,
            y_grid,
            lyapunov_2d,
            levels=10,
            colors="white",
            alpha=0.6,
            linewidths=0.8,
        )
        plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

        # Mark target position
        plt.plot(
            set_point[0],
            set_point[1],
            "r*",
            markersize=15,
            label=f"Target: ({set_point[0]:.1f}, {set_point[1]:.1f}, {set_point[2]:.2f})",
        )

        # Add orientation arrow at target
        arrow_length = 0.3
        arrow_dx = arrow_length * np.cos(set_point[2])
        arrow_dy = arrow_length * np.sin(set_point[2])
        plt.arrow(
            set_point[0],
            set_point[1],
            arrow_dx,
            arrow_dy,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
            alpha=0.8,
        )

        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.title(f"Differential Robot Lyapunov Function\n(θ = {set_point[2]:.2f} rad)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")

        if interactive:
            plt.show()
        else:
            plt.savefig(f"{fname}_lyapunov.png", dpi=150, bbox_inches="tight")
            print(f"Lyapunov plot saved as: {fname}_lyapunov.png")

    # Generate the visualization
    draw_lyapunov_contours(set_point)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test differential robot with Lyapunov control"
    )
    parser.add_argument("--model", type=Path, help="model_path", default=None)
    parser.add_argument(
        "--random_actor", action="store_true", help="Use random actions"
    )
    parser.add_argument("--low_actor", action="store_true", help="Use zero actions")
    parser.add_argument(
        "--no_lyapunov", action="store_true", help="Skip Lyapunov visualization"
    )
    parser.add_argument("--no_test", action="store_true", help="Skip interactive test")
    parser.add_argument(
        "--interactive", action="store_true", help="Show interactive plots"
    )
    parser.add_argument("--target_x", type=float, default=0.0, help="Target X position")
    parser.add_argument("--target_y", type=float, default=0.0, help="Target Y position")
    parser.add_argument(
        "--target_theta", type=float, default=0.0, help="Target orientation (radians)"
    )
    parser.add_argument(
        "--seed", type=int, default=np.random.randint(100000), help="Random seed"
    )
    args = parser.parse_args()

    # Load model and extract environment name
    try:
        checkpoint_path = args.model if args.model else utils.latest_model()
        print("using checkpoint:", checkpoint_path)
    except:
        print("there are no trained models")
        exit()

    dynamics = utils.load_checkpoint(checkpoint_path)
    env_name = utils.extract_env_name(checkpoint_path)
    print(f"Environment: {env_name}")

    # Set up actor policy
    if args.random_actor:
        action_space = gym.make(env_name).action_space

        def actor(obs_dict, **ignored):
            raw_action = action_space.sample()
            return ensure_numpy_action(raw_action)

        print("Using random actor")
    elif args.low_actor:

        def actor(obs_dict, **ignored):
            return np.array([0.0, 0.0], dtype=np.float32)  # [v=0, w=0]

        print("Using zero actor")
    else:
        try:
            loaded_actor = keras.models.load_model(
                checkpoint_path.parent / "actor.keras"
            )
            loaded_actor.summary()

            def actor(obs_dict, **ignored):
                """Wrapped actor with proper tensor-to-array conversion"""
                raw_action = loaded_actor(obs_dict, **ignored)
                return ensure_numpy_action(raw_action)

            print("Using trained actor with tensor conversion")
        except Exception as e:
            print(f"Failed to load trained actor: {e}")
            print("Using random actor as fallback")
            action_space = gym.make(env_name).action_space

            def actor(obs_dict, **ignored):
                raw_action = action_space.sample()
                return ensure_numpy_action(raw_action)

    # Set up target setpoint for differential robot
    setpoint = position_to_setpoint(args.target_x, args.target_y, args.target_theta)
    print(
        f"Target setpoint: x={args.target_x}, y={args.target_y}, theta={args.target_theta:.2f}"
    )

    # Lyapunov function visualization
    lyapunov = None
    if not args.no_lyapunov:
        try:
            lyapunov = keras.models.load_model(
                checkpoint_path.parent / "lyapunov.keras"
            )
            print("Lyapunov function loaded, generating visualization...")
            plot_diff_robot_lyapunov_enhanced(
                lyapunov,
                actor,
                dynamics,
                setpoint,
                f"DiffRobot_target_{args.target_x}_{args.target_y}_{args.target_theta:.2f}",
                interactive=args.interactive,
            )
        except Exception as e:
            print(f"Could not load or visualize Lyapunov function: {e}")
            lyapunov = None

    # Exit if only visualization was requested
    if args.no_test:
        print("Visualization complete, exiting...")
        exit()

    def run_test(num_steps=2000):
        """
        Interactive Test Environment for Differential Robot

        Test Features:
        1. Side-by-side comparison: learned dynamics (left) vs analytical (right)
        2. Real-time Lyapunov values for monitoring controller performance
        3. Synchronized environments with identical initial conditions
        4. Performance metrics and episode management
        """
        print("Starting interactive test environment...")

        # Initialize pygame for visualization
        pygame.init()
        pygame.display.init()
        window = pygame.display.set_mode((500 * 2, 500))
        pygame.display.set_caption("Differential Robot: Learned vs Analytical Dynamics")
        surface1 = pygame.Surface((500, 500))  # Learned dynamics
        surface2 = pygame.Surface((500, 500))  # Analytical dynamics

        # Create both environments
        try:
            modeled_env = gym.make(
                "Modeled" + env_name,
                model_path=checkpoint_path,
                render_mode="human",
                screen=surface1,
            )
            print("✓ Modeled environment created")
        except Exception as e:
            print(f"❌ Failed to create modeled environment: {e}")
            return

        try:
            orig_env = gym.make(
                env_name,
                render_mode="human",
                screen=surface2,
            )
            print("✓ Analytical environment created")
        except Exception as e:
            print(f"❌ Failed to create analytical environment: {e}")
            return

        # Reset both environments with same seed for comparison
        print(f"Resetting environments with seed: {args.seed}")
        env_obs, _ = modeled_env.reset(
            seed=args.seed,
            options={
                "target_x": args.target_x,
                "target_y": args.target_y,
                "target_theta": args.target_theta,
            },
        )

        orig_env_obs, _ = orig_env.reset(
            seed=args.seed,
            options={
                "target_x": args.target_x,
                "target_y": args.target_y,
                "target_theta": args.target_theta,
            },
        )

        def create_observation_dict(obs, target_setpoint):
            """
            Create properly formatted observation dictionary for actor network

            The actor expects batched inputs, so we add the batch dimension here.
            """
            return {
                "state": np.expand_dims(obs, axis=0),  # (3,) → (1,3)
                "setpoint": np.expand_dims(target_setpoint, axis=0),  # (3,) → (1,3)
            }

        # Main simulation loop
        step_count = 0
        total_reward_modeled = 0.0
        total_reward_analytical = 0.0

        print("Starting simulation loop...")
        print("Press ESC to exit, SPACE to reset environments")

        for i in range(num_steps):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_SPACE:
                        print("Resetting environments...")
                        env_obs, _ = modeled_env.reset()
                        orig_env_obs, _ = orig_env.reset()
                        step_count = 0
                        total_reward_modeled = 0.0
                        total_reward_analytical = 0.0

            # Update display
            window.blits(((surface1, (0, 0)), (surface2, (500, 0))))
            pygame.display.flip()

            # Generate control actions for both environments
            modeled_obs_dict = create_observation_dict(env_obs, setpoint)
            analytical_obs_dict = create_observation_dict(orig_env_obs, setpoint)

            # Actor generates actions (already converted to NumPy by wrapper)
            modeled_action = actor(modeled_obs_dict, training=False)
            analytical_action = actor(analytical_obs_dict, training=False)

            # Optional: Display Lyapunov function value for monitoring
            if lyapunov and step_count % 10 == 0:  # Print every 10 steps to avoid spam
                lyap_value = lyapunov(modeled_obs_dict, training=False)
                print(
                    f"Step {step_count}: Lyapunov = {lyap_value.numpy().squeeze():.4f}"
                )

            # Execute environment steps
            try:
                env_obs, env_reward, env_done, env_term, env_info = modeled_env.step(
                    modeled_action
                )
                (
                    orig_env_obs,
                    orig_env_reward,
                    orig_env_done,
                    orig_env_term,
                    orig_env_info,
                ) = orig_env.step(analytical_action)

                # Accumulate rewards for performance comparison
                total_reward_modeled += env_reward
                total_reward_analytical += orig_env_reward
                step_count += 1

            except Exception as e:
                print(f"Error during environment step: {e}")
                break

            # Handle episode termination
            if env_done or env_term or orig_env_done or orig_env_term:
                print(f"\nEpisode completed after {step_count} steps!")
                print(f"Modeled environment total reward: {total_reward_modeled:.2f}")
                print(
                    f"Analytical environment total reward: {total_reward_analytical:.2f}"
                )
                print(
                    f"Reward difference: {abs(total_reward_modeled - total_reward_analytical):.2f}"
                )

                # Reset for next episode
                env_obs, _ = modeled_env.reset()
                orig_env_obs, _ = orig_env.reset()
                step_count = 0
                total_reward_modeled = 0.0
                total_reward_analytical = 0.0

            # Small delay to make visualization visible
            time.sleep(0.01)

        print(f"\nSimulation completed after {num_steps} steps")
        pygame.quit()

    # Run the interactive test
    run_test()
