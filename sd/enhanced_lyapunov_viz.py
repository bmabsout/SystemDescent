import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_diff_robot_lyapunov_enhanced(
    lyapunov, actor, dynamics, set_point, fname, interactive=False
):
    """
    Enhanced Lyapunov Function Visualization for Differential Robot
    
    Multiple visualization modes:
    1. 2D slices at different orientations
    2. 3D surface plot
    3. Trajectory-based visualization
    4. Gradient field visualization
    """
    
    def calculate_lyapunov_2d(set_point, fixed_theta=0.0):
        """Calculate 2D slice of Lyapunov function at fixed orientation"""
        pts = 50  # Reduced for faster computation
        
        x_range = np.linspace(set_point[0] - 2.0, set_point[0] + 2.0, pts)
        y_range = np.linspace(set_point[1] - 2.0, set_point[1] + 2.0, pts)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        
        theta_grid = np.full_like(x_grid, fixed_theta)
        states = np.stack([x_grid.flatten(), y_grid.flatten(), theta_grid.flatten()], axis=1)
        setpoints = np.tile(set_point.reshape(1, -1), (states.shape[0], 1))
        
        lyapunov_inputs = {"state": states, "setpoint": setpoints}
        lyapunov_values = lyapunov(lyapunov_inputs, training=False)
        lyapunov_2d = lyapunov_values.numpy().reshape(x_grid.shape)
        
        return x_grid, y_grid, lyapunov_2d

    def plot_multiple_theta_slices(set_point):
        """Plot 2D slices at different theta values"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for idx, theta in enumerate(theta_values):
            ax = axes[idx // 2, idx % 2]
            
            # Calculate slice at this theta
            x_grid, y_grid, lyapunov_2d = calculate_lyapunov_2d(set_point, theta)
            
            # Plot contours
            contour = ax.contourf(x_grid, y_grid, lyapunov_2d, levels=15, cmap="viridis", alpha=0.8)
            ax.contour(x_grid, y_grid, lyapunov_2d, levels=10, colors="white", alpha=0.6, linewidths=0.5)
            
            # Mark target
            ax.plot(set_point[0], set_point[1], "r*", markersize=10)
            
            # Add orientation arrow
            arrow_length = 0.3
            arrow_dx = arrow_length * np.cos(theta)
            arrow_dy = arrow_length * np.sin(theta)
            ax.arrow(set_point[0], set_point[1], arrow_dx, arrow_dy,
                    head_width=0.1, head_length=0.1, fc="red", ec="red", alpha=0.8)
            
            ax.set_title(f"θ = {theta:.2f} rad ({theta*180/np.pi:.0f}°)")
            ax.set_xlabel("X Position [m]")
            ax.set_ylabel("Y Position [m]")
            ax.grid(True, alpha=0.3)
            ax.axis("equal")
        
        plt.tight_layout()
        plt.suptitle(f"Lyapunov Function at Different Orientations\nTarget: ({set_point[0]:.1f}, {set_point[1]:.1f})", y=1.02)
        
        if interactive:
            plt.show()
        else:
            plt.savefig(f"{fname}_lyapunov_multi_theta.png", dpi=150, bbox_inches="tight")
            print(f"Multi-theta plot saved as: {fname}_lyapunov_multi_theta.png")

    def plot_3d_surface(set_point):
        """Plot 3D surface of Lyapunov function (fixing theta at target value)"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate 3D surface data
        x_grid, y_grid, lyapunov_2d = calculate_lyapunov_2d(set_point, set_point[2])
        
        # Create 3D surface plot
        surf = ax.plot_surface(x_grid, y_grid, lyapunov_2d, 
                              cmap='viridis', alpha=0.8, antialiased=True)
        
        # Add contour lines at bottom
        ax.contour(x_grid, y_grid, lyapunov_2d, zdir='z', 
                  offset=lyapunov_2d.min(), cmap='viridis', alpha=0.5)
        
        # Mark target point
        target_V = lyapunov({"state": set_point[None,:], "setpoint": set_point[None,:]})
        ax.scatter([set_point[0]], [set_point[1]], [target_V.numpy()[0,0]], 
                  color='red', s=100, label='Target')
        
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_zlabel('Lyapunov Value V(x,y,θ)')
        ax.set_title(f'3D Lyapunov Surface\nθ = {set_point[2]:.2f} rad')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        if interactive:
            plt.show()
        else:
            plt.savefig(f"{fname}_lyapunov_3d.png", dpi=150, bbox_inches="tight")
            print(f"3D surface plot saved as: {fname}_lyapunov_3d.png")

    def plot_gradient_field(set_point):
        """Plot gradient field showing direction of steepest descent"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate Lyapunov values and gradients
        x_grid, y_grid, lyapunov_2d = calculate_lyapunov_2d(set_point, set_point[2])
        
        # Calculate numerical gradients
        grad_y, grad_x = np.gradient(lyapunov_2d)
        
        # Subsample for cleaner arrow display
        step = 3
        x_arrows = x_grid[::step, ::step]
        y_arrows = y_grid[::step, ::step]
        u_arrows = -grad_x[::step, ::step]  # Negative for descent direction
        v_arrows = -grad_y[::step, ::step]
        
        # Plot contours
        contour = ax.contourf(x_grid, y_grid, lyapunov_2d, levels=15, cmap="viridis", alpha=0.6)
        ax.contour(x_grid, y_grid, lyapunov_2d, levels=10, colors="white", alpha=0.8, linewidths=0.5)
        
        # Plot gradient arrows
        ax.quiver(x_arrows, y_arrows, u_arrows, v_arrows, 
                 color='red', alpha=0.7, scale=10, width=0.003)
        
        # Mark target
        ax.plot(set_point[0], set_point[1], "yellow", marker="*", markersize=15, 
               markeredgecolor="black", markeredgewidth=1, label="Target")
        
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_title(f"Lyapunov Gradient Field (Descent Direction)\nθ = {set_point[2]:.2f} rad")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        
        plt.colorbar(contour, ax=ax, label="Lyapunov Value")
        
        if interactive:
            plt.show()
        else:
            plt.savefig(f"{fname}_lyapunov_gradient.png", dpi=150, bbox_inches="tight")
            print(f"Gradient field plot saved as: {fname}_lyapunov_gradient.png")

    def plot_trajectory_lyapunov(set_point, actor, dynamics, steps=50):
        """Plot how Lyapunov value changes along a trajectory"""
        # Initialize random starting position
        start_state = np.array([
            set_point[0] + np.random.uniform(-1.5, 1.5),
            set_point[1] + np.random.uniform(-1.5, 1.5),
            np.random.uniform(-np.pi, np.pi)
        ])
        
        # Simulate trajectory
        states = [start_state.copy()]
        lyapunov_values = []
        current_state = start_state.copy()
        
        for _ in range(steps):
            # Get Lyapunov value
            state_input = {"state": current_state[None,:], "setpoint": set_point[None,:]}
            V_val = lyapunov(state_input, training=False)
            lyapunov_values.append(V_val.numpy()[0,0])
            
            # Get action from controller
            action = actor(state_input, training=False)
            
            # Apply dynamics (simplified for visualization)
            dt = 0.1
            v, omega = action[0], action[1]
            current_state[0] += v * np.cos(current_state[2]) * dt
            current_state[1] += v * np.sin(current_state[2]) * dt
            current_state[2] += omega * dt
            
            states.append(current_state.copy())
        
        states = np.array(states)
        
        # Create trajectory plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Trajectory in space
        ax1.plot(states[:, 0], states[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        ax1.plot(states[0, 0], states[0, 1], 'go', markersize=8, label='Start')
        ax1.plot(set_point[0], set_point[1], 'r*', markersize=15, label='Target')
        
        # Color trajectory by Lyapunov value
        scatter = ax1.scatter(states[:-1, 0], states[:-1, 1], c=lyapunov_values, 
                             cmap='viridis', s=30, alpha=0.8)
        plt.colorbar(scatter, ax=ax1, label='Lyapunov Value')
        
        ax1.set_xlabel('X Position [m]')
        ax1.set_ylabel('Y Position [m]')
        ax1.set_title('Robot Trajectory (colored by V value)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Lyapunov value over time
        ax2.plot(lyapunov_values, 'b-', linewidth=2)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Lyapunov Value V(x,x*)')
        ax2.set_title('Lyapunov Value Along Trajectory')
        ax2.grid(True, alpha=0.3)
        
        # Add horizontal line at V=0
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='V=0 (target)')
        ax2.legend()
        
        plt.tight_layout()
        
        if interactive:
            plt.show()
        else:
            plt.savefig(f"{fname}_lyapunov_trajectory.png", dpi=150, bbox_inches="tight")
            print(f"Trajectory plot saved as: {fname}_lyapunov_trajectory.png")

    # Generate all visualizations
    print("Generating enhanced Lyapunov visualizations...")
    
    try:
        plot_multiple_theta_slices(set_point)
        plot_3d_surface(set_point)
        plot_gradient_field(set_point)
        plot_trajectory_lyapunov(set_point, actor, dynamics)
        print("All visualizations completed!")
    except Exception as e:
        print(f"Error in enhanced visualization: {e}")
        # Fallback to original visualization
        print("Falling back to original 2D visualization...")
        plot_diff_robot_lyapunov_original(lyapunov, actor, dynamics, set_point, fname, interactive)

def plot_diff_robot_lyapunov_original(lyapunov, actor, dynamics, set_point, fname, interactive=False):
    """Original 2D visualization as fallback"""
    # Your existing implementation here
    pass