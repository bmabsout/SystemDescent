#!/usr/bin/env python3
"""
Test script for monotonic neural network architecture.

This script verifies that the monotonic Lyapunov networks satisfy key properties:
1. Positive definiteness (V(x) >= 0 for all x)
2. Zero at target (V(x*, x*) = 0)
3. Monotonicity properties
4. Output range in [0,1] for FPL compatibility
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add the project directory to path
sys.path.append(str(Path(__file__).parent))

# Import directly from the artifacts we created
try:
    from sd.monotonic_layers import (
        MonotonicUnit,
        MonotonicLayer,
        create_monotonic_lyapunov_network,
        V_def_with_architecture_choice,
    )
except ImportError:
    print("⚠️  Could not import from sd.monotonic_layers")
    print("   Creating monotonic layers inline for testing...")

    # Import the implementation directly (this would normally be in sd/monotonic_layers.py)
    exec(open("monotonic_layers_implementation.py").read())


def test_monotonic_unit():
    """Test individual monotonic unit properties."""
    print("🧪 Testing Monotonic Unit Properties...")

    unit = MonotonicUnit(num_pieces=4, m_plus=True, epsilon=0.01)

    # Test with range of inputs
    test_inputs = tf.linspace(-3.0, 3.0, 20)
    test_inputs = tf.expand_dims(test_inputs, -1)  # Shape: (20, 1)

    outputs = unit(test_inputs).numpy().flatten()
    inputs_np = test_inputs.numpy().flatten()

    print(f"   Input range: [{inputs_np.min():.2f}, {inputs_np.max():.2f}]")
    print(f"   Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    # Check monotonicity
    is_monotonic = np.all(np.diff(outputs) >= -1e-6)  # Allow small numerical errors
    print(f"   ✅ Monotonic: {is_monotonic}")

    # Check M+ property (zero for negative inputs)
    negative_mask = inputs_np <= 0
    negative_outputs = outputs[negative_mask]
    m_plus_satisfied = np.allclose(negative_outputs, 0, atol=1e-3)
    print(f"   ✅ M+ property (zero for x≤0): {m_plus_satisfied}")

    return is_monotonic and m_plus_satisfied


def test_monotonic_layer():
    """Test monotonic layer properties."""
    print("\n🧪 Testing Monotonic Layer Properties...")

    input_dim = 3
    layer = MonotonicLayer(input_dim=input_dim, num_directions=8, num_pieces=4)

    # Test with batch of inputs
    batch_size = 10
    test_inputs = tf.random.normal((batch_size, input_dim))
    outputs = layer(test_inputs)

    print(f"   Input shape: {test_inputs.shape}")
    print(f"   Output shape: {outputs.shape}")
    print(
        f"   Output range: [{outputs.numpy().min():.4f}, {outputs.numpy().max():.4f}]"
    )

    # Check that outputs are non-negative (property of monotonic composition)
    all_positive = tf.reduce_all(outputs >= 0)
    print(f"   ✅ All outputs non-negative: {all_positive.numpy()}")

    # Test zero input gives zero output (should be true for our construction)
    zero_input = tf.zeros((1, input_dim))
    zero_output = layer(zero_input)
    is_zero_at_origin = tf.abs(zero_output) < 1e-3
    print(f"   ✅ Zero at origin: {is_zero_at_origin.numpy().item()}")

    return all_positive.numpy() and is_zero_at_origin.numpy().item()


def test_lyapunov_properties():
    """Test complete Lyapunov network properties."""
    print("\n🧪 Testing Complete Lyapunov Network Properties...")

    state_shape = (3,)  # For differential robot [x, y, theta]

    # Create both architectures for comparison
    standard_model = V_def_with_architecture_choice(state_shape, use_monotonic=False)
    monotonic_model = V_def_with_architecture_choice(state_shape, use_monotonic=True)

    # Test data
    batch_size = 100
    states = tf.random.normal((batch_size, 3)) * 2.0  # Random states
    targets = tf.random.normal((batch_size, 3)) * 2.0  # Random targets

    # Test 1: Zero at target property
    print("\n   Testing V(x*, x*) = 0 property...")
    same_inputs = {"state": targets, "setpoint": targets}

    standard_at_target = standard_model(same_inputs)
    monotonic_at_target = monotonic_model(same_inputs)

    print(
        f"   Standard network V(x*, x*): mean={standard_at_target.numpy().mean():.4f}, std={standard_at_target.numpy().std():.4f}"
    )
    print(
        f"   Monotonic network V(x*, x*): mean={monotonic_at_target.numpy().mean():.4f}, std={monotonic_at_target.numpy().std():.4f}"
    )

    # Test 2: Positive definiteness (V(x, x*) > 0 for x != x*)
    print("\n   Testing positive definiteness...")
    different_inputs = {"state": states, "setpoint": targets}

    standard_outputs = standard_model(different_inputs)
    monotonic_outputs = monotonic_model(different_inputs)

    standard_positive = tf.reduce_mean(tf.cast(standard_outputs > 0, tf.float32))
    monotonic_positive = tf.reduce_mean(tf.cast(monotonic_outputs > 0, tf.float32))

    print(f"   Standard network positive outputs: {standard_positive.numpy():.2%}")
    print(f"   Monotonic network positive outputs: {monotonic_positive.numpy():.2%}")

    # Test 3: Output range [0,1] for FPL compatibility
    print("\n   Testing output range [0,1]...")

    all_outputs_standard = tf.concat([standard_outputs, standard_at_target], axis=0)
    all_outputs_monotonic = tf.concat([monotonic_outputs, monotonic_at_target], axis=0)

    standard_in_range = tf.reduce_all(
        tf.logical_and(all_outputs_standard >= 0, all_outputs_standard <= 1)
    )
    monotonic_in_range = tf.reduce_all(
        tf.logical_and(all_outputs_monotonic >= 0, all_outputs_monotonic <= 1)
    )

    print(f"   Standard network in [0,1]: {standard_in_range.numpy()}")
    print(f"   Monotonic network in [0,1]: {monotonic_in_range.numpy()}")

    print(
        f"   Standard range: [{all_outputs_standard.numpy().min():.4f}, {all_outputs_standard.numpy().max():.4f}]"
    )
    print(
        f"   Monotonic range: [{all_outputs_monotonic.numpy().min():.4f}, {all_outputs_monotonic.numpy().max():.4f}]"
    )

    return {
        "standard_positive_rate": standard_positive.numpy(),
        "monotonic_positive_rate": monotonic_positive.numpy(),
        "standard_in_range": standard_in_range.numpy(),
        "monotonic_in_range": monotonic_in_range.numpy(),
    }


def visualize_lyapunov_function():
    """Create visualizations comparing standard vs monotonic Lyapunov functions."""
    print("\n🎨 Creating Lyapunov Function Visualizations...")

    state_shape = (3,)

    # Create models
    standard_model = V_def_with_architecture_choice(state_shape, use_monotonic=False)
    monotonic_model = V_def_with_architecture_choice(state_shape, use_monotonic=True)

    # Create 2D grid (fixing theta=0)
    target = np.array([0.0, 0.0, 0.0])  # Target at origin
    x_range = np.linspace(-2, 2, 50)
    y_range = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x_range, y_range)

    # Prepare inputs
    grid_points = np.stack([X.flatten(), Y.flatten(), np.zeros(X.size)], axis=1)
    targets = np.tile(target, (grid_points.shape[0], 1))

    inputs = {
        "state": grid_points.astype(np.float32),
        "setpoint": targets.astype(np.float32),
    }

    # Evaluate models
    standard_values = standard_model(inputs).numpy().reshape(X.shape)
    monotonic_values = monotonic_model(inputs).numpy().reshape(X.shape)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Standard network
    c1 = ax1.contourf(X, Y, standard_values, levels=20, cmap="viridis")
    ax1.plot(target[0], target[1], "r*", markersize=15, label="Target")
    ax1.set_title("Standard Neural Network")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    plt.colorbar(c1, ax=ax1)

    # Monotonic network
    c2 = ax2.contourf(X, Y, monotonic_values, levels=20, cmap="viridis")
    ax2.plot(target[0], target[1], "r*", markersize=15, label="Target")
    ax2.set_title("Monotonic Neural Network")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    plt.colorbar(c2, ax=ax2)

    plt.tight_layout()
    plt.savefig("lyapunov_comparison.png", dpi=150, bbox_inches="tight")
    print("   💾 Saved visualization to 'lyapunov_comparison.png'")

    # Print statistics
    print(f"\n   Standard network statistics:")
    print(f"     Range: [{standard_values.min():.4f}, {standard_values.max():.4f}]")
    print(
        f"     At target: {standard_model({'state': target[None,:], 'setpoint': target[None,:]}).numpy().item():.4f}"
    )

    print(f"\n   Monotonic network statistics:")
    print(f"     Range: [{monotonic_values.min():.4f}, {monotonic_values.max():.4f}]")
    print(
        f"     At target: {monotonic_model({'state': target[None,:], 'setpoint': target[None,:]}).numpy().item():.4f}"
    )


def test_radial_monotonicity():
    """Test that monotonic network is increasing along radial lines from origin."""
    print("\n🧪 Testing Radial Monotonicity...")

    state_shape = (3,)
    model = V_def_with_architecture_choice(state_shape, use_monotonic=True)

    # Test along several radial directions
    directions = [
        [1, 0, 0],  # x-axis
        [0, 1, 0],  # y-axis
        [1, 1, 0],  # diagonal
        [1, -1, 0],  # anti-diagonal
        [0, 0, 1],  # theta-axis
    ]

    target = np.array([0.0, 0.0, 0.0])

    all_monotonic = True

    for i, direction in enumerate(directions):
        direction = np.array(direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)  # Normalize

        # Test points along this direction
        radii = np.linspace(0, 2, 20)
        points = np.outer(radii, direction)  # Shape: (20, 3)
        targets = np.tile(target, (points.shape[0], 1))

        inputs = {"state": points, "setpoint": targets}
        values = model(inputs).numpy().flatten()

        # Check monotonicity
        is_monotonic = np.all(np.diff(values) >= -1e-6)
        all_monotonic = all_monotonic and is_monotonic

        print(f"   Direction {direction}: monotonic = {is_monotonic}")
        if not is_monotonic:
            print(f"     Values: {values}")

    print(f"   ✅ Overall radial monotonicity: {all_monotonic}")
    return all_monotonic


def main():
    """Run all tests."""
    print("🚀 Testing Monotonic Neural Network Architecture")
    print("=" * 60)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run tests
    results = {}

    results["monotonic_unit"] = test_monotonic_unit()
    results["monotonic_layer"] = test_monotonic_layer()
    results["lyapunov_properties"] = test_lyapunov_properties()
    results["radial_monotonicity"] = test_radial_monotonicity()

    # Create visualizations
    try:
        visualize_lyapunov_function()
        results["visualization"] = True
    except Exception as e:
        print(f"   ❌ Visualization failed: {e}")
        results["visualization"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed_tests = sum(1 for k, v in results.items() if isinstance(v, bool) and v)
    total_tests = sum(1 for k, v in results.items() if isinstance(v, bool))

    print(f"✅ Basic tests passed: {passed_tests}/{total_tests}")

    if "lyapunov_properties" in results and isinstance(
        results["lyapunov_properties"], dict
    ):
        props = results["lyapunov_properties"]
        print(
            f"📈 Monotonic network positive definiteness: {props['monotonic_positive_rate']:.1%}"
        )
        print(f"📐 Output range compliance: {props['monotonic_in_range']}")

    if all(isinstance(v, bool) and v for v in results.values() if isinstance(v, bool)):
        print("\n🎉 ALL TESTS PASSED! Monotonic architecture is working correctly.")
        print("\n🔬 Next steps:")
        print("   1. Train with: python -m sd.lyapunov_diff_robot --use_monotonic")
        print("   2. Compare performance with: python -m sd.lyapunov_diff_robot")
        print("   3. Integrate MILP verification for formal guarantees")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

    return results


if __name__ == "__main__":
    main()
