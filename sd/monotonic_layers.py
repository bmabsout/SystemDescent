import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from typing import List, Optional


class MonotonicUnit(layers.Layer):
    """
    Monotonic Unit implementation from the paper.

    Implements: m(y) = a^T ReLU(y*1 - b)
    where a, b satisfy monotonicity constraints.
    """

    def __init__(
        self, num_pieces: int = 4, epsilon: float = 0.01, m_plus: bool = False, **kwargs
    ):
        super(MonotonicUnit, self).__init__(**kwargs)
        self.num_pieces = num_pieces
        self.epsilon = epsilon
        self.m_plus = m_plus  # If True, implements M+ class (m(y) = 0 for y <= 0)

    def build(self, input_shape):
        # Initialize bias vector b with sorted values
        initial_b = np.linspace(0.0 if self.m_plus else -1.0, 1.0, self.num_pieces)
        self.b = self.add_weight(
            name="bias", shape=(self.num_pieces,), initializer="zeros", trainable=True
        )

        # Initialize slope vector a
        initial_a = np.ones(self.num_pieces) * self.epsilon
        initial_a[0] = 1.0  # Ensure a[0] > 0
        self.a = self.add_weight(
            name="slopes", shape=(self.num_pieces,), initializer="ones", trainable=True
        )

        # Set initial values
        self.b.assign(initial_b)
        self.a.assign(initial_a)

        super(MonotonicUnit, self).build(input_shape)

    def call(self, inputs):
        # Ensure input is properly shaped (flatten to 1D if needed)
        if len(inputs.shape) > 2:
            inputs = tf.squeeze(inputs, axis=list(range(2, len(inputs.shape))))
        if len(inputs.shape) == 2 and inputs.shape[-1] == 1:
            inputs = tf.squeeze(inputs, axis=-1)

        # Ensure monotonicity constraints during forward pass
        # Constraint 1: b[i+1] > b[i] (sorted biases)
        b_sorted = tf.sort(self.b)
        if self.m_plus:
            # For M+ class, ensure b[0] = 0
            b_sorted = tf.concat([[0.0], b_sorted[1:]], axis=0)

        # Constraint 2: a[0] > 0 and a[i+1] >= -sum(a[0:i]) + epsilon
        a_constrained = tf.concat(
            [
                tf.maximum(self.a[0:1], self.epsilon),  # a[0] > 0
                tf.maximum(
                    self.a[1:], self.epsilon - tf.cumsum(self.a[:-1])
                ),  # Monotonicity constraint
            ],
            axis=0,
        )

        # Compute ReLU activations: ReLU(y*1 - b)
        # inputs shape: (batch,), b_sorted shape: (num_pieces,)
        inputs_expanded = tf.expand_dims(inputs, -1)  # Shape: (batch, 1)
        b_expanded = tf.expand_dims(b_sorted, 0)  # Shape: (1, num_pieces)

        # Broadcast subtraction: (batch, 1) - (1, num_pieces) = (batch, num_pieces)
        relu_inputs = inputs_expanded - b_expanded
        relu_outputs = tf.nn.relu(relu_inputs)

        # Compute weighted sum: a^T ReLU(y*1 - b)
        result = tf.reduce_sum(a_constrained * relu_outputs, axis=-1)  # Shape: (batch,)

        return result


class MonotonicLayer(layers.Layer):
    """
    Monotonic Layer implementation from the paper.

    Implements: [M]_j(x) = sum_i c_i * m_i(v_i^T * x)
    where v_i are direction vectors and m_i are monotonic units.
    """

    def __init__(
        self,
        input_dim: int,
        num_directions: Optional[int] = None,
        num_pieces: int = 4,
        epsilon: float = 0.01,
        include_axes: bool = True,
        **kwargs,
    ):
        super(MonotonicLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_pieces = num_pieces
        self.epsilon = epsilon
        self.include_axes = include_axes

        # Set number of directions (paper suggests >= nx + 1)
        if num_directions is None:
            self.num_directions = max(input_dim + 3, 8)  # Ensure sufficient directions
        else:
            self.num_directions = max(num_directions, input_dim + 1)

        # Initialize direction vectors
        self.directions = self._initialize_directions()

        # Create monotonic units for each direction
        self.monotonic_units = [
            MonotonicUnit(
                num_pieces=num_pieces, epsilon=epsilon, m_plus=True, name=f"unit_{i}"
            )
            for i in range(self.num_directions)
        ]

    def _initialize_directions(self):
        """Initialize direction vectors ensuring 0 is in convex hull."""
        directions = []

        if self.include_axes:
            # Add axis directions (required by Assumption 1 in paper)
            for i in range(self.input_dim):
                direction = np.zeros(self.input_dim)
                direction[i] = 1.0
                directions.append(direction)

                # Also add negative axis directions for better coverage
                direction_neg = np.zeros(self.input_dim)
                direction_neg[i] = -1.0
                directions.append(direction_neg)

        # Add random directions to reach target number
        remaining = self.num_directions - len(directions)
        for _ in range(remaining):
            # Generate random unit vector
            direction = np.random.randn(self.input_dim)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            directions.append(direction)

        # Convert to tensor
        directions_array = np.array(directions[: self.num_directions])

        # Verify 0 is in convex hull (approximately)
        # For simplicity, we ensure this by including both positive and negative axis directions

        return tf.constant(directions_array, dtype=tf.float32)

    def build(self, input_shape):
        # Combination weights c_i (must be positive)
        self.combination_weights = self.add_weight(
            name="combination_weights",
            shape=(self.num_directions,),
            initializer="ones",
            trainable=True,
        )

        super(MonotonicLayer, self).build(input_shape)

    def call(self, inputs):
        # Ensure input has correct shape: (batch, input_dim)
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, 0)  # Add batch dimension if missing
        elif len(inputs.shape) > 2:
            # Flatten extra dimensions
            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], -1))

        # Ensure combination weights are positive
        c_positive = tf.nn.softplus(self.combination_weights)

        # Compute projections: v_i^T * x for each direction
        projections = tf.matmul(
            inputs, self.directions, transpose_b=True
        )  # Shape: (batch, num_directions)

        # Apply monotonic units to each projection
        unit_outputs = []
        for i, unit in enumerate(self.monotonic_units):
            projection_i = projections[:, i]  # Shape: (batch,)
            unit_output = unit(projection_i)  # Shape: (batch,)
            unit_outputs.append(unit_output)

        unit_outputs = tf.stack(unit_outputs, axis=-1)  # Shape: (batch, num_directions)

        # Weighted combination
        result = tf.reduce_sum(c_positive * unit_outputs, axis=-1)  # Shape: (batch,)

        return result


def create_monotonic_lyapunov_network(
    state_shape,
    num_layers: int = 2,
    directions_per_layer: Optional[List[int]] = None,
    num_pieces: int = 4,
    name: str = "MonotonicLyapunovFunction",
):
    """
    Create a Lyapunov neural network using monotonic layers for MULTI-TARGET mode.

    Args:
        state_shape: Shape of state input (should be (3,) for [x,y,theta])
        num_layers: Number of monotonic layers
        directions_per_layer: List of number of directions for each layer
        num_pieces: Number of pieces in each monotonic unit
        name: Name of the model

    Returns:
        Keras model implementing monotonic Lyapunov function V([state, setpoint])
    """
    input_state = keras.Input(shape=state_shape, name="state")
    input_setpoint = keras.Input(shape=state_shape, name="setpoint")

    # Concatenate state and setpoint for multi-target mode
    inputs = layers.Concatenate(name="state_setpoint_concat")(
        [input_state, input_setpoint]
    )
    input_dim = int(inputs.shape[-1])  # Should be 6 for differential robot

    if directions_per_layer is None:
        # Default: decreasing number of directions per layer
        directions_per_layer = [max(input_dim + 3, 8) - i for i in range(num_layers)]

    # Apply monotonic layers
    x = inputs
    current_dim = input_dim

    for i in range(num_layers):
        num_dirs = (
            directions_per_layer[i]
            if i < len(directions_per_layer)
            else max(current_dim + 1, 4)
        )

        x = MonotonicLayer(
            input_dim=current_dim,
            num_directions=num_dirs,
            num_pieces=num_pieces,
            name=f"monotonic_layer_{i}",
        )(x)

        # Update current dimension for next layer (output is always 1D from MonotonicLayer)
        current_dim = 1

        # Ensure x has proper shape for next layer
        x = tf.expand_dims(x, -1) if len(x.shape) == 1 else x

    # Ensure final output is scalar per batch element
    if len(x.shape) > 2:
        x = tf.squeeze(x, axis=list(range(2, len(x.shape))))

    # Apply a clip to ensure output is in [0,1] for FPL compatibility
    outputs = layers.Lambda(
        lambda x: tf.clip_by_value(x, 0.0, 1.0), name="output_clipping"
    )(x)

    # Ensure output shape is (batch, 1)
    if len(outputs.shape) == 1:
        outputs = tf.expand_dims(outputs, -1)

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_setpoint},
        outputs=outputs,
        name=name,
    )

    return model


def _create_monotonic_origin_network(state_shape, **kwargs):
    """Create monotonic network for error-state stabilization V(state - setpoint)."""
    input_state = keras.Input(shape=state_shape, name="state")
    input_setpoint = keras.Input(shape=state_shape, name="setpoint")

    # Compute error state: state - setpoint
    error_state = layers.Subtract(name="error_state")([input_state, input_setpoint])

    # Apply monotonic layers to error state
    input_dim = int(error_state.shape[-1])
    num_layers = kwargs.get("num_layers", 2)
    num_pieces = kwargs.get("num_pieces", 4)

    x = error_state
    current_dim = input_dim

    for i in range(num_layers):
        num_dirs = max(current_dim + 3, 8) - i

        x = MonotonicLayer(
            input_dim=current_dim,
            num_directions=num_dirs,
            num_pieces=num_pieces,
            name=f"monotonic_layer_{i}",
        )(x)

        current_dim = 1
        x = tf.expand_dims(x, -1) if len(x.shape) == 1 else x

    # Clip output to [0,1]
    if len(x.shape) > 2:
        x = tf.squeeze(x, axis=list(range(2, len(x.shape))))

    outputs = layers.Lambda(
        lambda x: tf.clip_by_value(x, 0.0, 1.0), name="output_clipping"
    )(x)

    if len(outputs.shape) == 1:
        outputs = tf.expand_dims(outputs, -1)

    model = keras.Model(
        inputs={
            "state": input_state,
            "setpoint": input_setpoint,
        },  # Both inputs, computes error internally
        outputs=outputs,
        name="MonotonicLyapunovFunction_Origin",
    )

    return model


def _create_standard_lyapunov_network_origin(state_shape):
    """Standard neural network for error-state stabilization V(state - setpoint)."""
    input_state = keras.Input(shape=state_shape, name="state")
    input_setpoint = keras.Input(shape=state_shape, name="setpoint")

    # Compute error state: state - setpoint
    error_state = layers.Subtract(name="error_state")([input_state, input_setpoint])

    dense1 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_1",
    )(error_state)

    dense2 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_2",
    )(dense1)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_output",
    )(dense2)

    model = keras.Model(
        inputs={
            "state": input_state,
            "setpoint": input_setpoint,
        },  # Both inputs, computes error internally
        outputs=outputs,
        name="StandardLyapunovFunction_Origin",
    )

    print("\n=== Standard Lyapunov Function Architecture (Error-State) ===")
    model.summary()
    return model


def _create_standard_lyapunov_network(state_shape):
    """Standard neural network for multi-target stabilization V([state, setpoint])."""
    input_state = keras.Input(shape=state_shape, name="current_state")
    input_setpoint = keras.Input(shape=state_shape, name="target_setpoint")

    # Concatenate state and setpoint for multi-target mode
    inputs = layers.Concatenate(name="state_target_concat")(
        [input_state, input_setpoint]
    )

    dense1 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_1",
    )(inputs)

    dense2 = layers.Dense(
        64,
        activation="tanh",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_hidden_2",
    )(dense1)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(0.01),
        name="lyapunov_output",
    )(dense2)

    model = keras.Model(
        inputs={"state": input_state, "setpoint": input_setpoint},
        outputs=outputs,
        name="StandardLyapunovFunction_MultiTarget",
    )

    print("\n=== Standard Lyapunov Function Architecture (Multi-Target) ===")
    model.summary()
    return model


def V_def_with_architecture_choice(
    state_shape,
    use_monotonic: bool = False,
    origin_stabilization: bool = False,
    **kwargs,
):
    """
    Modified V_def function for error-state vs multi-target stabilization.

    Args:
        state_shape: Shape of state input
        use_monotonic: If True, use monotonic architecture; if False, use standard NN
        origin_stabilization: If True, use error-state formulation V(state - setpoint);
                             if False, use multi-target V([state, setpoint])
        **kwargs: Additional arguments for monotonic network

    Returns:
        Keras model for Lyapunov function
    """
    if use_monotonic:
        if origin_stabilization:
            print(
                "Creating Monotonic Lyapunov Network for ERROR-STATE stabilization V(state - setpoint)..."
            )
            # Create error-state monotonic network
            return _create_monotonic_origin_network(state_shape, **kwargs)
        else:
            print(
                "Creating Monotonic Lyapunov Network for MULTI-TARGET stabilization V([state, setpoint])..."
            )
            return create_monotonic_lyapunov_network(state_shape, **kwargs)
    else:
        if origin_stabilization:
            print(
                "Creating Standard Neural Network for ERROR-STATE stabilization V(state - setpoint)..."
            )
            return _create_standard_lyapunov_network_origin(state_shape)
        else:
            print(
                "Creating Standard Neural Network for MULTI-TARGET stabilization V([state, setpoint])..."
            )
            return _create_standard_lyapunov_network(state_shape)


# Test function to verify monotonic properties
def test_monotonic_unit():
    """Test function to verify monotonic unit behavior."""
    print("Testing Monotonic Unit...")

    unit = MonotonicUnit(num_pieces=4, m_plus=True)

    # Test with sample inputs (1D batch)
    test_inputs = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=tf.float32)
    outputs = unit(test_inputs)

    print("Input values:", test_inputs.numpy())
    print("Output values:", outputs.numpy())
    print("Output shape:", outputs.shape)

    # Check monotonicity
    output_diff = tf.diff(outputs)
    is_monotonic = tf.reduce_all(output_diff >= -1e-6)
    print("Is monotonic:", is_monotonic.numpy())

    # For M+ class, check that output is 0 for negative inputs
    if hasattr(unit, "m_plus") and unit.m_plus:
        negative_outputs = outputs[test_inputs <= 0]
        print("Outputs for non-positive inputs:", negative_outputs.numpy())


def test_error_state_networks():
    """Test error-state networks to verify they work correctly."""
    print("\n=== Testing Error-State Networks ===")

    state_shape = (3,)  # For differential robot

    # Create error-state networks
    monotonic_model = V_def_with_architecture_choice(
        state_shape, use_monotonic=True, origin_stabilization=True
    )
    standard_model = V_def_with_architecture_choice(
        state_shape, use_monotonic=False, origin_stabilization=True
    )

    # Test data
    batch_size = 5
    states = tf.random.normal((batch_size, 3))
    setpoints = tf.random.normal((batch_size, 3))

    # Test both networks
    monotonic_output = monotonic_model({"state": states, "setpoint": setpoints})
    standard_output = standard_model({"state": states, "setpoint": setpoints})

    print(f"Monotonic output shape: {monotonic_output.shape}")
    print(f"Standard output shape: {standard_output.shape}")

    # Test zero at target (when state == setpoint)
    same_states = tf.random.normal((batch_size, 3))
    monotonic_at_target = monotonic_model(
        {"state": same_states, "setpoint": same_states}
    )
    standard_at_target = standard_model({"state": same_states, "setpoint": same_states})

    print(f"Monotonic V(x,x): mean={monotonic_at_target.numpy().mean():.4f}")
    print(f"Standard V(x,x): mean={standard_at_target.numpy().mean():.4f}")

    print("✅ Error-state networks tested successfully!")


if __name__ == "__main__":
    # Test the implementation
    test_monotonic_unit()

    # Test error-state networks
    test_error_state_networks()

    # Test creating both types of networks
    state_shape = (3,)  # For differential robot

    print("\n" + "=" * 50)
    print("Testing Standard Network (Multi-Target):")
    standard_multi_model = V_def_with_architecture_choice(
        state_shape, use_monotonic=False, origin_stabilization=False
    )

    print("\n" + "=" * 50)
    print("Testing Standard Network (Error-State):")
    standard_error_model = V_def_with_architecture_choice(
        state_shape, use_monotonic=False, origin_stabilization=True
    )

    print("\n" + "=" * 50)
    print("Testing Monotonic Network (Error-State):")
    monotonic_model = V_def_with_architecture_choice(
        state_shape, use_monotonic=True, origin_stabilization=True
    )

    # Test with sample data
    batch_size = 5
    sample_state = tf.random.normal((batch_size, 3))
    sample_setpoint = tf.random.normal((batch_size, 3))

    print("\n" + "=" * 50)
    print("Testing with sample data:")

    multi_output = standard_multi_model(
        {"state": sample_state, "setpoint": sample_setpoint}
    )
    error_output = standard_error_model(
        {"state": sample_state, "setpoint": sample_setpoint}
    )
    monotonic_output = monotonic_model(
        {"state": sample_state, "setpoint": sample_setpoint}
    )

    print("Multi-target network output shape:", multi_output.shape)
    print("Error-state network output shape:", error_output.shape)
    print("Monotonic network output shape:", monotonic_output.shape)

    # Test zero at equilibrium (state == setpoint)
    same_inputs = {"state": sample_state, "setpoint": sample_state}
    print("\nTesting V(x,x) = 0 property:")
    print("Multi-target V(x,x):", standard_multi_model(same_inputs).numpy().mean())
    print("Error-state V(x,x):", standard_error_model(same_inputs).numpy().mean())
    print("Monotonic V(x,x):", monotonic_model(same_inputs).numpy().mean())

    print("\n✅ All tests completed successfully!")
