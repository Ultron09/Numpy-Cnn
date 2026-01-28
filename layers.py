"""
Core Layer Classes for NumPy CNN Implementation

This module contains the foundational layer classes used to build CNN architectures.
All operations are implemented using only NumPy for educational purposes.

Mathematical foundations are documented inline to aid understanding.
"""

import numpy as np
from typing import Optional, Tuple


class Layer:
    """
    Base class for all neural network layers.
    
    Defines the interface that all layers must implement:
    - forward(): Computes the forward pass
    - backward(): Computes gradients during backpropagation
    - get_params(): Returns trainable parameters
    - set_params(): Updates trainable parameters
    """
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass computation.
        
        Args:
            inputs: Input data of shape depending on the layer type
            training: Whether the layer is in training mode (affects dropout, etc.)
            
        Returns:
            Output of the layer
        """
        raise NotImplementedError
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass computation.
        
        Args:
            grad_output: Gradient of loss with respect to layer output
            
        Returns:
            Gradient of loss with respect to layer input
        """
        raise NotImplementedError
    
    def get_params(self) -> dict:
        """Returns dictionary of trainable parameters and their gradients."""
        return {}
    
    def set_params(self, params: dict):
        """Updates trainable parameters."""
        pass


class Conv2D(Layer):
    """
    2D Convolution Layer
    
    Performs 2D convolution operation on input images.
    
    Mathematical Operation:
    For input X with shape (batch_size, in_channels, height, width) and
    kernel W with shape (out_channels, in_channels, kernel_h, kernel_w):
    
    Output[n, c, h, w] = Σ Σ Σ X[n, k, h*stride+i, w*stride+j] * W[c, k, i, j] + bias[c]
                        k i j
    
    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output filters/channels
        kernel_size: Size of the convolution kernel (int or tuple)
        stride: Stride of the convolution (default: 1)
        padding: Padding added to input (default: 0)
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using He initialization
        # He et al. "Delving Deep into Rectifiers" - optimal for ReLU activations
        # W ~ N(0, sqrt(2 / n_in)) where n_in is the number of input units
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weights = np.random.randn(out_channels, in_channels, 
                                       self.kernel_size[0], self.kernel_size[1]) * np.sqrt(2.0 / fan_in)
        
        # Initialize biases to zero
        self.bias = np.zeros(out_channels)
        
        # Gradients (initialized during backward pass)
        self.grad_weights = None
        self.grad_bias = None
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of 2D convolution.
        
        Args:
            inputs: Input tensor of shape (batch_size, in_channels, height, width)
            training: Whether in training mode
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        batch_size, in_channels, in_h, in_w = inputs.shape
        
        # Apply padding if necessary
        if self.padding > 0:
            inputs = np.pad(inputs, 
                          ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant', constant_values=0)
        
        # Calculate output dimensions
        out_h = (in_h + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Cache input for backward pass
        self.cache = inputs
        
        # Perform convolution
        # This is the naive implementation for educational clarity
        # Production implementations use im2col for better performance
        for n in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract receptive field
                        receptive_field = inputs[n, :, h_start:h_end, w_start:w_end]
                        
                        # Compute convolution: element-wise multiply and sum
                        output[n, c_out, h, w] = np.sum(receptive_field * self.weights[c_out]) + self.bias[c_out]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of 2D convolution.
        
        Computes gradients with respect to:
        - Input (returned)
        - Weights (stored in self.grad_weights)
        - Bias (stored in self.grad_bias)
        
        Args:
            grad_output: Gradient of loss w.r.t. output, shape (batch_size, out_channels, out_h, out_w)
            
        Returns:
            Gradient of loss w.r.t. input
        """
        inputs = self.cache
        batch_size, in_channels, in_h, in_w = inputs.shape
        _, out_channels, out_h, out_w = grad_output.shape
        
        # Initialize gradients
        grad_input = np.zeros_like(inputs)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
        # Compute gradients
        for n in range(batch_size):
            for c_out in range(out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size[1]
                        
                        # Gradient w.r.t. bias: sum of upstream gradients
                        self.grad_bias[c_out] += grad_output[n, c_out, h, w]
                        
                        # Gradient w.r.t. weights: upstream gradient * input
                        self.grad_weights[c_out] += grad_output[n, c_out, h, w] * inputs[n, :, h_start:h_end, w_start:w_end]
                        
                        # Gradient w.r.t. input: upstream gradient * weights
                        grad_input[n, :, h_start:h_end, w_start:w_end] += grad_output[n, c_out, h, w] * self.weights[c_out]
        
        # Remove padding from gradient if it was applied
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input
    
    def get_params(self) -> dict:
        """Returns trainable parameters and their gradients."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'grad_weights': self.grad_weights,
            'grad_bias': self.grad_bias
        }
    
    def set_params(self, params: dict):
        """Updates trainable parameters."""
        if 'weights' in params:
            self.weights = params['weights']
        if 'bias' in params:
            self.bias = params['bias']


class MaxPool2D(Layer):
    """
    Max Pooling Layer
    
    Performs max pooling operation to reduce spatial dimensions.
    
    Mathematical Operation:
    Output[n, c, h, w] = max(X[n, c, h*pool_size:(h+1)*pool_size, w*pool_size:(w+1)*pool_size])
    
    For each pooling window, outputs the maximum value.
    
    Parameters:
        pool_size: Size of the pooling window (int or tuple)
        stride: Stride for pooling (default: same as pool_size)
    """
    
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of max pooling.
        
        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width)
            training: Whether in training mode
            
        Returns:
            Output tensor with reduced spatial dimensions
        """
        batch_size, channels, in_h, in_w = inputs.shape
        
        # Calculate output dimensions
        out_h = (in_h - self.pool_size[0]) // self.stride + 1
        out_w = (in_w - self.pool_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        
        # Cache for backward pass: store indices of maximum values
        self.cache = {'input_shape': inputs.shape, 'max_indices': {}}
        
        for n in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size[1]
                        
                        # Extract pooling window
                        pool_region = inputs[n, c, h_start:h_end, w_start:w_end]
                        
                        # Find maximum value
                        output[n, c, h, w] = np.max(pool_region)
                        
                        # Store the position of max value for backward pass
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.cache['max_indices'][(n, c, h, w)] = (
                            h_start + max_idx[0],
                            w_start + max_idx[1]
                        )
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of max pooling.
        
        Gradient flows only through the positions that were selected as maximum.
        
        Args:
            grad_output: Gradient of loss w.r.t. output
            
        Returns:
            Gradient of loss w.r.t. input
        """
        grad_input = np.zeros(self.cache['input_shape'])
        batch_size, channels, out_h, out_w = grad_output.shape
        
        # Route gradients to the positions of maximum values
        for n in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        max_h, max_w = self.cache['max_indices'][(n, c, h, w)]
                        grad_input[n, c, max_h, max_w] += grad_output[n, c, h, w]
        
        return grad_input


class AvgPool2D(Layer):
    """
    Average Pooling Layer
    
    Performs average pooling operation.
    
    Mathematical Operation:
    Output[n, c, h, w] = mean(X[n, c, h*pool_size:(h+1)*pool_size, w*pool_size:(w+1)*pool_size])
    
    Parameters:
        pool_size: Size of the pooling window
        stride: Stride for pooling (default: same as pool_size)
    """
    
    def __init__(self, pool_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else pool_size
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of average pooling."""
        batch_size, channels, in_h, in_w = inputs.shape
        
        out_h = (in_h - self.pool_size[0]) // self.stride + 1
        out_w = (in_w - self.pool_size[1]) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.cache = inputs.shape
        
        for n in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size[1]
                        
                        pool_region = inputs[n, c, h_start:h_end, w_start:w_end]
                        output[n, c, h, w] = np.mean(pool_region)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of average pooling.
        
        Gradient is distributed equally to all positions in the pooling window.
        """
        grad_input = np.zeros(self.cache)
        batch_size, channels, out_h, out_w = grad_output.shape
        
        # Each position in the pooling window gets equal share of gradient
        gradient_per_position = 1.0 / (self.pool_size[0] * self.pool_size[1])
        
        for n in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size[0]
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size[1]
                        
                        grad_input[n, c, h_start:h_end, w_start:w_end] += (
                            grad_output[n, c, h, w] * gradient_per_position
                        )
        
        return grad_input


class Flatten(Layer):
    """
    Flatten Layer
    
    Flattens input from (batch_size, channels, height, width) to (batch_size, features).
    Used to transition from convolutional layers to fully connected layers.
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Flatten spatial dimensions."""
        self.cache = inputs.shape
        batch_size = inputs.shape[0]
        return inputs.reshape(batch_size, -1)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Reshape gradient back to original spatial dimensions."""
        return grad_output.reshape(self.cache)


class Dense(Layer):
    """
    Fully Connected (Dense) Layer
    
    Mathematical Operation:
    Output = X @ W + b
    
    where X is input, W is weight matrix, b is bias vector, @ is matrix multiplication.
    
    Parameters:
        in_features: Number of input features
        out_features: Number of output features
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        # He initialization
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        
        self.grad_weights = None
        self.grad_bias = None
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass: output = input @ weights + bias
        
        Args:
            inputs: Shape (batch_size, in_features)
            
        Returns:
            Output of shape (batch_size, out_features)
        """
        self.cache = inputs
        return inputs @ self.weights + self.bias
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using chain rule.
        
        d_Loss/d_input = d_Loss/d_output @ W^T
        d_Loss/d_W = X^T @ d_Loss/d_output
        d_Loss/d_b = sum(d_Loss/d_output, axis=0)
        """
        inputs = self.cache
        
        # Gradient w.r.t. input
        grad_input = grad_output @ self.weights.T
        
        # Gradient w.r.t. weights
        self.grad_weights = inputs.T @ grad_output
        
        # Gradient w.r.t. bias
        self.grad_bias = np.sum(grad_output, axis=0)
        
        return grad_input
    
    def get_params(self) -> dict:
        """Returns trainable parameters and their gradients."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'grad_weights': self.grad_weights,
            'grad_bias': self.grad_bias
        }
    
    def set_params(self, params: dict):
        """Updates trainable parameters."""
        if 'weights' in params:
            self.weights = params['weights']
        if 'bias' in params:
            self.bias = params['bias']


class Dropout(Layer):
    """
    Dropout Regularization Layer
    
    During training: Randomly sets a fraction of inputs to 0 with probability p.
    During inference: Passes inputs unchanged (scaling is handled during training).
    
    Mathematical Operation (training):
    Output = input * mask / (1 - p)
    
    where mask is a binary matrix with probability (1-p) of being 1.
    
    This implements "inverted dropout" which scales during training instead of inference.
    
    Parameters:
        p: Dropout probability (fraction of units to drop)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout during training, pass through during inference."""
        if training and self.p > 0:
            # Create binary mask
            self.mask = (np.random.rand(*inputs.shape) > self.p).astype(np.float32)
            # Apply inverted dropout
            return inputs * self.mask / (1 - self.p)
        else:
            return inputs
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Pass gradient only through non-dropped units."""
        if self.mask is not None:
            return grad_output * self.mask / (1 - self.p)
        return grad_output


class BatchNorm2D(Layer):
    """
    Batch Normalization for 2D Convolutional Layers
    
    Normalizes activations across the batch dimension.
    
    Mathematical Operation:
    1. Compute batch mean: μ = mean(X, axis=(0,2,3))
    2. Compute batch variance: σ² = var(X, axis=(0,2,3))
    3. Normalize: X_norm = (X - μ) / sqrt(σ² + ε)
    4. Scale and shift: Y = γ * X_norm + β
    
    where γ (gamma) and β (beta) are learnable parameters.
    
    Parameters:
        num_features: Number of channels
        eps: Small constant for numerical stability
        momentum: Momentum for running mean/variance
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass of batch normalization.
        
        Args:
            inputs: Shape (batch_size, num_features, height, width)
            training: Whether in training mode
            
        Returns:
            Normalized output
        """
        if training:
            # Compute batch statistics
            batch_mean = np.mean(inputs, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(inputs, axis=(0, 2, 3), keepdims=True)
            
            # Normalize
            x_normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
            
            # Cache for backward pass
            self.cache = {
                'x_normalized': x_normalized,
                'batch_mean': batch_mean,
                'batch_var': batch_var,
                'inputs': inputs
            }
        else:
            # Use running statistics during inference
            x_normalized = (inputs - self.running_mean.reshape(1, -1, 1, 1)) / \
                          np.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)
        output = gamma * x_normalized + beta
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of batch normalization."""
        x_normalized = self.cache['x_normalized']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        inputs = self.cache['inputs']
        
        batch_size = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
        
        # Gradients w.r.t. gamma and beta
        gamma = self.gamma.reshape(1, -1, 1, 1)
        self.grad_gamma = np.sum(grad_output * x_normalized, axis=(0, 2, 3))
        self.grad_beta = np.sum(grad_output, axis=(0, 2, 3))
        
        # Gradient w.r.t. normalized input
        grad_x_normalized = grad_output * gamma
        
        # Gradient w.r.t. input (complex due to mean and variance dependencies)
        grad_var = np.sum(grad_x_normalized * (inputs - batch_mean) * -0.5 * (batch_var + self.eps)**(-1.5),
                         axis=(0, 2, 3), keepdims=True)
        
        grad_mean = np.sum(grad_x_normalized * -1 / np.sqrt(batch_var + self.eps), axis=(0, 2, 3), keepdims=True)
        grad_mean += grad_var * np.sum(-2 * (inputs - batch_mean), axis=(0, 2, 3), keepdims=True) / batch_size
        
        grad_input = grad_x_normalized / np.sqrt(batch_var + self.eps)
        grad_input += grad_var * 2 * (inputs - batch_mean) / batch_size
        grad_input += grad_mean / batch_size
        
        return grad_input
    
    def get_params(self) -> dict:
        """Returns trainable parameters and their gradients."""
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'grad_gamma': self.grad_gamma,
            'grad_beta': self.grad_beta
        }
    
    def set_params(self, params: dict):
        """Updates trainable parameters."""
        if 'gamma' in params:
            self.gamma = params['gamma']
        if 'beta' in params:
            self.beta = params['beta']
