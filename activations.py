"""
Activation Functions for Neural Networks

This module implements common activation functions used in CNNs.
Each activation is implemented as a Layer with forward and backward passes.

Activation functions introduce non-linearity, allowing neural networks to learn
complex patterns. Without activation functions, stacking layers would be equivalent
to a single linear transformation.
"""

import numpy as np
from layers import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) Activation
    
    Mathematical Operation:
    f(x) = max(0, x)
    
    Derivative:
    f'(x) = 1 if x > 0, else 0
    
    Properties:
    - Helps mitigate vanishing gradient problem
    - Computationally efficient
    - Sparse activation (many zeros)
    - Most commonly used activation in CNNs
    - Can suffer from "dying ReLU" problem when neurons output 0 for all inputs
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass: output = max(0, input)
        
        Args:
            inputs: Any shape
            
        Returns:
            Same shape as input, with negative values clipped to 0
        """
        self.cache = inputs
        return np.maximum(0, inputs)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: gradient flows through where input was positive
        
        Args:
            grad_output: Gradient from upstream
            
        Returns:
            Gradient w.r.t. input (same shape)
        """
        grad_input = grad_output.copy()
        grad_input[self.cache <= 0] = 0
        return grad_input


class LeakyReLU(Layer):
    """
    Leaky ReLU Activation
    
    Mathematical Operation:
    f(x) = x if x > 0, else α * x
    
    where α is a small positive constant (typically 0.01)
    
    Derivative:
    f'(x) = 1 if x > 0, else α
    
    Properties:
    - Addresses the "dying ReLU" problem by allowing small negative gradients
    - α is typically 0.01
    
    Parameters:
        alpha: Slope for negative values (default: 0.01)
    """
    
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with small slope for negative values."""
        self.cache = inputs
        return np.where(inputs > 0, inputs, inputs * self.alpha)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass with gradient α for negative values."""
        grad_input = grad_output.copy()
        grad_input[self.cache <= 0] *= self.alpha
        return grad_input


class Sigmoid(Layer):
    """
    Sigmoid Activation Function
    
    Mathematical Operation:
    f(x) = 1 / (1 + e^(-x))
    
    Derivative:
    f'(x) = f(x) * (1 - f(x))
    
    Properties:
    - Outputs in range (0, 1)
    - Useful for binary classification
    - Can suffer from vanishing gradients for large |x|
    - Less commonly used in hidden layers of modern CNNs
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass: sigmoid function
        
        Uses numerically stable computation to avoid overflow.
        """
        # Numerically stable sigmoid
        output = np.where(
            inputs >= 0,
            1 / (1 + np.exp(-inputs)),
            np.exp(inputs) / (1 + np.exp(inputs))
        )
        self.cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass using cached sigmoid output.
        
        gradient = output * (1 - output) * grad_output
        """
        sigmoid_output = self.cache
        return grad_output * sigmoid_output * (1 - sigmoid_output)


class Tanh(Layer):
    """
    Hyperbolic Tangent Activation
    
    Mathematical Operation:
    f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Derivative:
    f'(x) = 1 - f(x)²
    
    Properties:
    - Outputs in range (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Can suffer from vanishing gradients
    - Stronger gradients than sigmoid (as derivatives are steeper)
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: tanh function."""
        output = np.tanh(inputs)
        self.cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: gradient = (1 - tanh²(x)) * grad_output."""
        tanh_output = self.cache
        return grad_output * (1 - tanh_output ** 2)


class Softmax(Layer):
    """
    Softmax Activation Function
    
    Mathematical Operation:
    f(x_i) = e^(x_i) / Σ_j e^(x_j)
    
    Properties:
    - Converts logits to probability distribution
    - Outputs sum to 1
    - Used in the final layer for multi-class classification
    - Often combined with cross-entropy loss for numerical stability
    
    Note: When used with CrossEntropyLoss, the backward pass is simplified
    to (predictions - targets), which is handled by the loss function.
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass: softmax function
        
        Uses numerically stable computation by subtracting max value.
        
        Args:
            inputs: Shape (batch_size, num_classes)
            
        Returns:
            Probability distribution over classes
        """
        # Numerical stability: subtract max
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        self.cache = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass of softmax.
        
        The full Jacobian is complex, but when combined with cross-entropy loss,
        it simplifies significantly. Here we implement the general case.
        
        For softmax combined with cross-entropy, the gradient is simply:
        gradient = softmax_output - one_hot_labels
        
        This is typically handled by the loss function for efficiency.
        """
        # This is a simplified implementation
        # In practice, softmax is usually combined with cross-entropy loss
        softmax_output = self.cache
        batch_size = grad_output.shape[0]
        
        grad_input = np.zeros_like(grad_output)
        
        for i in range(batch_size):
            # Jacobian matrix for single sample
            s = softmax_output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        
        return grad_input


class ELU(Layer):
    """
    Exponential Linear Unit (ELU)
    
    Mathematical Operation:
    f(x) = x if x > 0, else α * (e^x - 1)
    
    Derivative:
    f'(x) = 1 if x > 0, else f(x) + α
    
    Properties:
    - Negative values saturate to -α
    - Self-normalizing properties
    - Can achieve higher accuracy than ReLU
    - More computationally expensive than ReLU
    
    Parameters:
        alpha: Controls saturation value for negative inputs (default: 1.0)
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of ELU."""
        self.cache = inputs
        return np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of ELU."""
        inputs = self.cache
        grad_input = grad_output.copy()
        
        # For x <= 0: derivative is α * e^x = f(x) + α
        negative_mask = inputs <= 0
        grad_input[negative_mask] *= self.alpha * np.exp(inputs[negative_mask])
        
        return grad_input


class GELU(Layer):
    """
    Gaussian Error Linear Unit (GELU)
    
    Mathematical Operation (approximation):
    f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Properties:
    - Smooth, non-monotonic activation
    - Used in transformers (BERT, GPT)
    - Better performance than ReLU on some tasks
    - More computationally expensive
    
    This implementation uses the tanh approximation for efficiency.
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass using tanh approximation of GELU."""
        self.cache = inputs
        
        # GELU approximation
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (inputs + 0.044715 * inputs ** 3)
        output = 0.5 * inputs * (1.0 + np.tanh(inner))
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of GELU (approximation)."""
        x = self.cache
        
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
        tanh_inner = np.tanh(inner)
        
        # Derivative components
        sech2 = 1 - tanh_inner ** 2
        inner_derivative = sqrt_2_over_pi * (1 + 3 * 0.044715 * x ** 2)
        
        grad_input = 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_derivative
        
        return grad_output * grad_input


class Swish(Layer):
    """
    Swish Activation (also known as SiLU - Sigmoid Linear Unit)
    
    Mathematical Operation:
    f(x) = x * sigmoid(x) = x / (1 + e^(-x))
    
    Derivative:
    f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    
    Properties:
    - Smooth, non-monotonic
    - Self-gated (uses its own value to gate)
    - Can outperform ReLU on deeper models
    - Used in EfficientNet and other modern architectures
    """
    
    def __init__(self):
        super().__init__()
        self.cache = None
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: x * sigmoid(x)."""
        # Numerically stable sigmoid
        sigmoid = np.where(
            inputs >= 0,
            1 / (1 + np.exp(-inputs)),
            np.exp(inputs) / (1 + np.exp(inputs))
        )
        
        self.cache = {'inputs': inputs, 'sigmoid': sigmoid}
        return inputs * sigmoid
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of Swish."""
        x = self.cache['inputs']
        sigmoid = self.cache['sigmoid']
        
        # Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        grad_input = sigmoid * (1 + x * (1 - sigmoid))
        
        return grad_output * grad_input
