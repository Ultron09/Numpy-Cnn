"""
Optimization Algorithms for Neural Network Training

Optimizers update network parameters based on gradients to minimize the loss function.
Different optimizers use different strategies to navigate the loss landscape.

General update rule:
θ_new = θ_old - learning_rate * update_direction

where update_direction varies by optimizer.
"""

import numpy as np
from typing import Dict, Any


class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """
        Update parameters based on gradients.
        
        Args:
            params: Dictionary of parameters {name: value}
            grads: Dictionary of gradients {name: gradient}
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD)
    
    Update Rule:
    θ = θ - η * ∇L(θ)
    
    where:
    - η is the learning rate
    - ∇L(θ) is the gradient of loss w.r.t. parameters
    
    With Momentum:
    v_t = β * v_{t-1} + ∇L(θ)
    θ = θ - η * v_t
    
    where β is the momentum coefficient (typically 0.9)
    
    Properties:
    - Simple and widely used
    - Can be slow to converge
    - Momentum helps accelerate convergence and reduce oscillations
    - Nesterov momentum provides look-ahead gradient
    
    Parameters:
        learning_rate: Step size (default: 0.01)
        momentum: Momentum coefficient (default: 0, no momentum)
        nesterov: Whether to use Nesterov momentum (default: False)
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using SGD with optional momentum."""
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize velocity on first update
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            if self.momentum > 0:
                # Momentum update
                self.velocity[key] = self.momentum * self.velocity[key] + grads[key]
                
                if self.nesterov:
                    # Nesterov Accelerated Gradient
                    update = self.momentum * self.velocity[key] + grads[key]
                else:
                    update = self.velocity[key]
                
                params[key] -= self.learning_rate * update
            else:
                # Standard SGD (no momentum)
                params[key] -= self.learning_rate * grads[key]


class Adam(Optimizer):
    """
    Adam Optimizer (Adaptive Moment Estimation)
    
    Update Rules:
    m_t = β₁ * m_{t-1} + (1 - β₁) * ∇L(θ)           # First moment (mean)
    v_t = β₂ * v_{t-1} + (1 - β₂) * (∇L(θ))²       # Second moment (variance)
    
    m̂_t = m_t / (1 - β₁ᵗ)                          # Bias correction
    v̂_t = v_t / (1 - β₂ᵗ)                          # Bias correction
    
    θ = θ - η * m̂_t / (√v̂_t + ε)
    
    where:
    - β₁ controls the exponential decay rate for first moment (typically 0.9)
    - β₂ controls the exponential decay rate for second moment (typically 0.999)
    - ε is a small constant for numerical stability (typically 1e-8)
    - t is the timestep
    
    Properties:
    - Combines ideas from momentum and RMSprop
    - Adaptive learning rates for each parameter
    - Works well in practice with little tuning
    - Most popular optimizer for deep learning
    - Bias correction prevents initial steps from being too small
    
    Parameters:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # First moment (mean of gradients)
        self.m = {}
        # Second moment (uncentered variance of gradients)
        self.v = {}
        # Timestep
        self.t = 0
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using Adam."""
        self.t += 1
        
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize moments on first update
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop(Optimizer):
    """
    RMSprop Optimizer (Root Mean Square Propagation)
    
    Update Rules:
    v_t = β * v_{t-1} + (1 - β) * (∇L(θ))²
    θ = θ - η * ∇L(θ) / (√v_t + ε)
    
    where:
    - β is the decay rate (typically 0.9)
    - v_t is the moving average of squared gradients
    - ε is a small constant for numerical stability
    
    Properties:
    - Adapts learning rate for each parameter
    - Divides learning rate by running average of gradient magnitudes
    - Good for non-convex optimization
    - Works well for RNNs
    - No bias correction (unlike Adam)
    
    Parameters:
        learning_rate: Step size (default: 0.001)
        beta: Decay rate for moving average (default: 0.9)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using RMSprop."""
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize moving average on first update
            if key not in self.v:
                self.v[key] = np.zeros_like(params[key])
            
            # Update moving average of squared gradients
            self.v[key] = self.beta * self.v[key] + (1 - self.beta) * (grads[key] ** 2)
            
            # Update parameters
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.v[key]) + self.epsilon)


class AdaGrad(Optimizer):
    """
    AdaGrad Optimizer (Adaptive Gradient Algorithm)
    
    Update Rules:
    G_t = G_{t-1} + (∇L(θ))²                # Accumulate squared gradients
    θ = θ - η * ∇L(θ) / (√G_t + ε)
    
    where G_t accumulates all past squared gradients.
    
    Properties:
    - Adapts learning rate for each parameter
    - Larger updates for infrequent parameters
    - Smaller updates for frequent parameters
    - Good for sparse data
    - Learning rate monotonically decreases (can be too aggressive)
    - No manual learning rate schedule needed
    
    Limitations:
    - Learning rate can become infinitesimally small
    - RMSprop and Adam address this with exponential moving averages
    
    Parameters:
        learning_rate: Initial step size (default: 0.01)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = {}  # Accumulated squared gradients
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using AdaGrad."""
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize accumulator on first update
            if key not in self.G:
                self.G[key] = np.zeros_like(params[key])
            
            # Accumulate squared gradients
            self.G[key] += grads[key] ** 2
            
            # Update parameters
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.G[key]) + self.epsilon)


class AdaDelta(Optimizer):
    """
    AdaDelta Optimizer
    
    Extension of AdaGrad that addresses its diminishing learning rates.
    Instead of accumulating all past squared gradients, uses exponential moving average.
    
    Update Rules:
    E[g²]_t = ρ * E[g²]_{t-1} + (1 - ρ) * g²
    RMS[g]_t = √(E[g²]_t + ε)
    
    Δθ_t = -(RMS[Δθ]_{t-1} / RMS[g]_t) * g_t
    
    E[Δθ²]_t = ρ * E[Δθ²]_{t-1} + (1 - ρ) * Δθ²_t
    RMS[Δθ]_t = √(E[Δθ²]_t + ε)
    
    θ = θ + Δθ_t
    
    Properties:
    - No learning rate parameter needed (self-adaptive)
    - Uses running averages of gradients and updates
    - More robust than AdaGrad
    - Corrects diminishing learning rate problem
    
    Parameters:
        rho: Decay rate for moving averages (default: 0.95)
        epsilon: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(self, rho: float = 0.95, epsilon: float = 1e-6):
        super().__init__(learning_rate=1.0)  # Learning rate not used
        self.rho = rho
        self.epsilon = epsilon
        self.E_g2 = {}  # Expected value of g²
        self.E_delta2 = {}  # Expected value of Δθ²
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using AdaDelta."""
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize on first update
            if key not in self.E_g2:
                self.E_g2[key] = np.zeros_like(params[key])
                self.E_delta2[key] = np.zeros_like(params[key])
            
            # Accumulate gradient
            self.E_g2[key] = self.rho * self.E_g2[key] + (1 - self.rho) * (grads[key] ** 2)
            
            # Compute update
            rms_g = np.sqrt(self.E_g2[key] + self.epsilon)
            rms_delta = np.sqrt(self.E_delta2[key] + self.epsilon)
            
            delta = -(rms_delta / rms_g) * grads[key]
            
            # Accumulate updates
            self.E_delta2[key] = self.rho * self.E_delta2[key] + (1 - self.rho) * (delta ** 2)
            
            # Update parameters
            params[key] += delta


class Nadam(Optimizer):
    """
    Nadam Optimizer (Nesterov-accelerated Adaptive Moment Estimation)
    
    Combines Adam with Nesterov momentum.
    
    Similar to Adam but uses Nesterov momentum instead of standard momentum.
    
    Properties:
    - Generally converges faster than Adam
    - Better for certain types of problems
    - More complex than Adam
    
    Parameters:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]):
        """Update parameters using Nadam."""
        self.t += 1
        
        for key in params:
            if key not in grads or grads[key] is None:
                continue
            
            # Initialize moments on first update
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Nesterov momentum
            m_bar = (self.beta1 * m_hat + 
                    (1 - self.beta1) * grads[key] / (1 - self.beta1 ** self.t))
            
            # Update parameters
            params[key] -= self.learning_rate * m_bar / (np.sqrt(v_hat) + self.epsilon)


class LearningRateScheduler:
    """
    Learning Rate Scheduler
    
    Adjusts learning rate during training for better convergence.
    
    Common strategies:
    - Step decay: Reduce LR by factor every N epochs
    - Exponential decay: LR = LR₀ * e^(-kt)
    - Cosine annealing: Smooth decrease following cosine curve
    - Reduce on plateau: Reduce when validation loss stops improving
    """
    
    @staticmethod
    def step_decay(initial_lr: float, epoch: int, drop_rate: float = 0.5, epochs_drop: int = 10) -> float:
        """
        Step decay: Reduce learning rate by drop_rate every epochs_drop epochs.
        
        LR = LR₀ * drop_rate^(floor(epoch / epochs_drop))
        """
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr: float, epoch: int, decay_rate: float = 0.95) -> float:
        """
        Exponential decay: LR = LR₀ * decay_rate^epoch
        """
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr: float, epoch: int, total_epochs: int, min_lr: float = 0.0) -> float:
        """
        Cosine annealing: Smooth decrease following cosine curve.
        
        LR = min_lr + (LR₀ - min_lr) * (1 + cos(π * epoch / total_epochs)) / 2
        """
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    
    @staticmethod
    def linear_warmup(initial_lr: float, epoch: int, warmup_epochs: int) -> float:
        """
        Linear warmup: Gradually increase LR from 0 to initial_lr.
        
        Used at the start of training to stabilize training.
        """
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        return initial_lr
