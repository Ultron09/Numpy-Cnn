"""
Loss Functions for Neural Networks

Loss functions measure how well the network's predictions match the targets.
During training, we minimize the loss to improve the model.

Each loss function computes:
1. Forward pass: Calculate the loss value
2. Backward pass: Calculate gradient w.r.t. predictions
"""

import numpy as np


class Loss:
    """Base class for loss functions."""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the loss value.
        
        Args:
            predictions: Model outputs
            targets: Ground truth labels
            
        Returns:
            Scalar loss value
        """
        raise NotImplementedError
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. predictions.
        
        Args:
            predictions: Model outputs
            targets: Ground truth labels
            
        Returns:
            Gradient with same shape as predictions
        """
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for Multi-class Classification
    
    Mathematical Operation:
    For a single sample with true class y:
    L = -log(p_y)
    
    where p_y is the predicted probability for the true class.
    
    For a batch:
    L = -1/N * Σ_i log(p_{y_i})
    
    When combined with softmax:
    L = -1/N * Σ_i [y_i * log(p_i)]  (one-hot encoding)
    
    Gradient (when combined with softmax):
    ∂L/∂z = p - y  (predictions - one_hot_targets)
    
    This simple gradient is one reason cross-entropy + softmax is popular.
    
    Properties:
    - Measures the difference between two probability distributions
    - Penalizes confident wrong predictions more
    - Numerical stability is important
    """
    
    def __init__(self):
        super().__init__()
        self.eps = 1e-8  # Small constant for numerical stability
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, num_classes)
                        Should be output of softmax
            targets: True labels, either:
                    - Class indices: shape (batch_size,)
                    - One-hot encoded: shape (batch_size, num_classes)
        
        Returns:
            Average loss across the batch
        """
        batch_size = predictions.shape[0]
        
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        
        # Handle both one-hot and class index formats
        if targets.ndim == 1:
            # Class indices format
            # Extract the predicted probability for the true class
            correct_class_probs = predictions[np.arange(batch_size), targets]
            loss = -np.mean(np.log(correct_class_probs))
        else:
            # One-hot format
            loss = -np.mean(np.sum(targets * np.log(predictions), axis=1))
        
        return loss
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss.
        
        When used with softmax, the gradient simplifies to:
        gradient = (predictions - targets) / batch_size
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, num_classes)
            targets: True labels (class indices or one-hot)
        
        Returns:
            Gradient w.r.t. predictions
        """
        batch_size = predictions.shape[0]
        
        # Convert class indices to one-hot if necessary
        if targets.ndim == 1:
            num_classes = predictions.shape[1]
            targets_one_hot = np.zeros_like(predictions)
            targets_one_hot[np.arange(batch_size), targets] = 1
        else:
            targets_one_hot = targets
        
        # Gradient: (predictions - targets) / batch_size
        gradient = (predictions - targets_one_hot) / batch_size
        
        return gradient


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy Loss for Binary Classification
    
    Mathematical Operation:
    L = -1/N * Σ [y * log(p) + (1-y) * log(1-p)]
    
    where:
    - y is the true label (0 or 1)
    - p is the predicted probability
    
    Gradient:
    ∂L/∂p = (p - y) / (p * (1 - p))
    
    For sigmoid output:
    ∂L/∂z = p - y
    
    Properties:
    - Special case of cross-entropy for two classes
    - Often used with sigmoid activation in final layer
    """
    
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, 1) or (batch_size,)
            targets: True labels (0 or 1), same shape as predictions
        
        Returns:
            Average loss
        """
        # Clip predictions for numerical stability
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        
        loss = -np.mean(
            targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)
        )
        
        return loss
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of binary cross-entropy.
        
        For sigmoid output, gradient simplifies to: p - y
        """
        batch_size = predictions.size
        
        # Clip for numerical stability
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        
        # Simplified gradient for sigmoid + BCE
        gradient = (predictions - targets) / batch_size
        
        return gradient


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) Loss
    
    Mathematical Operation:
    L = 1/N * Σ (y - ŷ)²
    
    where y is true value and ŷ is predicted value.
    
    Gradient:
    ∂L/∂ŷ = 2/N * (ŷ - y)
    
    Properties:
    - Used primarily for regression tasks
    - Sensitive to outliers (due to squaring)
    - Assumes Gaussian noise in the data
    - Less common for classification
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute MSE loss.
        
        Args:
            predictions: Predicted values
            targets: True values
        
        Returns:
            Average squared error
        """
        return np.mean((predictions - targets) ** 2)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE.
        
        gradient = 2 * (predictions - targets) / N
        """
        batch_size = predictions.size
        return 2 * (predictions - targets) / batch_size


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error (MAE) Loss (L1 Loss)
    
    Mathematical Operation:
    L = 1/N * Σ |y - ŷ|
    
    Gradient:
    ∂L/∂ŷ = 1/N * sign(ŷ - y)
    
    Properties:
    - More robust to outliers than MSE
    - Gradient has constant magnitude
    - Can be harder to optimize (non-smooth at 0)
    """
    
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MAE loss."""
        return np.mean(np.abs(predictions - targets))
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of MAE."""
        batch_size = predictions.size
        
        # Gradient is sign of difference
        diff = predictions - targets
        gradient = np.sign(diff) / batch_size
        
        # Handle the case where diff is exactly 0 (use subgradient)
        gradient[np.abs(diff) < self.eps] = 0
        
        return gradient


class HuberLoss(Loss):
    """
    Huber Loss (Smooth L1 Loss)
    
    Mathematical Operation:
    L(x) = 0.5 * x²                if |x| <= δ
           δ * (|x| - 0.5 * δ)     otherwise
    
    where x = prediction - target
    
    Gradient:
    ∂L/∂x = x                      if |x| <= δ
            δ * sign(x)            otherwise
    
    Properties:
    - Combines MSE (for small errors) and MAE (for large errors)
    - Less sensitive to outliers than MSE
    - Smoother than MAE
    - δ controls the transition point
    
    Parameters:
        delta: Threshold for switching between quadratic and linear (default: 1.0)
    """
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Huber loss."""
        diff = predictions - targets
        abs_diff = np.abs(diff)
        
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        
        loss = np.where(abs_diff <= self.delta, quadratic, linear)
        
        return np.mean(loss)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of Huber loss."""
        batch_size = predictions.size
        diff = predictions - targets
        abs_diff = np.abs(diff)
        
        gradient = np.where(
            abs_diff <= self.delta,
            diff,
            self.delta * np.sign(diff)
        )
        
        return gradient / batch_size


class HingeLoss(Loss):
    """
    Hinge Loss for Binary Classification
    
    Mathematical Operation:
    L = 1/N * Σ max(0, 1 - y * ŷ)
    
    where:
    - y ∈ {-1, +1} (true label)
    - ŷ is the predicted score (not probability)
    
    Gradient:
    ∂L/∂ŷ = -y/N    if y * ŷ < 1
             0       otherwise
    
    Properties:
    - Used in Support Vector Machines (SVM)
    - Encourages correct classifications with margin
    - Only penalizes when margin is violated
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute hinge loss.
        
        Args:
            predictions: Predicted scores (not probabilities)
            targets: True labels, should be -1 or +1
        
        Returns:
            Average hinge loss
        """
        margin = 1 - targets * predictions
        loss = np.mean(np.maximum(0, margin))
        return loss
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of hinge loss."""
        batch_size = predictions.size
        
        margin = 1 - targets * predictions
        gradient = np.where(margin > 0, -targets, 0) / batch_size
        
        return gradient


class FocalLoss(Loss):
    """
    Focal Loss for Addressing Class Imbalance
    
    Mathematical Operation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the probability of the correct class
    - α_t is a weighting factor for class t
    - γ is the focusing parameter (typically 2)
    
    The (1 - p_t)^γ term down-weights easy examples and focuses on hard examples.
    
    Properties:
    - Addresses class imbalance
    - Focuses training on hard examples
    - Reduces weight of easy examples
    - Used in object detection (RetinaNet)
    
    Parameters:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-8
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute focal loss.
        
        Args:
            predictions: Predicted probabilities, shape (batch_size, num_classes)
            targets: True labels (class indices or one-hot)
        
        Returns:
            Average focal loss
        """
        batch_size = predictions.shape[0]
        
        # Clip predictions
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        
        # Convert to one-hot if necessary
        if targets.ndim == 1:
            num_classes = predictions.shape[1]
            targets_one_hot = np.zeros_like(predictions)
            targets_one_hot[np.arange(batch_size), targets] = 1
        else:
            targets_one_hot = targets
        
        # Get probability of correct class
        p_t = np.sum(predictions * targets_one_hot, axis=1)
        
        # Compute focal loss
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -np.log(p_t)
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return np.mean(focal_loss)
    
    def backward(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute gradient of focal loss.
        
        This is a simplified implementation.
        """
        batch_size = predictions.shape[0]
        
        # Clip predictions
        predictions = np.clip(predictions, self.eps, 1 - self.eps)
        
        # Convert to one-hot if necessary
        if targets.ndim == 1:
            num_classes = predictions.shape[1]
            targets_one_hot = np.zeros_like(predictions)
            targets_one_hot[np.arange(batch_size), targets] = 1
        else:
            targets_one_hot = targets
        
        # Simplified gradient (approximate)
        p_t = np.sum(predictions * targets_one_hot, axis=1, keepdims=True)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Approximate gradient
        gradient = self.alpha * focal_weight * (predictions - targets_one_hot) / batch_size
        
        return gradient
