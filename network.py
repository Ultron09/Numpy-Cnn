"""
Neural Network Architecture

This module provides the main Network class that chains layers together
and handles the training loop.
"""

import numpy as np
from typing import List, Tuple, Optional
from layers import Layer
from losses import Loss
from optimizers import Optimizer


class NeuralNetwork:
    """
    Neural Network Class
    
    Chains together layers to form a complete neural network.
    Handles forward pass, backward pass, and training loop.
    
    Example Usage:
        model = NeuralNetwork()
        model.add(Conv2D(1, 32, kernel_size=3, padding=1))
        model.add(ReLU())
        model.add(MaxPool2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(7*7*32, 10))
        
        model.compile(loss=CrossEntropyLoss(), optimizer=Adam())
        model.fit(X_train, y_train, epochs=10, batch_size=32)
    """
    
    def __init__(self):
        """Initialize an empty neural network."""
        self.layers: List[Layer] = []
        self.loss_function: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def add(self, layer: Layer):
        """
        Add a layer to the network.
        
        Args:
            layer: Layer instance to add
        """
        self.layers.append(layer)
    
    def compile(self, loss: Loss, optimizer: Optimizer):
        """
        Configure the network for training.
        
        Args:
            loss: Loss function to minimize
            optimizer: Optimization algorithm
        """
        self.loss_function = loss
        self.optimizer = optimizer
    
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            inputs: Input data
            training: Whether in training mode (affects dropout, batchnorm, etc.)
            
        Returns:
            Output of the network
        """
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output
    
    def backward(self, loss_gradient: np.ndarray):
        """
        Backward pass through all layers.
        
        Args:
            loss_gradient: Gradient of loss w.r.t. network output
        """
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def update_parameters(self):
        """Update all trainable parameters using the optimizer."""
        for layer in self.layers:
            params_dict = layer.get_params()
            if not params_dict:
                continue
            
            # Extract parameters and gradients
            params = {}
            grads = {}
            
            for key, value in params_dict.items():
                if key.startswith('grad_'):
                    param_name = key[5:]  # Remove 'grad_' prefix
                    grads[param_name] = value
                else:
                    params[key] = value
            
            # Update parameters
            self.optimizer.update(params, grads)
            
            # Set updated parameters back to layer
            layer.set_params(params)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss value.
        
        Args:
            predictions: Network predictions
            targets: Ground truth labels
            
        Returns:
            Scalar loss value
        """
        return self.loss_function.forward(predictions, targets)
    
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            predictions: Network predictions (probabilities or logits)
            targets: Ground truth labels
            
        Returns:
            Accuracy as a fraction in [0, 1]
        """
        # Get predicted classes
        if predictions.ndim == 2:
            pred_classes = np.argmax(predictions, axis=1)
        else:
            pred_classes = (predictions > 0.5).astype(int)
        
        # Handle both class indices and one-hot targets
        if targets.ndim == 2:
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets
        
        return np.mean(pred_classes == true_classes)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            epochs: int = 10, batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: int = 1, shuffle: bool = True):
        """
        Train the network.
        
        Args:
            X_train: Training data, shape (num_samples, ...)
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            validation_data: Optional tuple (X_val, y_val) for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            shuffle: Whether to shuffle training data each epoch
        """
        if self.loss_function is None or self.optimizer is None:
            raise ValueError("Network must be compiled before training. Call compile() first.")
        
        num_samples = X_train.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            if shuffle:
                indices = np.random.permutation(num_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            
            # Mini-batch training
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.compute_loss(predictions, y_batch)
                epoch_loss += loss
                
                # Compute accuracy
                accuracy = self.compute_accuracy(predictions, y_batch)
                epoch_accuracy += accuracy
                
                # Backward pass
                loss_gradient = self.loss_function.backward(predictions, y_batch)
                self.backward(loss_gradient)
                
                # Update parameters
                self.update_parameters()
                
                # Print progress
                if verbose == 1:
                    progress = (batch_idx + 1) / num_batches
                    bar_length = 30
                    filled = int(bar_length * progress)
                    bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
                    print(f'\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{num_batches} - '
                          f'loss: {loss:.4f} - acc: {accuracy:.4f}', end='')
            
            # Average metrics over batches
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            
            # Validation
            val_message = ""
            if validation_data is not None:
                X_val, y_val = validation_data
                val_predictions = self.forward(X_val, training=False)
                val_loss = self.compute_loss(val_predictions, y_val)
                val_accuracy = self.compute_accuracy(val_predictions, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                val_message = f' - val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f}'
            
            if verbose >= 1:
                if verbose == 1:
                    print(f'\rEpoch {epoch+1}/{epochs} [{bar}] {num_batches}/{num_batches} - '
                          f'loss: {avg_loss:.4f} - acc: {avg_accuracy:.4f}{val_message}')
                else:  # verbose == 2
                    print(f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - '
                          f'acc: {avg_accuracy:.4f}{val_message}')
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            batch_size: Batch size for prediction
            
        Returns:
            Model predictions
        """
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        predictions = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            X_batch = X[start_idx:end_idx]
            batch_predictions = self.forward(X_batch, training=False)
            predictions.append(batch_predictions)
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluate the network on test data.
        
        Args:
            X: Test data
            y: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        predictions = self.predict(X, batch_size=batch_size)
        loss = self.compute_loss(predictions, y)
        accuracy = self.compute_accuracy(predictions, y)
        
        return loss, accuracy
    
    def save_weights(self, filepath: str):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save weights
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            if params:
                weights[f'layer_{i}'] = {k: v for k, v in params.items() 
                                        if not k.startswith('grad_')}
        
        np.save(filepath, weights)
        print(f"Weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load weights from
        """
        weights = np.load(filepath, allow_pickle=True).item()
        
        for i, layer in enumerate(self.layers):
            layer_key = f'layer_{i}'
            if layer_key in weights:
                layer.set_params(weights[layer_key])
        
        print(f"Weights loaded from {filepath}")
    
    def summary(self):
        """
        Print a summary of the network architecture.
        
        Displays layer types, output shapes, and parameter counts.
        """
        print("=" * 80)
        print("Model Summary")
        print("=" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
        print("=" * 80)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{i}"
            
            # Count parameters
            params = layer.get_params()
            num_params = 0
            for key, value in params.items():
                if not key.startswith('grad_') and value is not None:
                    num_params += value.size
            
            total_params += num_params
            
            # This is approximate - actual shape depends on input
            output_shape = "Multiple"
            
            print(f"{layer_name:<30} {output_shape:<25} {num_params:<15,}")
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print("=" * 80)
