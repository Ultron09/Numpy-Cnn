"""
Visualization Utilities for CNN Training

Provides functions to visualize:
- Training history
- Convolution filters
- Feature maps
- Predictions
"""

import numpy as np
from typing import List, Dict


def plot_training_history_text(history: Dict[str, List[float]]):
    """
    Print a text-based visualization of training history.
    
    Args:
        history: Dictionary with 'loss', 'accuracy', 'val_loss', 'val_accuracy' keys
    """
    print("\n" + "=" * 80)
    print("Training History")
    print("=" * 80)
    
    epochs = len(history['loss'])
    
    print(f"\n{'Epoch':<10} {'Loss':<15} {'Accuracy':<15} {'Val Loss':<15} {'Val Acc':<15}")
    print("-" * 80)
    
    for epoch in range(epochs):
        loss = history['loss'][epoch]
        acc = history['accuracy'][epoch]
        
        val_loss_str = f"{history['val_loss'][epoch]:.4f}" if history['val_loss'] else "N/A"
        val_acc_str = f"{history['val_accuracy'][epoch]:.4f}" if history['val_accuracy'] else "N/A"
        
        print(f"{epoch+1:<10} {loss:<15.4f} {acc:<15.4f} {val_loss_str:<15} {val_acc_str:<15}")
    
    print("-" * 80)
    
    # Summary statistics
    final_loss = history['loss'][-1]
    final_acc = history['accuracy'][-1]
    best_acc = max(history['accuracy'])
    
    print(f"\nFinal Training Loss: {final_loss:.4f}")
    print(f"Final Training Accuracy: {final_acc:.4f}")
    print(f"Best Training Accuracy: {best_acc:.4f}")
    
    if history['val_accuracy']:
        final_val_acc = history['val_accuracy'][-1]
        best_val_acc = max(history['val_accuracy'])
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")


def plot_loss_curve_ascii(history: Dict[str, List[float]], height: int = 15):
    """
    ASCII art plot of loss curve.
    
    Args:
        history: Training history
        height: Height of the plot in characters
    """
    losses = history['loss']
    epochs = len(losses)
    
    if epochs == 0:
        return
    
    # Normalize to plot height
    min_loss = min(losses)
    max_loss = max(losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1
    
    print("\n" + "=" * 80)
    print("Loss Curve (ASCII)")
    print("=" * 80)
    
    # Create plot
    for row in range(height):
        line = ""
        threshold = max_loss - (row / height) * loss_range
        
        for epoch_idx, loss in enumerate(losses):
            if abs(loss - threshold) < loss_range / (2 * height):
                line += "●"
            elif loss > threshold:
                line += "|"
            else:
                line += " "
        
        # Add y-axis label
        print(f"{threshold:6.3f} │ {line}")
    
    # X-axis
    print("       └" + "─" * epochs)
    print(f"        Epochs 1-{epochs}")


def visualize_conv_filters(weights: np.ndarray, title: str = "Convolutional Filters"):
    """
    Visualize convolutional filter weights as ASCII art.
    
    Args:
        weights: Filter weights, shape (out_channels, in_channels, kernel_h, kernel_w)
        title: Title for the visualization
    """
    out_channels, in_channels, kh, kw = weights.shape
    
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"Shape: {out_channels} filters, {in_channels} channels, {kh}x{kw} kernel")
    print()
    
    # Show first few filters
    num_to_show = min(4, out_channels)
    
    for out_c in range(num_to_show):
        print(f"Filter {out_c + 1}:")
        
        # For visualization, average across input channels
        filter_avg = np.mean(weights[out_c], axis=0)
        
        # Normalize to 0-1 range
        f_min, f_max = filter_avg.min(), filter_avg.max()
        if f_max != f_min:
            filter_norm = (filter_avg - f_min) / (f_max - f_min)
        else:
            filter_norm = filter_avg
        
        # Convert to ASCII
        chars = " .:-=+*#%@"
        for row in range(kh):
            line = ""
            for col in range(kw):
                val = filter_norm[row, col]
                char_idx = int(val * (len(chars) - 1))
                line += chars[char_idx] + " "
            print(f"  {line}")
        print()


def visualize_feature_map(feature_map: np.ndarray, title: str = "Feature Map"):
    """
    Visualize a single feature map as ASCII art.
    
    Args:
        feature_map: 2D array representing a feature map
        title: Title for the visualization
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"Shape: {feature_map.shape}")
    print()
    
    h, w = feature_map.shape
    
    # Subsample if too large
    max_size = 40
    if h > max_size or w > max_size:
        step_h = max(1, h // max_size)
        step_w = max(1, w // max_size)
        feature_map = feature_map[::step_h, ::step_w]
        h, w = feature_map.shape
    
    # Normalize
    f_min, f_max = feature_map.min(), feature_map.max()
    if f_max != f_min:
        feature_norm = (feature_map - f_min) / (f_max - f_min)
    else:
        feature_norm = feature_map
    
    # Convert to ASCII
    chars = " .:-=+*#%@"
    for row in range(h):
        line = ""
        for col in range(w):
            val = feature_norm[row, col]
            char_idx = int(val * (len(chars) - 1))
            line += chars[char_idx]
        print(line)


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    """
    Print a text-based confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    """
    # Create confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        conf_matrix[true_label, pred_label] += 1
    
    print("\n" + "=" * 80)
    print("Confusion Matrix")
    print("=" * 80)
    print("(Rows: True labels, Columns: Predicted labels)")
    print()
    
    # Header
    header = "True\\Pred |"
    for i in range(num_classes):
        header += f" {i:4d}"
    print(header)
    print("-" * (11 + 5 * num_classes))
    
    # Rows
    for i in range(num_classes):
        row = f"    {i:2d}    |"
        for j in range(num_classes):
            row += f" {conf_matrix[i, j]:4d}"
        print(row)
    
    # Compute per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(num_classes):
        total = np.sum(conf_matrix[i])
        if total > 0:
            accuracy = conf_matrix[i, i] / total
            print(f"  Class {i}: {accuracy:.2%} ({conf_matrix[i, i]}/{total})")


def print_prediction_samples(images: np.ndarray, y_true: np.ndarray, 
                             y_pred: np.ndarray, num_samples: int = 5):
    """
    Print sample predictions with confidence scores.
    
    Args:
        images: Input images
        y_true: True labels
        y_pred: Prediction probabilities
        num_samples: Number of samples to show
    """
    print("\n" + "=" * 80)
    print("Sample Predictions")
    print("=" * 80)
    
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for idx in indices:
        true_label = y_true[idx] if y_true.ndim == 1 else np.argmax(y_true[idx])
        pred_label = np.argmax(y_pred[idx])
        confidence = y_pred[idx, pred_label]
        
        status = "✓" if true_label == pred_label else "✗"
        
        print(f"\nSample {idx}:")
        print(f"  {status} True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2%}")
        
        # Show top-3 predictions
        top3_idx = np.argsort(y_pred[idx])[-3:][::-1]
        print(f"  Top-3: ", end="")
        for i, class_idx in enumerate(top3_idx):
            print(f"{class_idx}({y_pred[idx, class_idx]:.2%})", end=" ")
        print()


def print_model_statistics(model):
    """
    Print detailed model statistics.
    
    Args:
        model: NeuralNetwork instance
    """
    print("\n" + "=" * 80)
    print("Model Statistics")
    print("=" * 80)
    
    total_params = 0
    trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        params = layer.get_params()
        layer_params = 0
        
        for key, value in params.items():
            if not key.startswith('grad_') and value is not None:
                layer_params += value.size
        
        total_params += layer_params
        trainable_params += layer_params  # All params are trainable in this implementation
        
        print(f"Layer {i:2d} ({layer.__class__.__name__:15s}): {layer_params:>10,} parameters")
    
    print("-" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: 0")
    
    # Memory estimate (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"Estimated model size: {memory_mb:.2f} MB")


# Example usage function
def visualize_training_results(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Comprehensive visualization of training results.
    
    Args:
        model: Trained NeuralNetwork
        X_test: Test data
        y_test: Test labels
    """
    # Training history
    plot_training_history_text(model.history)
    plot_loss_curve_ascii(model.history)
    
    # Model statistics
    print_model_statistics(model)
    
    # Make predictions
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)
    
    # Sample predictions
    print_prediction_samples(X_test, y_test, predictions, num_samples=10)
    
    # Confusion matrix
    num_classes = predictions.shape[1]
    print_confusion_matrix(true_classes, pred_classes, num_classes)
    
    # Visualize first conv layer filters
    first_conv = None
    for layer in model.layers:
        if layer.__class__.__name__ == 'Conv2D':
            first_conv = layer
            break
    
    if first_conv is not None:
        visualize_conv_filters(first_conv.weights, "First Conv Layer Filters")
