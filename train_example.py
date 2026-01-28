"""
Training Example: Simple CNN for Image Classification

This example demonstrates how to build, train, and evaluate
a Convolutional Neural Network using only NumPy.

Network Architecture:
    Conv2D(1, 16, 3x3) -> ReLU -> MaxPool(2x2) ->
    Conv2D(16, 32, 3x3) -> ReLU -> MaxPool(2x2) ->
    Flatten -> Dense(1568, 128) -> ReLU -> Dropout(0.5) ->
    Dense(128, 10) -> Softmax

This is a classic CNN architecture similar to LeNet-5.
"""

import numpy as np
import sys

# Import our CNN components
from layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from optimizers import Adam
from network import NeuralNetwork
from data_utils import load_mnist_like, normalize, train_val_split

# Set random seed for reproducibility
np.random.seed(42)


def build_cnn(input_channels: int = 1, num_classes: int = 10) -> NeuralNetwork:
    """
    Build a simple CNN architecture.
    
    Args:
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
        
    Returns:
        Compiled neural network
    """
    print("Building CNN architecture...")
    
    model = NeuralNetwork()
    
    # First convolutional block
    model.add(Conv2D(input_channels, 16, kernel_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    
    # Second convolutional block
    model.add(Conv2D(16, 32, kernel_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    
    # Fully connected layers
    # After two 2x2 pooling layers, 28x28 becomes 7x7
    # So we have 32 channels * 7 * 7 = 1568 features
    model.add(Flatten())
    model.add(Dense(1568, 128))
    model.add(ReLU())
    model.add(Dropout(p=0.5))
    
    # Output layer
    model.add(Dense(128, num_classes))
    model.add(Softmax())
    
    # Compile model
    model.compile(
        loss=CrossEntropyLoss(),
        optimizer=Adam(learning_rate=0.001)
    )
    
    print("\nModel architecture built successfully!")
    model.summary()
    
    return model


def main():
    """Main training function."""
    print("=" * 80)
    print("CNN Training Example - NumPy Implementation")
    print("=" * 80)
    print()
    
    # 1. Load and preprocess data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_mnist_like(
        num_samples=1000,
        image_size=28,
        num_classes=10
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}")
    print()
    
    # 2. Normalize data
    print("Normalizing data...")
    X_train, mean, std = normalize(X_train)
    X_test, _, _ = normalize(X_test, mean=mean, std=std)
    print(f"Data normalized with mean={mean:.4f}, std={std:.4f}")
    print()
    
    # 3. Create validation split
    print("Creating validation split...")
    X_train, X_val, y_train, y_val = train_val_split(
        X_train, y_train, val_ratio=0.2, shuffle=True
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print()
    
    # 4. Build model
    model = build_cnn(input_channels=1, num_classes=10)
    print()
    
    # 5. Train model
    print("=" * 80)
    print("Training Model")
    print("=" * 80)
    
    model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        shuffle=True
    )
    
    print()
    print("=" * 80)
    print("Training completed!")
    print("=" * 80)
    print()
    
    # 6. Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # 7. Make predictions on a few samples
    print("Making predictions on sample data...")
    sample_indices = np.random.choice(X_test.shape[0], size=5, replace=False)
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    predictions = model.predict(X_sample)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\nSample Predictions:")
    print("-" * 40)
    for i in range(len(sample_indices)):
        print(f"Sample {i+1}:")
        print(f"  True label: {y_sample[i]}")
        print(f"  Predicted: {predicted_classes[i]}")
        print(f"  Confidence: {predictions[i, predicted_classes[i]]:.4f}")
        print()
    
    # 8. Save model weights
    print("Saving model weights...")
    model.save_weights("cnn_weights.npy")
    
    print("\n" + "=" * 80)
    print("Training example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
