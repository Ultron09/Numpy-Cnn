"""
Inference Example: Using a Trained CNN

This example demonstrates how to load a trained model
and use it for inference on new data.
"""

import numpy as np
from layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from activations import ReLU, Softmax
from network import NeuralNetwork
from data_utils import normalize

# Set random seed
np.random.seed(42)


def build_cnn_for_inference(input_channels: int = 1, num_classes: int = 10) -> NeuralNetwork:
    """
    Build the same CNN architecture used during training.
    
    Note: The architecture must match the one used during training
    for the weights to load correctly.
    """
    model = NeuralNetwork()
    
    # Build the same architecture as in training
    model.add(Conv2D(input_channels, 16, kernel_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    
    model.add(Conv2D(16, 32, kernel_size=3, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    
    model.add(Flatten())
    model.add(Dense(1568, 128))
    model.add(ReLU())
    model.add(Dropout(p=0.5))
    
    model.add(Dense(128, num_classes))
    model.add(Softmax())
    
    return model


def predict_with_confidence(model: NeuralNetwork, image: np.ndarray, 
                           threshold: float = 0.5) -> dict:
    """
    Make prediction with confidence score.
    
    Args:
        model: Trained model
        image: Input image, shape (1, channels, height, width)
        threshold: Confidence threshold for prediction
        
    Returns:
        Dictionary with prediction results
    """
    # Get prediction probabilities
    probabilities = model.predict(image)
    
    # Get predicted class
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0, predicted_class]
    
    # Get top-5 predictions
    top5_indices = np.argsort(probabilities[0])[-5:][::-1]
    top5_probs = probabilities[0, top5_indices]
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'is_confident': confidence >= threshold,
        'top5_classes': top5_indices,
        'top5_probabilities': top5_probs,
        'all_probabilities': probabilities[0]
    }
    
    return result


def batch_inference(model: NeuralNetwork, images: np.ndarray, 
                   batch_size: int = 32) -> np.ndarray:
    """
    Perform inference on a batch of images efficiently.
    
    Args:
        model: Trained model
        images: Input images, shape (num_images, channels, height, width)
        batch_size: Batch size for inference
        
    Returns:
        Predictions for all images
    """
    return model.predict(images, batch_size=batch_size)


def main():
    """Main inference function."""
    print("=" * 80)
    print("CNN Inference Example - NumPy Implementation")
    print("=" * 80)
    print()
    
    # 1. Build model architecture
    print("Building model architecture...")
    model = build_cnn_for_inference(input_channels=1, num_classes=10)
    print("Model architecture loaded.")
    print()
    
    # 2. Load trained weights
    print("Loading trained weights...")
    try:
        model.load_weights("cnn_weights.npy")
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("WARNING: No saved weights found. Using random weights.")
        print("Please run train_example.py first to train the model.")
    print()
    
    # 3. Create sample data for inference
    print("Creating sample data for inference...")
    # In a real scenario, you would load your own images here
    num_samples = 10
    sample_images = np.random.rand(num_samples, 1, 28, 28).astype(np.float32)
    
    # Add some patterns (similar to training data generation)
    for i in range(num_samples):
        label = i % 10
        sample_images[i, 0, label:label+3, label:label+3] = 1.0
    
    # Normalize using the same parameters as training
    # In production, you would save and load these parameters
    mean, std = 0.5, 0.25  # Example values
    sample_images, _, _ = normalize(sample_images, mean=mean, std=std)
    
    print(f"Created {num_samples} sample images for inference.")
    print()
    
    # 4. Single image inference with detailed output
    print("=" * 80)
    print("Single Image Inference")
    print("=" * 80)
    
    single_image = sample_images[0:1]  # Keep batch dimension
    result = predict_with_confidence(model, single_image, threshold=0.7)
    
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"High Confidence: {'Yes' if result['is_confident'] else 'No'}")
    print("\nTop-5 Predictions:")
    for i, (cls, prob) in enumerate(zip(result['top5_classes'], result['top5_probabilities'])):
        print(f"  {i+1}. Class {cls}: {prob:.4f}")
    print()
    
    # 5. Batch inference
    print("=" * 80)
    print("Batch Inference")
    print("=" * 80)
    
    print(f"\nPerforming inference on {num_samples} images...")
    predictions = batch_inference(model, sample_images, batch_size=4)
    predicted_classes = np.argmax(predictions, axis=1)
    max_confidences = np.max(predictions, axis=1)
    
    print("\nResults:")
    print("-" * 60)
    print(f"{'Image':<10} {'Predicted Class':<20} {'Confidence':<15}")
    print("-" * 60)
    for i in range(num_samples):
        print(f"{i:<10} {predicted_classes[i]:<20} {max_confidences[i]:.4f}")
    print("-" * 60)
    print()
    
    # 6. Statistics
    print("=" * 80)
    print("Inference Statistics")
    print("=" * 80)
    
    avg_confidence = np.mean(max_confidences)
    std_confidence = np.std(max_confidences)
    high_confidence_count = np.sum(max_confidences >= 0.7)
    
    print(f"\nAverage Confidence: {avg_confidence:.4f}")
    print(f"Confidence Std Dev: {std_confidence:.4f}")
    print(f"High Confidence Predictions (>0.7): {high_confidence_count}/{num_samples}")
    print()
    
    # 7. Class distribution
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    print("Predicted Class Distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} samples")
    print()
    
    print("=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
