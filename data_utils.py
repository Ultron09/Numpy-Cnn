"""
Data Utilities for CNN Training

Includes data loading, preprocessing, and augmentation functions.
"""

import numpy as np
from typing import Tuple, Optional


def load_mnist_like(num_samples: int = 1000, image_size: int = 28, num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic MNIST-like data for testing.
    
    In a real scenario, you would load actual MNIST data, but this function
    creates synthetic data for demonstration purposes.
    
    Args:
        num_samples: Total number of samples to generate
        image_size: Size of square images
        num_classes: Number of classes
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    np.random.seed(42)
    
    # Generate random images
    X = np.random.rand(num_samples, 1, image_size, image_size).astype(np.float32)
    
    # Generate random labels
    y = np.random.randint(0, num_classes, size=num_samples)
    
    # Create some pattern in the data (so it's learnable)
    for i in range(num_samples):
        label = y[i]
        # Add a pattern based on the label
        X[i, 0, label:label+3, label:label+3] = 1.0
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * num_samples)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test


def normalize(X: np.ndarray, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to have zero mean and unit variance.
    
    Normalization formula:
    X_normalized = (X - μ) / σ
    
    where μ is mean and σ is standard deviation.
    
    Args:
        X: Data to normalize
        mean: Pre-computed mean (if None, computed from X)
        std: Pre-computed std (if None, computed from X)
        
    Returns:
        Tuple of (normalized_X, mean, std)
    """
    if mean is None:
        mean = np.mean(X)
    if std is None:
        std = np.std(X)
    
    X_normalized = (X - mean) / (std + 1e-8)
    
    return X_normalized, mean, std


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class indices to one-hot encoded vectors.
    
    Example:
        Input: [0, 2, 1] with num_classes=3
        Output: [[1, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0]]
    
    Args:
        y: Class indices, shape (num_samples,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array, shape (num_samples, num_classes)
    """
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


def random_crop(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    """
    Randomly crop image to specified size.
    
    Args:
        image: Input image, shape (channels, height, width)
        crop_size: Desired crop size (height, width)
        
    Returns:
        Cropped image
    """
    _, h, w = image.shape
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        raise ValueError(f"Image size {(h, w)} is smaller than crop size {crop_size}")
    
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    
    return image[:, top:top+crop_h, left:left+crop_w]


def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Randomly flip image horizontally with probability p.
    
    Args:
        image: Input image, shape (channels, height, width)
        p: Probability of flipping
        
    Returns:
        Flipped or original image
    """
    if np.random.rand() < p:
        return image[:, :, ::-1].copy()
    return image


def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Randomly rotate image by angle in [-max_angle, max_angle] degrees.
    
    Note: This is a simplified rotation that only handles 90-degree increments
    for efficiency with NumPy. For arbitrary angles, use scipy or cv2.
    
    Args:
        image: Input image, shape (channels, height, width)
        max_angle: Maximum rotation angle in degrees
        
    Returns:
        Rotated image
    """
    # Simplified: only 90-degree rotations
    k = np.random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
    
    if k == 0:
        return image
    
    # Rotate around spatial dimensions (height, width)
    return np.rot90(image, k=k, axes=(1, 2)).copy()


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Create batches from data.
    
    Generator that yields batches of data.
    
    Args:
        X: Input data
        y: Labels
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Yields:
        Tuples of (X_batch, y_batch)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]


class DataAugmenter:
    """
    Data Augmentation Pipeline
    
    Applies a series of augmentation operations to images during training.
    
    Example:
        augmenter = DataAugmenter(
            horizontal_flip=True,
            rotation=True,
            noise_std=0.05
        )
        
        augmented_image = augmenter(image)
    """
    
    def __init__(self, horizontal_flip: bool = False, rotation: bool = False,
                 crop_size: Optional[Tuple[int, int]] = None, noise_std: float = 0.0):
        """
        Initialize augmenter.
        
        Args:
            horizontal_flip: Apply random horizontal flip
            rotation: Apply random rotation
            crop_size: Apply random crop to this size
            noise_std: Standard deviation of Gaussian noise to add
        """
        self.horizontal_flip = horizontal_flip
        self.rotation = rotation
        self.crop_size = crop_size
        self.noise_std = noise_std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations to image."""
        if self.crop_size is not None:
            image = random_crop(image, self.crop_size)
        
        if self.horizontal_flip:
            image = random_horizontal_flip(image)
        
        if self.rotation:
            image = random_rotation(image)
        
        if self.noise_std > 0:
            image = add_gaussian_noise(image, std=self.noise_std)
        
        return image


def compute_mean_std(X: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and standard deviation across entire dataset.
    
    Args:
        X: Dataset
        
    Returns:
        Tuple of (mean, std)
    """
    return np.mean(X), np.std(X)


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.
    
    Args:
        X: Input data
        y: Labels
        val_ratio: Fraction of data to use for validation
        shuffle: Whether to shuffle before splitting
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    split_idx = int((1 - val_ratio) * num_samples)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]
