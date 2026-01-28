# Advanced CNN Training with NumPy
## Educational Implementation from Scratch

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Only-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive, educational implementation of **Convolutional Neural Networks (CNNs)** using **only NumPy**. This project implements CNNs entirely from scratch to provide deep understanding of how these powerful models work under the hood.

---

## 📚 Table of Contents
- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Documentation](#detailed-documentation)
  - [Convolutional Layers](#convolutional-layers)
  - [Pooling Layers](#pooling-layers)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)
- [Training a CNN](#training-a-cnn)
- [Inference](#inference)
- [Educational Goals](#educational-goals)
- [Advanced Topics](#advanced-topics)

---

## 🎯 Overview

This project implements a complete CNN framework from **ground up** using only NumPy. No deep learning frameworks (PyTorch, TensorFlow, etc.) are used - everything is implemented manually to provide educational insight into:

- **Forward propagation** through convolutional layers
- **Backpropagation** with gradient calculation
- **Weight initialization** strategies
- **Optimization algorithms** (SGD, Adam, RMSprop, etc.)
- **Regularization techniques** (Dropout, Batch Normalization)
- **Data augmentation** and preprocessing

### Why This Project?

Understanding how CNNs work at the implementation level is crucial for:
- Debugging deep learning models
- Developing novel architectures
- Understanding computational complexity
- Appreciating the design decisions in modern frameworks

---

## 🧮 Mathematical Foundations

### Convolution Operation

The core operation in CNNs is the **discrete convolution**:

```
Output[n, c, h, w] = Σ Σ Σ Input[n, k, h*s+i, w*s+j] * Kernel[c, k, i, j] + Bias[c]
                     k i j
```

Where:
- `n` = batch index
- `c` = output channel
- `k` = input channel
- `h, w` = spatial position
- `s` = stride
- `i, j` = kernel indices

**Key Properties:**
1. **Local connectivity**: Each neuron only connects to a small region
2. **Parameter sharing**: Same kernel applied across spatial locations
3. **Translation equivariance**: If input shifts, output shifts correspondingly

### Backpropagation Through Convolution

For gradient computation, we use the chain rule:

```
∂L/∂Input = ∂L/∂Output ⊗ Kernel (convolution with flipped kernel)
∂L/∂Kernel = Input ⊗ ∂L/∂Output
∂L/∂Bias = Σ ∂L/∂Output
```

### Pooling Operations

**Max Pooling**: Routes gradient only to the maximum value

```
Forward: y = max(x₁, x₂, ..., xₙ)
Backward: ∂L/∂xᵢ = ∂L/∂y if xᵢ = y, else 0
```

**Average Pooling**: Distributes gradient equally

```
Forward: y = (1/n) Σ xᵢ
Backward: ∂L/∂xᵢ = (1/n) * ∂L/∂y
```

### Activation Functions

**ReLU** (most common):
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

**Softmax** (for classification):
```
f(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

### Loss Functions

**Cross-Entropy** (classification):
```
L = -Σᵢ yᵢ * log(ŷᵢ)

Gradient (with softmax): ∂L/∂z = ŷ - y
```

This elegant gradient is why softmax + cross-entropy is the standard for classification.

---

## 📁 Project Structure

```
numpy-cnn/
│
├── README.md                 # This file
├── layers.py                 # Core layer implementations
│   ├── Layer (base class)
│   ├── Conv2D
│   ├── MaxPool2D
│   ├── AvgPool2D
│   ├── Flatten
│   ├── Dense
│   ├── Dropout
│   └── BatchNorm2D
│
├── activations.py            # Activation functions
│   ├── ReLU
│   ├── LeakyReLU
│   ├── Sigmoid
│   ├── Tanh
│   ├── Softmax
│   ├── ELU
│   ├── GELU
│   └── Swish
│
├── losses.py                 # Loss functions
│   ├── CrossEntropyLoss
│   ├── BinaryCrossEntropyLoss
│   ├── MeanSquaredError
│   ├── MeanAbsoluteError
│   ├── HuberLoss
│   ├── HingeLoss
│   └── FocalLoss
│
├── optimizers.py             # Optimization algorithms
│   ├── SGD (with momentum)
│   ├── Adam
│   ├── RMSprop
│   ├── AdaGrad
│   ├── AdaDelta
│   ├── Nadam
│   └── LearningRateScheduler
│
├── network.py                # Neural network class
│   ├── NeuralNetwork
│   ├── Training loop
│   ├── Evaluation
│   └── Weight saving/loading
│
├── data_utils.py             # Data preprocessing
│   ├── Normalization
│   ├── Data augmentation
│   ├── Batch creation
│   └── Train/val split
│
├── train_example.py          # Complete training example
└── inference_example.py      # Inference example
```

---

## 🚀 Installation

### Requirements
- Python 3.7+
- NumPy 1.19+

```bash
# Clone or navigate to the project directory
cd numpy-cnn

# Install NumPy if not already installed
pip install numpy

# No other dependencies needed!
```

---

## ⚡ Quick Start

### Training a CNN

```python
import numpy as np
from layers import Conv2D, MaxPool2D, Flatten, Dense
from activations import ReLU, Softmax
from losses import CrossEntropyLoss
from optimizers import Adam
from network import NeuralNetwork

# Create model
model = NeuralNetwork()

# Add layers
model.add(Conv2D(1, 32, kernel_size=3, padding=1))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(32, 64, kernel_size=3, padding=1))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2))

model.add(Flatten())
model.add(Dense(3136, 128))
model.add(ReLU())
model.add(Dense(128, 10))
model.add(Softmax())

# Compile
model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

### Running Examples

```bash
# Train a CNN on synthetic data
python train_example.py

# Run inference with trained model
python inference_example.py
```

---

## 📖 Detailed Documentation

### Convolutional Layers

**Purpose**: Extract spatial features through learned filters

**Implementation Details**:

```python
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        # Initialize weights with He initialization
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = np.random.randn(out_channels, in_channels, 
                                       kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
```

**Why He Initialization?**
- For ReLU activations, He initialization `W ~ N(0, √(2/fan_in))` prevents vanishing/exploding gradients
- Derived from analyzing variance propagation through layers

**Key Concepts**:

1. **Receptive Field**: Region of input that affects one output neuron
   ```
   Receptive field grows with depth:
   - After 1 conv (3x3): 3x3
   - After 2 conv (3x3): 5x5
   - After 3 conv (3x3): 7x7
   ```

2. **Output Size Calculation**:
   ```
   out_h = (in_h + 2*padding - kernel_size) / stride + 1
   out_w = (in_w + 2*padding - kernel_size) / stride + 1
   ```

3. **Parameter Count**:
   ```
   params = (kernel_h * kernel_w * in_channels * out_channels) + out_channels
            └────────── weights ──────────┘   └─ biases ─┘
   ```

**Computational Complexity**:
- Time: O(batch_size × out_channels × in_channels × kernel_h × kernel_w × out_h × out_w)
- Space: O(out_channels × in_channels × kernel_h × kernel_w)

### Pooling Layers

**Purpose**: Reduce spatial dimensions and provide translation invariance

**Max Pooling**:
- Selects maximum value in each window
- Provides strong invariance to small translations
- Retains the most "activated" feature
- Gradient flows only to max position

**Average Pooling**:
- Computes mean of each window
- Smoother downsampling
- Gradient distributed equally
- Often used in final layers (Global Average Pooling)

**Best Practices**:
- Use 2×2 pooling with stride 2 (standard)
- Max pooling for early layers (feature detection)
- Average pooling for later layers (feature aggregation)

### Activation Functions

#### ReLU (Rectified Linear Unit)
```python
f(x) = max(0, x)
```

**Advantages**:
- ✅ Computationally efficient
- ✅ Helps with vanishing gradient
- ✅ Sparse activation (biological plausibility)

**Disadvantages**:
- ❌ "Dying ReLU" problem (neurons can stop learning)
- ❌ Not zero-centered

**When to use**: Default choice for hidden layers

#### Leaky ReLU
```python
f(x) = x if x > 0 else α*x  (α ≈ 0.01)
```

**Advantages**:
- ✅ Solves dying ReLU problem
- ✅ Small negative slope allows gradient flow

**When to use**: When experiencing dying ReLU issues

#### GELU (Gaussian Error Linear Unit)
```python
f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
```

**Advantages**:
- ✅ Smooth, non-monotonic
- ✅ Used in transformers (BERT, GPT)
- ✅ Better performance on some tasks

**When to use**: Modern architectures, NLP tasks

#### Softmax
```python
f(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Advantages**:
- ✅ Outputs valid probability distribution
- ✅ Differentiable

**When to use**: Final layer for multi-class classification

### Loss Functions

#### Cross-Entropy Loss

**Formula**:
```
L = -Σᵢ yᵢ * log(ŷᵢ)
```

**Intuition**: Measures the "surprise" of predictions given true labels
- When prediction is correct and confident: low loss
- When prediction is wrong or uncertain: high loss

**Why with Softmax?**
The gradient ∂L/∂z = ŷ - y is remarkably simple, leading to stable training

#### Focal Loss

**Formula**:
```
FL(pₜ) = -αₜ * (1 - pₜ)^γ * log(pₜ)
```

**Purpose**: Address class imbalance by focusing on hard examples

**When to use**:
- Severe class imbalance (e.g., 1:1000)
- Object detection
- Hard example mining

### Optimizers

#### SGD (Stochastic Gradient Descent)

```python
θ = θ - η * ∇L(θ)
```

**With Momentum**:
```python
v = β * v + ∇L(θ)
θ = θ - η * v
```

**Hyperparameters**:
- Learning rate η: 0.01 - 0.1 (common range)
- Momentum β: 0.9 (standard)

**When to use**: Simple problems, well-understood learning rate

#### Adam (Adaptive Moment Estimation)

```python
m = β₁ * m + (1 - β₁) * ∇L(θ)        # First moment
v = β₂ * v + (1 - β₂) * (∇L(θ))²    # Second moment
θ = θ - η * m̂ / (√v̂ + ε)
```

**Hyperparameters** (standard values):
- Learning rate η: 0.001
- β₁: 0.9
- β₂: 0.999
- ε: 1e-8

**Advantages**:
- ✅ Adapts learning rate per parameter
- ✅ Works well with little tuning
- ✅ Efficient on sparse gradients

**When to use**: Default optimizer for most problems

#### RMSprop

```python
v = β * v + (1 - β) * (∇L(θ))²
θ = θ - η * ∇L(θ) / (√v + ε)
```

**When to use**: Recurrent networks, online learning

### Learning Rate Schedules

**Step Decay**:
```python
LR = LR₀ * 0.5^(epoch // 10)
```
Reduce LR by half every 10 epochs

**Cosine Annealing**:
```python
LR = min_lr + (max_lr - min_lr) * (1 + cos(π * epoch / total)) / 2
```
Smooth decrease following cosine curve

**When to use**:
- Step decay: Simple baseline
- Cosine: Better final performance
- Warmup + decay: Complex/large models

---

## 🎓 Training a CNN

### Complete Training Pipeline

```python
# 1. DATA PREPARATION
from data_utils import load_mnist_like, normalize, train_val_split

# Load data
X_train, y_train, X_test, y_test = load_mnist_like(num_samples=5000)

# Normalize (critical for convergence!)
X_train, mean, std = normalize(X_train)
X_test, _, _ = normalize(X_test, mean=mean, std=std)

# Create validation split
X_train, X_val, y_train, y_val = train_val_split(X_train, y_train, val_ratio=0.2)

# 2. MODEL ARCHITECTURE
from network import NeuralNetwork
from layers import *
from activations import *

model = NeuralNetwork()

# Convolutional feature extraction
model.add(Conv2D(1, 32, kernel_size=3, padding=1))
model.add(ReLU())
model.add(Conv2D(32, 32, kernel_size=3, padding=1))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2))

model.add(Conv2D(32, 64, kernel_size=3, padding=1))
model.add(ReLU())
model.add(Conv2D(64, 64, kernel_size=3, padding=1))
model.add(ReLU())
model.add(MaxPool2D(pool_size=2))

# Classification head
model.add(Flatten())
model.add(Dense(3136, 256))
model.add(ReLU())
model.add(Dropout(0.5))
model.add(Dense(256, 10))
model.add(Softmax())

# 3. COMPILATION
from losses import CrossEntropyLoss
from optimizers import Adam

model.compile(
    loss=CrossEntropyLoss(),
    optimizer=Adam(learning_rate=0.001)
)

model.summary()

# 4. TRAINING
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val),
    verbose=1
)

# 5. EVALUATION
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2%}")

# 6. SAVE MODEL
model.save_weights("trained_model.npy")
```

### Architecture Design Guidelines

**Rule of Thumb**:
1. **Input**: Normalize to zero mean, unit variance
2. **Early layers**: Small kernels (3×3), moderate channels (32-64)
3. **Middle layers**: Increase channels while reducing spatial size
4. **Late layers**: Large channel count (256-512)
5. **Output**: Softmax for classification

**Common Patterns**:

```
Conv-Conv-Pool Pattern (VGG-style):
    Conv(3×3) → ReLU → Conv(3×3) → ReLU → Pool(2×2)
    Double spatial features before downsampling

Residual Pattern (ResNet-style):
    Input → Conv → ReLU → Conv → Add(Input) → ReLU
    Skip connections improve gradient flow
```

---

## 🔮 Inference

### Single Image Prediction

```python
# Load model
model = build_model()
model.load_weights("trained_model.npy")

# Prepare image
image = load_image("test.jpg")  # Shape: (1, channels, height, width)
image = normalize(image, mean=saved_mean, std=saved_std)

# Predict
prediction = model.predict(image)
class_idx = np.argmax(prediction, axis=1)[0]
confidence = prediction[0, class_idx]

print(f"Predicted class: {class_idx}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Inference

```python
# Efficient batch processing
batch_images = load_images(image_paths)  # (N, C, H, W)
predictions = model.predict(batch_images, batch_size=32)

# Get top-5 predictions for each
top5_classes = np.argsort(predictions, axis=1)[:, -5:][:, ::-1]
```

---

## 🎯 Educational Goals

This project helps you understand:

### 1. **Forward Propagation**
- How data flows through layers
- Shape transformations at each layer
- Computational requirements

### 2. **Backpropagation**
- Gradient calculation via chain rule
- Why some operations are expensive
- Numerical stability considerations

### 3. **Optimization**
- How gradients update weights
- Why adaptive methods work better
- Learning rate importance

### 4. **Architectural Choices**
- Why we use convolutions (vs. fully connected)
- Role of pooling layers
- Depth vs. width tradeoffs

### 5. **Practical Considerations**
- Batch normalization for stability
- Dropout for regularization
- Data augmentation for generalization

---

## 🚀 Advanced Topics

### Batch Normalization

**Why it works**:
1. Reduces internal covariate shift
2. Allows higher learning rates
3. Acts as regularization

**Implementation insight**:
```python
# Forward pass
mean = np.mean(x, axis=(0, 2, 3))
var = np.var(x, axis=(0, 2, 3))
x_norm = (x - mean) / sqrt(var + eps)
out = gamma * x_norm + beta
```

During inference, use running statistics instead of batch statistics.

### Data Augmentation

**Why augment?**
- Increases effective dataset size
- Improves generalization
- Reduces overfitting

**Common techniques**:
```python
# Random horizontal flip
if np.random.rand() > 0.5:
    image = image[:, :, ::-1]

# Random crop
top = np.random.randint(0, h - crop_h)
left = np.random.randint(0, w - crop_w)
image = image[:, top:top+crop_h, left:left+crop_w]

# Color jitter (if RGB)
image = image * (0.8 + 0.4 * np.random.rand())
```

### Weight Initialization

**Why initialization matters**:
- Too small → vanishing gradients
- Too large → exploding gradients

**Strategies**:

```python
# Xavier (for Tanh/Sigmoid)
W = np.random.randn(fan_in, fan_out) * np.sqrt(1 / fan_in)

# He (for ReLU)
W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)

# LeCun (for SELU)
W = np.random.randn(fan_in, fan_out) * np.sqrt(1 / fan_in)
```

### Gradient Checking

Verify backpropagation implementation:

```python
def gradient_check(layer, x, eps=1e-7):
    # Analytical gradient
    out = layer.forward(x)
    grad_analytical = layer.backward(np.ones_like(out))
    
    # Numerical gradient
    grad_numerical = np.zeros_like(x)
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += eps
        out_plus = layer.forward(x_plus)
        
        x_minus = x.copy()
        x_minus.flat[i] -= eps
        out_minus = layer.forward(x_minus)
        
        grad_numerical.flat[i] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)
    
    # Compare
    diff = np.abs(grad_analytical - grad_numerical).max()
    assert diff < 1e-5, f"Gradient check failed: {diff}"
```

---

## 📊 Performance Considerations

### Time Complexity

| Operation | Forward | Backward | Space |
|-----------|---------|----------|-------|
| Conv2D | O(N×C_out×C_in×K²×H×W) | O(N×C_out×C_in×K²×H×W) | O(C_out×C_in×K²) |
| MaxPool | O(N×C×H×W) | O(N×C×H×W) | O(1) |
| Dense | O(N×D_in×D_out) | O(N×D_in×D_out) | O(D_in×D_out) |

Where:
- N = batch size
- C = channels
- K = kernel size
- H, W = spatial dimensions
- D = dense layer dimensions

### Optimization Tips

1. **Use smaller batch sizes** if memory constrained
2. **Cache intermediate results** in forward pass for backward
3. **Vectorize operations** - avoid Python loops
4. **Use appropriate dtypes** (float32 vs float64)

---

## 🤝 Contributing

Educational improvements welcome! Focus areas:
- Additional architectures (ResNet, U-Net, etc.)
- Performance optimizations
- More comprehensive examples
- Better documentation

---

## 📚 References

### Foundational Papers
1. **LeCun et al. (1998)**: "Gradient-Based Learning Applied to Document Recognition" - Original LeNet
2. **Krizhevsky et al. (2012)**: "ImageNet Classification with Deep CNNs" - AlexNet
3. **Simonyan & Zisserman (2014)**: "Very Deep Convolutional Networks" - VGGNet
4. **He et al. (2015)**: "Deep Residual Learning for Image Recognition" - ResNet

### Books
- **Deep Learning** by Goodfellow, Bengio, Courville
- **Neural Networks and Deep Learning** by Michael Nielsen

### Online Resources
- **CS231n**: Convolutional Neural Networks for Visual Recognition (Stanford)
- **distill.pub**: Interactive visualizations of ML concepts

---

## 📄 License

MIT License - Feel free to use for educational purposes

---

## 🙏 Acknowledgments

This implementation is inspired by:
- CS231n course materials
- PyTorch and TensorFlow source code
- NumPy documentation and community

---

## 💡 Final Notes

**Remember**: This is an *educational* implementation. For production use:
- Use established frameworks (PyTorch, TensorFlow)
- Leverage GPU acceleration
- Use optimized libraries (cuDNN, MKL)

However, understanding these fundamentals will make you a better deep learning practitioner!

**Happy Learning! 🎓**

---

*For questions or suggestions, please open an issue or submit a pull request.*
