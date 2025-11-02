# Module 8: Deep Learning

## Overview
Deep Learning represents the cutting edge of artificial intelligence, enabling machines to learn complex patterns and representations from data. This comprehensive module covers neural networks, convolutional neural networks, recurrent neural networks, and advanced architectures that power modern AI applications.

## Learning Objectives
By the end of this module, you will be able to:
- Understand the fundamentals of neural networks and deep learning
- Implement convolutional neural networks for computer vision
- Build recurrent neural networks for sequential data
- Apply transfer learning and fine-tuning techniques
- Understand advanced architectures and attention mechanisms
- Deploy deep learning models in production environments
- Optimize model performance and computational efficiency

## 1. Introduction to Deep Learning

### 1.1 What is Deep Learning?

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to model complex patterns in data. Unlike traditional machine learning, deep learning can automatically learn hierarchical feature representations.

#### Key Characteristics
- **Hierarchical Learning**: Learns features at multiple levels of abstraction
- **Automatic Feature Extraction**: No need for manual feature engineering
- **Scalability**: Performance improves with more data and computational power
- **Flexibility**: Can handle various data types (images, text, sequences)

### 1.2 Neural Network Basics

#### Biological Inspiration
- **Neurons**: Basic computational units that receive inputs and produce outputs
- **Synapses**: Connections between neurons with associated weights
- **Activation**: Neurons fire when input exceeds a threshold
- **Learning**: Connection strengths (weights) are modified based on experience

#### Artificial Neural Networks
```python
import numpy as np

class SimpleNeuron:
    """Simple neuron implementation to understand neural network basics"""

    def __init__(self, n_inputs: int):
        # Initialize weights and bias randomly
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs: np.ndarray) -> float:
        """Forward pass through the neuron"""
        # Linear combination: z = w*x + b
        z = np.dot(self.weights, inputs) + self.bias

        # Activation function (sigmoid)
        output = 1 / (1 + np.exp(-z))

        return output

    def __repr__(self):
        return f"SimpleNeuron(weights={self.weights}, bias={self.bias:.3f})"

# Example usage
neuron = SimpleNeuron(n_inputs=3)
inputs = np.array([0.5, -0.2, 0.8])
output = neuron.forward(inputs)
print(f"Neuron output: {output:.4f}")
```

## 2. Feedforward Neural Networks

### 2.1 Multi-Layer Perceptron (MLP)

#### Architecture
- **Input Layer**: Receives raw input features
- **Hidden Layers**: Learn intermediate representations
- **Output Layer**: Produces final predictions
- **Fully Connected**: Each neuron connects to all neurons in the next layer

#### Implementation with TensorFlow/Keras
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

def create_mlp_model(input_shape: int, hidden_layers: list, output_shape: int,
                    activation: str = 'relu', output_activation: str = 'softmax'):
    """Create a Multi-Layer Perceptron model"""

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(input_shape,)))

    # Hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(0.2))  # Regularization

    # Output layer
    model.add(layers.Dense(output_shape, activation=output_activation))

    return model

# Example: Binary classification
model = create_mlp_model(
    input_shape=784,  # 28x28 flattened image
    hidden_layers=[128, 64, 32],
    output_shape=10,  # 10 classes
    output_activation='softmax'
)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
model.summary()

# Generate sample data for demonstration
X_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)
y_train = keras.utils.to_categorical(y_train, 10)

X_test = np.random.randn(200, 784)
y_test = np.random.randint(0, 10, 200)
y_test = keras.utils.to_categorical(y_test, 10)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('mlp_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 2.2 Activation Functions

#### Common Activation Functions
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_functions():
    """Plot common activation functions"""

    x = np.linspace(-3, 3, 100)

    # Define activation functions
    activations = {
        'Sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'Tanh': lambda x: np.tanh(x),
        'ReLU': lambda x: np.maximum(0, x),
        'Leaky ReLU': lambda x: np.where(x > 0, x, 0.01 * x),
        'ELU': lambda x: np.where(x > 0, x, np.exp(x) - 1),
        'Swish': lambda x: x * (1 / (1 + np.exp(-x))),
        'GELU': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Common Activation Functions', fontsize=16, fontweight='bold')

    for i, (name, func) in enumerate(activations.items()):
        row, col = i // 4, i % 4
        y = func(x)

        axes[row, col].plot(x, y, linewidth=2)
        axes[row, col].set_title(name)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[row, col].axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot activation functions
plot_activation_functions()
```

### 2.3 Loss Functions and Optimization

#### Common Loss Functions
```python
def binary_crossentropy(y_true, y_pred):
    """Binary cross-entropy loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def mean_squared_error(y_true, y_pred):
    """Mean squared error loss"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Mean absolute error loss"""
    return np.mean(np.abs(y_true - y_pred))

# Example usage
y_true_binary = np.array([0, 1, 1, 0])
y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2])

bce_loss = binary_crossentropy(y_true_binary, y_pred_binary)
print(f"Binary Cross-Entropy Loss: {bce_loss:.4f}")

y_true_regression = np.array([1.0, 2.0, 3.0, 4.0])
y_pred_regression = np.array([1.1, 1.9, 3.2, 3.8])

mse_loss = mean_squared_error(y_true_regression, y_pred_regression)
mae_loss = mean_absolute_error(y_true_regression, y_pred_regression)

print(f"MSE Loss: {mse_loss:.4f}")
print(f"MAE Loss: {mae_loss:.4f}")
```

#### Gradient Descent Optimization
```python
class GradientDescentOptimizer:
    """Simple gradient descent optimizer implementation"""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray):
        """Update parameter using gradient descent with momentum"""

        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(grad)

        # Update velocity (momentum)
        self.velocity[param_name] = self.momentum * self.velocity[param_name] - self.learning_rate * grad

        # Update parameter
        param += self.velocity[param_name]

        return param

# Example: Training a simple linear model
np.random.seed(42)

# Generate synthetic data
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Initialize parameters
w = np.random.randn(1, 1)
b = np.random.randn(1, 1)

# Training loop
optimizer = GradientDescentOptimizer(learning_rate=0.01, momentum=0.9)
losses = []

for epoch in range(100):
    # Forward pass
    y_pred = X @ w + b

    # Compute loss
    loss = mean_squared_error(y, y_pred)
    losses.append(loss)

    # Compute gradients
    dw = (2/len(X)) * X.T @ (y_pred - y)
    db = (2/len(X)) * np.sum(y_pred - y)

    # Update parameters
    w = optimizer.update('w', w, dw)
    b = optimizer.update('b', b, db)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")

print(f"Final parameters: w = {w[0,0]:.4f}, b = {b[0,0]:.4f}")
print("True parameters: w = 2.0, b = 1.0"

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, X @ w + b, color='red', linewidth=2, label='Fitted Line')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_descent_training.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 3. Convolutional Neural Networks (CNNs)

### 3.1 CNN Architecture

#### Key Components
- **Convolutional Layers**: Extract spatial features using filters
- **Pooling Layers**: Reduce spatial dimensions and computational complexity
- **Fully Connected Layers**: Perform classification based on extracted features
- **Activation Functions**: Introduce non-linearity

#### Building a CNN with Keras
```python
def create_cnn_model(input_shape: tuple, num_classes: int):
    """Create a Convolutional Neural Network"""

    model = keras.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Example: CIFAR-10 image classification
cnn_model = create_cnn_model(input_shape=(32, 32, 3), num_classes=10)

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# Data preprocessing for CIFAR-10
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Train the model
history = cnn_model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True)
    ]
)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 3.2 Advanced CNN Architectures

#### Residual Networks (ResNet)
```python
def create_resnet_block(input_tensor, filters: int, kernel_size: int = 3):
    """Create a residual block"""

    # Main path
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    if input_tensor.shape[-1] != filters:
        # Adjust dimensions if needed
        input_tensor = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)

    # Add skip connection
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)

    return x

def create_resnet_model(input_shape: tuple, num_classes: int, num_blocks: list = [2, 2, 2, 2]):
    """Create a ResNet-like architecture"""

    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual blocks
    filters = 64
    for i, blocks in enumerate(num_blocks):
        for j in range(blocks):
            x = create_resnet_block(x, filters)
        if i < len(num_blocks) - 1:  # Don't downsample after last block group
            x = layers.Conv2D(filters * 2, (1, 1), strides=2)(x)
            filters *= 2

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

# Create ResNet model
resnet_model = create_resnet_model((224, 224, 3), 1000)
resnet_model.summary()
```

#### Transfer Learning with Pre-trained Models
```python
def create_transfer_learning_model(base_model_name: str = 'VGG16', num_classes: int = 1000):
    """Create a model using transfer learning"""

    # Load pre-trained model
    if base_model_name == 'VGG16':
        base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'ResNet50':
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = keras.applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )

    # Freeze base model layers
    base_model.trainable = False

    # Add custom classification head
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    return model

# Create transfer learning model
tl_model = create_transfer_learning_model('ResNet50', num_classes=10)

tl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tl_model.summary()

# Fine-tuning: Unfreeze some layers for fine-tuning
def unfreeze_model(model, base_model, num_layers_to_unfreeze: int = 10):
    """Unfreeze layers for fine-tuning"""

    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers except the last N
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Unfreeze for fine-tuning
tl_model = unfreeze_model(tl_model, tl_model.layers[1], num_layers_to_unfreeze=10)
```

## 4. Recurrent Neural Networks (RNNs)

### 4.1 RNN Fundamentals

#### Basic RNN Architecture
```python
class SimpleRNN:
    """Simple RNN implementation for understanding"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output

        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

        # Hidden state
        self.h = np.zeros((hidden_size, 1))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through RNN"""
        outputs = []

        for x in inputs:
            # Update hidden state
            self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)

            # Compute output
            y = np.dot(self.Why, self.h) + self.by
            outputs.append(y)

        return np.array(outputs)

# Example usage
rnn = SimpleRNN(input_size=10, hidden_size=20, output_size=5)

# Generate sample sequence data
sequence_length = 5
input_size = 10
inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]

outputs = rnn.forward(inputs)
print(f"RNN outputs shape: {outputs.shape}")
```

### 4.2 Long Short-Term Memory (LSTM)

#### LSTM Implementation with Keras
```python
def create_lstm_model(vocab_size: int, embedding_dim: int = 100,
                     lstm_units: int = 128, max_length: int = 100):
    """Create an LSTM model for text classification"""

    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])

    return model

# Example: Sentiment analysis
vocab_size = 10000  # Assume we have 10k unique words
max_length = 200    # Maximum sequence length

lstm_model = create_lstm_model(vocab_size, max_length=max_length)

lstm_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

lstm_model.summary()

# Generate sample data (in practice, you'd use real text data)
num_samples = 1000
X_train = np.random.randint(0, vocab_size, (num_samples, max_length))
y_train = np.random.randint(0, 2, num_samples)

X_test = np.random.randint(0, vocab_size, (200, max_length))
y_test = np.random.randint(0, 2, 200)

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
test_loss, test_accuracy = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {test_accuracy:.4f}")
```

### 4.3 Bidirectional RNNs and Attention

#### Bidirectional LSTM
```python
def create_bidirectional_lstm(vocab_size: int, embedding_dim: int = 100,
                             lstm_units: int = 128, max_length: int = 100):
    """Create a Bidirectional LSTM model"""

    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

# Create bidirectional model
bi_lstm_model = create_bidirectional_lstm(vocab_size, max_length=max_length)
bi_lstm_model.summary()
```

#### Attention Mechanism
```python
class AttentionLayer(layers.Layer):
    """Simple attention layer implementation"""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='normal')
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros')
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=1)

        # Apply attention weights
        output = x * a
        return keras.backend.sum(output, axis=1)

def create_attention_model(vocab_size: int, embedding_dim: int = 100,
                          lstm_units: int = 128, max_length: int = 100):
    """Create a model with attention mechanism"""

    inputs = layers.Input(shape=(max_length,))

    # Embedding layer
    embedding = layers.Embedding(vocab_size, embedding_dim)(inputs)

    # LSTM layer (return sequences for attention)
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(embedding)

    # Attention layer
    attention_out = AttentionLayer()(lstm_out)

    # Dense layers
    dense = layers.Dense(64, activation='relu')(attention_out)
    dropout = layers.Dropout(0.5)(dense)
    outputs = layers.Dense(1, activation='sigmoid')(dropout)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create attention model
attention_model = create_attention_model(vocab_size, max_length=max_length)
attention_model.summary()
```

## 5. Advanced Topics and Best Practices

### 5.1 Regularization Techniques

#### Dropout and Batch Normalization
```python
def create_regularized_model(input_shape: int, num_classes: int):
    """Create a model with various regularization techniques"""

    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),

        # Dense layers with dropout
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Create regularized model
reg_model = create_regularized_model(784, 10)

# Compile with L2 regularization
reg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reg_model.summary()
```

### 5.2 Hyperparameter Tuning

#### Automated Hyperparameter Search
```python
from kerastuner import HyperModel, RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

class CNNHyperModel(HyperModel):
    """CNN model for hyperparameter tuning"""

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()

        # Tune number of convolutional layers
        num_conv_layers = hp.Int('num_conv_layers', 1, 3)

        for i in range(num_conv_layers):
            # Tune number of filters
            filters = hp.Int(f'filters_{i}', 32, 256, step=32)

            if i == 0:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu',
                                       input_shape=self.input_shape))
            else:
                model.add(layers.Conv2D(filters, (3, 3), activation='relu'))

            model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())

        # Tune dense layer units
        dense_units = hp.Int('dense_units', 64, 512, step=64)
        model.add(layers.Dense(dense_units, activation='relu'))

        # Tune dropout rate
        dropout_rate = hp.Float('dropout_rate', 0.0, 0.5, step=0.1)
        model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Tune learning rate
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

# Perform hyperparameter search
hypermodel = CNNHyperModel((28, 28, 1), 10)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='hyperparameter_tuning',
    project_name='cnn_tuning'
)

# Generate sample data
X_train = np.random.randn(1000, 28, 28, 1)
y_train = keras.utils.to_categorical(np.random.randint(0, 10, 1000), 10)

tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

# Get best model
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
for param, value in best_hyperparameters.values.items():
    print(f"{param}: {value}")
```

### 5.3 Model Interpretability

#### SHAP (SHapley Additive exPlanations)
```python
import shap

def explain_model_predictions(model, X_train, X_test, feature_names=None):
    """Explain model predictions using SHAP"""

    # Create explainer
    explainer = shap.DeepExplainer(model, X_train[:100])  # Use subset for speed

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test[:10])  # Explain first 10 test samples

    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:10],
                     feature_names=feature_names, show=False)
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot waterfall for single prediction
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explainer.expected_value[0],
                        shap_values[0][0],
                        X_test[0],
                        feature_names=feature_names,
                        show=False)
    plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    return shap_values

# Example usage (requires trained model)
# shap_values = explain_model_predictions(trained_model, X_train, X_test, feature_names)
```

## 6. Production Deployment

### 6.1 Model Serialization and Serving

#### TensorFlow Serving
```python
# Save model for TensorFlow Serving
def save_model_for_serving(model, model_version: int = 1):
    """Save model in TensorFlow Serving format"""

    import os

    # Create version directory
    model_dir = f"models/my_model/{model_version}"
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model.save(model_dir)
    print(f"Model saved to {model_dir}")

    return model_dir

# Save model
saved_model_path = save_model_for_serving(trained_model, model_version=1)
```

#### FastAPI for Model Serving
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI(title="Deep Learning Model API")

# Load model (in practice, load from saved model)
model = None  # keras.models.load_model('path/to/model')

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict on uploaded image"""

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess image
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    if model:
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions[0])

        return JSONResponse({
            "predicted_class": int(predicted_class),
            "confidence": float(predictions[0][predicted_class]),
            "all_probabilities": predictions[0].tolist()
        })
    else:
        return JSONResponse({"error": "Model not loaded"})

@app.post("/predict/text")
async def predict_text(text: str):
    """Predict on text input"""

    # Preprocess text (tokenization, etc.)
    # processed_text = preprocess_text(text)

    # Make prediction
    if model:
        # prediction = model.predict(processed_text)
        return JSONResponse({"prediction": "placeholder"})
    else:
        return JSONResponse({"error": "Model not loaded"})

# Run with: uvicorn main:app --reload
```

## 7. Best Practices and Common Pitfalls

### 7.1 Training Best Practices

#### Data Preparation
- **Normalization**: Scale features appropriately
- **Augmentation**: Use data augmentation for small datasets
- **Validation**: Always use separate validation set
- **Cross-validation**: Use k-fold CV for robust evaluation

#### Training Strategies
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Adjust learning rate during training
- **Gradient Clipping**: Prevent exploding gradients
- **Mixed Precision**: Use float16 for faster training

### 7.2 Common Pitfalls to Avoid

#### Overfitting
- **Symptoms**: High training accuracy, low validation accuracy
- **Solutions**: Regularization, dropout, early stopping, more data

#### Vanishing/Exploding Gradients
- **Symptoms**: Training stalls, NaN losses
- **Solutions**: Proper initialization, gradient clipping, batch normalization

#### Data Leakage
- **Symptoms**: Unrealistically high validation performance
- **Solutions**: Proper train/validation/test splits, no future data in training

#### Class Imbalance
- **Symptoms**: Poor performance on minority classes
- **Solutions**: Class weighting, oversampling, undersampling

## 8. Resources and Further Reading

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron
- "Deep Learning for Computer Vision" by Adrian Rosebrock

### Online Courses
- Coursera: Deep Learning Specialization (Andrew Ng)
- fast.ai: Practical Deep Learning for Coders
- Udacity: Deep Learning Nanodegree

### Research Papers
- "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Attention Is All You Need" (Transformer architecture)

### Tools and Frameworks
- **TensorFlow**: Comprehensive deep learning framework
- **PyTorch**: Research-focused deep learning framework
- **Keras**: High-level neural networks API
- **JAX**: High-performance numerical computing

## Next Steps

Congratulations on mastering deep learning fundamentals! You now understand neural networks, CNNs, RNNs, and advanced architectures. In the next module, we'll explore big data technologies for handling large-scale datasets.

**Ready to continue?** Proceed to [Module 10: Big Data Technologies](../10_big_data_technologies/)
