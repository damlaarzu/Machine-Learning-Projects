import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import warnings

# Supress future and user warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load and preprocess the MNIST dataset
# Load the MNIST dataset (only treining data is used here)
(x_train, y_train), (_, _) = mnist.load_data()
# Normalize pixel values to the range [0, 1] and flatten images into 1D arrays
x_train = x_train.reshape(-1, 784) / 255.0

# Sample 1,000 examples while maintaining class distribution
sampled_indices = []
for i in range(10):
    # Find indices of all samples belonging to class `i`
    class_indices = np.where(y_train == i)[0]
    # Randomly select 100 examples from class `i` without replacement
    sampled_indices.extend(np.random.choice(class_indices, 100, replace=False))
sampled_indices = np.array(sampled_indices)

# Select sampled examples and their corresponding labels
x_sampled = x_train[sampled_indices]
y_sampled = y_train[sampled_indices]

# Step 2: Perform dimensionality reduction using PCA
# Reduce the input dimension from 784 to 3 components
pca = PCA(n_components=3)
x_encoded = pca.fit_transform(x_sampled)

# Step 3: Define a simple MLP model for decoding
# Hidden layer configurations
hidden_layer_sizes = [10, 50]
# Input dimension for the MLP (3dimensional PCA encoding)
input_dim = x_encoded.shape[1]
# Output dimension for the MLP (reconstructed 784dimensional image)
output_dim = 784

def create_mlp():
    """
    Creates and compiles a simple feedforward neural network model.
    """
    model = Sequential([
        Dense(hidden_layer_sizes[0], activation='relu', input_dim=input_dim),
        Dense(hidden_layer_sizes[1], activation='relu'),
        Dense(output_dim, activation='sigmoid')  # Output in range [0, 1]
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model

# Step 4: Train the model and evaluate performance at different epoch values
# List of epochs to evaluate
epochs_list = [10, 25, 500]
# Dictionary to store reconstructed images for each epoch setting
reconstructed_images = {}

for epochs in epochs_list:
    # Create a new model instance
    model = create_mlp()
    # Train the model on the PCA-encoded data
    model.fit(x_encoded, x_sampled, epochs=epochs, batch_size=32, verbose=0)
    # Store reconstructed images for the current epoch setting
    reconstructed_images[epochs] = model.predict(x_encoded)

# Step 5: Visualize original and reconstructed images
n_digits = 10  # Number of digits to display,
fig, axes = plt.subplots(n_digits, len(epochs_list) + 1, figsize=(10, 10))

for i in range(n_digits):
    # Get the index of the first example of the curent digit
    digit_indices = np.where(y_sampled == i)[0][0]
    # Extract and reshape the original image
    original_image = x_sampled[digit_indices].reshape(28, 28)

    # Display the original image in the first column
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')

    # Display reconstructed images for each epoch setting
    for j, epochs in enumerate(epochs_list):
        # Extract and reshape the reconstructed image
        reconstructed_image = reconstructed_images[epochs][digit_indices].reshape(28, 28)
        axes[i, j + 1].imshow(reconstructed_image, cmap='gray')
        axes[i, j + 1].set_title(f"{epochs} epochs")
        axes[i, j + 1].axis('off')

# Adjust layout to prevent overlapping titles
plt.tight_layout()
plt.show()

# Suppress TensorFlow oneDNN messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
