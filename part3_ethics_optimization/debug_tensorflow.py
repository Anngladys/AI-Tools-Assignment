# debug_tensorflow.py (CORRECTED VERSION)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input # Import Input layer
from tensorflow.keras.optimizers import Adam
import numpy as np

print("Attempting to run corrected code...")

# --- FIXED BUGS ---

# Bug 1 & 5 FIXED: Correct data generation/types and shape for MNIST-like structure
# Assuming we want 28x28 grayscale images, 10 classes, and float32 type
num_samples = 100
image_height = 28
image_width = 28
num_channels = 1 # Grayscale images have 1 channel
num_classes = 10 # Model output will be 10 classes (digits 0-9)

# Generate dummy image data (pixel values between 0 and 1)
X_train_corrected = np.random.rand(num_samples, image_height, image_width, num_channels).astype('float32')
# Generate dummy integer labels (0 to 9)
y_train_corrected = np.random.randint(0, num_classes, num_samples)

print(f"Generated X_train_corrected shape: {X_train_corrected.shape}")
print(f"Generated y_train_corrected shape: {y_train_corrected.shape}")

# Bug 2 FIXED: Model architecture with explicit Input layer and correct output Dense layer
model = Sequential([
    Input(shape=(image_height, image_width, num_channels)), # Explicitly define input shape (height, width, channels)
    Flatten(), # Flatten the 28x28x1 image into a 784-element vector
    Dense(128, activation='relu'), # Hidden layer with ReLU activation
    Dense(num_classes, activation='softmax') # Output layer for 10 classes with softmax for probabilities
])

# Bug 3 & 4 FIXED: Correct loss function for integer labels (sparse_categorical_crossentropy)
# Adam optimizer and accuracy metric are appropriate
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # Corrected: use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

print("\nModel compiled successfully.")
model.summary() # Print model summary to verify architecture

# Attempt training with corrected data
try:
    print("\nStarting dummy training for 1 epoch...")
    # Use verbose=0 to suppress per-batch output, just show epoch summary
    model.fit(X_train_corrected, y_train_corrected, epochs=1, batch_size=32, verbose=1)
    print("\nCorrected model training attempted and succeeded for 1 epoch!")
except Exception as e:
    print(f"\nError during training (this should not happen with corrected code): {e}")
    print("Please double-check the pasted code and ensure your venv is active.")

print("\n--- End of Corrected Code Execution ---")