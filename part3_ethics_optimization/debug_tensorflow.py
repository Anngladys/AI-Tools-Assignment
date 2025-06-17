# debug_tensorflow.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

print("Attempting to run buggy code...")

# Problem: Data loading issues and incorrect shaping/types
# Let's assume we want to work with a simplified dataset, e_g_, a small dummy one for quick debugging.
# Original intention might have been MNIST, but simplifying for debug.

# Bug 1: Incorrect data generation/types
# Trying to create dummy data that might cause issues later
X_train_buggy = np.random.rand(100, 28, 28) * 255 # Should be float, not necessarily int and wrong channel
y_train_buggy = np.random.randint(0, 5, 100) # Only 5 classes, but model output might be different

# Bug 2: Reshaping issue for Dense layers
# Flatten layer expects specific input, but if it gets wrong shape, it will fail
# X_train_buggy = X_train_buggy.reshape(100, 28 * 28) # If directly feeding to dense layer

# Bug 3: Model architecture mismatch with expected output (e.g., num classes)
model = Sequential([
    Flatten(input_shape=(28, 28)), # Input shape missing channel
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') # Output for 10 classes, but data is 5
])

# Bug 4: Incorrect loss function for the labels
# If y_train_buggy is integer (sparse), but we use categorical_crossentropy
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # Should be sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

print("Model compiled.")

# Bug 5: Training data mismatch
# If the data type or shape is wrong, fit will fail
try:
    model.fit(X_train_buggy, y_train_buggy, epochs=1) # Training with buggy data
    print("Buggy model training attempted.")
except Exception as e:
    print(f"\nError during training: {e}")
    print("Hint: Check data types, shapes, and loss function.")

print("\n--- End of Buggy Code ---")