import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models


# Function to preprocess data
def preprocess_data(spectrograms, labels):
    # Normalize the spectrograms
    spectrograms = np.array(spectrograms)
    spectrograms = spectrograms / np.max(spectrograms)  # Normalize to [0, 1]

    # Ensure labels are in the correct format (0 or 1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Reshape spectrograms for CNN input
    spectrograms = spectrograms.reshape(
        spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1
    )

    return spectrograms, labels


# Example usage:
# Assuming spectrograms and labels are ready
# Let's split the data into training and testing sets

labels = ["healthy", "diseased", "healthy", "diseased", "healthy"]  # Example labels
spectrograms, labels = preprocess_data(spectrograms, labels)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    spectrograms, labels, test_size=0.2, random_state=42
)


# CNN Model
def build_cnn_model(input_shape):
    model = models.Sequential()
    # First convolutional block
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional block
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    model.add(
        layers.Dense(1, activation="sigmoid")
    )  # Sigmoid for binary classification

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Build the model
input_shape = X_train.shape[1:]  # Input shape should match the spectrogram dimensions
cnn_model = build_cnn_model(input_shape)

# Summary of the model architecture
cnn_model.summary()

# Train the model
history = cnn_model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
)

# Evaluate the model on test data
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Optionally, you can visualize the training process
# Plot the accuracy and loss curves
plt.figure(figsize=(12, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
