import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Function to extract spectrogram from audio
def extract_spectrogram(audio_path, n_mels=128, fmax=8000):
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, fmax=fmax
        )  # Compute Mel spectrogram
        log_S = librosa.power_to_db(S, ref=np.max)  # Convert to decibel scale

        # Check for zero maximum values and avoid normalization by zero
        if np.max(log_S) == 0:
            print(f"Warning: Spectrogram for {audio_path} has no variance.")
            return None
        return log_S
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# Function to extract MFCCs from audio
def extract_mfcc(audio_path, n_mfcc=13):
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Load the audio file
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # Compute MFCCs
        return mfccs
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


# Function to process all audio files in a given directory
def process_audio_files(dataset_path, extension=".wav", batch_size=10):
    audio_files = glob.glob(
        os.path.join(dataset_path, f"**/*{extension}"), recursive=True
    )  # Get all audio files recursively
    spectrograms = []
    mfccs = []

    # Process audio files in batches
    for i, audio_path in enumerate(audio_files):
        spectrogram = extract_spectrogram(audio_path)
        mfcc = extract_mfcc(audio_path)

        if spectrogram is not None:
            spectrograms.append(spectrogram)
        if mfcc is not None:
            mfccs.append(mfcc)

        # Process in batches to reduce memory usage
        if len(spectrograms) >= batch_size:
            print(f"Processed {i + 1}/{len(audio_files)} files")
            break  # Optional: Remove this line if you want to process all files

    # Check if any spectrograms were generated
    if not spectrograms:
        print("No spectrograms were generated.")

    return spectrograms, mfccs


# Function to preprocess the data
def preprocess_data(spectrograms, labels):
    if len(spectrograms) == 0:
        raise ValueError("No valid spectrograms to preprocess.")

    # Normalize the spectrograms
    spectrograms = np.array(spectrograms)
    spectrograms = spectrograms / np.max(
        spectrograms, axis=(1, 2), keepdims=True
    )  # Normalize per spectrogram

    # Ensure labels are in the correct format (0 or 1)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Reshape spectrograms for CNN input (samples, height, width, channels)
    spectrograms = spectrograms.reshape(
        spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1
    )

    return spectrograms, labels


# Example usage:
dataset_path = "/root/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2/respiratory_sound_database/Respiratory_Sound_Database/audio_and_txt_files"

# Get and print the list of files
audio_files = glob.glob(os.path.join(dataset_path, "**/*.wav"), recursive=True)
print(f"Found {len(audio_files)} audio files.")

# Process a small batch of audio files
spectrograms, mfccs = process_audio_files(dataset_path, batch_size=5)

# Check if we have valid data for training
if spectrograms:
    print(f"Spectrograms shape: {np.array(spectrograms).shape}")
else:
    print("No spectrograms to process.")

# Labels (make sure they correspond to the correct files)
labels = ["healthy", "diseased", "healthy", "diseased", "healthy"]  # Example labels

# Preprocess the data
spectrograms, labels = preprocess_data(spectrograms, labels)

# Split into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    spectrograms, labels, test_size=0.2, random_state=42
)

# CNN Model
from tensorflow.keras import layers, models


def build_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="sigmoid"))  # Binary classification

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


# Build the model
input_shape = X_train.shape[1:]  # Shape of the input data
cnn_model = build_cnn_model(input_shape)

# Model Summary
cnn_model.summary()

# Train the model
history = cnn_model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
