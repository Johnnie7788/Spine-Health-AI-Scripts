#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import cv2

# Define the input shape for images
INPUT_SHAPE = (256, 256, 3)

# Load and preprocess the CT scan image
def load_and_preprocess_image(image_path, target_size=INPUT_SHAPE[:2]):
    """
    Loads and preprocesses a CT scan image.
    :param image_path: Path to the CT scan image
    :param target_size: Target size for image resizing
    :return: Preprocessed image
    """
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Advanced data augmentation and image generator
def create_data_generator(image_dir, batch_size=32, target_size=INPUT_SHAPE[:2], validation_split=0.2):
    """
    Creates an ImageDataGenerator to load and augment images.
    :param image_dir: Directory containing CT scan images
    :param batch_size: Batch size for data loading
    :param target_size: Target size for image resizing
    :param validation_split: Proportion of data to be used for validation
    :return: Training and validation data generators
    """
    datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale pixel values
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,  # Added rotation augmentation
        brightness_range=[0.5, 1.5],  # Added brightness variation for robustness
        validation_split=validation_split  # Split data into training and validation
    )

    train_generator = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',  # Binary classification (fracture or no fracture)
        subset='training'  # Use this subset for training
    )

    validation_generator = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',  # Binary classification
        subset='validation'  # Use this subset for validation
    )

    return train_generator, validation_generator

# Build the CNN model for fracture detection
def build_cnn_model(input_shape=INPUT_SHAPE):
    """
    Builds a Convolutional Neural Network (CNN) model for fracture detection.
    :param input_shape: Shape of input CT scan image
    :return: Compiled CNN model
    """
    model = Sequential()

    # Convolutional layer block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional layer block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional layer block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (fracture or no fracture)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Train the model with k-fold cross-validation
def train_model_with_cross_validation(model, image_dir, epochs=25, batch_size=32, num_splits=5):
    """
    Train the CNN model for fracture detection using k-fold cross-validation.
    :param model: Compiled CNN model
    :param image_dir: Directory containing CT scan images
    :param epochs: Number of epochs for training
    :param batch_size: Batch size for training
    :param num_splits: Number of splits for cross-validation
    :return: Trained model
    """
    train_generator, validation_generator = create_data_generator(image_dir, batch_size=batch_size)

    checkpoint_callback = ModelCheckpoint(
        'best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )

    # Define the StratifiedKFold cross-validator
    kfold = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kfold.split(train_generator, validation_generator):
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[checkpoint_callback]
        )

    # Save the model for future use
    model.save('fracture_detection_model.h5')
    return model

# Evaluate the model with precision, recall, F1-score, and confusion matrix
def evaluate_model(model, validation_generator):
    """
    Evaluates the model on the validation set and outputs performance metrics.
    :param model: Trained model
    :param validation_generator: Generator for validation data
    :return: None
    """
    validation_loss, validation_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # Generate predictions on the validation data
    y_true = validation_generator.classes
    y_pred = (model.predict(validation_generator) > 0.5).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1], ['No Fracture', 'Fracture'])
    plt.yticks([0, 1], ['No Fracture', 'Fracture'])
    plt.show()

    # Additional Metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Save and load model using TensorFlow's saving/loading
def save_model(model, model_filename='fracture_detection_model.h5'):
    """
    Save the trained model for future use.
    :param model: Trained model
    :param model_filename: Filename to save the model
    :return: None
    """
    model.save(model_filename)

def load_model(model_filename='fracture_detection_model.h5'):
    """
    Load a previously trained model.
    :param model_filename: Filename of the saved model
    :return: Loaded model
    """
    return load_model(model_filename)

# Example of using the code
if __name__ == "__main__":
    # Define the directory containing your CT scan images (training and validation split)
    image_dir = 'path_to_ct_scan_images'

    # Create data generators for training and validation
    train_generator, validation_generator = create_data_generator(image_dir, batch_size=32)

    # Build and compile the CNN model
    model = build_cnn_model()

    # Train the model using k-fold cross-validation
    trained_model = train_model_with_cross_validation(model, image_dir, epochs=25)

    # Evaluate the model on validation data
    evaluate_model(trained_model, validation_generator)

    # Save the trained model for future use
    save_model(trained_model)

    # Load the model for future predictions
    model = load_model('fracture_detection_model.h5')

    # Make a prediction on a new CT scan image
    new_image_path = 'path_to_new_ct_scan_image.jpg'
    prediction = predict_fracture(new_image_path, model)
    print(f"Prediction for {new_image_path}: {prediction}")

