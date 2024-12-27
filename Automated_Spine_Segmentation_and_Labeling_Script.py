#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import SimpleITK as sitk  # For handling medical images like CT scans
from sklearn.preprocessing import StandardScaler

# 1. Load the CT scan data with error handling
def load_ct_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess the CT scan image.
    :param image_path: Path to the CT scan image
    :param target_size: Size to resize image for the model
    :return: Preprocessed image
    """
    try:
        image = load_img(image_path, target_size=target_size, color_mode="grayscale")
        image_array = img_to_array(image) / 255.0  # Normalize the image
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# 2. U-Net Architecture for Spine Segmentation with Batch Normalization and Dropout
def build_unet(input_shape=(256, 256, 1)):
    """
    Build U-Net model for spine segmentation with batch normalization and dropout.
    :param input_shape: Shape of input CT scan image
    :return: U-Net model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encoder: Convolutional layers with batch normalization and dropout
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.3)(p1)  # Dropout to prevent overfitting

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.3)(p2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.3)(p3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.3)(p4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)

    # Decoder: Upsampling layers
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    u6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    
    u7 = layers.UpSampling2D((2, 2))(u6)
    u7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    u7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    
    u8 = layers.UpSampling2D((2, 2))(u7)
    u8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    u8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    
    u9 = layers.UpSampling2D((2, 2))(u8)
    u9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    u9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u9)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 3. Model Training with Early Stopping
def train_unet_model(model, train_images, train_labels, epochs=10, batch_size=16):
    """
    Train U-Net model with early stopping.
    :param model: U-Net model
    :param train_images: Training images
    :param train_labels: Training labels
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    :return: Trained model
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return model

# 4. Dynamic Labeling for Vertebrae
def label_vertebrae(segmented_image):
    """
    Label the segmented vertebrae dynamically based on region properties.
    :param segmented_image: Segmented image from U-Net model
    :return: Labeled vertebrae
    """
    labeled_image = np.zeros_like(segmented_image)
    vertebra_labels = ['T1', 'T2', 'L1', 'L2', 'L3', 'L4', 'L5']
    num_vertebrae = len(vertebra_labels)
    
    # Dynamically label vertebrae based on unique segments
    for i in range(1, segmented_image.max() + 1):
        labeled_image[segmented_image == i] = i
    return labeled_image, vertebra_labels

# 5. Advanced Anomaly Detection with Autoencoders
def detect_anomalies(segmented_image):
    """
    Detect transitional vertebrae or other anomalies using an autoencoder.
    :param segmented_image: Segmented image from U-Net model
    :return: Anomalies detected in the CT scan
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    
    # Autoencoder model for anomaly detection
    input_img = Input(shape=(256, 256, 1))
    encoded = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)
    
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(encoded)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Train autoencoder on normal data
    autoencoder.fit(segmented_image, segmented_image, epochs=10, batch_size=16)

    # Detect anomalies
    reconstructed_image = autoencoder.predict(segmented_image)
    mse = np.mean(np.square(segmented_image - reconstructed_image), axis=(1, 2))
    
    anomalies = []
    for i in range(len(mse)):
        if mse[i] > np.percentile(mse, 95):  # Flagging anomalies above the 95th percentile of error
            anomalies.append(f"Anomaly detected in vertebra {i}")
    
    return anomalies

# 6. 3D Visualization (Optional, if CT data is 3D)
def visualize_3d(ct_scan_3d):
    """
    Visualize the 3D CT scan using PyVista for better understanding of anatomy.
    :param ct_scan_3d: 3D CT scan data
    """
    import pyvista as pv
    mesh = pv.wrap(ct_scan_3d)
    plotter = pv.Plotter()
    plotter.add_volume(mesh, cmap="bone")
    plotter.show()

# Main execution
if __name__ == '__main__':
    # Load and preprocess the CT image
    ct_image = load_ct_image('path_to_ct_image.jpg')

    # Build U-Net model
    model = build_unet(input_shape=(256, 256, 1))

    # Train the model on your dataset (you'll need labeled data for training)
    # Assuming you have `train_images` and `train_labels` for training the model
    # model = train_unet_model(model, train_images, train_labels)

    # Segment the spine from a CT scan
    segmented_image = model.predict(ct_image)

    # Label the vertebrae
    labeled_image, vertebra_labels = label_vertebrae(segmented_image)

    # Detect anomalies (e.g., transitional vertebrae)
    anomalies = detect_anomalies(segmented_image)

    # Visualize the results
    visualize_segmentation(segmented_image, labeled_image, anomalies)

    # If 3D CT scan is available, use 3D visualization
    # visualize_3d(ct_scan_3d)

