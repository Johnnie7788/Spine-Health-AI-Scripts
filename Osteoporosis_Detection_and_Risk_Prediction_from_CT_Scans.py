#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import SimpleITK as sitk
from skimage.feature import greycomatrix, greycoprops  # For texture features (Radiomics)
import joblib  # For model persistence

# Load CT scan image and preprocess
def load_ct_image(image_path, target_size=(256, 256)):
    """
    Loads and preprocesses a CT scan image.
    :param image_path: Path to the CT scan image
    :param target_size: Target size for image resizing
    :return: Preprocessed image
    """
    try:
        image = load_img(image_path, target_size=target_size, color_mode="grayscale")
        image_array = img_to_array(image) / 255.0  # Normalize the image
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Advanced feature extraction from CT scan image
def extract_features_from_ct_image(image):
    """
    Extracts features such as bone density and texture features from the CT scan image.
    :param image: Preprocessed CT scan image
    :return: Extracted features (e.g., bone density, texture features)
    """
    # Bone density: Compute mean pixel value (bone density proxy)
    bone_density = np.mean(image)

    # Texture features: Use GLCM (Gray Level Co-occurrence Matrix) for texture analysis
    glcm = greycomatrix(image.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]

    # Combine all features into a single vector
    return np.array([bone_density, contrast, correlation, energy, homogeneity])

# U-Net model for spine segmentation
def build_unet(input_shape=(256, 256, 1)):
    """
    Build a U-Net model for spine segmentation.
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
    p1 = layers.Dropout(0.3)(p1)

    # Decoder: Upsampling layers
    u6 = layers.UpSampling2D((2, 2))(p1)
    u6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    u6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

    # Output layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(u6)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Risk prediction model for osteoporosis using Random Forest
def train_osteoporosis_predictor(features, labels):
    """
    Trains a random forest model for osteoporosis risk prediction based on extracted features.
    :param features: Extracted features from CT scan images
    :param labels: Ground truth labels for osteoporosis (0 = no risk, 1 = high risk)
    :return: Trained model
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # Hyperparameter tuning with RandomizedSearchCV
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_model = RandomForestClassifier(random_state=42)
    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42)
    random_search.fit(X_train, y_train)

    # Best model evaluation
    best_rf_model = random_search.best_estimator_
    predictions = best_rf_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, best_rf_model.predict_proba(X_test)[:, 1])
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # Save the model
    joblib.dump(best_rf_model, 'osteoporosis_risk_predictor.pkl')

    return best_rf_model

# Load pre-trained model
def load_model(model_path='osteoporosis_risk_predictor.pkl'):
    """
    Loads the pre-trained Random Forest model for osteoporosis risk prediction.
    :param model_path: Path to the model file
    :return: Loaded model
    """
    return joblib.load(model_path)

# Predict osteoporosis risk based on CT scan and extracted features
def predict_osteoporosis_risk(ct_image_path, rf_model):
    """
    Predicts the risk of osteoporosis based on CT scan image using a pre-trained model.
    :param ct_image_path: Path to the CT scan image
    :param rf_model: Trained random forest model for risk prediction
    :return: Predicted osteoporosis risk (0 = no risk, 1 = high risk)
    """
    ct_image = load_ct_image(ct_image_path)
    if ct_image is None:
        return None

    # Feature extraction
    extracted_features = extract_features_from_ct_image(ct_image[0])

    # Make prediction using the trained Random Forest model
    risk_prediction = rf_model.predict([extracted_features])
    return risk_prediction[0]

# Example of using the code
if __name__ == "__main__":
    # Load and preprocess CT scan for osteoporosis prediction
    ct_scan_path = 'path_to_ct_scan.jpg'
    features = []  # List to store extracted features
    labels = []    # List to store labels (0 or 1 for osteoporosis risk)

    # Example loop for loading multiple CT scans and extracting features
    for ct_scan_path in ['path_to_ct_scan_1.jpg', 'path_to_ct_scan_2.jpg']:
        ct_image = load_ct_image(ct_scan_path)
        if ct_image is not None:
            features.append(extract_features_from_ct_image(ct_image[0]))
            labels.append(1 if 'high_risk' in ct_scan_path else 0)  # Example labeling logic

    # Train the risk predictor model
    features_array = np.array(features)
    labels_array = np.array(labels)
    rf_model = train_osteoporosis_predictor(features_array, labels_array)

    # Save the trained model for future use
    joblib.dump(rf_model, 'osteoporosis_risk_predictor.pkl')

    # Load the model and predict osteoporosis risk for a new CT scan
    rf_model = load_model('osteoporosis_risk_predictor.pkl')
    new_ct_scan_path = 'path_to_new_ct_scan.jpg'
    risk = predict_osteoporosis_risk(new_ct_scan_path, rf_model)
    print(f"Osteoporosis Risk for {new_ct_scan_path}: {'High Risk' if risk == 1 else 'No Risk'}")

