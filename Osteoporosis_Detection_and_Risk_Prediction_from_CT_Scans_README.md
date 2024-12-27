
# Osteoporosis Detection and Risk Prediction from CT Scans

## Overview
This project implements a **deep learning and machine learning-based solution** for **osteoporosis detection** and **risk prediction** using **CT scan images**. The workflow involves **spine segmentation** using a **U-Net** model and **feature extraction** for **osteoporosis risk prediction**. It utilizes **Random Forest** for predicting the risk based on extracted features such as **bone density** and **texture patterns** from the segmented CT scans.

This approach aims to assist clinicians and radiologists in detecting osteoporosis early, improving diagnosis, and enabling **predictive healthcare**.

## Key Features:
- **Spine Segmentation**: Automatic segmentation of the spine in CT scans using a **U-Net** deep learning model.
- **Feature Extraction**: Extracts critical features such as **bone density** and **texture** (via **GLCM** for radiomics) for risk prediction.
- **Osteoporosis Risk Prediction**: Uses **Random Forest** model to predict osteoporosis risk (high or no risk) based on the extracted features.
- **Model Persistence**: Trains and saves the model using **joblib** for future predictions and reusability.
- **Model Evaluation**: Evaluates model performance with **ROC curve** and **AUC score**, ensuring that the model performs well for clinical use.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- SimpleITK (for handling CT images)
- scikit-learn
- scikit-image (for texture analysis)
- joblib (for model persistence)

## Installation

1. Navigate to the project directory:
   ```bash
   cd osteoporosis-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Load and Preprocess the CT Scan
The first step is to load and preprocess the CT scan image using the `load_ct_image` function. Ensure the image is in **grayscale** format (e.g., `.jpg`, `.png`, `.dicom`).

```python
ct_image = load_ct_image('path_to_ct_image.jpg')
```

### Step 2: Build the U-Net Model
Build the **U-Net model** for spine segmentation with batch normalization and dropout layers for regularization.

```python
model = build_unet(input_shape=(256, 256, 1))
```

### Step 3: Train the Model
Train the model using the `train_unet_model` function. The dataset should include **CT images** and **labeled vertebrae**.

```python
model = train_unet_model(model, train_images, train_labels)
```

### Step 4: Segment the Spine
Use the trained model to segment the spine from a CT scan.

```python
segmented_image = model.predict(ct_image)
```

### Step 5: Extract Features
Extract relevant features from the segmented CT scan for osteoporosis prediction.

```python
features = extract_features_from_ct_image(segmented_image)
```

### Step 6: Train the Risk Prediction Model
Train the **Random Forest** model using the extracted features.

```python
rf_model = train_osteoporosis_predictor(features, labels)
```

### Step 7: Predict Osteoporosis Risk
Predict the osteoporosis risk for a new CT scan using the trained model.

```python
risk = predict_osteoporosis_risk('path_to_new_ct_scan.jpg', rf_model)
print(f"Osteoporosis Risk: {'High Risk' if risk == 1 else 'No Risk'}")
```

### Step 8: Save and Load the Model
Save the trained **Random Forest** model for future use.

```python
joblib.dump(rf_model, 'osteoporosis_risk_predictor.pkl')
```

Load the model for future predictions.

```python
rf_model = joblib.load('osteoporosis_risk_predictor.pkl')
```

## Model Evaluation
The model is evaluated using **cross-validation** during training and **ROC curve** and **AUC score** during testing to assess its **performance**.

## Contributing
Contributions are welcome! 

## License
The project is licensed under the MIT License. 

