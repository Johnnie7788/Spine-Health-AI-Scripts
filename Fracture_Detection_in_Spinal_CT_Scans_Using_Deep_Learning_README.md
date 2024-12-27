
# Fracture Detection in Spinal CT Scans Using Deep Learning

## Overview
This project leverages deep learning techniques to detect fractures in spinal CT scans. Using a **Convolutional Neural Network (CNN)**, the model is trained to classify CT scans as either containing a **fracture** or **no fracture**. The workflow involves advanced **data augmentation**, **cross-validation**, and the use of **TensorFlow** for model saving/loading, ensuring robust and scalable performance in clinical applications.

## Key Features:
- **Spine Fracture Detection**: Uses a **CNN** to classify CT scans into two categories: fracture vs. no fracture.
- **Advanced Data Augmentation**: Implements various augmentation techniques like **rotation**, **zoom**, and **brightness variations** to improve generalization.
- **Cross-Validation**: Uses **StratifiedKFold** to ensure the model generalizes well across different data splits.
- **Model Persistence**: The model is saved and loaded using **TensorFlow’s built-in `model.save()` and `load_model()`**, which ensures proper handling of deep learning models for future predictions.
- **Evaluation Metrics**: Includes performance evaluation with **F1-score**, **Precision**, **Recall**, and **Confusion Matrix** for comprehensive model assessment.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- joblib

## Installation
1. Navigate to the project directory:
   ```bash
   cd Fracture_Detection_in_Spinal_CT_Scans_Using_Deep_Learning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Load and Preprocess the CT Scan
Use the `load_and_preprocess_image` function to load and preprocess the CT scan images before feeding them into the model.

```python
image = load_and_preprocess_image('path_to_ct_image.jpg')
```

### Step 2: Build the CNN Model
Create the CNN model for fracture detection.

```python
model = build_cnn_model()
```

### Step 3: Train the Model
Train the model with k-fold cross-validation.

```python
trained_model = train_model_with_cross_validation(model, 'path_to_image_directory', epochs=25)
```

### Step 4: Evaluate the Model
Evaluate the model using the validation dataset.

```python
evaluate_model(trained_model, validation_generator)
```

### Step 5: Save the Trained Model
After training, save the model to disk.

```python
save_model(trained_model, 'fracture_detection_model.h5')
```

### Step 6: Load the Trained Model
For future predictions, load the trained model.

```python
model = load_model('fracture_detection_model.h5')
```

### Step 7: Predict Fractures in New CT Scans
Use the model to predict whether a CT scan contains a fracture.

```python
prediction = predict_fracture('path_to_new_ct_scan_image.jpg', model)
print(f"Prediction for {new_image_path}: {prediction}")
```

## Model Evaluation
- **Accuracy**: The model’s accuracy is tracked during training and evaluated on the validation set.
- **F1-Score, Precision, Recall**: These metrics provide a more balanced view of the model's performance, especially for imbalanced datasets.
- **Confusion Matrix**: The confusion matrix visualizes the performance of the model with respect to true positives, false positives, true negatives, and false negatives.

## Contributing
Contributions are welcome! 

## License
The project is licensed under the MIT License. 