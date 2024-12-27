
# Automated Spine Segmentation and Labeling Script

## Overview
The **Automated Spine Segmentation and Labeling Script** is designed to automate the process of **spine segmentation** and **vertebrae labeling** from **CT scans** using an **AI-driven U-Net model**. It also includes **anomaly detection**, specifically for **transitional vertebrae**, and provides detailed **visualizations** of the segmented spine. This tool aims to assist **radiologists** and **clinicians** in evaluating **bone health** and improving **diagnostic accuracy** for conditions like **osteoporosis**.

The project is intended for healthcare applications, where it can aid in **automated bone health assessments** and **spinal analysis**.

## Key Features:
- **Spine Segmentation**: Automatically segments the spine in CT scans using a **U-Net** deep learning model.
- **Vertebrae Labeling**: Labels each vertebra (e.g., **T1**, **T2**, **L1**, **L2**) in the segmented image.
- **Anomaly Detection**: Identifies **transitional vertebrae** and other anomalies that might impact diagnosis and treatment decisions.
- **Model Training**: Trains the **U-Net model** on labeled data for spine segmentation.
- **Visualization**: Generates visualizations of the segmented spine and labeled vertebrae, highlighting any detected anomalies.
- **3D Visualization**: Supports 3D rendering for CT scans, providing a spatial understanding of the anatomy (optional).
  
## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- SimpleITK (for handling CT images)
- scikit-learn
- PyVista (optional for 3D visualization)

## Installation

1. Navigate to the project directory:
   ```bash
   cd spine-segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install **PyVista** for 3D visualization (optional):
   ```bash
   pip install pyvista
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

### Step 5: Label the Vertebrae
Label the vertebrae in the segmented spine.

```python
labeled_image, vertebra_labels = label_vertebrae(segmented_image)
```

### Step 6: Detect Anomalies
Detect anomalies such as **transitional vertebrae** or other abnormal vertebral structures.

```python
anomalies = detect_anomalies(segmented_image)
```

### Step 7: Visualize the Results
Visualize the segmented spine and labeled vertebrae.

```python
visualize_segmentation(segmented_image, labeled_image, anomalies)
```

### Step 8: 3D Visualization (Optional)
If the CT scan is 3D, use **PyVista** for 3D visualization.

```python
visualize_3d(ct_scan_3d)
```

## Contributing
Contributions are welcome! 

## License
The project is licensed under the MIT License. 
