# Detailed Documentation for AirCommand-OS Pipeline

## Overview
The AirCommand-OS project is designed to classify hand gestures using a pipeline that includes data collection, augmentation, preprocessing, model training, evaluation, and deployment. This document provides a detailed explanation of the pipeline, model architecture, and the processes involved.

---

## Pipeline Stages

### 1. Data Collection
The `process_dataset.py` script is responsible for extracting hand landmarks from images using MediaPipe. The dataset is structured as follows:

```
dataset/
    train/
        3 fingers/
        fist/
        L/
        palm/
        peace/
        thumbs-up/
    test/
        3 fingers/
        fist/
        L/
        palm/
        peace/
        thumbs-up/
```

#### Key Steps:
- **MediaPipe Integration**: Uses the MediaPipe Hand Landmarker model to extract 3D landmarks for each hand gesture.
- **CSV Conversion**: Converts the extracted landmarks into a structured CSV file for training and testing.
- **Class Mapping**: Maps folder names to gesture classes (e.g., `fist` → `fist`).

---

### 2. Data Augmentation
The `augment_data.py` script augments the raw dataset to improve model generalization and class balance. The following augmentation techniques are applied:

#### Techniques:
1. **Horizontal Flip**: Simulates left-hand gestures by negating all x-coordinates.
2. **Gaussian Noise Injection**: Adds random noise to landmarks to simulate jitter and improve robustness.
3. **Class Balancing**: Oversamples minority classes to ensure equal representation.

#### Implementation:
- **Feature Columns**: The script identifies columns corresponding to x, y, and z coordinates for each landmark.
- **Random Seed**: Ensures reproducibility of augmentations.

---

### 3. Model Architecture
The `model.py` script defines the architecture of the Multi-Layer Perceptron (MLP) used for gesture classification.

#### Design Decisions:
- **Batch Normalization**: Applied after each dense layer for training stability and faster convergence.
- **Dropout**: Regularization technique to prevent overfitting.
- **L2 Regularization**: Adds a penalty to large weights to further reduce overfitting.
- **Softmax Output**: Produces a probability distribution over the six gesture classes.

#### Architecture:
```
Input(63)
→ Dense(256, relu) + BatchNorm + Dropout(0.3)
→ Dense(128, relu) + BatchNorm + Dropout(0.3)
→ Dense(64, relu) + BatchNorm + Dropout(0.3)
→ Dense(6, softmax)
```

#### Hyperparameters:
- **Hidden Layers**: [256, 128, 64]
- **Dropout Rate**: 0.3
- **Learning Rate**: Configurable via `config.py`

---

### 4. Training
The training process involves feeding the augmented dataset into the MLP model. While the `train.py` file could not be accessed, the general steps include:
- Splitting the dataset into training, validation, and test sets.
- Compiling the model with an optimizer (e.g., Adam) and a loss function (e.g., categorical crossentropy).
- Training the model over multiple epochs with early stopping to prevent overfitting.

---

### 5. Evaluation
The evaluation script (`evaluate.py`) assesses the model's performance on the test set. Metrics such as accuracy, precision, recall, and F1-score are calculated.

---

### 6. Exporting the Model
The `export_tflite.py` script converts the trained model into TensorFlow Lite format for deployment on edge devices. This involves:
- Quantizing the model to reduce size and improve inference speed.
- Saving the model as `gesture_model.tflite`.

---

## Conclusion
This document outlines the key components of the AirCommand-OS pipeline, from data collection to model deployment. For further details, refer to the respective scripts in the `src/` directory.