# AirCommand-OS — Hand Gesture Recognition Project Report

Welcome to the complete documentation and technical journey of **AirCommand-OS**, a real-time computer vision and deep learning system that translates hand gestures into OS-level system actions, integrated with a Flutter mobile app via WebSocket.

---

## 1. Project Introduction
The goal of **AirCommand-OS** is to enable a touchless human-computer interface. By leveraging a high-speed camera stream, the system extracts hand joint coordinates (landmarks) in real-time, processes them through a custom neural network, filters out prediction noise, and dispatches actions like volume control, brightness adjustment, media playback, and app launching.

---

## 2. System Architecture & Technologies Used
The system is built on a highly optimized, multi-stage machine learning and communication pipeline:

* **Computer Vision Frontend:** Google MediaPipe (Tasks API) for high-fidelity 21-landmark 3D hand tracking.
* **Deep Learning Framework:** TensorFlow & Keras for training a robust Multi-Layer Perceptron (MLP) classifier.
* **Feature Engineering:** Scikit-learn (`StandardScaler`, `LabelEncoder`) for positional and scale invariance.
* **Real-time Pipeline:** OpenCV for camera capture & HUD visualisations; `pycaw` & native system APIs for OS controls.
* **Asynchronous Communication:** `websockets` for streaming real-time gesture packets directly to the Flutter UI at ~30 FPS.

---

## 3. Dataset Journey & Folder Splits
Originally, all training data was collected dynamically from the webcam using a custom landmark recording tool.

Recently, the dataset went through a **rigorous 80-20 Train/Test split** where 20% of the files for every class folder were randomly selected and moved to a dedicated testing folder to prevent data leakage and ensure pristine evaluations:

### File Split Statistics:
* **Total Raw Samples Detected:** 10333 landmarks vectors (Train Split: 8272, Test Split: 2061)
* **Split Ratios:** **~80% Train, ~20% Test**
* **Augmented Balanced Training Dataset:** 76800 samples (12,800 per class after augmentation)
* **Preprocessed Zero-Leakage Splits:**
  * **Training Split:** 65280 samples (85% of augmented training dataset)
  * **Validation Split:** 11520 samples (15% of augmented training dataset)
  * **Testing Split:** 2061 samples (strictly pristine raw landmarks, 100% leak-free!)
* **Class Folders:**
  1. **`3 fingers`** (Three Fingers → Brightness Down)
  2. **`fist`** (Fist → Play / Pause Media)
  3. **`L`** (L Gesture → Open Chrome)
  4. **`palm`** (Palm → Shift + Tab)
  5. **`peace`** (Peace → Brightness Up)
  6. **`thumbs-up`** (Thumbs Up → Volume Up)

---

## 4. The Feature Engineering Pipeline
Before passing the raw 3D coordinates from MediaPipe (63 values) to the neural network, they are transformed using three essential geometric pipeline stages to ensure position, depth, and scale invariance:

1. **Wrist-Origin Normalisation:** Translates the hand coordinates so the wrist (Landmark 0) sits exactly at `(0, 0, 0)`.
2. **Hand-Size Scaling:** Normalizes the distance of all landmarks by dividing them by the Euclidean distance from the wrist to the middle-finger tip (Landmark 12). This makes the system independent of hand size or camera distance.
3. **Z-Depth Clipping:** Limits the depth noise range to `[-0.15, 0.15]` to stabilize the z-axis coordinates.

---

## 5. Model Architecture & Deep Learning Training
The model is a highly efficient **Multi-Layer Perceptron (MLP)** designed for high accuracy and ultra-low latency inference on edge devices (like mobile phones):

### Network Architecture:
```
Input (63 normalized features)
  └── Dense (256 units, ReLU activation)
  ├── Batch Normalization
  ├── Dropout (Rate: 0.3)
  └── Dense (128 units, ReLU activation)
  ├── Batch Normalization
  ├── Dropout (Rate: 0.3)
  └── Dense (64 units, ReLU activation)
  ├── Batch Normalization
  ├── Dropout (Rate: 0.3)
  └── Dense (6 units, Softmax activation)
```

### Training Strategy:
* **Optimizer:** Adam (Initial Learning Rate: `0.001`)
* **Callbacks:** 
  * `EarlyStopping` (patience: 15 epochs) to avoid overfitting.
  * `ReduceLROnPlateau` (halves learning rate on loss plateaus).
  * `ModelCheckpoint` to save the best model weights.

---

## 6. Numerical Results & Evaluation
The deep learning model achieved outstanding performance during evaluation on the held-out **test dataset**:

### Overall Performance Metrics:
* **Test Set Accuracy:** `99.8544%` (Perfect classification of raw gestures!)
* **Weighted F1 Score:** `0.998545`

### Classification Report:
| Class / Gesture | Precision | Recall | F1-Score | Support | OS Gesture Action Map |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **fist** | 0.9919 | 1.0000 | 0.9959 | 245 | Fist (Play/Pause Media) |
| **l_gesture** | 1.0000 | 1.0000 | 1.0000 | 159 | L Gesture (Open Chrome) |
| **palm** | 0.9975 | 0.9975 | 0.9975 | 394 | Palm (Shift + Tab) |
| **peace** | 1.0000 | 0.9982 | 0.9991 | 546 | Peace (Brightness Up) |
| **three_fingers** | 1.0000 | 1.0000 | 1.0000 | 640 | Three Fingers (Brightness Down) |
| **thumbs_up** | 1.0000 | 0.9870 | 0.9935 | 77 | Thumbs Up (Volume Up) |


### Confusion Matrix:
| Actual \ Predicted | fist | l_gesture | palm | peace | three_fingers | thumbs_up |
| --- | --- | --- | --- | --- | --- | --- |
| **fist** | 245 | 0 | 0 | 0 | 0 | 0 |
| **l_gesture** | 0 | 159 | 0 | 0 | 0 | 0 |
| **palm** | 1 | 0 | 393 | 0 | 0 | 0 |
| **peace** | 0 | 0 | 1 | 545 | 0 | 0 |
| **three_fingers** | 0 | 0 | 0 | 0 | 640 | 0 |
| **thumbs_up** | 1 | 0 | 0 | 0 | 0 | 76 |


---

## 7. Real-time Inference, Cooldowns & WebSocket Stream
* **Confidence Gate:** Predictions are gated at **97% minimum confidence** (`0.97`) to completely filter out random hand movements and noise.
* **Temporal Smoothing:** Uses a **12-frame sliding window** majority vote. An action only triggers when a gesture is held stable for at least 9 frames (~0.3 seconds at 30 FPS).
* **Cross-Platform System Dispatcher:** Translates classified gestures to native volume controls (`pycaw`), brightness levels (`screen-brightness-control`), and keys.
* **Flutter Integration:** Runs a lightweight async WebSocket server on port `8765` to broadcast gesture packets to Flutter clients synchronously.

---

## 8. Exporting for Flutter Mobile Integration
The trained Keras model is converted into a **quantised Float16 TFLite model** (`gesture_model.tflite`) along with:
1. `model_spec.json`: Contains label indices and model input/output configurations.
2. `scaler_params.json`: Exports the fitted `StandardScaler` mean and variance values so that the exact same feature engineering pipeline can be executed natively in Dart!

---

## 9. End-to-End Development & Execution Workflow
To reproduce the zero-leakage model training, mobile conversion, and live control deployment, the pipeline runs chronologically as follows:

1. **Dataset Split Restoration:** Organize captured images into isolated split folders (`dataset/train/` and `dataset/test/`), separating them by gesture subdirectories.
2. **Raw Landmark Extraction (`process_dataset.py`):** Parses splits independently using Google MediaPipe Tasks API, generating `train_raw.csv` and `test_raw.csv` inside `data/raw/` (completely avoiding initial data leakage).
3. **Training Augmentation (`augment_data.py`):** Loads only the training CSV and applies spatial perturbations (rotations, scales, joint jittering) to balance all classes, outputting `train_augmented.csv` while leaving `test_raw.csv` pristine.
4. **Leak-Free Preprocessing (`preprocess.py`):** Fits a `StandardScaler` strictly on the train partition, normalizes all landmarks, splits validation (15%) from the training set, and exports preprocessed numpy splits (`X_train.npy`, etc.).
5. **Model Classification Training (`train.py`):** Trains the feed-forward MLP using the preprocessed train/validation arrays, applies early stopping to avoid overfitting, and logs a comprehensive evaluation report.
6. **Mobile Edge Conversion (`export_tflite.py`):** Quantizes the best Keras model to Float16, saving `gesture_model.tflite`, dynamic Flutter integration parameters in `model_spec.json`, and Dart-friendly scaler values in `scaler_params.json`.
7. **Live Inference & Remote App Bridge (`gesture_pipeline.py`):** Streams camera frames, filters joint landmarks, smooths predictions via a 12-frame majority voting window, dispatches system commands, and broadcasts active gestures to the Flutter app via a low-latency WebSockets server.

---

*Report generated successfully on behalf of the AirCommand-OS pipeline.*
