"""
model.py — MLP Gesture Classifier architecture.

Design decisions:
  - BatchNormalization after each Dense → training stability, faster convergence
  - Dropout for regularisation → prevents overfitting on a small dataset
  - L2 weight regularisation → additional overfitting guard
  - Softmax output → clean probability distribution over 6 classes
  - Invalid-gesture detection by confidence threshold (not a 7th class)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    FEATURE_SIZE, NUM_CLASSES, GESTURE_CLASSES,
    MLP_HIDDEN_LAYERS, MLP_DROPOUT_RATE,
    MLP_ACTIVATION, MLP_OUTPUT_ACTIVATION,
    LEARNING_RATE,
    KERAS_MODEL_PATH,
)


def build_model(
    input_dim: int = FEATURE_SIZE,
    num_classes: int = NUM_CLASSES,
    hidden_layers: list[int] = MLP_HIDDEN_LAYERS,
    dropout_rate: float = MLP_DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """
    Builds and compiles the MLP classifier.

    Architecture:
        Input(63)
        → Dense(256, relu) + BN + Dropout(0.3)
        → Dense(128, relu) + BN + Dropout(0.3)
        → Dense(64,  relu) + BN + Dropout(0.3)
        → Dense(6, softmax)

    Returns:
        Compiled keras.Model
    """
    l2_reg = regularizers.l2(1e-4)

    model = keras.Sequential(name="GestureClassifier_MLP")

    # Input layer
    model.add(layers.Input(shape=(input_dim,), name="landmarks_input"))

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units,
            activation=None,
            kernel_regularizer=l2_reg,
            bias_regularizer=l2_reg,
            name=f"dense_{i+1}",
        ))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.Activation(MLP_ACTIVATION, name=f"relu_{i+1}"))
        model.add(layers.Dropout(dropout_rate, name=f"dropout_{i+1}"))

    # Output layer
    model.add(layers.Dense(
        num_classes,
        activation=MLP_OUTPUT_ACTIVATION,
        name="gesture_output",
    ))

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def model_summary(model: keras.Model) -> None:
    model.summary(line_length=70)
    total  = model.count_params()
    train  = sum(np.prod(v.shape) for v in model.trainable_weights)
    frozen = total - train
    print(f"\n  Trainable params : {train:,}")
    print(f"  Non-trainable    : {frozen:,}")
    print(f"  Total            : {total:,}\n")


def load_model(path: Path = KERAS_MODEL_PATH) -> Optional[keras.Model]:
    if not path.exists():
        print(f"[Model] No saved model found at {path}")
        return None
    model = keras.models.load_model(path)
    print(f"[Model] Loaded from {path}")
    return model


def predict_gesture(
    model: keras.Model,
    feature_vector: np.ndarray,
    confidence_threshold: float = 0.85,
) -> dict:
    """
    Run a single inference. Returns a result dict with:
        gesture    : str  — predicted class or "invalid"
        confidence : float — probability of top class
        all_probs  : list[float] — probabilities for all 6 classes
        valid      : bool — True if confidence exceeds threshold
    """
    if feature_vector.ndim == 1:
        feature_vector = feature_vector[np.newaxis, :]   # (1, 63)

    probs = model.predict(feature_vector, verbose=0)[0]  # (6,)
    top_idx  = int(np.argmax(probs))
    top_conf = float(probs[top_idx])
    gesture  = GESTURE_CLASSES[top_idx]

    return {
        "gesture":    gesture if top_conf >= confidence_threshold else "invalid",
        "raw_gesture": gesture,
        "confidence": round(top_conf, 4),
        "all_probs":  {g: round(float(p), 4) for g, p in zip(GESTURE_CLASSES, probs)},
        "valid":      top_conf >= confidence_threshold,
    }


if __name__ == "__main__":
    print("Building model with default config...")
    m = build_model()
    model_summary(m)
