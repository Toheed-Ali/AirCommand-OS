"""
export_tflite.py — Converts the trained Keras model to TFLite format for Flutter.

Optimisations applied:
  - DEFAULT quantisation (float16 weights) -> ~50% size reduction, negligible accuracy drop
  - Optional FULL_INTEGER (int8) quantisation -> maximum compression for mobile
  - Metadata embedding (input/output tensor names, gesture labels)

Usage:
    python training/export_tflite.py
    python training/export_tflite.py --int8   (requires representative dataset)
"""

import sys
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    KERAS_MODEL_PATH, TFLITE_MODEL_PATH, SCALER_PATH,
    PROCESSED_DIR, EXPORTS_DIR, GESTURE_CLASSES,
    FEATURE_SIZE, NUM_CLASSES,
)


def load_representative_dataset(n_samples: int = 200):
    """Generator for full-integer quantisation calibration."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy").astype(np.float32)
    indices = np.random.choice(len(X_train), size=n_samples, replace=False)
    for idx in indices:
        sample = X_train[idx:idx+1]   # shape (1, 63)
        yield [sample]


def export(use_int8: bool = False) -> None:
    if not KERAS_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Keras model not found at {KERAS_MODEL_PATH}.\n"
            f"Run training/train.py first."
        )

    print(f"[Export] Loading Keras model from {KERAS_MODEL_PATH}")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    model.summary(line_length=60)

    # ── Convert ──
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if use_int8:
        print("[Export] Applying full INT8 quantisation (requires calibration data)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = load_representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8
        out_path = EXPORTS_DIR / "gesture_model_int8.tflite"
    else:
        print("[Export] Applying DEFAULT (float16) quantisation...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        out_path = TFLITE_MODEL_PATH

    tflite_model = converter.convert()

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = out_path.stat().st_size / 1024
    print(f"[Export] TFLite model saved -> {out_path}  ({size_kb:.1f} KB)")

    # ── Verify: run a test inference ──
    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()

    in_details  = interpreter.get_input_details()
    out_details = interpreter.get_output_details()

    print(f"\n[Export] Model I/O verification:")
    print(f"  Input  : shape={in_details[0]['shape']}, dtype={in_details[0]['dtype'].__name__}")
    print(f"  Output : shape={out_details[0]['shape']}, dtype={out_details[0]['dtype'].__name__}")

    # Feed zeros
    dummy = np.zeros(in_details[0]["shape"], dtype=in_details[0]["dtype"])
    interpreter.set_tensor(in_details[0]["index"], dummy)
    interpreter.invoke()
    output = interpreter.get_tensor(out_details[0]["index"])
    print(f"  Test inference output: {output.round(3)}")
    print(f"  Sum of probs: {output.sum():.4f} (should be ~1.0 for float)")

    # ── Save Flutter integration spec ──
    spec = {
        "model_file":    out_path.name,
        "input_shape":   in_details[0]["shape"].tolist(),
        "input_dtype":   in_details[0]["dtype"].__name__,
        "output_shape":  out_details[0]["shape"].tolist(),
        "output_dtype":  out_details[0]["dtype"].__name__,
        "gesture_labels": GESTURE_CLASSES,
        "label_to_index": {g: i for i, g in enumerate(GESTURE_CLASSES)},
        "confidence_threshold": 0.85,
        "feature_size":  FEATURE_SIZE,
        "preprocessing": {
            "wrist_normalise": True,
            "hand_size_normalise": True,
            "z_clip": [-0.15, 0.15],
            "scaler": "StandardScaler — load scaler.pkl or embed means/scales",
            "note": "Apply wrist_normalise → hand_size_normalise → z_clip → StandardScaler before inference",
        },
    }
    spec_path = EXPORTS_DIR / "model_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"\n[Export] Flutter integration spec -> {spec_path}")

    # ── Export scaler parameters as JSON (so Flutter can apply them without pickle) ──
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        scaler_json = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "n_features": int(scaler.n_features_in_),
        }
        scaler_json_path = EXPORTS_DIR / "scaler_params.json"
        with open(scaler_json_path, "w") as f:
            json.dump(scaler_json, f, indent=2)
        print(f"[Export] Scaler params (JSON) -> {scaler_json_path}")

    print("\n[Export] Done (success)  Copy these files to your Flutter assets folder:")
    print(f"  {out_path.name}")
    print(f"  model_spec.json")
    print(f"  scaler_params.json")


def main():
    parser = argparse.ArgumentParser(description="Export trained model to TFLite")
    parser.add_argument("--int8", action="store_true",
                        help="Use full INT8 quantisation (smaller, needs calibration data)")
    args = parser.parse_args()
    export(use_int8=args.int8)


if __name__ == "__main__":
    main()
