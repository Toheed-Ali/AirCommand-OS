"""
train.py — Complete model training pipeline.

What this does:
  1. Loads preprocessed train/val/test splits
  2. Builds the MLP with architecture from config
  3. Trains with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  4. Evaluates on held-out test set (accuracy, precision, recall, F1)
  5. Generates confusion matrix
  6. Saves training history as JSON (for plotting)
  7. Saves best model in Keras format

Usage:
    python training/train.py
    python training/train.py --epochs 200 --batch-size 32
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    PROCESSED_DIR, KERAS_MODEL_PATH, TRAINING_HISTORY_PATH,
    GESTURE_CLASSES, NUM_CLASSES, FEATURE_SIZE,
    MLP_HIDDEN_LAYERS, MLP_DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR, MIN_LR, RANDOM_SEED,
)
from src.model.model import build_model, model_summary


# ─── Reproducibility ─────────────────────────────────────────────────────────
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_splits():
    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    splits = {}
    for name in required:
        p = PROCESSED_DIR / f"{name}.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Processed split '{name}.npy' not found.\n"
                f"Run src/preprocessing/preprocess.py first."
            )
        splits[name] = np.load(p)
        print(f"  Loaded {name}: {splits[name].shape}")
    return splits


def build_callbacks(checkpoint_path: Path) -> list:
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(KERAS_MODEL_PATH.parent / "training_log.csv"),
            append=False,
        ),
    ]
    return callbacks


def evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    # Inference
    probs    = model.predict(X_test, verbose=0)
    y_pred   = np.argmax(probs, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\nOverall Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Weighted F1      : {f1:.4f}\n")

    # Per-class report
    report = classification_report(
        y_test, y_pred,
        target_names=GESTURE_CLASSES,
        digits=4,
    )
    print("Classification Report:")
    print(report)

    # Confusion matrix (pretty printed)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    header = f"{'':>15}  " + "  ".join(f"{g[:6]:>6}" for g in GESTURE_CLASSES)
    print(header)
    for i, row in enumerate(cm):
        label = GESTURE_CLASSES[i][:15]
        vals  = "  ".join(f"{v:>6}" for v in row)
        print(f"{label:>15}  {vals}")

    return {
        "accuracy":  float(round(acc, 6)),
        "f1_weighted": float(round(f1, 6)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_history(history: keras.callbacks.History) -> None:
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"\n[Train] History saved -> {TRAINING_HISTORY_PATH}")


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    print("\n" + "="*60)
    print(" GESTURE CLASSIFIER — TRAINING")
    print("="*60)

    # ── Load data ──
    print("\n[Train] Loading splits...")
    splits = load_splits()
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]

    # Verify shapes
    assert X_train.shape[1] == FEATURE_SIZE, (
        f"Feature size mismatch: expected {FEATURE_SIZE}, got {X_train.shape[1]}"
    )

    # ── Build model ──
    print("\n[Train] Building model...")
    model = build_model(
        input_dim=FEATURE_SIZE,
        num_classes=NUM_CLASSES,
        hidden_layers=MLP_HIDDEN_LAYERS,
        dropout_rate=MLP_DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )
    model_summary(model)

    # ── Train ──
    print(f"[Train] Training for up to {epochs} epochs | Batch size: {batch_size}")
    print(f"[Train] Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs\n")

    KERAS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(KERAS_MODEL_PATH)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Save history ──
    save_history(history)

    # ── Final evaluation ──
    eval_results = evaluate(model, X_test, y_test)

    # ── Save evaluation report ──
    report_path = KERAS_MODEL_PATH.parent / "evaluation_report.json"
    with open(report_path, "w") as f:
        eval_report = {k: v for k, v in eval_results.items() if k != "classification_report"}
        json.dump(eval_report, f, indent=2)
    print(f"[Train] Evaluation report -> {report_path}")

    # ── Model already saved by checkpoint callback ──
    print(f"\n[Train] Best model -> {KERAS_MODEL_PATH}")
    print("[Train] Training complete (success)")

    return model, history, eval_results


def main():
    parser = argparse.ArgumentParser(description="Train gesture MLP classifier")
    parser.add_argument("--epochs",     type=int, default=EPOCHS,     help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
