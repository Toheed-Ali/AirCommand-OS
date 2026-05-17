"""
evaluate.py — Detailed model evaluation with visualisations.

Generates:
  - Confusion matrix heatmap (PNG)
  - Training/validation accuracy + loss curves (PNG)
  - Per-class precision/recall/F1 bar chart (PNG)
  - Confidence distribution histogram (PNG)
  - Full text report

Usage:
    python training/evaluate.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    KERAS_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
    PROCESSED_DIR, MODELS_DIR, GESTURE_CLASSES,
    TRAINING_HISTORY_PATH, CONFIDENCE_THRESHOLD,
)

PLOTS_DIR = MODELS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Consistent color palette
PALETTE = ["#5C5FD1", "#2DB98C", "#E8593C", "#F2A623",
           "#A855E8", "#3B8BD4"]


def load_all():
    model   = tf.keras.models.load_model(KERAS_MODEL_PATH)
    X_test  = np.load(PROCESSED_DIR / "X_test.npy")
    y_test  = np.load(PROCESSED_DIR / "y_test.npy")

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(TRAINING_HISTORY_PATH) as f:
        history = json.load(f)

    return model, X_test, y_test, scaler, encoder, history


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold", y=1.02)

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw counts",     "d"),
        (axes[1], cm_norm, "Normalised (%)", ".2%"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix → {save_path}")


def plot_training_curves(history: dict, save_path: Path):
    epochs = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(epochs, history["accuracy"],     color=PALETTE[0], lw=2, label="Train")
    axes[0].plot(epochs, history["val_accuracy"], color=PALETTE[1], lw=2, linestyle="--", label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(max(history["val_accuracy"]), color=PALETTE[1],
                    lw=0.8, linestyle=":", alpha=0.6,
                    label=f"Best val: {max(history['val_accuracy']):.4f}")
    axes[0].legend(fontsize=9)

    # Loss
    axes[1].plot(epochs, history["loss"],     color=PALETTE[2], lw=2, label="Train")
    axes[1].plot(epochs, history["val_loss"], color=PALETTE[3], lw=2, linestyle="--", label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sparse categorical crossentropy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Training curves → {save_path}")


def plot_per_class_metrics(y_true, y_pred, labels, save_path: Path):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels)))
    )

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, precision, width, label="Precision", color=PALETTE[0], alpha=0.85)
    ax.bar(x,          recall,   width, label="Recall",    color=PALETTE[1], alpha=0.85)
    ax.bar(x + width,  f1,       width, label="F1 Score",  color=PALETTE[2], alpha=0.85)

    # Annotate with values
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics (Precision / Recall / F1)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Support text
    for i, s in enumerate(support):
        ax.text(x[i], -0.07, f"n={s}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Per-class metrics → {save_path}")


def plot_confidence_distribution(probs: np.ndarray, y_true, y_pred, labels, save_path: Path):
    top_confs = probs.max(axis=1)
    correct   = (y_true == y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Confidence Distribution", fontsize=13, fontweight="bold")

    # Overall distribution
    axes[0].hist(top_confs[correct],  bins=40, color=PALETTE[1], alpha=0.7, label="Correct")
    axes[0].hist(top_confs[~correct], bins=40, color=PALETTE[2], alpha=0.7, label="Wrong")
    axes[0].axvline(CONFIDENCE_THRESHOLD, color="black", lw=1.5, linestyle="--",
                    label=f"Threshold ({CONFIDENCE_THRESHOLD})")
    axes[0].set_xlabel("Model confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Correct vs Wrong predictions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Per-class median confidence
    medians = [
        np.median(top_confs[y_pred == i]) for i in range(len(labels))
    ]
    axes[1].bar(labels, medians, color=PALETTE, alpha=0.85)
    axes[1].set_ylabel("Median confidence")
    axes[1].set_title("Median confidence per class")
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.3)
    for i, m in enumerate(medians):
        axes[1].text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confidence distribution → {save_path}")


def main():
    print("[Eval] Loading model and test data...")
    try:
        model, X_test, y_test, scaler, encoder, history = load_all()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"[Eval] Test set: {len(X_test)} samples")

    probs  = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    print("\n" + "="*60)
    print(classification_report(y_test, y_pred, target_names=GESTURE_CLASSES, digits=4))
    print("="*60 + "\n")

    # ── Plots ──
    plot_confusion_matrix(y_test, y_pred, GESTURE_CLASSES,
                          PLOTS_DIR / "confusion_matrix.png")

    plot_training_curves(history,
                         PLOTS_DIR / "training_curves.png")

    plot_per_class_metrics(y_test, y_pred, GESTURE_CLASSES,
                           PLOTS_DIR / "per_class_metrics.png")

    plot_confidence_distribution(probs, y_test, y_pred, GESTURE_CLASSES,
                                 PLOTS_DIR / "confidence_distribution.png")

    print(f"\n[Eval] All plots saved to {PLOTS_DIR}")
    print("[Eval] Evaluation complete ✓")


if __name__ == "__main__":
    main()
