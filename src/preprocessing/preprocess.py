"""
preprocess.py — Feature engineering and dataset preparation.

Pipeline:
  1. Load separate train and test raw/augmented CSV files.
  2. Wrist-origin normalisation (translate hand to origin).
  3. Hand-size normalisation (scale by wrist→middle-finger-tip distance).
  4. Z-depth clipping (MediaPipe z is noisy relative to xy).
  5. Split training data into Train (85%) and Validation (15%).
  6. Fit StandardScaler strictly on the training set only to prevent data leakage.
  7. Apply scale transform to train, val, and test splits.
  8. Save processed splits + fitted scaler + label encoder.

Usage:
    python src/preprocessing/preprocess.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    RAW_TRAIN_CSV_PATH, RAW_TEST_CSV_PATH, AUGMENTED_TRAIN_CSV_PATH,
    PROCESSED_DIR, GESTURE_CLASSES, NUM_LANDMARKS, FEATURE_SIZE,
    NORMALIZE_BY_WRIST, SCALE_BY_HAND_SIZE, CLIP_Z_DEPTH, Z_CLIP_RANGE,
    VALIDATION_SPLIT, RANDOM_SEED, SCALER_PATH, LABEL_ENCODER_PATH,
)

FEATURE_COLS = [
    f"{ax}{i}"
    for i in range(NUM_LANDMARKS)
    for ax in ("x", "y", "z")
]


# ─── Normalisation ────────────────────────────────────────────────────────────

def wrist_normalise(X: np.ndarray) -> np.ndarray:
    """
    Translate all landmarks so wrist (landmark 0) is at origin.
    X shape: (N, 63)
    Wrist x=X[:,0], y=X[:,1], z=X[:,2]
    """
    X_norm = X.copy()
    wrist_x = X_norm[:, 0:1]   # (N, 1)
    wrist_y = X_norm[:, 1:2]
    wrist_z = X_norm[:, 2:3]

    x_indices = list(range(0, FEATURE_SIZE, 3))
    y_indices = list(range(1, FEATURE_SIZE, 3))
    z_indices = list(range(2, FEATURE_SIZE, 3))

    X_norm[:, x_indices] -= wrist_x
    X_norm[:, y_indices] -= wrist_y
    X_norm[:, z_indices] -= wrist_z

    return X_norm


def hand_size_normalise(X: np.ndarray) -> np.ndarray:
    """
    Scale landmarks by the distance from wrist (lm 0) to middle-finger tip (lm 12).
    After wrist_normalise, wrist is at origin, so distance = |lm12|.

    lm 12 (middle tip): indices [36, 37, 38] (landmark 12 × 3)
    """
    X_norm = X.copy()
    tip_x = X_norm[:, 36]   # lm12 x
    tip_y = X_norm[:, 37]   # lm12 y
    tip_z = X_norm[:, 38]   # lm12 z

    hand_size = np.sqrt(tip_x**2 + tip_y**2 + tip_z**2)
    hand_size = np.where(hand_size < 1e-6, 1.0, hand_size)  # avoid div-by-zero

    X_norm = X_norm / hand_size[:, np.newaxis]
    return X_norm


def clip_z(X: np.ndarray, z_range=Z_CLIP_RANGE) -> np.ndarray:
    X_clip = X.copy()
    z_indices = list(range(2, FEATURE_SIZE, 3))
    X_clip[:, z_indices] = np.clip(X_clip[:, z_indices], z_range[0], z_range[1])
    return X_clip


def apply_feature_pipeline(X: np.ndarray) -> np.ndarray:
    """Apply the full deterministic transform (no fitting required)."""
    if NORMALIZE_BY_WRIST:
        X = wrist_normalise(X)
    if SCALE_BY_HAND_SIZE:
        X = hand_size_normalise(X)
    if CLIP_Z_DEPTH:
        X = clip_z(X)
    return X


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # Load training data (prefer augmented train if it exists)
    train_src = AUGMENTED_TRAIN_CSV_PATH if AUGMENTED_TRAIN_CSV_PATH.exists() else RAW_TRAIN_CSV_PATH
    if not train_src.exists():
        raise FileNotFoundError(
            f"Training dataset not found. Run process_dataset.py and augment_data.py first.\n"
            f"Expected at: {train_src}"
        )
    train_df = pd.read_csv(train_src)
    print(f"[Preprocess] Loaded {len(train_df)} training samples from {train_src.name}")

    # Load testing data (always raw and untouched to prevent leakage)
    test_src = RAW_TEST_CSV_PATH
    if not test_src.exists():
        raise FileNotFoundError(
            f"Testing dataset not found. Run process_dataset.py first.\n"
            f"Expected at: {test_src}"
        )
    test_df = pd.read_csv(test_src)
    print(f"[Preprocess] Loaded {len(test_df)} testing samples from {test_src.name}")

    print(f"\n[Preprocess] Training Class distribution:")
    for g, c in train_df["gesture"].value_counts().items():
        print(f"  {g:>15}: {c:5d}")

    print(f"[Preprocess] Testing Class distribution:")
    for g, c in test_df["gesture"].value_counts().items():
        print(f"  {g:>15}: {c:5d}")

    return train_df, test_df


def preprocess_and_split():
    train_df, test_df = load_data()

    X_train_raw = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train_raw = train_df["gesture"].values

    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test_raw = test_df["gesture"].values

    # ── Label encoding ──
    le = LabelEncoder()
    le.classes_ = np.array(GESTURE_CLASSES)   # fixed order from config
    y_train_encoded = le.transform(y_train_raw)
    y_test_encoded  = le.transform(y_test_raw)
    print(f"\n[Preprocess] Label encoding: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # ── Feature pipeline (deterministic) ──
    X_train_processed = apply_feature_pipeline(X_train_raw)
    X_test_processed  = apply_feature_pipeline(X_test_raw)
    print(f"[Preprocess] Feature pipeline applied successfully.")

    # ── Validation Split from Training Data ──
    # Split the processed training data into final Train and Validation splits
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_processed, y_train_encoded,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y_train_encoded,
    )

    # Test data remains untouched and isolated
    X_test = X_test_processed
    y_test = y_test_encoded

    print(f"\n[Preprocess] Split sizes (Zero Data Leakage):")
    print(f"  Train:      {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test:       {len(X_test)}")

    # ── StandardScaler (fit on train ONLY) ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n[Preprocess] Scaler fitted on training split (no validation or test leakage).")
    print(f"  Feature means  (first 5): {scaler.mean_[:5].round(4)}")
    print(f"  Feature scales (first 5): {scaler.scale_[:5].round(4)}")

    # ── Save splits ──
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }
    for name, arr in splits.items():
        path = PROCESSED_DIR / f"{name}.npy"
        np.save(path, arr)
        print(f"  Saved {name} -> {path}")

    # ── Save scaler & encoder ──
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"\n[Preprocess] Scaler  -> {SCALER_PATH}")
    print(f"[Preprocess] Encoder -> {LABEL_ENCODER_PATH}")

    # ── Preprocessing report ──
    report = {
        "total_train_samples": int(len(train_df)),
        "total_test_samples":  int(len(test_df)),
        "train_size":          int(len(X_train)),
        "val_size":            int(len(X_val)),
        "test_size":           int(len(X_test)),
        "feature_size":        int(X_train.shape[1]),
        "wrist_normalise":     NORMALIZE_BY_WRIST,
        "hand_size_normalise": SCALE_BY_HAND_SIZE,
        "z_clip":              CLIP_Z_DEPTH,
        "label_mapping": {g: int(i) for i, g in enumerate(le.classes_)},
    }
    report_path = PROCESSED_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Preprocess] Report  -> {report_path}")

    return splits, scaler, le


if __name__ == "__main__":
    preprocess_and_split()
