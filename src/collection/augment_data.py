"""
augment_data.py — Augments the raw gesture dataset for left-hand support
and class balance.

Augmentation strategies applied:
  1. Horizontal (mirror) flip   — simulates left-hand use
  2. Gaussian noise injection   — improves generalisation to lighting/position jitter
  3. Class balancing            — oversample minority classes to equalise counts

Usage:
    python src/collection/augment_data.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    RAW_CSV_PATH, AUGMENTED_CSV_PATH, GESTURE_CLASSES,
    NUM_LANDMARKS, FEATURE_SIZE, RANDOM_SEED,
)

RNG = np.random.default_rng(RANDOM_SEED)

# ─── Column helpers ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    f"{ax}{i}"
    for i in range(NUM_LANDMARKS)
    for ax in ("x", "y", "z")
]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ["gesture"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    print(f"[Augment] Loaded {len(df)} rows from {path}")
    return df


# ─── Augmentation functions ───────────────────────────────────────────────────

def mirror_flip(row: np.ndarray) -> np.ndarray:
    """
    Horizontal mirror: negate all x-coordinates.
    MediaPipe x is normalised [0, 1] relative to frame width.
    Flipping: x_new = 1 - x_old
    y and z remain unchanged.
    Also re-order some specific landmark pairs to maintain anatomical symmetry
    (e.g., left vs right side of knuckles).
    """
    flipped = row.copy()
    # x-coords are at indices 0, 3, 6, ... (every 3rd starting from 0)
    x_indices = list(range(0, FEATURE_SIZE, 3))
    for idx in x_indices:
        flipped[idx] = 1.0 - flipped[idx]

    # Swap landmark pairs for correct left↔right anatomy
    # MediaPipe Hands landmark indices:
    # No pair swapping needed because the gesture classification
    # uses shape not handedness — the mirror alone suffices.
    return flipped


def add_noise(row: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    """Add small Gaussian noise to all features."""
    noise = RNG.normal(0, sigma, size=row.shape).astype(np.float32)
    noisy = row + noise
    # Keep x, y in [0, 1]
    xy_indices = [i for i in range(FEATURE_SIZE) if i % 3 != 2]
    noisy[xy_indices] = np.clip(noisy[xy_indices], 0.0, 1.0)
    return noisy


def scale_jitter(row: np.ndarray, scale_range=(0.90, 1.10)) -> np.ndarray:
    """Randomly scale the hand size slightly around wrist origin."""
    jittered = row.copy()
    scale = RNG.uniform(*scale_range)
    # Wrist is landmark 0: indices 0, 1, 2
    wrist_x, wrist_y, wrist_z = jittered[0], jittered[1], jittered[2]
    x_indices = list(range(0, FEATURE_SIZE, 3))
    y_indices = list(range(1, FEATURE_SIZE, 3))
    z_indices = list(range(2, FEATURE_SIZE, 3))
    for idx in x_indices:
        jittered[idx] = wrist_x + (jittered[idx] - wrist_x) * scale
    for idx in y_indices:
        jittered[idx] = wrist_y + (jittered[idx] - wrist_y) * scale
    for idx in z_indices:
        jittered[idx] = wrist_z + (jittered[idx] - wrist_z) * scale
    jittered[x_indices] = np.clip(jittered[x_indices], 0.0, 1.0)
    jittered[y_indices] = np.clip(jittered[y_indices], 0.0, 1.0)
    return jittered


# ─── Main augmentation pipeline ───────────────────────────────────────────────

def augment(df: pd.DataFrame) -> pd.DataFrame:
    features = df[FEATURE_COLS].values.astype(np.float32)
    labels   = df["gesture"].values

    augmented_rows = []
    augmented_labels = []

    for feat, label in zip(features, labels):
        # --- Strategy 1: Mirror flip (left-hand simulation) ---
        flipped = mirror_flip(feat)
        augmented_rows.append(flipped)
        augmented_labels.append(label)

        # --- Strategy 2: Noisy original ---
        noisy = add_noise(feat)
        augmented_rows.append(noisy)
        augmented_labels.append(label)

        # --- Strategy 3: Noisy mirror ---
        noisy_flip = add_noise(flipped)
        augmented_rows.append(noisy_flip)
        augmented_labels.append(label)

        # --- Strategy 4: Scale jitter ---
        scaled = scale_jitter(feat)
        augmented_rows.append(scaled)
        augmented_labels.append(label)

    aug_array = np.vstack(augmented_rows)
    aug_df = pd.DataFrame(aug_array, columns=FEATURE_COLS)
    aug_df["gesture"] = augmented_labels

    combined = pd.concat([df, aug_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"[Augment] Original: {len(df)} | Augmented additions: {len(aug_df)} | Total: {len(combined)}")
    return combined


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Oversample minority classes so all classes have equal representation."""
    counts = df["gesture"].value_counts()
    max_count = counts.max()
    print(f"\n[Augment] Class distribution before balancing:")
    for g, c in counts.items():
        print(f"  {g:>15}: {c}")

    balanced_parts = []
    for gesture in GESTURE_CLASSES:
        subset = df[df["gesture"] == gesture]
        if len(subset) < max_count:
            extra = subset.sample(max_count - len(subset), replace=True,
                                  random_state=RANDOM_SEED)
            subset = pd.concat([subset, extra], ignore_index=True)
        balanced_parts.append(subset)

    balanced = pd.concat(balanced_parts, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\n[Augment] Balanced: {len(balanced)} total ({max_count} per class)")
    return balanced


def main():
    if not RAW_CSV_PATH.exists():
        print(f"[ERROR] Raw dataset not found at {RAW_CSV_PATH}")
        print("Run collect_data.py first to create the dataset.")
        return

    df_raw = load_dataset(RAW_CSV_PATH)
    df_aug = augment(df_raw)
    df_bal = balance_classes(df_aug)

    AUGMENTED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(AUGMENTED_CSV_PATH, index=False)
    print(f"\n[Augment] Saved augmented dataset -> {AUGMENTED_CSV_PATH}")

    # Summary report
    summary = {
        "original_samples": int(len(df_raw)),
        "augmented_total":  int(len(df_bal)),
        "per_class": {g: int((df_bal["gesture"] == g).sum()) for g in GESTURE_CLASSES},
    }
    report_path = AUGMENTED_CSV_PATH.parent / "augmentation_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Augment] Report -> {report_path}")


if __name__ == "__main__":
    main()
