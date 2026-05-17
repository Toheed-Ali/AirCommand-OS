"""
process_dataset.py — Extracts MediaPipe landmarks from an existing image dataset.
Converts folders of images into a single CSV of raw landmarks.

Dataset Structure expected:
    dataset/
        train/
            fist/ ...
            palm/ ...
        test/
            fist/ ...
"""

import os
import sys
import csv
import logging
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    ROOT_DIR, RAW_DIR, NUM_LANDMARKS, COORDS_PER_LM,
    MP_HAND_LANDMARKER_MODEL_PATH, RAW_TRAIN_CSV_PATH, RAW_TEST_CSV_PATH,
)

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_DIR = ROOT_DIR / "dataset"

# Map folder names to config classes
FOLDER_TO_CLASS = {
    "3 fingers": "three_fingers",
    "fist":      "fist",
    "L":         "l_gesture",
    "palm":      "palm",
    "peace":     "peace",
    "thumbs-up": "thumbs_up",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("process_dataset")


def extract_landmark_vector(hand_landmarks) -> list[float]:
    vec = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return vec


def process():
    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        return

    # ── MediaPipe Task setup ──
    base_options = python.BaseOptions(model_asset_path=str(MP_HAND_LANDMARKER_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Prepare CSV Header
    header = [f"x{i}" if j == 0 else f"y{i}" if j == 1 else f"z{i}"
              for i in range(NUM_LANDMARKS) for j in range(COORDS_PER_LM)]
    header.append("gesture")

    # Iterate through splits (train/test)
    for split in ["train", "test"]:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue

        output_csv = RAW_TRAIN_CSV_PATH if split == "train" else RAW_TEST_CSV_PATH
        logger.info(f"Processing split: {split} -> Saving to {output_csv}")

        rows_count = 0
        fail_count = 0

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Iterate through gesture folders
            for folder_name, class_name in FOLDER_TO_CLASS.items():
                gesture_dir = split_dir / folder_name
                if not gesture_dir.exists():
                    continue

                images = list(gesture_dir.glob("*.jpg")) + list(gesture_dir.glob("*.png"))
                logger.info(f"  - {folder_name} ({len(images)} images) -> {class_name}")

                for img_path in tqdm(images, desc=f"    {class_name}"):
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    
                    result = detector.detect(mp_image)

                    if result.hand_landmarks:
                        hl = result.hand_landmarks[0]
                        vec = extract_landmark_vector(hl)
                        writer.writerow(vec + [class_name])
                        rows_count += 1
                    else:
                        fail_count += 1

        logger.info(f"Split '{split}' completed: {rows_count} landmarks, {fail_count} failed detections.\n")

    detector.close()
    logger.info("[Process Dataset] All splits successfully processed!")


if __name__ == "__main__":
    process()
