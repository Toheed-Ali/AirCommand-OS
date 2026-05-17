"""
collect_data.py — Interactive webcam-based gesture data collector.
Updated to use MediaPipe Tasks API for Python 3.13+ compatibility.

Usage:
    python src/collection/collect_data.py --gesture thumbs_up --samples 800
    python src/collection/collect_data.py --gesture palm --samples 800 --append

Controls during recording:
    SPACE  → capture current frame landmark
    R      → reset / discard last capture
    Q      → quit and save
    P      → pause / resume live preview
"""

import sys
import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    GESTURE_CLASSES, PROCESSED_DIR, CSV_PATH,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    NUM_LANDMARKS, COORDS_PER_LM, FEATURE_SIZE,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    MP_HAND_LANDMARKER_MODEL_PATH,
)

# ─── Hand Connections (for manual drawing) ────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # Index
    (5, 9), (9, 10), (10, 11), (11, 12),        # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky + Palm
]


def extract_landmark_vector(hand_landmarks) -> list[float]:
    """
    Returns a flat list of 63 raw floats [x0,y0,z0, x1,y1,z1, ...].
    """
    vec = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return vec


def draw_landmarks_manual(frame, landmarks, connections):
    h, w = frame.shape[:2]
    # Draw connections
    for start_idx, end_idx in connections:
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)), (0, 255, 0), 2)
    # Draw dots
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (255, 0, 255), -1)


def draw_hud(frame, gesture: str, captured: int, target: int,
             paused: bool, last_msg: str) -> None:
    h, w = frame.shape[:2]
    bar_pct = captured / target if target > 0 else 0

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Gesture: {gesture.upper()}",
                (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 220, 60), 1)
    cv2.putText(frame, f"Captured: {captured} / {target}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)

    bar_x, bar_y, bar_w, bar_h = 12, 68, w - 24, 10
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_w * bar_pct), bar_y + bar_h),
                  (60, 200, 120), -1)

    ctrl = "SPACE=capture  R=undo  P=pause  Q=save & quit"
    cv2.putText(frame, ctrl, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    if paused:
        cv2.putText(frame, "PAUSED", (w // 2 - 55, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 80, 255), 2)

    if last_msg:
        cv2.putText(frame, last_msg, (12, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1)


def save_to_csv(rows: list[list], gesture: str, append: bool) -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mode = "a" if (append and CSV_PATH.exists()) else "w"
    header_needed = not CSV_PATH.exists() or not append

    header = [f"x{i}" if j == 0 else f"y{i}" if j == 1 else f"z{i}"
              for i in range(NUM_LANDMARKS) for j in range(COORDS_PER_LM)]
    header.append("gesture")

    with open(CSV_PATH, mode, newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row + [gesture])

    return len(rows)


def collect(gesture: str, target_samples: int, append: bool) -> None:
    if gesture not in GESTURE_CLASSES:
        raise ValueError(f"Unknown gesture '{gesture}'. Choose from: {GESTURE_CLASSES}")

    # ── MediaPipe Task setup ──
    base_options = python.BaseOptions(model_asset_path=str(MP_HAND_LANDMARKER_MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=MP_MAX_HANDS,
        min_hand_detection_confidence=MP_MIN_DETECTION_CONF,
        min_hand_presence_confidence=MP_MIN_TRACKING_CONF,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    collected_rows: list[list] = []
    paused = False
    last_msg = ""
    msg_until = 0.0

    print(f"\n[Collector] Gesture: '{gesture}' | Target: {target_samples} samples")
    print("[Collector] Show your hand and press SPACE to capture.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural UX
        
        hand_detected = False
        vec = None

        if not paused:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                hand_detected = True
                hl = result.hand_landmarks[0]
                draw_landmarks_manual(frame, hl, HAND_CONNECTIONS)
                vec = extract_landmark_vector(hl)

        # Flash message timeout
        if time.time() > msg_until:
            last_msg = ""

        draw_hud(frame, gesture, len(collected_rows), target_samples,
                 paused, last_msg)

        # Hand-present indicator
        colour = (0, 220, 0) if hand_detected else (0, 0, 220)
        cv2.circle(frame, (frame.shape[1] - 24, 24), 10, colour, -1)

        cv2.imshow("Gesture Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):      # CAPTURE
            if hand_detected and vec is not None:
                collected_rows.append(vec)
                last_msg = f"✓ Captured #{len(collected_rows)}"
                msg_until = time.time() + 1.0
                if len(collected_rows) >= target_samples:
                    last_msg = "Target reached! Press Q to save."
            else:
                last_msg = "No hand detected — try again."
                msg_until = time.time() + 0.8

        elif key == ord("r") or key == ord("R"):     # UNDO
            if collected_rows:
                collected_rows.pop()
                last_msg = f"Removed last sample ({len(collected_rows)} remaining)"
                msg_until = time.time() + 1.0

        elif key == ord("p") or key == ord("P"):     # PAUSE
            paused = not paused

        elif key == ord("q") or key == ord("Q"):     # QUIT & SAVE
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    if collected_rows:
        saved = save_to_csv(collected_rows, gesture, append)
        print(f"\n[Collector] Saved {saved} samples for '{gesture}' → {CSV_PATH}")
    else:
        print("\n[Collector] Nothing captured. CSV not modified.")


def main():
    parser = argparse.ArgumentParser(description="Gesture sample collector")
    parser.add_argument("--gesture", required=True, choices=GESTURE_CLASSES,
                        help="Which gesture to collect")
    parser.add_argument("--samples", type=int, default=800,
                        help="Target number of samples to collect")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV rather than overwriting")
    args = parser.parse_args()

    collect(args.gesture, args.samples, args.append)


if __name__ == "__main__":
    main()
