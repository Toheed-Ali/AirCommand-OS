"""
gesture_pipeline.py — Main real-time gesture recognition loop.
Updated to use MediaPipe Tasks API for Python 3.13+ compatibility.

Complete pipeline per frame:
  Webcam → MediaPipe landmarks → Feature engineering → Scaler → MLP predict
  → Confidence check → Temporal smoother → Action engine → WebSocket broadcast

Usage:
    python src/realtime/gesture_pipeline.py
    python src/realtime/gesture_pipeline.py --no-display   (headless server mode)
    python src/realtime/gesture_pipeline.py --dry-run      (no system actions)
"""

import sys
import time
import json
import pickle
import logging
import asyncio
import argparse
import threading
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    KERAS_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, DISPLAY_FPS,
    MP_MAX_HANDS, MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    FEATURE_SIZE, NUM_LANDMARKS, GESTURE_CLASSES,
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW_SIZE,
    WEBSOCKET_HOST, WEBSOCKET_PORT,
    MP_HAND_LANDMARKER_MODEL_PATH,
    GESTURE_ACTION_MAP, GESTURE_DISPLAY_NAMES,
)
from src.realtime.temporal_smoother import TemporalSmoother
from src.realtime.action_engine import ActionEngine
from src.preprocessing.preprocess import apply_feature_pipeline

logger = logging.getLogger("gesture_pipeline")

# ─── Hand Connections (for manual drawing) ────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),             # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),             # Index
    (5, 9), (9, 10), (10, 11), (11, 12),        # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),      # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Pinky + Palm
]

# Palette (BGR)
COLOR_VALID   = (60, 220, 60)
COLOR_INVALID = (0, 80, 220)
COLOR_HOLD    = (0, 200, 255)
COLOR_TEXT    = (240, 240, 240)
COLOR_OVERLAY = (20, 20, 20)


# ─── Model loader ─────────────────────────────────────────────────────────────

class ModelBundle:
    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.encoder = None
        self.loaded  = False

    def load(self):
        logger.info("Loading Keras model...")
        if not KERAS_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {KERAS_MODEL_PATH}")
        self.model = tf.keras.models.load_model(KERAS_MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)

        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.encoder = pickle.load(f)

        logger.info("Model loaded -> OK")
        self.loaded = True

    def predict(self, raw_landmarks: list[float]) -> dict:
        """
        Full inference pipeline: raw landmarks -> prediction dict.
        """
        x = np.array(raw_landmarks, dtype=np.float32).reshape(1, FEATURE_SIZE)

        # Feature engineering (deterministic)
        x = apply_feature_pipeline(x)

        # StandardScaler
        x = self.scaler.transform(x)

        # MLP inference
        probs   = self.model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        gesture  = GESTURE_CLASSES[top_idx]

        return {
            "gesture":    gesture if top_conf >= CONFIDENCE_THRESHOLD else "invalid",
            "raw_gesture": gesture,
            "confidence": round(top_conf, 4),
            "all_probs":  {g: round(float(p), 4) for g, p in zip(GESTURE_CLASSES, probs)},
            "valid":      top_conf >= CONFIDENCE_THRESHOLD,
        }


def extract_landmarks(hand_landmarks) -> list[float]:
    vec = []
    for lm in hand_landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return vec


def draw_landmarks_manual(frame, landmarks, colour):
    h, w = frame.shape[:2]
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]
        cv2.line(frame, (int(p1.x * w), int(p1.y * h)),
                 (int(p2.x * w), int(p2.y * h)), colour, 2)
    # Draw dots
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (240, 240, 240), -1)


# ─── HUD renderer ─────────────────────────────────────────────────────────────

def draw_hud(frame, prediction: dict, smoother_result: dict,
             action_label: str, fps: float) -> None:
    h, w = frame.shape[:2]

    gesture    = smoother_result["stable_gesture"]
    confidence = prediction.get("confidence", 0.0)
    valid      = prediction.get("valid", False)
    hold       = smoother_result.get("hold_detected", False)

    colour = COLOR_HOLD if hold else (COLOR_VALID if valid else COLOR_INVALID)

    # Dark top strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), COLOR_OVERLAY, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Gesture name
    display = gesture.replace("_", " ").upper() if gesture != "unknown" else "—"
    cv2.putText(frame, display, (14, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2)

    # Confidence bar
    bar_w = int((w - 28) * confidence)
    cv2.rectangle(frame, (14, 52), (w - 14, 66), (50, 50, 50), -1)
    cv2.rectangle(frame, (14, 52), (14 + bar_w, 66), colour, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%", (14, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1)

    # Action label
    if action_label:
        cv2.putText(frame, f"→  {action_label}", (14, 104),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HOLD, 1)

    # FPS
    if DISPLAY_FPS:
        cv2.putText(frame, f"{fps:.1f} fps", (w - 90, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

    # Window fill indicator (how full the smoother buffer is)
    fill_pct = smoother_result.get("window_fill", 0) / SMOOTHING_WINDOW_SIZE
    fill_w = int(80 * fill_pct)
    cv2.rectangle(frame, (w - 90, 36), (w - 10, 44), (50, 50, 50), -1)
    cv2.rectangle(frame, (w - 90, 36), (w - 90 + fill_w, 44), (120, 120, 200), -1)

    # Controls
    cv2.putText(frame, "Q = quit   R = reset smoother",
                (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Styled Glassmorphism Gesture & Action Card in Bottom-Left
    overlay_bl = frame.copy()
    cv2.rectangle(overlay_bl, (14, h - 100), (320, h - 35), (20, 20, 20), -1)
    cv2.addWeighted(overlay_bl, 0.75, frame, 0.25, 0, frame)

    accent_colour = COLOR_HOLD if hold else (COLOR_VALID if valid else COLOR_INVALID)
    stable_gesture = smoother_result.get("stable_gesture", "unknown")
    if stable_gesture not in ("unknown", "invalid"):
        g_disp = GESTURE_DISPLAY_NAMES.get(stable_gesture, stable_gesture).upper()
        act_disp = GESTURE_ACTION_MAP.get(stable_gesture, {}).get("label", "—").upper()
    else:
        g_disp = "—"
        act_disp = "IDLE"

    cv2.putText(frame, f"GESTURE: {g_disp}", (26, h - 77),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, accent_colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTION:  {act_disp}", (26, h - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(display: bool = True, dry_run: bool = False) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # ── Load model ──
    bundle = ModelBundle()
    try:
        # We only try to load the model if it exists; 
        # otherwise we just run MediaPipe for visualization
        if KERAS_MODEL_PATH.exists():
            bundle.load()
        else:
            logger.warning("Keras model not found. Running in visualization mode only.")
    except Exception as e:
        logger.error(f"Model load error: {e}")

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

    # ── Setup components ──
    smoother = TemporalSmoother()
    action_engine = ActionEngine(dry_run=dry_run)

    # ── Open camera ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open camera.")
        return

    logger.info(f"Camera opened. Running gesture pipeline... (display={display}, dry_run={dry_run})")

    # FPS tracking
    frame_times: list[float] = []
    fps = 0.0

    # Last action label for HUD
    last_action_label = ""
    action_label_until = 0.0

    # Blank prediction / smoother result for first frames
    blank_pred   = {"gesture": "invalid", "confidence": 0.0, "all_probs": {}, "valid": False}
    blank_smooth = {"stable_gesture": "unknown", "confidence_avg": 0.0,
                    "hold_detected": False, "hold_gesture": None, "window_fill": 0}

    while True:
        t_start = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            logger.error("Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)   # mirror for natural feel
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        prediction   = blank_pred.copy()
        smooth_result = blank_smooth.copy()
        action_event  = None

        if result.hand_landmarks:
            hl  = result.hand_landmarks[0]
            raw = extract_landmarks(hl)

            # Predict (if model loaded)
            if bundle.loaded:
                prediction = bundle.predict(raw)

            # Draw skeleton
            valid = prediction.get("valid", False)
            colour = COLOR_VALID if valid else COLOR_INVALID
            if display:
                draw_landmarks_manual(frame, hl, colour)

            # Smooth
            smooth_result = smoother.update(
                prediction["gesture"],
                prediction["confidence"],
            )

            # Tap action
            if smooth_result["stable_gesture"] not in ("unknown", "invalid"):
                action_event = action_engine.on_gesture(smooth_result["stable_gesture"])

            # Hold action
            if smooth_result["hold_detected"]:
                action_event = action_engine.on_hold(smooth_result["hold_gesture"])

            if action_event:
                last_action_label = action_event.get("label", "")
                action_label_until = time.monotonic() + 2.0
                logger.info(f"Action: {action_event['action']}  gesture={action_event['gesture']}")

        else:
            # No hand detected — push invalid into smoother
            smooth_result = smoother.update("invalid", 0.0)

        # Clear action label after timeout
        if time.monotonic() > action_label_until:
            last_action_label = ""

        # FPS
        t_end = time.monotonic()
        frame_times.append(t_end - t_start)
        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

        # Draw HUD
        if display:
            draw_hud(frame, prediction, smooth_result, last_action_label, fps)
            cv2.imshow("Gesture Control System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
            elif key == ord("r") or key == ord("R"):
                smoother.reset()
                logger.info("Smoother reset.")

    cap.release()
    detector.close()
    if display:
        cv2.destroyAllWindows()
    logger.info("Pipeline stopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time gesture recognition pipeline")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no OpenCV window)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log actions but don't execute system calls")
    args = parser.parse_args()

    run_pipeline(display=not args.no_display, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
