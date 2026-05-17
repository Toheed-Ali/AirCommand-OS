"""
config.py — Central configuration for the Gesture Control System.
All tunable parameters live here. Nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
AUGMENTED_DIR   = DATA_DIR / "augmented"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = ROOT_DIR / "models" / "saved"
MP_MODELS_DIR   = ROOT_DIR / "models" / "mediapipe"
EXPORTS_DIR     = ROOT_DIR / "exports"
LOGS_DIR        = ROOT_DIR / "logs"

# MediaPipe Task Model
MP_HAND_LANDMARKER_MODEL_PATH = MP_MODELS_DIR / "hand_landmarker.task"

for d in [RAW_DIR, AUGMENTED_DIR, PROCESSED_DIR, MODELS_DIR, MP_MODELS_DIR, EXPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Gesture Classes ───────────────────────────────────────────────────────────
GESTURE_CLASSES = [
    "fist",
    "l_gesture",
    "palm",
    "peace",
    "three_fingers",
    "thumbs_up",
]

NUM_CLASSES      = len(GESTURE_CLASSES)
GESTURE_TO_INDEX = {g: i for i, g in enumerate(GESTURE_CLASSES)}
INDEX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_CLASSES)}

# Human-readable display names (for UI / Flutter)
GESTURE_DISPLAY_NAMES = {
    "fist":          "Fist",
    "l_gesture":     "L Gesture",
    "palm":          "Palm",
    "peace":         "Peace",
    "three_fingers": "Three Fingers",
    "thumbs_up":     "Thumbs Up",
}

# ─── Action Mapping ────────────────────────────────────────────────────────────
# Each gesture maps to a primary and optional secondary action
GESTURE_ACTION_MAP = {
    "thumbs_up":     {"action": "volume_up",        "label": "Volume Up"},
    "fist":          {"action": "play_pause",        "label": "Play / Pause"},
    "l_gesture":     {"action": "open_app",          "label": "Open Chrome"},
    "palm":          {"action": "shift_tab",         "label": "Shift + Tab"},
    "peace":         {"action": "brightness_up",     "label": "Brightness Up"},
    "three_fingers": {"action": "brightness_down",   "label": "Brightness Down"},
}

# No hold actions enabled
HOLD_ACTION_MAP = {}

# ─── MediaPipe ────────────────────────────────────────────────────────────────
NUM_LANDMARKS    = 21
COORDS_PER_LM    = 3          # x, y, z
FEATURE_SIZE     = NUM_LANDMARKS * COORDS_PER_LM  # 63

MP_MAX_HANDS          = 1
MP_MIN_DETECTION_CONF = 0.7
MP_MIN_TRACKING_CONF  = 0.6

# ─── Dataset ──────────────────────────────────────────────────────────────────
RAW_TRAIN_CSV_PATH    = RAW_DIR / "train_raw.csv"
RAW_TEST_CSV_PATH     = RAW_DIR / "test_raw.csv"
AUGMENTED_TRAIN_CSV_PATH = AUGMENTED_DIR / "train_augmented.csv"
CSV_PATH              = RAW_DIR / "custom_recorded.csv"
SAMPLES_PER_CLASS     = 800      # target per class before augmentation
VALIDATION_SPLIT      = 0.15     # validation split ratio from training set
RANDOM_SEED           = 42

# ─── Preprocessing ────────────────────────────────────────────────────────────
NORMALIZE_BY_WRIST    = True     # translate so wrist (lm 0) is origin
SCALE_BY_HAND_SIZE    = True     # divide by max distance from wrist to tip
CLIP_Z_DEPTH          = True     # z coords are noisy; we clip to [-0.15, 0.15]
Z_CLIP_RANGE          = (-0.15, 0.15)

# ─── MLP Model Architecture ───────────────────────────────────────────────────
MLP_HIDDEN_LAYERS     = [256, 128, 64]   # neurons per hidden layer
MLP_DROPOUT_RATE      = 0.3
MLP_ACTIVATION        = "relu"
MLP_OUTPUT_ACTIVATION = "softmax"

# ─── Training ─────────────────────────────────────────────────────────────────
LEARNING_RATE         = 1e-3
BATCH_SIZE            = 64
EPOCHS                = 150
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE    = 8
LR_REDUCE_FACTOR      = 0.5
MIN_LR                = 1e-6

# Saved artefacts
KERAS_MODEL_PATH      = MODELS_DIR / "gesture_mlp.keras"
TFLITE_MODEL_PATH     = EXPORTS_DIR / "gesture_model.tflite"
SCALER_PATH           = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.pkl"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"

# ─── Real-Time Inference ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.97    # below this -> "invalid gesture"
CAMERA_INDEX          = 0
FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720
DISPLAY_FPS           = True

# ─── Temporal Smoothing ───────────────────────────────────────────────────────
SMOOTHING_WINDOW_SIZE = 12       # frames kept in history
MAJORITY_VOTE_MIN     = 9        # minimum consistent votes to accept prediction

# ─── Action Engine ────────────────────────────────────────────────────────────
ACTION_COOLDOWN_SEC   = 1.5      # seconds between repeated action triggers
HOLD_TRIGGER_SEC      = 2.0      # seconds gesture must be held to fire hold-action
POWER_OFF_HOLD_SEC    = 3.0      # extra safety for power-off (must hold longer)

# ─── Communication (Python ↔ Flutter) ─────────────────────────────────────────
WEBSOCKET_HOST        = "0.0.0.0"
WEBSOCKET_PORT        = 8765
HTTP_HOST             = "0.0.0.0"
HTTP_PORT             = 8766
MAX_WS_CONNECTIONS    = 5

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL             = "INFO"
LOG_FILE              = LOGS_DIR / "gesture_system.log"
