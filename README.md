# Gesture Control System — ML Pipeline

Real-time hand gesture recognition system using MediaPipe + MLP neural network,
designed to control mobile device functions via webcam-detected gestures.

---

## Gesture → Action Mapping

| Gesture        | Action               | Trigger |
|----------------|----------------------|---------|
| Thumbs Up      | Volume Up            | Tap     |
| Thumbs Up (hold) | Volume Down        | Hold 2s |
| Fist           | Play / Pause Media   | Tap     |
| L Gesture      | Open App             | Tap     |
| Palm           | Power Off            | Hold 3s |
| Peace          | Brightness Up        | Tap     |
| Three Fingers  | Brightness Down      | Tap     |

---

## Project Structure

```
gesture_control/
│
├── config/
│   └── config.py              ← Single source of truth for all parameters
│
├── data/
│   ├── raw/                   ← Raw collected CSVs (before augmentation)
│   ├── augmented/             ← After mirror + noise + scale augmentation
│   └── processed/             ← Train/val/test .npy splits + scaler/encoder
│
├── src/
│   ├── collection/
│   │   ├── collect_data.py    ← Interactive webcam data collector
│   │   └── augment_data.py    ← Augmentation: mirror + noise + balancing
│   │
│   ├── preprocessing/
│   │   └── preprocess.py      ← Wrist-norm → scale → scaler → split
│   │
│   ├── model/
│   │   └── model.py           ← MLP architecture + predict helper
│   │
│   └── realtime/
│       ├── temporal_smoother.py  ← Sliding-window majority vote
│       ├── action_engine.py      ← Gesture → OS action dispatcher
│       ├── websocket_server.py   ← Python ↔ Flutter WebSocket bridge
│       └── gesture_pipeline.py  ← Main real-time recognition loop
│
├── training/
│   ├── train.py               ← Full training + evaluation script
│   ├── evaluate.py            ← Plots: confusion matrix, curves, metrics
│   └── export_tflite.py       ← Convert to .tflite for Flutter
│
├── models/
│   └── saved/                 ← gesture_mlp.keras + scaler.pkl + encoder.pkl
│
├── exports/                   ← gesture_model.tflite + model_spec.json
├── logs/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU (optional but faster training)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Step-by-Step Workflow

### Phase 1 — Collect data

Collect 800 samples per gesture class. Run once per gesture:

```bash
# Collect each gesture (SPACE to capture, Q to save)
python src/collection/collect_data.py --gesture fist          --samples 800
python src/collection/collect_data.py --gesture l_gesture     --samples 800 --append
python src/collection/collect_data.py --gesture palm          --samples 800 --append
python src/collection/collect_data.py --gesture peace         --samples 800 --append
python src/collection/collect_data.py --gesture three_fingers --samples 800 --append
python src/collection/collect_data.py --gesture thumbs_up     --samples 800 --append
```

Controls during collection:
- `SPACE` — capture current frame
- `R` — undo last capture
- `P` — pause/resume
- `Q` — save and quit

### Phase 2 — Augment dataset

```bash
# Adds mirror flip, noise, scale jitter → ~5× more samples
# Also balances class counts
python src/collection/augment_data.py
```

### Phase 3 — Preprocess

```bash
# Wrist-origin normalisation → hand-size scaling → StandardScaler → train/val/test split
python src/preprocessing/preprocess.py
```

### Phase 4 — Train

```bash
python training/train.py
# Or with custom params:
python training/train.py --epochs 200 --batch-size 32
```

Expected output on a good dataset: **val_accuracy ≈ 0.97+**

### Phase 5 — Evaluate

```bash
python training/evaluate.py
# Generates confusion matrix, training curves, per-class F1 bar chart
# Saved to models/saved/plots/
```

### Phase 6 — Export to TFLite (for Flutter)

```bash
python training/export_tflite.py
# Output: exports/gesture_model.tflite
#         exports/model_spec.json
#         exports/scaler_params.json
```

### Phase 7 — Run real-time pipeline

```bash
# With display window
python src/realtime/gesture_pipeline.py

# Headless (for server / no monitor)
python src/realtime/gesture_pipeline.py --no-display

# Dry run (no actual system actions)
python src/realtime/gesture_pipeline.py --dry-run
```

---

## Key Design Decisions

### Invalid gesture protection
Confidence threshold = 0.85. Any frame where the top class probability
falls below this is classified as "invalid" — no action is triggered.
This prevents spurious commands from ambiguous or transitional hand poses.

### Left-hand support
Dataset collected right-hand only. Augmentation applies horizontal mirror
(x_new = 1 - x_old) to every sample. The model learns both orientations
without requiring a separate left-hand collection session.

### Temporal smoothing
Raw per-frame predictions are inherently noisy. A sliding window of 10 frames
with a majority-vote threshold of 7 means a gesture must be held steadily for
~0.33 seconds (at 30 fps) before it fires. This eliminates flicker.

### Hold actions
The temporal smoother tracks how long a stable gesture has been held.
- Regular tap → primary action fires immediately on reaching vote threshold
- Hold 2s+ → secondary action fires (e.g., thumbs-up hold → volume down)
- Palm hold 3s+ → power off (extra safety margin)

### Feature engineering (wrist normalisation)
Raw MediaPipe coordinates are in image-space (0–1 relative to frame).
We translate so wrist (landmark 0) is at origin, then scale by the
wrist→middle-finger-tip distance. This makes features invariant to:
  - Where in frame the hand is placed
  - How close/far the hand is from camera
  - Hand physical size differences between users

---

## Configuration

All parameters in `config/config.py`. Key tunable values:

| Parameter                | Default | Description |
|--------------------------|---------|-------------|
| `CONFIDENCE_THRESHOLD`   | 0.85    | Min confidence to accept a prediction |
| `SMOOTHING_WINDOW_SIZE`  | 10      | Frames in sliding window |
| `MAJORITY_VOTE_MIN`      | 7       | Votes needed to confirm gesture |
| `ACTION_COOLDOWN_SEC`    | 1.5     | Seconds between repeated action triggers |
| `HOLD_TRIGGER_SEC`       | 2.0     | Seconds to activate hold action |
| `MLP_HIDDEN_LAYERS`      | [256, 128, 64] | Neurons per hidden layer |
| `MLP_DROPOUT_RATE`       | 0.3     | Dropout regularisation |
| `EPOCHS`                 | 150     | Max training epochs |
| `SAMPLES_PER_CLASS`      | 800     | Target samples per gesture |

---

## Flutter Integration

After exporting, copy these to your Flutter project's `assets/` folder:
- `gesture_model.tflite`
- `model_spec.json`
- `scaler_params.json`

The Flutter app connects to the Python WebSocket server at
`ws://localhost:8765` and receives JSON action events:

```json
{
  "type": "action_triggered",
  "action": "volume_up",
  "gesture": "thumbs_up",
  "label": "Volume Up",
  "confidence": 0.97,
  "timestamp": 1720000000.0
}
```
