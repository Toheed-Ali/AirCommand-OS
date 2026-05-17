# AirCommand-OS — Hand Gesture Control System

> Real-time hand gesture recognition pipeline that translates webcam-detected hand poses into OS-level actions (volume, brightness, media, app launch). Exports a TFLite model for Flutter mobile integration via WebSocket.

---

## How It Works

```
Webcam → MediaPipe Hand Landmarker → 63 raw coordinates
       → Wrist normalisation + hand-size scaling + Z-clipping
       → StandardScaler → MLP (256→128→64→6) → Softmax
       → Confidence gate (0.97) → Temporal smoother (12-frame majority vote)
       → Action Engine → OS system call  +  WebSocket → Flutter
```

---

## Gesture → Action Mapping

| Gesture | Action | Trigger |
|---|---|---|
| 👍 Thumbs Up | Volume Up | Tap |
| ✊ Fist | Play / Pause Media | Tap |
| 🤟 L Gesture | Open Chrome | Tap |
| 🖐 Palm | Shift + Tab | Tap |
| ✌️ Peace | Brightness Up | Tap |
| 🤟 Three Fingers | Brightness Down | Tap |

> Hold actions can be added in `config/config.py → HOLD_ACTION_MAP`.

---

## Project Structure

```
AirCommand-OS/
│
├── config/
│   └── config.py              ← Single source of truth for ALL parameters
│
├── src/
│   ├── collection/
│   │   ├── collect_data.py    ← Interactive webcam landmark collector
│   │   ├── augment_data.py    ← Mirror flip + noise + scale jitter (~5× data)
│   │   └── process_dataset.py
│   │
│   ├── preprocessing/
│   │   └── preprocess.py      ← Wrist-norm → hand-scale → z-clip → scaler → split
│   │
│   ├── model/
│   │   └── model.py           ← MLP architecture (Dense + BN + Dropout)
│   │
│   └── realtime/
│       ├── gesture_pipeline.py   ← Main loop: camera → predict → act
│       ├── temporal_smoother.py  ← Sliding-window majority vote + hold detection
│       ├── action_engine.py      ← Gesture → OS action dispatcher (cross-platform)
│       └── websocket_server.py   ← Async WebSocket bridge to Flutter clients
│
├── training/
│   ├── train.py               ← Train MLP with EarlyStopping + ReduceLR
│   ├── evaluate.py            ← Confusion matrix, curves, F1 charts
│   └── export_tflite.py       ← Convert to .tflite (float16 or int8) for Flutter
│
├── data/                      ← Raw / augmented / processed CSVs + .npy splits
├── models/                    ← gesture_mlp.keras + scaler.pkl + label_encoder.pkl
├── exports/                   ← gesture_model.tflite + model_spec.json + scaler_params.json
├── logs/
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MediaPipe Hand Landmarker model
#    Place hand_landmarker.task inside:  models/mediapipe/
#    Download: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

# 4. (Optional) Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## End-to-End Workflow

### Phase 1 — Collect Training Data

```bash
# Collect 800 samples per gesture (SPACE to capture, Q to save)
python src/collection/collect_data.py --gesture fist          --samples 800
python src/collection/collect_data.py --gesture l_gesture     --samples 800 --append
python src/collection/collect_data.py --gesture palm          --samples 800 --append
python src/collection/collect_data.py --gesture peace         --samples 800 --append
python src/collection/collect_data.py --gesture three_fingers --samples 800 --append
python src/collection/collect_data.py --gesture thumbs_up     --samples 800 --append
```

| Key | Action |
|-----|--------|
| `SPACE` | Capture current frame |
| `R` | Undo last capture |
| `P` | Pause / resume preview |
| `Q` | Save and quit |

### Phase 2 — Augment Dataset

```bash
# Adds mirror flip, Gaussian noise, scale jitter → ~5× more samples + class balancing
python src/collection/augment_data.py
```

### Phase 3 — Preprocess

```bash
# Wrist-origin normalisation → hand-size scaling → z-clipping → StandardScaler → stratified split
python src/preprocessing/preprocess.py
# Outputs: data/processed/{X,y}_{train,val,test}.npy + scaler.pkl + label_encoder.pkl
```

### Phase 4 — Train

```bash
python training/train.py
# Or override defaults:
python training/train.py --epochs 200 --batch-size 32

# Expected: val_accuracy ≈ 0.97+
```

Callbacks active during training:
- **ModelCheckpoint** — saves best `val_accuracy` model
- **EarlyStopping** — patience 15 epochs
- **ReduceLROnPlateau** — halves LR when val_loss plateaus (patience 8)

### Phase 5 — Evaluate

```bash
python training/evaluate.py
# Generates: confusion matrix, training curves, per-class F1 bar chart
# Saved to: models/saved/plots/
```

### Phase 6 — Export to TFLite

```bash
# Float16 quantisation (~50% size reduction, negligible accuracy drop)
python training/export_tflite.py

# Full INT8 quantisation (maximum compression, needs calibration data)
python training/export_tflite.py --int8

# Output files:
#   exports/gesture_model.tflite
#   exports/model_spec.json       ← input/output shapes, gesture labels
#   exports/scaler_params.json    ← StandardScaler mean/scale as JSON for Flutter
```

### Phase 7 — Run Real-Time Pipeline

```bash
# Standard (with display window)
python src/realtime/gesture_pipeline.py

# Headless / server mode (no OpenCV window)
python src/realtime/gesture_pipeline.py --no-display

# Dry run (logs actions but doesn't execute system calls)
python src/realtime/gesture_pipeline.py --dry-run
```

HUD controls during runtime:
- `Q` — quit
- `R` — reset temporal smoother

---

## Model Architecture

```
Input (63 features)
  → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(64)  + BatchNorm + ReLU + Dropout(0.3)
  → Dense(6, Softmax)
```

- **Optimizer**: Adam (lr=1e-3)
- **Loss**: Sparse Categorical Crossentropy
- **Regularisation**: L2 (1e-4) on all Dense kernels + Dropout
- **Invalid gesture**: handled by confidence threshold (not an extra class)

---

## Key Design Decisions

### Feature Engineering
Raw MediaPipe coordinates are image-space floats (0–1). Three transforms make features position-, distance-, and user-invariant:

1. **Wrist-origin normalisation** — translates all 21 landmarks so wrist (lm 0) is at `(0, 0, 0)`
2. **Hand-size scaling** — divides by the Euclidean distance from wrist to middle-finger tip (lm 12)
3. **Z-depth clipping** — clips z to `[-0.15, 0.15]` (MediaPipe z is noisy relative to xy)

### Left-Hand Support
Only right-hand data is collected. Augmentation applies a horizontal mirror (`x_new = 1 − x_old`) so the model learns both handedness without a second collection session.

### Temporal Smoothing
Raw per-frame predictions are noisy. A 12-frame sliding window accepts a gesture only when it appears ≥ 9 times — requiring ~0.4 s of stable hold at 30 fps before any action fires.

### Hold Actions
The `TemporalSmoother` tracks uninterrupted gesture duration. Secondary actions fire after `HOLD_TRIGGER_SEC` (2 s default); `palm` requires `POWER_OFF_HOLD_SEC` (3 s) as a safety margin.

### Confidence Gate
Any frame where the top-class probability is below `CONFIDENCE_THRESHOLD` (0.97) is classified as `"invalid"` — no action is triggered and the smoother receives a negative vote.

---

## Configuration Reference

All parameters live in `config/config.py` — nothing is hardcoded elsewhere.

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.97` | Min probability to accept a prediction |
| `SMOOTHING_WINDOW_SIZE` | `12` | Frames in sliding window |
| `MAJORITY_VOTE_MIN` | `9` | Votes needed to confirm a gesture |
| `ACTION_COOLDOWN_SEC` | `1.5` | Minimum seconds between repeated triggers |
| `HOLD_TRIGGER_SEC` | `2.0` | Seconds held to fire secondary action |
| `MLP_HIDDEN_LAYERS` | `[256, 128, 64]` | Neurons per hidden layer |
| `MLP_DROPOUT_RATE` | `0.3` | Dropout regularisation |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `EPOCHS` | `150` | Max training epochs |
| `BATCH_SIZE` | `64` | Training batch size |
| `SAMPLES_PER_CLASS` | `800` | Target raw samples per gesture |
| `WEBSOCKET_PORT` | `8765` | Python ↔ Flutter bridge port |

---

## Flutter Integration

### Asset Files
After export, copy these three files to your Flutter project's `assets/` folder:

```
gesture_model.tflite   ← quantised MLP model
model_spec.json        ← input/output tensor shapes + gesture labels
scaler_params.json     ← StandardScaler mean + scale arrays
```

### Preprocessing in Dart (before inference)
Apply the same pipeline the Python side uses:
1. Wrist-origin normalisation
2. Hand-size scaling (wrist → middle-finger-tip distance)
3. Z-depth clipping to `[-0.15, 0.15]`
4. StandardScaler: `(x − mean) / scale` using values from `scaler_params.json`

### WebSocket Protocol
The Python server binds on `ws://0.0.0.0:8765`. Connect from Flutter and receive JSON events:

**Server → Flutter:**
```json
{ "type": "action_triggered", "action": "volume_up", "gesture": "thumbs_up",
  "label": "Volume Up", "confidence": 0.98, "timestamp": 1720000000.0 }

{ "type": "gesture_update", "gesture": "thumbs_up", "confidence": 0.98,
  "valid": true, "all_probs": { "thumbs_up": 0.98, "fist": 0.01, ... } }

{ "type": "system_status", "fps": 28.4, "clients_connected": 1, "model_loaded": true }
```

**Flutter → Server:**
```json
{ "type": "ping" }
{ "type": "update_settings", "confidence_threshold": 0.90 }
{ "type": "get_status" }
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow >= 2.15` | MLP training + TFLite export |
| `mediapipe >= 0.10.9` | Hand landmark detection (Tasks API) |
| `opencv-python >= 4.9` | Webcam capture + HUD rendering |
| `scikit-learn >= 1.4` | StandardScaler, train/test split, metrics |
| `numpy`, `pandas` | Data handling |
| `websockets >= 12.0` | Async Python ↔ Flutter bridge |
| `pycaw`, `screen-brightness-control` | Windows volume / brightness control |
| `matplotlib`, `seaborn` | Evaluation plots |

---

## Platform Support

| Feature | Windows | Linux | macOS |
|---|---|---|---|
| Volume control | ✅ `ctypes` VK keys | ✅ `amixer` | ✅ `osascript` |
| Brightness control | ✅ `screen-brightness-control` | ✅ `brightnessctl` | ⚠️ Not implemented |
| Media play/pause | ✅ VK_MEDIA_PLAY_PAUSE | ✅ `playerctl` | ✅ `osascript` |
| App launch | ✅ | ✅ | ✅ |

---

## License

MIT — see `LICENSE` for details.
