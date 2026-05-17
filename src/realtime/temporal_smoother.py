"""
temporal_smoother.py — Stabilises real-time gesture predictions.

Problem: Raw frame-by-frame predictions fluctuate between gesture classes
even when the hand is held steady, due to:
  - Slight hand tremor
  - MediaPipe landmark jitter
  - Ambiguous frames during transitions

Solution: Sliding window majority vote.
  - Keep last N predictions in a deque
  - A gesture is accepted only if it appears ≥ M times in the window
  - Otherwise output "unknown" (wait for stability)

Additional feature: Hold detection
  - If the same gesture is consistently predicted for HOLD_TRIGGER_SEC seconds,
    trigger the secondary (hold) action (e.g., volume down for thumbs-up hold)
"""

import time
from collections import deque, Counter

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    SMOOTHING_WINDOW_SIZE, MAJORITY_VOTE_MIN,
    HOLD_TRIGGER_SEC, POWER_OFF_HOLD_SEC,
    GESTURE_CLASSES, HOLD_ACTION_MAP,
)


class TemporalSmoother:
    """
    Sliding-window gesture stabiliser.

    Usage:
        smoother = TemporalSmoother()
        for frame in video_stream:
            raw = model.predict(frame)   # e.g. "thumbs_up" or "invalid"
            result = smoother.update(raw, confidence)
            if result["stable_gesture"] != "unknown":
                trigger_action(result["stable_gesture"])
    """

    def __init__(
        self,
        window_size: int = SMOOTHING_WINDOW_SIZE,
        min_votes: int = MAJORITY_VOTE_MIN,
        hold_trigger_sec: float = HOLD_TRIGGER_SEC,
    ):
        self.window_size     = window_size
        self.min_votes       = min_votes
        self.hold_trigger_sec = hold_trigger_sec

        self._history: deque[str]    = deque(maxlen=window_size)
        self._conf_history: deque[float] = deque(maxlen=window_size)

        self._hold_start: float | None = None
        self._hold_gesture: str | None = None
        self._hold_fired: bool = False

        self._last_stable: str | None = None
        self._stable_since: float | None = None

    def update(self, raw_gesture: str, confidence: float) -> dict:
        """
        Feed a new raw prediction.

        Args:
            raw_gesture : predicted class ("invalid" if below threshold)
            confidence  : model confidence for the raw prediction

        Returns dict:
            stable_gesture  : smoothed gesture or "unknown"
            confidence_avg  : mean confidence over window
            hold_detected   : True if hold threshold crossed
            hold_gesture    : gesture being held (if hold_detected)
            window_counts   : Counter of recent predictions (debug)
        """
        self._history.append(raw_gesture)
        self._conf_history.append(confidence)

        counts = Counter(self._history)
        top_gesture, top_count = counts.most_common(1)[0]
        conf_avg = sum(self._conf_history) / len(self._conf_history)

        # Stability check
        if (top_gesture != "invalid"
                and top_count >= self.min_votes
                and top_gesture in GESTURE_CLASSES):
            stable = top_gesture
        else:
            stable = "unknown"

        # ── Hold detection ──
        now = time.monotonic()
        hold_detected = False
        hold_gesture_name = None

        if stable != "unknown" and stable == self._hold_gesture:
            if self._hold_start is not None:
                elapsed = now - self._hold_start
                threshold = (POWER_OFF_HOLD_SEC
                             if stable == "palm"
                             else self.hold_trigger_sec)
                if elapsed >= threshold and not self._hold_fired:
                    hold_detected = True
                    hold_gesture_name = stable
                    self._hold_fired = True
        else:
            # Gesture changed → reset hold state
            self._hold_gesture = stable if stable != "unknown" else None
            self._hold_start   = now if stable != "unknown" else None
            self._hold_fired   = False

        return {
            "stable_gesture":  stable,
            "confidence_avg":  round(conf_avg, 4),
            "hold_detected":   hold_detected,
            "hold_gesture":    hold_gesture_name,
            "window_counts":   dict(counts),
            "window_fill":     len(self._history),
        }

    def reset(self) -> None:
        """Clear all history (call on scene change or camera reconnect)."""
        self._history.clear()
        self._conf_history.clear()
        self._hold_start   = None
        self._hold_gesture = None
        self._hold_fired   = False
        self._last_stable  = None
        self._stable_since = None

    def get_window_fill_pct(self) -> float:
        """How full is the window? 0.0→1.0. Low = not enough data yet."""
        return len(self._history) / self.window_size


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    smoother = TemporalSmoother(window_size=10, min_votes=7)

    print("Simulating 15 frames of 'thumbs_up'...")
    for _ in range(15):
        r = smoother.update("thumbs_up", 0.95)
        print(f"  stable={r['stable_gesture']:<14} hold={r['hold_detected']} counts={r['window_counts']}")
        time.sleep(0.1)

    print("\nSimulating 5 noisy frames...")
    for g in ["thumbs_up", "invalid", "thumbs_up", "peace", "thumbs_up"]:
        r = smoother.update(g, 0.70)
        print(f"  raw={g:<14} → stable={r['stable_gesture']}")
