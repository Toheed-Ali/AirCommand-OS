"""
action_engine.py — Maps stable gesture predictions to mobile control actions.

On the Python side this triggers:
  - Volume up/down (via system calls or pycaw on Windows / amixer on Linux)
  - Media play/pause
  - Brightness up/down
  - Application launch
  - System power off

On the Flutter side, the Flutter app receives the action name via WebSocket
and executes the actual mobile-level system call (volume, brightness, etc.)
through platform channels.

The action engine here handles:
  - Cooldown enforcement (prevent rapid-fire repeats)
  - Hold-action dispatch
  - Broadcasting to WebSocket clients
"""

import time
import logging
import subprocess
import platform
import asyncio
from typing import Optional, Callable

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    ACTION_COOLDOWN_SEC, GESTURE_ACTION_MAP, HOLD_ACTION_MAP,
    GESTURE_DISPLAY_NAMES,
)

logger = logging.getLogger("action_engine")


# ─── Action implementations (Python side for testing without Flutter) ─────────

class SystemActions:
    """
    Executes system-level actions locally (for desktop testing).
    In production, these trigger via WebSocket → Flutter platform channel.
    """

    OS = platform.system()   # "Windows", "Linux", "Darwin"

    @staticmethod
    def volume_up():
        if SystemActions.OS == "Linux":
            subprocess.run(["amixer", "-q", "sset", "Master", "5%+"], check=False)
        elif SystemActions.OS == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 5)"], check=False)
        elif SystemActions.OS == "Windows":
            try:
                # Use native Windows key events. Windows volume step is 2 points.
                # To get 10 points, we fire 5 times.
                import ctypes
                VK_VOLUME_UP = 0xAF
                for _ in range(5):
                    ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 2, 0) # Key up
            except Exception as e:
                logger.warning(f"Windows volume error: {e}")
        logger.info("ACTION: volume_up")

    @staticmethod
    def volume_down():
        if SystemActions.OS == "Linux":
            subprocess.run(["amixer", "-q", "sset", "Master", "5%-"], check=False)
        elif SystemActions.OS == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 5)"], check=False)
        elif SystemActions.OS == "Windows":
            try:
                import ctypes
                VK_VOLUME_DOWN = 0xAE
                for _ in range(5):
                    ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 2, 0)
            except Exception as e:
                logger.warning(f"Windows volume error: {e}")
        logger.info("ACTION: volume_down")

    @staticmethod
    def play_pause():
        if SystemActions.OS == "Linux":
            subprocess.run(["playerctl", "play-pause"], check=False)
        elif SystemActions.OS == "Darwin":
            subprocess.run(["osascript", "-e", 'tell application "System Events" to key code 16'], check=False)
        elif SystemActions.OS == "Windows":
            try:
                import ctypes
                VK_MEDIA_PLAY_PAUSE = 0xB3
                ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
                ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
            except Exception as e:
                logger.warning(f"Windows play_pause error: {e}")
        logger.info("ACTION: play_pause")

    @staticmethod
    def brightness_up():
        if SystemActions.OS == "Linux":
            subprocess.run(["brightnessctl", "set", "+10%"], check=False)
        elif SystemActions.OS == "Windows":
            try:
                import screen_brightness_control as sbc
                # Try setting without specifying display first
                current = sbc.get_brightness()
                if current:
                    val = current[0] if isinstance(current, list) else current
                    sbc.set_brightness(min(100, int(val) + 10))
                else:
                    # Fallback for some systems
                    sbc.fade_brightness(min(100, 50), increment=10)
            except Exception as e:
                logger.warning(f"Windows brightness error: {e}")
        logger.info("ACTION: brightness_up")

    @staticmethod
    def brightness_down():
        if SystemActions.OS == "Linux":
            subprocess.run(["brightnessctl", "set", "10%-"], check=False)
        elif SystemActions.OS == "Windows":
            try:
                import screen_brightness_control as sbc
                current = sbc.get_brightness()
                if current:
                    val = current[0] if isinstance(current, list) else current
                    sbc.set_brightness(max(0, int(val) - 10))
            except Exception as e:
                logger.warning(f"Windows brightness error: {e}")
        logger.info("ACTION: brightness_down")

    @staticmethod
    def open_app():
        # Configurable: launch a specific application
        target_app = "chrome"  # override from settings
        if SystemActions.OS == "Linux":
            subprocess.Popen([target_app])
        elif SystemActions.OS == "Darwin":
            subprocess.Popen(["open", "-a", target_app])
        elif SystemActions.OS == "Windows":
            subprocess.Popen(["start", target_app], shell=True)
        logger.info(f"ACTION: open_app ({target_app})")

    @staticmethod
    def shift_tab():
        logger.info("ACTION: shift_tab")
        if SystemActions.OS == "Windows":
            try:
                import ctypes
                import time
                VK_LSHIFT = 0xA0
                VK_TAB    = 0x09
                # Press Shift
                ctypes.windll.user32.keybd_event(VK_LSHIFT, 0, 0, 0)
                time.sleep(0.05)
                # Press Tab
                ctypes.windll.user32.keybd_event(VK_TAB, 0, 0, 0)
                time.sleep(0.05)
                # Release Tab
                ctypes.windll.user32.keybd_event(VK_TAB, 0, 2, 0)
                time.sleep(0.05)
                # Release Shift
                ctypes.windll.user32.keybd_event(VK_LSHIFT, 0, 2, 0)
            except Exception as e:
                logger.warning(f"Windows shift_tab error: {e}")

    @staticmethod
    def power_off():
        # Keeping this for internal use if needed, but removed from gesture map
        logger.warning("ACTION: power_off requested — executing with 3s delay")
        time.sleep(3.0)
        if SystemActions.OS == "Linux":
            subprocess.run(["shutdown", "-h", "now"], check=False)
        elif SystemActions.OS == "Darwin":
            subprocess.run(["shutdown", "-h", "now"], check=False)
        elif SystemActions.OS == "Windows":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=False)


ACTION_HANDLERS: dict[str, Callable] = {
    "volume_up":      SystemActions.volume_up,
    "volume_down":    SystemActions.volume_down,
    "play_pause":     SystemActions.play_pause,
    "brightness_up":  SystemActions.brightness_up,
    "brightness_down": SystemActions.brightness_down,
    "open_app":       SystemActions.open_app,
    "shift_tab":      SystemActions.shift_tab,
    "power_off":      SystemActions.power_off,
}


# ─── Action Engine ────────────────────────────────────────────────────────────

class ActionEngine:
    """
    Central dispatcher for gesture → action mapping.

    Features:
      - Per-action cooldown timers (prevent rapid repeats)
      - Hold-action support (secondary action on sustained gesture)
      - Optional broadcast callback (send event to WebSocket clients)
      - Dry-run mode (log only, no system calls)
    """

    def __init__(
        self,
        broadcast_cb: Optional[Callable] = None,
        dry_run: bool = False,
    ):
        self.broadcast_cb = broadcast_cb
        self.dry_run = dry_run
        self._last_fired: dict[str, float] = {}   # action → timestamp

    def _can_fire(self, action: str) -> bool:
        now = time.monotonic()
        last = self._last_fired.get(action, 0.0)
        return (now - last) >= ACTION_COOLDOWN_SEC

    def _fire(self, action: str, gesture: str, triggered_by: str = "tap") -> dict | None:
        if not self._can_fire(action):
            return None

        self._last_fired[action] = time.monotonic()
        display_name = GESTURE_DISPLAY_NAMES.get(gesture, gesture)
        label = GESTURE_ACTION_MAP.get(gesture, {}).get("label", action)

        event = {
            "type":         "action_triggered",
            "action":       action,
            "gesture":      gesture,
            "gesture_name": display_name,
            "label":        label,
            "triggered_by": triggered_by,
            "timestamp":    time.time(),
        }

        logger.info(f"ACTION FIRED: {action}  (gesture={gesture}, by={triggered_by})")

        if self.dry_run:
            logger.debug(f"[DRY RUN] Would execute: {action}")
        else:
            handler = ACTION_HANDLERS.get(action)
            if handler:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Action handler failed for '{action}': {e}")
                    event["error"] = str(e)

        if self.broadcast_cb:
            self.broadcast_cb(event)

        return event

    def on_gesture(self, stable_gesture: str) -> dict | None:
        """
        Call this when a stable gesture is detected (tap trigger).

        Returns the fired action event or None (cooldown / unknown).
        """
        if not stable_gesture or stable_gesture in ("unknown", "invalid"):
            return None

        mapping = GESTURE_ACTION_MAP.get(stable_gesture)
        if not mapping:
            return None

        return self._fire(mapping["action"], stable_gesture, triggered_by="tap")

    def on_hold(self, hold_gesture: str) -> dict | None:
        """
        Call this when a hold is detected.

        If the gesture has a hold-action mapping, fire it.
        Otherwise fire the regular action again.
        """
        if not hold_gesture:
            return None

        hold_mapping = HOLD_ACTION_MAP.get(hold_gesture)
        if hold_mapping:
            return self._fire(hold_mapping["action"], hold_gesture, triggered_by="hold")

        # Fallback: repeat primary action
        return self.on_gesture(hold_gesture)

    def get_status(self) -> dict:
        now = time.monotonic()
        return {
            "cooldowns": {
                action: max(0.0, ACTION_COOLDOWN_SEC - (now - ts))
                for action, ts in self._last_fired.items()
            }
        }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = ActionEngine(dry_run=True)

    print("Testing tap actions...")
    for gesture in ["thumbs_up", "fist", "peace", "three_fingers", "l_gesture"]:
        result = engine.on_gesture(gesture)
        print(f"  {gesture:<15} → {result['action'] if result else 'COOLDOWN'}")
        time.sleep(0.1)

    print("\nTesting hold action (thumbs_up hold → volume_down)...")
    result = engine.on_hold("thumbs_up")
    print(f"  hold thumbs_up → {result['action'] if result else 'None'}")
