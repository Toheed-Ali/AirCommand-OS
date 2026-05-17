"""
websocket_server.py — Async WebSocket server that streams gesture events
to connected Flutter clients in real-time.

Protocol (JSON messages):
  Server → Client:
    {type: "gesture_update",  gesture, confidence, valid, all_probs}
    {type: "action_triggered", action, gesture, label, triggered_by, timestamp}
    {type: "system_status",   fps, clients_connected, model_loaded}

  Client → Server:
    {type: "update_settings", confidence_threshold, action_map}
    {type: "ping"}

Usage (standalone):
    python src/realtime/websocket_server.py
"""

import asyncio
import json
import logging
import time
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.config import (
    WEBSOCKET_HOST, WEBSOCKET_PORT, MAX_WS_CONNECTIONS,
    GESTURE_CLASSES, GESTURE_DISPLAY_NAMES, GESTURE_ACTION_MAP,
)

logger = logging.getLogger("websocket_server")

# Global client registry
CONNECTED_CLIENTS: set[WebSocketServerProtocol] = set()

# Runtime settings (can be overridden by Flutter client)
RUNTIME_SETTINGS: dict = {
    "confidence_threshold": 0.85,
    "action_map": GESTURE_ACTION_MAP,
}


async def broadcast(message: dict) -> None:
    """Send a JSON message to all connected Flutter clients."""
    if not CONNECTED_CLIENTS:
        return
    payload = json.dumps(message)
    disconnected = set()
    for ws in CONNECTED_CLIENTS:
        try:
            await ws.send(payload)
        except websockets.exceptions.ConnectionClosed:
            disconnected.add(ws)
    CONNECTED_CLIENTS.difference_update(disconnected)


def make_broadcast_callback():
    """Returns a synchronous callback usable from non-async code."""
    loop = asyncio.get_event_loop()

    def cb(event: dict) -> None:
        asyncio.run_coroutine_threadsafe(broadcast(event), loop)

    return cb


async def handle_client(websocket: WebSocketServerProtocol) -> None:
    """Handles an individual Flutter client connection."""
    if len(CONNECTED_CLIENTS) >= MAX_WS_CONNECTIONS:
        await websocket.close(code=1008, reason="Server at capacity")
        logger.warning(f"Rejected client — capacity ({MAX_WS_CONNECTIONS}) reached")
        return

    CONNECTED_CLIENTS.add(websocket)
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}  [{len(CONNECTED_CLIENTS)} active]")

    # Send welcome / initial state
    await websocket.send(json.dumps({
        "type": "connected",
        "message": "Gesture Control Server ready",
        "gesture_classes": GESTURE_CLASSES,
        "gesture_display_names": GESTURE_DISPLAY_NAMES,
        "action_map": GESTURE_ACTION_MAP,
        "settings": RUNTIME_SETTINGS,
    }))

    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
                await handle_incoming(websocket, msg)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON",
                }))
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Client disconnected: {client_addr} — {e}")
    finally:
        CONNECTED_CLIENTS.discard(websocket)
        logger.info(f"Client removed. [{len(CONNECTED_CLIENTS)} active]")


async def handle_incoming(
    websocket: WebSocketServerProtocol, msg: dict
) -> None:
    """Process messages received from the Flutter client."""
    msg_type = msg.get("type", "")

    if msg_type == "ping":
        await websocket.send(json.dumps({
            "type": "pong",
            "timestamp": time.time(),
        }))

    elif msg_type == "update_settings":
        if "confidence_threshold" in msg:
            RUNTIME_SETTINGS["confidence_threshold"] = float(msg["confidence_threshold"])
        if "action_map" in msg:
            RUNTIME_SETTINGS["action_map"].update(msg["action_map"])

        logger.info(f"Settings updated: {RUNTIME_SETTINGS}")
        await websocket.send(json.dumps({
            "type": "settings_updated",
            "settings": RUNTIME_SETTINGS,
        }))

    elif msg_type == "get_status":
        await websocket.send(json.dumps({
            "type": "system_status",
            "clients_connected": len(CONNECTED_CLIENTS),
            "gesture_classes": GESTURE_CLASSES,
        }))

    else:
        await websocket.send(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
        }))


# ── Public broadcast functions (called by the gesture pipeline) ───────────────

async def send_gesture_update(
    gesture: str,
    confidence: float,
    valid: bool,
    all_probs: dict[str, float],
) -> None:
    await broadcast({
        "type":       "gesture_update",
        "gesture":    gesture,
        "confidence": confidence,
        "valid":      valid,
        "all_probs":  all_probs,
        "timestamp":  time.time(),
    })


async def send_action_event(event: dict) -> None:
    await broadcast(event)


async def send_system_status(fps: float, model_loaded: bool) -> None:
    await broadcast({
        "type":              "system_status",
        "fps":               round(fps, 1),
        "clients_connected": len(CONNECTED_CLIENTS),
        "model_loaded":      model_loaded,
        "timestamp":         time.time(),
    })


# ── Server entry point ────────────────────────────────────────────────────────

async def run_server() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info(f"Starting WebSocket server on ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")

    async with websockets.serve(
        handle_client,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
        ping_interval=20,
        ping_timeout=10,
    ) as server:
        logger.info("Server ready. Waiting for Flutter clients...")
        await asyncio.Future()   # run forever


if __name__ == "__main__":
    asyncio.run(run_server())
