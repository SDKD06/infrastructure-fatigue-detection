"""
telemetry_server.py  -  receives ESP32 anomaly POSTs and serves them to the dashboard.

The firmware POSTs JSON like:
    {"score": -0.61, "node": "Node_01", "features": [...30...]}
to  http://<laptop-ip>:8765/api/v1/telemetry

This Flask app stores the most recent events in memory and exposes:
    POST /api/v1/telemetry   <- ESP32 sends here
    GET  /api/v1/latest      <- dashboard polls here (returns recent events as JSON)
    GET  /api/v1/health      <- quick "is it up" check

Run from project root:   python src/telemetry_server.py
Point the ESP32 firmware's serverUrl at  http://<laptop-ip>:8765/api/v1/telemetry
"""

import os
import sys
import time
from collections import deque
from flask import Flask, request, jsonify

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from feature_utils import IF_PATH
    import joblib
    _pkg = joblib.load(IF_PATH) if os.path.exists(IF_PATH) else None
    IF_THRESHOLD = _pkg["threshold"] if _pkg else -0.5
except Exception:
    IF_THRESHOLD = -0.5

app = Flask(__name__)
EVENTS = deque(maxlen=500)   # rolling buffer of recent ESP32 events


@app.route("/api/v1/telemetry", methods=["POST"])
def telemetry():
    data = request.get_json(force=True, silent=True) or {}
    score = float(data.get("score", 0.0))
    event = {
        "t": time.time(),
        "score": score,
        "node": data.get("node", "unknown"),
        "anomaly": score < IF_THRESHOLD,
        "n_features": len(data.get("features", [])),
    }
    EVENTS.append(event)
    print(f"[{event['node']}] score={score:.4f}  "
          f"{'ANOMALY' if event['anomaly'] else 'normal'}  "
          f"(features={event['n_features']})")
    return jsonify({"ok": True, "anomaly": event["anomaly"]})


@app.route("/api/v1/latest", methods=["GET"])
def latest():
    n = int(request.args.get("n", 100))
    return jsonify({"threshold": IF_THRESHOLD,
                    "events": list(EVENTS)[-n:]})


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "stored": len(EVENTS), "threshold": IF_THRESHOLD})


if __name__ == "__main__":
    print("Telemetry server on http://0.0.0.0:8765")
    print(f"  IF threshold = {IF_THRESHOLD}")
    print("  ESP32 -> POST /api/v1/telemetry   dashboard -> GET /api/v1/latest")
    app.run(host="0.0.0.0", port=8765, debug=False)