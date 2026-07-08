"""
fake_esp32.py  -  pretends to be the ESP32 so you can test the telemetry server
                  + dashboard edge panel TODAY, with no hardware.

Streams real feature vectors from data/features.npz to the telemetry server,
exactly like the firmware will once the MPU6050 arrives.

Run (in a second terminal, after telemetry_server.py is up):
    python src/fake_esp32.py
"""

import os
import sys
import time
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import FEATURES_PATH

URL = "http://127.0.0.1:8765/api/v1/telemetry"


def main():
    if not os.path.exists(FEATURES_PATH):
        print("Run features.py first to create data/features.npz")
        return
    X = np.load(FEATURES_PATH, allow_pickle=True)["features"]
    print(f"Streaming {len(X)} feature vectors to {URL} (Ctrl+C to stop)")

    import joblib
    from feature_utils import IF_PATH
    model = joblib.load(IF_PATH)["model"]   # use the real IF to make scores realistic

    for i, feat in enumerate(X):
        score = float(model.score_samples(feat.reshape(1, -1))[0])
        try:
            requests.post(URL, json={"score": score, "node": "FakeESP32",
                                     "features": feat.tolist()}, timeout=2)
        except Exception as e:
            print(f"POST failed: {e}"); break
        if i % 20 == 0:
            print(f"  sent {i+1}/{len(X)}  score={score:.4f}")
        time.sleep(0.3)


if __name__ == "__main__":
    main()