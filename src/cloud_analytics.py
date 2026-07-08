"""
cloud_analytics.py  -  cloud LOF diagnostic over a VIBRATION feature file.
(Renamed from acoustic/audio: the data is accelerometer vibration, not sound.)
"""

import os
import joblib
import numpy as np

from feature_utils import LOF_PATH, FEATURES_PATH


def analyze_vibration_logs(data_file_path):
    if not os.path.exists(LOF_PATH) or not os.path.exists(data_file_path):
        raise FileNotFoundError("Missing trained LOF model or feature file.")

    pkg = joblib.load(LOF_PATH)
    model, threshold = pkg["model"], pkg["threshold"]

    data = np.load(data_file_path, allow_pickle=True)
    if "labels" not in data.files:
        raise KeyError("features.npz has no labels — re-run features.py "
                       "(you may have loaded a stale/mock file).")
    X = data["features"]

    scores = model.score_samples(X)
    n_anom = int(np.sum(scores < threshold))
    rate = n_anom / len(scores) * 100

    return {
        "avg_density":        float(scores.mean()),
        "peak_deviation":     float(scores.min()),     # lowest density = worst
        "trailing_trend":     float(scores[-100:].mean()),
        "total_anomalies":    n_anom,
        "anomaly_rate_pct":   f"{rate:.2f}%",
    }


if __name__ == "__main__":
    try:
        r = analyze_vibration_logs(FEATURES_PATH)
        print("\nCLOUD VIBRATION DIAGNOSTIC\n" + "-" * 40)
        print(f"  Mean density score       : {r['avg_density']:.4f}")
        print(f"  Peak anomalous deviation : {r['peak_deviation']:.4f}")
        print(f"  Active anomaly rate      : {r['anomaly_rate_pct']}")
        print(f"  Total anomaly windows    : {r['total_anomalies']}")
        print("-" * 40)
    except Exception as e:
        print(f"Halted: {e}")