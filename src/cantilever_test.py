"""
cantilever_test.py

Reproduces the constant-amplitude frequency-shift result reported in the paper
(cantilever experiment): the Isolation Forest detects the added-mass frequency
shift that the RMS-energy baseline is blind to.

Setup (see paper, Section VI-E):
  - A metal scale clamped with a fixed overhang; MPU6050 mounted at the free tip.
  - Healthy  = bare tip.
  - Damaged  = a small mass added at the tip, which lowers the natural
               frequency (~16 Hz -> ~8 Hz) WITHOUT raising excitation energy.
  - Same tapping, same 45 s / 256 Hz pipeline as the main dataset.

Expected result (5 healthy / 6 damaged recordings):
  Isolation Forest -> ~72% of damaged windows flagged, ~5% false-positive rate
  RMS baseline     -> ~0% of damaged windows flagged
  Energy matched   -> RMS ratio ~0.92 ; dominant freq ~16 Hz vs ~8 Hz

Usage:
    python cantilever_test.py --data-dir data/cantilever
    (expects healthy/*.csv and damaged/*.csv underneath)
"""

import argparse
import os
import glob
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest

FS = 256
WIN = 512
HOP = 256
SEED = 0

# Column index of the sensing axis used for the cantilever mode (z-axis).
# acc_x=1, acc_y=2, acc_z=3 in the logger CSV.
SENSE_AXIS = 3


def load_recording(path):
    """Read a logger CSV (time,acc_x,acc_y,acc_z), skipping boot-log junk."""
    with open(path, "rb") as f:
        lines = f.read().decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith("time,acc_x"))
    rows = []
    for l in lines[start + 1:]:
        parts = l.strip().split(",")
        if len(parts) == 4:
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                pass
    return np.array(rows)


def axis_features(sig):
    """10 features for one axis of one window (same set as the main pipeline)."""
    a = sig - sig.mean()
    rms = np.sqrt(np.mean(a ** 2))
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    freqs = np.fft.rfftfreq(len(a), 1.0 / FS)
    p = spec / (spec.sum() + 1e-12)
    return [
        rms,
        a.std(),
        kurtosis(a),
        skew(a),
        a.max() - a.min(),
        np.max(np.abs(a)) / (rms + 1e-12),
        freqs[2:][np.argmax(spec[2:])],          # dominant frequency
        np.sum(spec ** 2),
        np.sum(freqs * spec) / (spec.sum() + 1e-12),
        -np.sum(p * np.log(p + 1e-12)),
    ]


def windows(sig3):
    """30 features per 2 s / 50%-overlap window."""
    X = []
    for s in range(0, len(sig3) - WIN + 1, HOP):
        row = []
        for ax in range(3):
            row += axis_features(sig3[s:s + WIN, ax])
        X.append(row)
    return np.array(X)


def to_windows(path):
    rec = load_recording(path)
    sig3 = np.stack([rec[:, 1], rec[:, 2], rec[:, 3]], axis=1)
    sig3 = sig3 - sig3.mean(axis=0)          # per-axis mean-centering
    return windows(sig3)


def dominant_freq(path):
    """Median per-window dominant frequency on the sensing axis (for reporting)."""
    rec = load_recording(path)
    a = rec[:, SENSE_AXIS] - rec[:, SENSE_AXIS].mean()
    doms = []
    for s in range(0, len(a) - WIN + 1, HOP):
        seg = a[s:s + WIN] * np.hanning(WIN)
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(WIN, 1.0 / FS)
        m = freqs >= 2
        doms.append(freqs[m][np.argmax(spec[m])])
    return float(np.median(doms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/cantilever",
                    help="dir with healthy/ and damaged/ subfolders of CSVs")
    args = ap.parse_args()

    healthy_paths = sorted(glob.glob(os.path.join(args.data_dir, "healthy", "*.csv")))
    damaged_paths = sorted(glob.glob(os.path.join(args.data_dir, "damaged", "*.csv")))
    if not healthy_paths or not damaged_paths:
        raise SystemExit(f"Expected CSVs under {args.data_dir}/healthy and /damaged")

    Xh = np.vstack([to_windows(p) for p in healthy_paths])
    Xd = np.vstack([to_windows(p) for p in damaged_paths])

    # Standardize on the healthy (normal) statistics
    mu, sd = Xh.mean(0), Xh.std(0) + 1e-9
    Zh, Zd = (Xh - mu) / sd, (Xd - mu) / sd

    # Train Isolation Forest on healthy only; threshold = 5th percentile
    clf = IsolationForest(n_estimators=100, contamination=0.05,
                          random_state=SEED).fit(Zh)
    thr = np.percentile(clf.score_samples(Zh), 5)

    if_detect = 100 * np.mean(clf.score_samples(Zd) < thr)
    if_fpr = 100 * np.mean(clf.score_samples(Zh) < thr)

    # RMS baseline on the sensing (z) axis, matching how energy was matched
    # during collection. Feature index 0 of each axis block is that axis' RMS;
    # z-axis is the 3rd block -> index 20.
    Z_RMS = 20
    rms_h = Xh[:, Z_RMS]
    rms_d = Xd[:, Z_RMS]
    rms_thr = np.percentile(rms_h, 95)
    rms_detect = 100 * np.mean(rms_d > rms_thr)

    freq_h = np.median([dominant_freq(p) for p in healthy_paths])
    freq_d = np.median([dominant_freq(p) for p in damaged_paths])

    print("=== Cantilever constant-amplitude frequency-shift test ===")
    print(f"Healthy recordings: {len(healthy_paths)}  ({len(Xh)} windows)")
    print(f"Damaged recordings: {len(damaged_paths)}  ({len(Xd)} windows)\n")
    print(f"Dominant frequency:  healthy ~{freq_h:.0f} Hz   damaged ~{freq_d:.0f} Hz")
    print(f"Energy ratio (dmg/healthy), z-axis: {rms_d.mean() / rms_h.mean():.2f}\n")
    print(f"Isolation Forest -> damaged detected: {if_detect:.0f}%   "
          f"(healthy false-positive: {if_fpr:.0f}%)")
    print(f"RMS baseline     -> damaged detected: {rms_detect:.0f}%")


if __name__ == "__main__":
    main()