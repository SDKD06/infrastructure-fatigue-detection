"""
preprocess.py  —  Phyphox CSVs -> resampled, per-recording windows.

Fixes vs. original:
  * detects each file's TRUE sample rate and resamples to TARGET_FS (the 414->200 bug)
  * windows each recording independently (no windows straddling two files)
  * tags every window with a recording/group id (enables leave-one-recording-out CV)
  * peak normalisation is OFF by default (it erased amplitude, the anomaly signal)
"""

import os
import numpy as np
import pandas as pd

from feature_utils import (
    LABEL_MAP, TARGET_FS, PROCESSED_PATH, AXES,
    resample_to_target, estimate_fs, windows_from_signal,
)

DATA_DIR       = "data"
NORMALISE_PEAK = False   # set True only to A/B test; off preserves amplitude


def clean_recording(df):
    """Zero-centre each axis (removes gravity/DC). Optional peak normalisation."""
    out = df.copy()
    for a in AXES:
        out[a] = out[a] - out[a].mean()
        if NORMALISE_PEAK:
            peak = out[a].abs().max()
            if peak > 0:
                out[a] = out[a] / peak
    return out


def load_recording(filepath):
    """Read one Phyphox CSV, map columns, resample to TARGET_FS, clean."""
    df = pd.read_csv(filepath)
    if len(df.columns) < 4:
        return None
    df = df.iloc[:, :4]
    df.columns = ["time", "acc_x", "acc_y", "acc_z"]
    df = df.dropna()
    if len(df) < 10:
        return None

    fs = estimate_fs(df["time"].values)
    sig = resample_to_target(df[AXES].values, fs, TARGET_FS)
    clean = clean_recording(pd.DataFrame(sig, columns=AXES))
    return clean, fs


def main():
    print("=" * 56)
    print("PREPROCESSING  (resample -> per-recording windows -> groups)")
    print("=" * 56)

    all_w, all_l, all_g, group_names = [], [], [], []
    gid = 0

    for condition, label in LABEL_MAP.items():
        folder = os.path.join(DATA_DIR, condition)
        print(f"\n-> {folder}  (label {label})")
        if not os.path.exists(folder):
            print("  [skip] folder missing")
            continue

        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".csv")):
            fpath = os.path.join(folder, fname)
            try:
                loaded = load_recording(fpath)
            except Exception as e:
                print(f"  [error] {fname}: {e}")
                continue
            if loaded is None:
                print(f"  [skip] {fname}: unreadable / too short")
                continue

            clean, fs = loaded
            wins = windows_from_signal(clean[AXES].values)
            if len(wins) == 0:
                print(f"  [skip] {fname}: too short for a window")
                continue

            all_w.extend(wins)
            all_l.extend([label] * len(wins))
            all_g.extend([gid] * len(wins))
            group_names.append(f"{condition}/{fname}")
            print(f"  ok {fname:28s} fs~{fs:6.1f}Hz -> {len(wins):4d} windows (group {gid})")
            gid += 1

    if not all_w:
        raise ValueError("No windows produced. Put Phyphox CSVs under data/<condition>/.")

    windows = np.array(all_w)
    labels  = np.array(all_l)
    groups  = np.array(all_g)
    names   = np.array(group_names)

    np.savez_compressed(PROCESSED_PATH,
                        windows=windows, labels=labels,
                        groups=groups, group_names=names)

    print(f"\nsaved -> {PROCESSED_PATH}")
    print(f"  windows : {windows.shape}")
    print(f"  labels  : {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  groups  : {len(np.unique(groups))} recordings")


if __name__ == "__main__":
    main()