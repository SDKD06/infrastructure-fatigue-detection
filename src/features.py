"""
features.py  —  windows -> 30 features, carrying labels AND groups through.
Feature math lives in feature_utils.py so the dashboard uses the exact same code.
"""

import numpy as np
from feature_utils import (
    PROCESSED_PATH, FEATURES_PATH, extract_features_one_window,
)


def main():
    print("=" * 50)
    print("FEATURE EXTRACTION")
    print("=" * 50)

    data    = np.load(PROCESSED_PATH, allow_pickle=True)
    windows = data["windows"]
    labels  = data["labels"]
    groups  = data["groups"]
    names   = data["group_names"]

    print(f"Loaded {len(windows)} windows of shape {windows[0].shape}")

    feats = np.empty((len(windows), 30), dtype=np.float64)
    for i, w in enumerate(windows):
        feats[i] = extract_features_one_window(w)
        if (i + 1) % 500 == 0 or (i + 1) == len(windows):
            print(f"  {i+1}/{len(windows)}")

    np.savez_compressed(FEATURES_PATH,
                        features=feats, labels=labels,
                        groups=groups, group_names=names)

    print(f"\nsaved -> {FEATURES_PATH}")
    print(f"  features : {feats.shape}")
    print(f"  groups   : {len(np.unique(groups))} recordings")


if __name__ == "__main__":
    main()