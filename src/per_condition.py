"""
per_condition.py

Reproduces the per-condition detection rates in Table VI of the paper
(Isolation Forest, held-out test; 132 anomalous windows per condition).

Uses the SAME precomputed features (features.npz) and the SAME fixed
recording-level split (split.npz) as the main pipeline, so the numbers match
the paper exactly. Run make_split.py / features.py first if those files
are absent.

Usage (from project root):
    python src/per_condition.py
"""

import os
import numpy as np
from sklearn.ensemble import IsolationForest

try:
    from feature_utils import (FEATURES_PATH, SPLIT_PATH, RANDOM_STATE)
except ImportError:  # allow running from project root
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_utils import (FEATURES_PATH, SPLIT_PATH, RANDOM_STATE)

# label value -> condition name. label 0 = normal.
CONDITION_NAMES = {
    1: "Tapping",
    2: "Surface vibration",
    3: "Load variation",
    4: "Structural disturbance",
}


def main():
    d = np.load(FEATURES_PATH, allow_pickle=True)
    X, y, groups = d["features"], d["labels"], d["groups"]

    sp = np.load(SPLIT_PATH, allow_pickle=True)
    train_groups = set(sp["train_groups"].tolist())

    train_mask = np.array([g in train_groups for g in groups])
    Xtr = X[train_mask]                      # normal training windows only

    # standardize on training statistics
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = IsolationForest(n_estimators=100, contamination=0.05,
                          random_state=RANDOM_STATE).fit((Xtr - mu) / sd)
    thr = np.percentile(clf.score_samples((Xtr - mu) / sd), 5)

    print(f"Training: {train_mask.sum()} normal windows "
          f"(groups {sorted(train_groups)}), threshold {thr:.4f}\n")

    print(f"{'Condition':<26}{'Detected':>14}{'Rate':>9}")
    print("-" * 49)
    # report each anomalous condition over ALL its windows
    for label, name in CONDITION_NAMES.items():
        mask = (y == label)
        if not mask.any():
            continue
        Xc = X[mask]
        flagged = int(np.sum(clf.score_samples((Xc - mu) / sd) < thr))
        total = int(mask.sum())
        print(f"{name:<26}{flagged:>7}/{total:<6}{100*flagged/total:>8.1f}%")


if __name__ == "__main__":
    main()