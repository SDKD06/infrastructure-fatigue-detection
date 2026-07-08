"""
make_split.py  —  one fixed RECORDING-LEVEL split, reused by train + evaluate.

Why recording-level: windows from the same continuous recording are correlated.
Splitting by window leaks information; splitting by recording does not.

  * normal recordings  -> split into train / test
  * anomaly recordings -> all go to test (we only ever TRAIN on normal)
"""

import numpy as np
from feature_utils import FEATURES_PATH, SPLIT_PATH, RANDOM_STATE

TEST_NORMAL_FRACTION = 0.34   # ~1 of every 3 normal recordings held out


def main():
    d = np.load(FEATURES_PATH, allow_pickle=True)
    y, groups = d["labels"], d["groups"]

    # label per recording (all windows in a group share a label)
    uniq = np.unique(groups)
    grp_label = {g: int(y[groups == g][0]) for g in uniq}

    normal_groups  = np.array([g for g in uniq if grp_label[g] == 0])
    anomaly_groups = np.array([g for g in uniq if grp_label[g] != 0])

    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(normal_groups)
    n_test = max(1, int(round(len(normal_groups) * TEST_NORMAL_FRACTION)))
    test_normal  = normal_groups[:n_test]
    train_normal = normal_groups[n_test:]

    train_groups = np.array(train_normal)
    test_groups  = np.concatenate([test_normal, anomaly_groups])

    np.savez(SPLIT_PATH, train_groups=train_groups, test_groups=test_groups)

    print("Recording-level split")
    print(f"  train (normal only) : {len(train_groups)} recordings")
    print(f"  test  (held-out)    : {len(test_groups)} recordings "
          f"({len(test_normal)} normal + {len(anomaly_groups)} anomalous)")
    if len(train_normal) < 2:
        print("  [warning] very few normal training recordings — collect more.")


if __name__ == "__main__":
    main()