"""
rms_baseline.py  -  naive energy detector vs your trained models.

Flags a window as anomalous if its raw RMS energy exceeds a threshold (no
features, no model). Compares AUC/F1 against your deployed IF and LOF on the
SAME held-out test split. If IF/LOF win, your method earns its complexity.

Run after train.py + evaluate.py:  python src/rms_baseline.py
"""
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, f1_score
from feature_utils import FEATURES_PATH, SPLIT_PATH, IF_PATH, LOF_PATH, CONTAMINATION

d = np.load(FEATURES_PATH, allow_pickle=True)
X, y, groups = d["features"], d["labels"], d["groups"]
y_bin = np.where(y != 0, 1, 0)
split = np.load(SPLIT_PATH)
train_norm = np.isin(groups, split["train_groups"].tolist()) & (y == 0)
test = np.isin(groups, split["test_groups"].tolist())

# RMS feature columns are 0,10,20 (per-axis RMS). Mean = overall energy.
rms_train = X[train_norm][:, [0, 10, 20]].mean(axis=1)
rms_test  = X[test][:, [0, 10, 20]].mean(axis=1)
yte = y_bin[test]
thr = np.percentile(rms_train, 100 * (1 - CONTAMINATION))

print("=" * 46)
print(f"{'method':<24}{'AUC':<8}{'F1':<8}")
print("=" * 46)
print(f"{'RMS energy (baseline)':<24}{roc_auc_score(yte, rms_test):<8.4f}"
      f"{f1_score(yte, (rms_test > thr).astype(int), zero_division=0):<8.4f}")

for name, path in [("Isolation Forest", IF_PATH), ("Local Outlier Factor", LOF_PATH)]:
    pkg = joblib.load(path); m, t = pkg["model"], pkg["threshold"]
    s = m.score_samples(X[test])
    print(f"{name:<24}{roc_auc_score(yte, -s):<8.4f}"
          f"{f1_score(yte, (s < t).astype(int), zero_division=0):<8.4f}")
print("=" * 46)
print("Your models should beat the RMS baseline -> features earn their place.")