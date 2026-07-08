"""
ablation.py  -  do the FFT features actually help?

Trains Isolation Forest on (a) time-domain features only, then (b) all 30
features, and compares AUC/F1 on the held-out test split. If full > time-only,
the frequency features earn their place.

Feature layout per axis (x3): [rms,std,kurt,skew,p2p | dom_freq,spec_energy,
spec_centroid,spec_entropy,crest]. Time = first 5 of each axis; freq = last 5.

Run:  python src/ablation.py
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, f1_score
from feature_utils import FEATURES_PATH, SPLIT_PATH, CONTAMINATION, RANDOM_STATE

TIME_COLS = [0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24]   # 15 time-domain features
ALL_COLS  = list(range(30))

d = np.load(FEATURES_PATH, allow_pickle=True)
X, y, groups = d["features"], d["labels"], d["groups"]
y_bin = np.where(y != 0, 1, 0)
split = np.load(SPLIT_PATH)
train_norm = np.isin(groups, split["train_groups"].tolist()) & (y == 0)
test = np.isin(groups, split["test_groups"].tolist())
yte = y_bin[test]

print("=" * 52)
print(f"{'feature set':<26}{'#feat':<7}{'AUC':<8}{'F1':<8}")
print("=" * 52)
for label, cols in [("Time-domain only", TIME_COLS), ("Time + frequency (full)", ALL_COLS)]:
    Xtr, Xte = X[train_norm][:, cols], X[test][:, cols]
    m = IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                        random_state=RANDOM_STATE).fit(Xtr)
    t = np.percentile(m.score_samples(Xtr), CONTAMINATION * 100)
    s = m.score_samples(Xte)
    print(f"{label:<26}{len(cols):<7}{roc_auc_score(yte, -s):<8.4f}"
          f"{f1_score(yte, (s < t).astype(int), zero_division=0):<8.4f}")
print("=" * 52)
print("If full > time-only, the FFT features are justified.")