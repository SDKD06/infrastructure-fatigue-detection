"""
train.py  —  trains the 3 classical detectors on TRAIN-split normal windows only.

The model you deploy is now the model you evaluate (same split as make_split.py).
Run order: preprocess -> features -> make_split -> train.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from feature_utils import (
    FEATURES_PATH, SPLIT_PATH, MODEL_DIR, IF_PATH, OCSVM_PATH, LOF_PATH,
    CONTAMINATION, RANDOM_STATE,
)


def main():
    print("=" * 56)
    print("CLASSICAL TRAINING (IF / One-Class SVM / LOF) on train-normal")
    print("=" * 56)

    d = np.load(FEATURES_PATH, allow_pickle=True)
    X, y, groups = d["features"], d["labels"], d["groups"]
    split = np.load(SPLIT_PATH)
    train_groups = set(split["train_groups"].tolist())

    train_mask = np.isin(groups, list(train_groups)) & (y == 0)
    X_train = X[train_mask]
    print(f"Training on {len(X_train)} normal windows "
          f"from {len(train_groups)} recordings.")
    if len(X_train) < 50:
        print("[warning] very little training data; results will be noisy.")

    os.makedirs(MODEL_DIR, exist_ok=True)

    def fit_and_save(name, model, path):
        model.fit(X_train)
        scores = model.score_samples(X_train)
        threshold = float(np.percentile(scores, CONTAMINATION * 100))
        joblib.dump({"model": model, "threshold": threshold}, path)
        print(f"  ok {name:22s} threshold={threshold:.6f} -> {path}")

    fit_and_save("Isolation Forest",
                 IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                                 random_state=RANDOM_STATE),
                 IF_PATH)
    fit_and_save("One-Class SVM",
                 OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION),
                 OCSVM_PATH)
    fit_and_save("Local Outlier Factor",
                 LocalOutlierFactor(n_neighbors=20, novelty=True,
                                    contamination=CONTAMINATION),
                 LOF_PATH)

    print("Done. Next: train_autoencoder.py, then evaluate.py / compare_models.py")


if __name__ == "__main__":
    main()