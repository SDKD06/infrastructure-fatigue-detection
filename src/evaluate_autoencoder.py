"""
evaluate_autoencoder.py  —  autoencoder on the HELD-OUT test split only.
No leakage: the AE was trained on train-split normal windows; we score the
held-out recordings here.
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)

from feature_utils import PROCESSED_PATH, SPLIT_PATH, AE_PATH, AE_THRES


def main():
    print("=" * 56)
    print("AUTOENCODER EVALUATION (held-out test split)")
    print("=" * 56)

    d = np.load(PROCESSED_PATH, allow_pickle=True)
    X, y, groups = d["windows"], d["labels"], d["groups"]
    split = np.load(SPLIT_PATH)
    test_mask = np.isin(groups, split["test_groups"].tolist())

    Xte = X[test_mask].astype("float32")
    yte = np.where(y[test_mask] != 0, 1, 0)

    model = load_model(AE_PATH)
    threshold = float(np.load(AE_THRES))

    pred = model.predict(Xte, verbose=0)
    err  = np.mean(np.square(Xte - pred), axis=(1, 2))
    yhat = np.where(err > threshold, 1, 0)

    print(f"\nTest windows: {len(yte)} "
          f"({np.sum(yte==0)} normal / {np.sum(yte==1)} anomalous)")
    print(classification_report(yte, yhat,
          target_names=["Normal", "Anomalous"], zero_division=0))
    cm = confusion_matrix(yte, yhat)
    print(f"TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    print(f"AUC={roc_auc_score(yte, err):.4f}  F1={f1_score(yte, yhat, zero_division=0):.4f}")
    print("\nReminder: with few normal recordings the AE likely overfits — "
          "report this honestly relative to IF/LOF.")


if __name__ == "__main__":
    main()