"""
compare_models.py  —  one ROC chart comparing all four models on the SAME
held-out test split (recording-level). Deployed classical models are loaded;
the AE is loaded and scored on the held-out windows.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

from feature_utils import (
    FEATURES_PATH, PROCESSED_PATH, SPLIT_PATH,
    IF_PATH, OCSVM_PATH, LOF_PATH, AE_PATH, AE_THRES,
)

OUT = "notebooks/master_model_comparison_roc.png"


def main():
    print("=" * 56)
    print("FOUR-WAY COMPARISON (held-out test split)")
    print("=" * 56)

    feat = np.load(FEATURES_PATH, allow_pickle=True)
    X, y, groups = feat["features"], feat["labels"], feat["groups"]
    split = np.load(SPLIT_PATH)
    test_mask = np.isin(groups, split["test_groups"].tolist())
    Xte = X[test_mask]
    yte = np.where(y[test_mask] != 0, 1, 0)

    results = {}

    for name, path, color in [
        ("Isolation Forest", IF_PATH, "darkorange"),
        ("One-Class SVM",    OCSVM_PATH, "green"),
        ("Local Outlier Factor", LOF_PATH, "purple"),
    ]:
        if not os.path.exists(path):
            continue
        pkg = joblib.load(path)
        scores = -pkg["model"].score_samples(Xte)   # higher = more anomalous
        preds  = np.where(-pkg["model"].score_samples(Xte) > -pkg["threshold"], 1, 0)
        fpr, tpr, _ = roc_curve(yte, scores)
        results[name] = (fpr, tpr, roc_auc_score(yte, scores),
                         f1_score(yte, preds, zero_division=0), color)

    # autoencoder (scored on raw windows of the same test recordings)
    if os.path.exists(AE_PATH):
        from tensorflow.keras.models import load_model
        win = np.load(PROCESSED_PATH, allow_pickle=True)
        Wmask = np.isin(win["groups"], split["test_groups"].tolist())
        Xw = win["windows"][Wmask].astype("float32")
        yw = np.where(win["labels"][Wmask] != 0, 1, 0)
        ae = load_model(AE_PATH)
        err = np.mean(np.square(Xw - ae.predict(Xw, verbose=0)), axis=(1, 2))
        thr = float(np.load(AE_THRES))
        fpr, tpr, _ = roc_curve(yw, err)
        results["Conv1D Autoencoder"] = (
            fpr, tpr, roc_auc_score(yw, err),
            f1_score(yw, np.where(err > thr, 1, 0), zero_division=0), "blue")

    os.makedirs("notebooks", exist_ok=True)
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc, f1, color) in results.items():
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC={auc:.3f}, F1={f1:.2f})")
    plt.plot([0, 1], [0, 1], "navy", lw=1, ls="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Model comparison — ROC (held-out, recording-level)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(OUT, dpi=300); plt.close()

    print(f"\nSaved -> {OUT}\n")
    print(f"  {'Model':<24} | {'AUC':<8} | {'F1':<8}")
    print("-" * 48)
    for name, (_, _, auc, f1, _) in results.items():
        print(f"  {name:<24} | {auc:<8.4f} | {f1:<8.4f}")


if __name__ == "__main__":
    main()