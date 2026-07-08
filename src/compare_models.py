"""
compare_models.py  -  four-model ROC comparison on the held-out test split.

TF-free: the classical models are loaded from their .pkl files; the autoencoder's
per-window errors are read from data/ae_test_scores.npz (written by
evaluate_autoencoder.py). This avoids importing TensorFlow here, which is
unstable on some Windows setups. If the AE cache is missing, the AE is skipped.
"""
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

from feature_utils import (FEATURES_PATH, SPLIT_PATH, IF_PATH, OCSVM_PATH, LOF_PATH)

OUT     = "notebooks/master_model_comparison_roc.png"
AE_CACHE = "data/ae_test_scores.npz"   # written by evaluate_autoencoder.py


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
        scores = -pkg["model"].score_samples(Xte)
        preds  = np.where(-pkg["model"].score_samples(Xte) > -pkg["threshold"], 1, 0)
        fpr, tpr, _ = roc_curve(yte, scores)
        results[name] = (fpr, tpr, roc_auc_score(yte, scores),
                         f1_score(yte, preds, zero_division=0), color)

    # Autoencoder from cache (no TensorFlow needed here)
    if os.path.exists(AE_CACHE):
        c = np.load(AE_CACHE)
        err, yw, thr = c["errors"], c["labels"], float(c["threshold"])
        fpr, tpr, _ = roc_curve(yw, err)
        results["Conv1D Autoencoder"] = (
            fpr, tpr, roc_auc_score(yw, err),
            f1_score(yw, (err > thr).astype(int), zero_division=0), "blue")
    else:
        print("[note] AE cache not found — run evaluate_autoencoder.py first to include it.")

    os.makedirs("notebooks", exist_ok=True)
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc, f1, color) in results.items():
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f}, F1={f1:.2f})")
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