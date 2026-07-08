"""
evaluate.py  —  honest evaluation, two views:

  (A) HELD-OUT TEST  — scores the DEPLOYED classical models on the test split.
      These are the numbers that match the .pkl files you ship.

  (B) LEAVE-ONE-RECORDING-OUT CV — refits each classical model while holding out
      one recording at a time. This is the honest, generalisation number for the
      paper (reported as mean +/- std). Skipped for the AE (too costly to refit).

Writes data/eval_results.npz so the dashboard shows REAL metrics, not literals.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                             classification_report, confusion_matrix)

from feature_utils import (
    FEATURES_PATH, SPLIT_PATH, RESULTS_PATH,
    IF_PATH, OCSVM_PATH, LOF_PATH, CONTAMINATION, RANDOM_STATE,
)

ROC_OUT = "notebooks/roc_holdout.png"

CLASSICAL = {
    "Isolation Forest":     ("darkorange", IF_PATH),
    "One-Class SVM":        ("green",      OCSVM_PATH),
    "Local Outlier Factor": ("purple",     LOF_PATH),
}


def make_model(name):
    if name == "Isolation Forest":
        return IsolationForest(n_estimators=100, contamination=CONTAMINATION,
                               random_state=RANDOM_STATE)
    if name == "One-Class SVM":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION)
    return LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=CONTAMINATION)


def held_out(X, y_bin, groups):
    split = np.load(SPLIT_PATH)
    test_mask = np.isin(groups, split["test_groups"].tolist())
    Xte, yte = X[test_mask], y_bin[test_mask]

    print("\n" + "=" * 56)
    print(f"(A) HELD-OUT TEST  -  {len(yte)} windows "
          f"({np.sum(yte==0)} normal / {np.sum(yte==1)} anomalous)")
    print("=" * 56)

    results, roc_curves = {}, {}
    for name, (color, path) in CLASSICAL.items():
        pkg = joblib.load(path)
        model, thr = pkg["model"], pkg["threshold"]
        scores = model.score_samples(Xte)
        preds  = np.where(scores < thr, 1, 0)
        auc    = roc_auc_score(yte, -scores)
        f1     = f1_score(yte, preds, zero_division=0)
        results[name] = (auc, f1, color)
        fpr, tpr, _ = roc_curve(yte, -scores)
        roc_curves[name] = (fpr, tpr, color, auc)

        print(f"\n--- {name} ---")
        print(classification_report(yte, preds,
              target_names=["Normal", "Anomalous"], zero_division=0))
        cm = confusion_matrix(yte, preds)
        print(f"  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}  "
              f"AUC={auc:.4f} F1={f1:.4f}")

    os.makedirs("notebooks", exist_ok=True)
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, color, auc) in roc_curves.items():
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "navy", lw=1, ls="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Held-out ROC (recording-level split)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(ROC_OUT, dpi=300); plt.close()
    print(f"\nROC saved -> {ROC_OUT}")
    return results


def loro_cv(X, y_bin, groups):
    print("\n" + "=" * 56)
    print("(B) LEAVE-ONE-RECORDING-OUT CV  (honest generalisation)")
    print("=" * 56)
    logo = LeaveOneGroupOut()
    summary = {}
    for name in CLASSICAL:
        aucs, f1s = [], []
        for tr_idx, te_idx in logo.split(X, y_bin, groups):
            tr_norm = tr_idx[y_bin[tr_idx] == 0]
            if len(tr_norm) < 10 or len(np.unique(y_bin[te_idx])) < 2:
                continue
            model = make_model(name).fit(X[tr_norm])
            thr = np.percentile(model.score_samples(X[tr_norm]), CONTAMINATION * 100)
            s = model.score_samples(X[te_idx])
            aucs.append(roc_auc_score(y_bin[te_idx], -s))
            f1s.append(f1_score(y_bin[te_idx], np.where(s < thr, 1, 0), zero_division=0))
        if aucs:
            summary[name] = (np.mean(aucs), np.std(aucs), np.mean(f1s), np.std(f1s))
            print(f"  {name:24s} AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
                  f"F1={np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  (n={len(aucs)} folds)")
        else:
            print(f"  {name:24s} not enough recordings for CV")
    return summary


def main():
    d = np.load(FEATURES_PATH, allow_pickle=True)
    X, y, groups = d["features"], d["labels"], d["groups"]
    y_bin = np.where(y != 0, 1, 0)

    held = held_out(X, y_bin, groups)
    loro_cv(X, y_bin, groups)

    lof_auc, lof_f1, _ = held.get("Local Outlier Factor", (0.0, 0.0, ""))
    np.savez(RESULTS_PATH,
             model_names=np.array(list(held.keys())),
             holdout_auc=np.array([held[m][0] for m in held]),
             holdout_f1=np.array([held[m][1] for m in held]),
             lof_holdout_auc=lof_auc, lof_holdout_f1=lof_f1)
    print(f"\nMetrics saved -> {RESULTS_PATH} (dashboard reads these)")
    print("Report the LORO-CV mean+/-std as your headline generalisation number.")


if __name__ == "__main__":
    main()