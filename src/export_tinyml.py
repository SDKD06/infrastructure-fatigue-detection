"""
export_tinyml.py  —  FAITHFUL Isolation Forest -> C/C++ exporter for ESP32.

Serialises every tree (feature, threshold, children, leaf sample counts) into
flat C arrays and emits an on-device walker reproducing sklearn's
IsolationForest.score_samples() to float precision. NOT a heuristic.

Anomaly rule matches train.py:  score_samples(x) < threshold  => anomaly.
"""

import os
import joblib
import numpy as np
from math import log

from feature_utils import IF_PATH

OUTPUT_H_FILE = "firmware/micro_forest.h"
EULER_GAMMA = 0.5772156649015329


def average_path_length(n):
    n = int(n)
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


def export_to_cpp():
    print("=" * 56)
    print("ISOLATION FOREST -> TINYML C++ EXPORT (faithful tree port)")
    print("=" * 56)
    if not os.path.exists(IF_PATH):
        raise FileNotFoundError(f"{IF_PATH} not found. Run train.py first.")

    pkg = joblib.load(IF_PATH)
    model = pkg["model"]
    threshold = float(pkg.get("threshold", -0.5))

    n_estimators = len(model.estimators_)
    n_features = int(model.n_features_in_)
    max_samples = int(model.max_samples_)
    c_norm = average_path_length(max_samples)

    feat, thr, left, right, nsamp, offsets = [], [], [], [], [], []
    for est in model.estimators_:
        t = est.tree_
        offsets.append(len(feat))
        feat.extend(int(v) for v in t.feature)
        thr.extend(float(v) for v in t.threshold)
        left.extend(int(v) for v in t.children_left)
        right.extend(int(v) for v in t.children_right)
        nsamp.extend(int(v) for v in t.n_node_samples)
    offsets.append(len(feat))
    n_nodes = len(feat)

    if max(nsamp) > 32767 or max(max(left), max(right)) > 32767:
        raise ValueError("Index/sample count exceeds int16; widen the C arrays.")

    print(f"  estimators={n_estimators}  features={n_features}  "
          f"max_samples={max_samples}  nodes={n_nodes}")
    approx_kb = (n_nodes * (2 + 4 + 2 + 2 + 2) + (n_estimators + 1) * 4) / 1024
    print(f"  approx flash footprint: ~{approx_kb:.1f} KB")
    print(f"  threshold={threshold:.6f}  c_norm={c_norm:.6f}")

    def arr(name, vals, ctype, fmt):
        return f"const {ctype} {name}[{len(vals)}] = {{{', '.join(fmt.format(v) for v in vals)}}};\n"

    header = f'''/*
 * micro_forest.h  -  AUTO-GENERATED faithful Isolation Forest inference.
 * Source: scikit-learn IsolationForest. Reproduces score_samples() on-device.
 * Anomaly rule: score < ANOMALY_THRESHOLD  => anomaly. DO NOT EDIT BY HAND.
 */
#ifndef MICRO_FOREST_H
#define MICRO_FOREST_H
#include <Arduino.h>
#include <math.h>

#define N_ESTIMATORS {n_estimators}
#define N_FEATURES   {n_features}
#define N_NODES      {n_nodes}
#define MAX_SAMPLES  {max_samples}

static const float ANOMALY_THRESHOLD = {threshold:.8f}f;
static const float C_NORM            = {c_norm:.8f}f;

const int32_t TREE_OFFSET[N_ESTIMATORS + 1] = {{{", ".join(str(v) for v in offsets)}}};
{arr("NODE_FEAT",  feat,  "int16_t", "{}")}{arr("NODE_LEFT",  left,  "int16_t", "{}")}{arr("NODE_RIGHT", right, "int16_t", "{}")}{arr("NODE_NSAMP", nsamp, "int16_t", "{}")}const float NODE_THR[N_NODES] = {{{", ".join(f"{v:.8e}f" for v in thr)}}};

static inline float avg_path_length(int n) {{
    if (n <= 1) return 0.0f;
    if (n == 2) return 1.0f;
    return 2.0f * (logf((float)(n - 1)) + 0.5772156649015329f)
           - 2.0f * (float)(n - 1) / (float)n;
}}

static float tree_path_length(int t, const float* x) {{
    int off = TREE_OFFSET[t];
    int node = 0;
    float depth = 0.0f;
    for (int guard = 0; guard < N_NODES; ++guard) {{
        int idx = off + node;
        if (NODE_LEFT[idx] == -1) return depth + avg_path_length(NODE_NSAMP[idx]);
        node = (x[NODE_FEAT[idx]] <= NODE_THR[idx]) ? NODE_LEFT[idx] : NODE_RIGHT[idx];
        depth += 1.0f;
    }}
    return depth;
}}

/* Faithful IsolationForest.score_samples() port. */
bool predict_anomaly(const float* features, float* out_score) {{
    float total = 0.0f;
    for (int t = 0; t < N_ESTIMATORS; ++t) total += tree_path_length(t, features);
    float avg = total / (float)N_ESTIMATORS;
    float score = -powf(2.0f, -avg / C_NORM);   // sklearn convention
    *out_score = score;
    return (score < ANOMALY_THRESHOLD);
}}

#endif // MICRO_FOREST_H
'''
    os.makedirs(os.path.dirname(OUTPUT_H_FILE), exist_ok=True)
    with open(OUTPUT_H_FILE, "w") as f:
        f.write(header)
    print(f"\nWrote {OUTPUT_H_FILE}. Validate: model.score_samples(X[:5]) vs C output ~1e-5.")


if __name__ == "__main__":
    export_to_cpp()