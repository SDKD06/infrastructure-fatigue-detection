# Infrastructure Fatigue Detection — Hybrid Edge–Cloud Anomaly Detection

Vibration-based structural anomaly detection using low-cost, phone-grade
accelerometer data. A **hybrid two-tier system**: a lightweight **Isolation
Forest runs on an ESP32 (edge)**, while a **Local Outlier Factor model runs in
the cloud** dashboard for aggregated analysis.

> Status: software complete and evaluated. ESP32 hardware deployment in progress
> (pending MPU6050 sensor).

---

## Overview

The system learns the _normal_ vibration signature of a structure and flags
deviations (tapping, surface vibration, load variation, structural
bending/flex) as anomalies. It is trained only on normal data
(one-class / novelty detection), so it needs no labelled fault data to deploy.

- **Sensor:** phone accelerometer (Phyphox app), 3-axis, resampled to 256 Hz
- **Windowing:** 2 s windows (512 samples), 50% overlap
- **Features:** 30 features (10 per axis) — time-domain (RMS, std, kurtosis,
  skew, peak-to-peak) + frequency-domain (dominant freq, spectral energy,
  centroid, entropy, crest factor)
- **Models compared:** Isolation Forest, One-Class SVM, Local Outlier Factor,
  Conv1D Autoencoder

---

## Results (held-out recording-level test)

| Model                 | AUC   | F1    |
| --------------------- | ----- | ----- |
| **Isolation Forest**  | 0.880 | 0.836 |
| Local Outlier Factor  | 0.868 | 0.832 |
| Conv1D Autoencoder    | 0.857 | 0.823 |
| One-Class SVM         | 0.770 | 0.755 |
| RMS energy (baseline) | 0.864 | 0.816 |

**Headline generalisation** (mean ± std over 10 repeated recording-level splits):

| Model                | AUC         | F1          |
| -------------------- | ----------- | ----------- |
| Isolation Forest     | 0.83 ± 0.11 | 0.73 ± 0.10 |
| Local Outlier Factor | 0.82 ± 0.11 | 0.69 ± 0.08 |
| One-Class SVM        | 0.75 ± 0.11 | 0.67 ± 0.09 |

Isolation Forest performs best and is chosen for the edge tier. The relatively
high variance reflects the small dataset (3 recordings per condition), noted as
a limitation and direction for future work.

---

## Dataset

Self-collected with the Phyphox app across 5 conditions
(3 recordings each): **normal** (baseline), **light tapping**,
**surface vibration**, **load variation**, **structural disturbance** (bending).
Recordings were captured at device-dependent rates (~200–414 Hz) and resampled
to a common 256 Hz.

> Raw data and trained models are **not** included in this repo (see
> `.gitignore`). Place recordings under `data/<condition>/` to reproduce.

---

## Pipeline — how to run

Run from the project root, in order:

```bash
python src/preprocess.py          # CSVs -> resampled, windowed, grouped
python src/features.py            # 30 features per window
python src/make_split.py          # recording-level train/test split
python src/train.py               # Isolation Forest / One-Class SVM / LOF
python src/train_autoencoder.py   # Conv1D autoencoder
python src/evaluate.py            # held-out + repeated-CV metrics
python src/evaluate_autoencoder.py
python src/compare_models.py      # four-model ROC comparison
python src/export_tinyml.py       # faithful IF -> firmware/micro_forest.h
python src/rms_baseline.py        # naive energy baseline
python src/ablation.py            # time-only vs time+frequency features
```

### Live dashboard (cloud tier)

```bash
python src/telemetry_server.py    # receiver + serves metrics
python src/fake_esp32.py          # streams real feature vectors (stands in for the board)
streamlit run src/dashboard.py    # open the monitor, pick "ESP32 Edge Feed"
```

---

## Repository structure

```
src/
  feature_utils.py        # shared config + 30-feature extraction
  preprocess.py           # resample + per-recording windowing
  features.py             # feature extraction driver
  make_split.py           # recording-level split
  train.py                # classical models
  train_autoencoder.py    # Conv1D autoencoder
  evaluate.py             # held-out + repeated-CV
  evaluate_autoencoder.py
  compare_models.py       # four-model ROC (TF-free)
  rms_baseline.py         # naive baseline
  ablation.py             # feature ablation
  export_tinyml.py        # faithful Isolation Forest -> C header
  cloud_analytics.py      # cloud LOF diagnostic
  telemetry_server.py     # ESP32 telemetry receiver + dashboard host
  fake_esp32.py           # hardware-free ESP32 simulator
  dashboard.py            # Streamlit monitor
firmware/
  firmware.ino            # ESP32 edge inference (MPU6050 + on-device features)
  esp32_validation.ino    # score-match + latency harness
  micro_forest.h          # auto-generated Isolation Forest (do not edit)
```

---

## Edge deployment (ESP32) — in progress

- **Board:** ESP32 DevKit + MPU6050 accelerometer (I2C: 3V3, GND, D21=SDA, D22=SCL)
- `export_tinyml.py` serialises the trained Isolation Forest into
  `micro_forest.h` (~122 KB) — a faithful port, not an approximation.
- `esp32_validation.ino` verifies on-chip scores match Python `score_samples`
  to ~1e-5 and measures inference latency.
  _Latency and memory-footprint measurements will be added after flashing._

---

## Requirements

Python 3.12, `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`,
`streamlit`, `plotly`, `flask`, `requests`, `tensorflow` (autoencoder only),
`joblib`. Firmware: Arduino IDE with ESP32 board support, `arduinoFFT`,
and an MPU6050 library.

---

## Limitations

- Small dataset (3 recordings/condition) → high variance in cross-validation.
- Anomalies are largely energy-distinguishable, so a simple RMS baseline is
  competitive; frequency features add marginal benefit on this data.
- Trained on phone-grade sensors; transfer to the ESP32's own accelerometer is
  part of ongoing hardware validation.
