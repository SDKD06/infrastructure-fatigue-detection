"""
feature_utils.py  —  SINGLE SOURCE OF TRUTH for constants + feature extraction.

Both the offline pipeline (features.py) and the live dashboard import from here,
so the 30 features are computed identically everywhere. If you change a feature,
change it once, here.
"""

import numpy as np
from scipy import stats
from scipy.signal import resample_poly
from fractions import Fraction

# ─────────────────────────────────────────────
# CANONICAL CONFIG  (must be consistent project-wide)
# ─────────────────────────────────────────────
TARGET_FS    = 256            # 256 Hz so a 2s window = 512 samples (power of two)
WINDOW_SEC   = 2              # -> lets the ESP32 FFT match numpy.rfft exactly
WINDOW_SIZE  = int(TARGET_FS * WINDOW_SEC)   # 512 samples
OVERLAP      = 0.5
STEP_SIZE    = int(WINDOW_SIZE * (1 - OVERLAP))
N_AXES       = 3
N_FEATURES   = 30             # 10 features x 3 axes
AXES         = ["acc_x", "acc_y", "acc_z"]

CONTAMINATION = 0.05
RANDOM_STATE  = 42

LABEL_MAP = {
    "normal":                 0,   # fan-on AND fan-off recordings live here
    "tapping":                1,   # light tapping
    "vibration":              2,   # surface vibration
    "load_variation":         3,
    "structural_disturbance": 4,
}

# Paths (relative to project root — run scripts from the root dir)
PROCESSED_PATH = "data/processed_windows.npz"
FEATURES_PATH  = "data/features.npz"
SPLIT_PATH     = "data/split.npz"
RESULTS_PATH   = "data/eval_results.npz"   # metrics the dashboard reads

MODEL_DIR  = "models"
IF_PATH    = "models/isolation_forest_model.pkl"
OCSVM_PATH = "models/one_class_svm_model.pkl"
LOF_PATH   = "models/lof_model.pkl"
AE_PATH    = "models/autoencoder_model.keras"
AE_THRES   = "models/autoencoder_threshold.npy"


# ─────────────────────────────────────────────
# RESAMPLING — fixes the 200-vs-414 Hz bug
# ─────────────────────────────────────────────
def resample_to_target(signal_2d, source_fs, target_fs=TARGET_FS):
    """signal_2d: (N, 3) array. Resamples each axis from source_fs to target_fs."""
    if abs(source_fs - target_fs) < 1e-6:
        return signal_2d
    frac = Fraction(target_fs / source_fs).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return np.column_stack([
        resample_poly(signal_2d[:, a], up, down) for a in range(signal_2d.shape[1])
    ])


def estimate_fs(time_col):
    """Estimate sampling rate (Hz) from a Phyphox time column (seconds)."""
    dt = np.median(np.diff(np.asarray(time_col, dtype=float)))
    return 1.0 / dt if dt > 0 else TARGET_FS


# ─────────────────────────────────────────────
# FEATURE EXTRACTION — 10 features x 3 axes = 30
# Order per axis (DO NOT REORDER — firmware mirrors this):
#   rms, std, kurtosis, skew, peak2peak,
#   dom_freq, spec_energy, spec_centroid, spec_entropy, crest_factor
# ─────────────────────────────────────────────
def extract_features_one_window(window, fs=TARGET_FS):
    """window: (WINDOW_SIZE, 3). Returns (30,) feature vector."""
    features = []
    for axis in range(N_AXES):
        signal = window[:, axis]

        # time domain
        rms  = np.sqrt(np.mean(signal ** 2))
        std  = np.std(signal)
        kurt = stats.kurtosis(signal)
        skew = stats.skew(signal)
        p2p  = np.max(signal) - np.min(signal)

        # frequency domain
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs    = np.fft.rfftfreq(len(signal), d=1.0 / fs)

        dom_freq    = freqs[np.argmax(fft_vals)]
        spec_energy = np.sum(fft_vals ** 2)
        spec_centroid = (np.sum(freqs * fft_vals) / np.sum(fft_vals)
                         if np.sum(fft_vals) > 0 else 0.0)

        psd = fft_vals ** 2
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spec_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        crest_factor = np.max(np.abs(signal)) / (rms + 1e-10)

        features.extend([
            rms, std, kurt, skew, p2p,
            dom_freq, spec_energy, spec_centroid, spec_entropy, crest_factor,
        ])
    return np.array(features)


def windows_from_signal(signal_2d):
    """Slice a (N,3) signal into overlapping windows -> (M, WINDOW_SIZE, 3)."""
    out = []
    n = len(signal_2d)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, N_AXES))
    for start in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        out.append(signal_2d[start:start + WINDOW_SIZE])
    return np.array(out)