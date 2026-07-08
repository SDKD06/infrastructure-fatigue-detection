"""
train_autoencoder.py  —  Conv1D autoencoder on TRAIN-split normal windows only.

NOTE for the paper: with very few normal recordings this network will tend to
overfit. That's expected and is part of the story — it justifies preferring the
classical detectors (IF/LOF) in this small-data regime. Report it honestly.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from feature_utils import (
    PROCESSED_PATH, SPLIT_PATH, MODEL_DIR, AE_PATH, AE_THRES,
    WINDOW_SIZE, N_AXES, RANDOM_STATE,
)

EPOCHS, BATCH_SIZE, LR = 50, 32, 1e-3
INPUT_SHAPE = (WINDOW_SIZE, N_AXES)


def build_autoencoder(input_shape):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv1D(16, 7, strides=2, padding="same", activation="relu")(x)
    enc = layers.Conv1D(8, 7, strides=2, padding="same", activation="relu",
                        name="bottleneck")(x)
    x = layers.Conv1DTranspose(16, 7, strides=2, padding="same", activation="relu")(enc)
    x = layers.Conv1DTranspose(32, 7, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv1DTranspose(N_AXES, 7, strides=2, padding="same",
                                 activation="linear")(x)
    return keras.Model(inp, out, name="conv_autoencoder")


def main():
    print("=" * 56)
    print("CONV1D AUTOENCODER TRAINING (train-normal only)")
    print("=" * 56)

    tf.random.set_seed(RANDOM_STATE)
    d = np.load(PROCESSED_PATH, allow_pickle=True)
    X, y, groups = d["windows"], d["labels"], d["groups"]
    split = np.load(SPLIT_PATH)
    train_groups = split["train_groups"].tolist()

    mask = np.isin(groups, train_groups) & (y == 0)
    X_norm = X[mask].astype("float32")
    print(f"{len(X_norm)} normal training windows.")
    if len(X_norm) < 20:
        raise ValueError("Not enough normal training windows for the autoencoder.")

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(X_norm))
    X_norm = X_norm[idx]
    cut = int(len(X_norm) * 0.8)
    X_tr, X_val = X_norm[:cut], X_norm[cut:]

    model = build_autoencoder(INPUT_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse", metrics=["mae"])
    model.fit(X_tr, X_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, X_val), verbose=1)

    val_pred = model.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - val_pred), axis=(1, 2))
    threshold = float(np.percentile(mse, 98))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(AE_PATH)
    np.save(AE_THRES, threshold)
    print(f"Threshold (98th pct val error): {threshold:.6f}")
    print(f"saved -> {AE_PATH}")


if __name__ == "__main__":
    main()