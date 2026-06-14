import os
import glob
import argparse
import numpy as np
import pandas as pd
import cdflib
from typing import List, Tuple

# Optional: if TensorFlow isn't installed, install it in your environment.
# import subprocess; subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])

import tensorflow as tf
from tensorflow.keras import layers, models


def preprocess_distributions(X: np.ndarray) -> np.ndarray:
    """
    Preprocess distribution arrays as provided by the user:
    - Replace zeros with minimum positive value per sample
    - Log10 transform
    - Min-max normalize per-sample across all voxels
    - Add channel dimension at the end

    X: (N, 32, 16, 32)
    returns: (N, 32, 16, 32, 1)
    """
    X = X.copy()
    # Replace zeros with the minimum positive value in the entire dataset or per-sample
    # Here we do per-sample to follow the provided function semantics
    for i in range(X.shape[0]):
        sample = X[i]
        pos = sample[sample > 0]
        if pos.size > 0:
            min_pos = np.min(pos)
            sample[sample == 0] = min_pos
            X[i] = sample
        else:
            # If a sample has no positive values, set tiny epsilon to avoid -inf
            X[i] = np.where(sample == 0, 1e-12, sample)

    X = np.log10(X)
    # Min-max normalize per-sample
    X_min = X.min(axis=(1, 2, 3), keepdims=True)
    X_max = X.max(axis=(1, 2, 3), keepdims=True)
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0
    X = (X - X_min) / denom
    return X[..., np.newaxis]


def build_mms_cnn(input_shape=(32, 16, 32, 1), num_classes=4, feature_dim: int = 0) -> tf.keras.Model:
    """
    3D CNN for MMS ion distribution classification.

    input_shape : tuple
        Shape of the input sample (32 energy, 16 theta, 32 phi, 1 channel).
    num_classes : int
        Number of output classes (default = 4: SW, IF, MSH, MSP).

    Returns:
        A compiled Keras model.
    """
    # CNN branch
    inp = layers.Input(shape=input_shape, name='dist_input')
    x = layers.Conv3D(
        filters=32, kernel_size=(5, 3, 5), strides=(2, 1, 2),
        activation='relu', padding='valid')(inp)
    x = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='valid')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Optional scalar features branch
    if feature_dim and feature_dim > 0:
        
        f_in = layers.Input(shape=(feature_dim,), name='feat_input')
        f = layers.Dense(32, activation='relu')(f_in)
        f = layers.Dense(16, activation='relu')(f)
        cat = layers.concatenate([x, f])
        h = layers.Dense(128, activation='relu')(cat)
        out = layers.Dense(num_classes, activation='softmax')(h)
        model = tf.keras.Model(inputs=[inp, f_in], outputs=out)
    else:
        out = layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def _match_epochs_and_indices(src_epochs, target_epochs) -> Tuple[List[int], List[int]]:
    """Return two lists: (indices_into_src, matched_target_indices).
    The lists are ordered according to target_epochs (i.e. the nth returned src index
    corresponds to target_epochs[matched_target_indices[n]]).
    """
    src_list = np.asarray(src_epochs).ravel().tolist()
    idx_map = {}
    for i, v in enumerate(src_list):
        idx_map.setdefault(v, []).append(i)
    src_indices: List[int] = []
    tgt_indices: List[int] = []
    for j, v in enumerate(np.asarray(target_epochs).ravel().tolist()):
        if v in idx_map:
            for si in idx_map[v]:
                src_indices.append(si)
                tgt_indices.append(j)
    return src_indices, tgt_indices


def load_dataset(sampled_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X_raw, y) by reading minimal sampled CDFs for labels/Epochs and
    pulling distributions from the corresponding original CDFs.

    sampled_dir: path to folder with minimal CDFs (Epoch, label, probability, energy, theta, phi)

    Returns: (X_raw, y)
      - X_raw shape: (N_total, 32, 16, 32)
      - y shape: (N_total,) integer labels 0..3
    """
    sampled_paths = sorted(glob.glob(os.path.join(sampled_dir, '*.cdf')))
    if not sampled_paths:
        raise FileNotFoundError(f'No CDF files found under {sampled_dir}')

    parent_dir = os.path.dirname(sampled_dir.rstrip('\\/'))  # original files live here

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    epochs_list: List[np.ndarray] = []

    for spath in sampled_paths:
        fname = os.path.basename(spath)
        src_path = os.path.join(parent_dir, fname)  # original file
        if not os.path.exists(src_path):
            print(f"[warn] Original file missing for {fname}; skipping.")
            continue

        # Read sampled file
        s_cdf = cdflib.CDF(spath)
        try:
            s_epochs = s_cdf.varget('Epoch')
            s_labels = s_cdf.varget('label')
        finally:
            try:
                s_cdf.close()
            except Exception:
                pass

        # Read original file to get distribution and original epochs
        o_cdf = cdflib.CDF(src_path)
        try:
            o_epochs = o_cdf.varget('Epoch')
            dist = o_cdf.varget('mms1_dis_dist_fast')  # expected shape (T, 32, 16, 32)
        finally:
            try:
                o_cdf.close()
            except Exception:
                pass

        # Find indices in original matching sampled epochs
        idx_src, idx_tgt = _match_epochs_and_indices(o_epochs, s_epochs)
        if not idx_src:
            print(f"[warn] No matching epochs in original for {fname}; skipping.")
            continue

        dist = np.asarray(dist)
        if dist.ndim != 4 or dist.shape[1:] != (32, 16, 32):
            print(f"[warn] Unexpected dist shape {dist.shape} in {fname}; expected (T,32,16,32). Skipping.")
            continue

        X_part = dist[np.asarray(idx_src, dtype=int)]
        y_arr = np.asarray(s_labels, dtype=np.int32).ravel()
        epochs_arr = np.asarray(s_epochs).ravel()
        y_part = y_arr[np.asarray(idx_tgt, dtype=int)]
        epochs_part = epochs_arr[np.asarray(idx_tgt, dtype=int)]

        # Align lengths just in case
        n = min(X_part.shape[0], y_part.shape[0], epochs_part.shape[0])
        if X_part.shape[0] != y_part.shape[0] or X_part.shape[0] != epochs_part.shape[0]:
            X_part = X_part[:n]
            y_part = y_part[:n]
            epochs_part = epochs_part[:n]

        X_list.append(X_part)
        y_list.append(y_part)
        epochs_list.append(epochs_part)
        print(f"[info] Loaded {fname}: {X_part.shape[0]} samples")

    if not X_list:
        raise RuntimeError('No data loaded. Check sampled_dir and original files.')

    X_raw = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    epochs_all = np.concatenate(epochs_list, axis=0)
    return X_raw, y, epochs_all


def main():
    parser = argparse.ArgumentParser(description='Train MMS 3D CNN with optional scalar features')
    parser.add_argument('--sampled-dir', default=r"D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2017\11\sampled")
    parser.add_argument('--features-csv', default=r"C:\Users\Meet Modi\Desktop\temporal_model\ootb\mms_features_for_training_201711.csv")
    parser.add_argument('--model-out', default=None, help='Optional output model path (overrides sampled-dir/models)')
    args = parser.parse_args()

    sampled_dir = args.sampled_dir
    features_csv = args.features_csv if args.features_csv and os.path.exists(args.features_csv) else None

    print('[stage] Loading dataset...')
    X_raw, y, epochs = load_dataset(sampled_dir)
    print(f"[stage] Loaded X_raw: {X_raw.shape}, y: {y.shape}, epochs: {epochs.shape}")

    print('[stage] Preprocessing...')
    X = preprocess_distributions(X_raw)
    num_classes = int(np.max(y) + 1) if y.size > 0 else 4
    num_classes = max(num_classes, 4)  # ensure at least 4
    y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # Load engineered scalar features if provided
    F = None
    feature_cols = ['beta', 'entropy', 'n_i', 'n_e', 'T_para_i', 'T_perp_i', 'T_para_e', 'T_perp_e']
    if features_csv:
        print(f"[stage] Loading features CSV from {features_csv}...")
        df_feat = pd.read_csv(features_csv)
        if 'epoch' not in df_feat.columns:
            raise RuntimeError("Features CSV must include an 'epoch' column for alignment. Regenerate CSV including epoch.")
        if not set(feature_cols).issubset(set(df_feat.columns)):
            missing = set(feature_cols) - set(df_feat.columns)
            raise RuntimeError(f"Features CSV missing required columns: {missing}")
        df_feat_indexed = df_feat.set_index('epoch')
        epochs_index = epochs.astype(np.int64)
        F = df_feat_indexed.reindex(epochs_index)[feature_cols].to_numpy(dtype=float)
        valid = ~np.isnan(F).any(axis=1)
        before = X.shape[0]
        X = X[valid]
        y_cat = y_cat[valid]
        epochs = epochs[valid]
        F = F[valid]
        after = X.shape[0]
        print(f"[stage] Dropped {before - after} samples due to missing feature rows; kept {after}.")

    # Simple train/val split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y_cat[tr_idx], y_cat[va_idx]

    print('[stage] Building model...')
    feat_dim = F.shape[1] if F is not None else 0
    model = build_mms_cnn(input_shape=X.shape[1:], num_classes=num_classes, feature_dim=feat_dim)
    model.summary()

    print('[stage] Training...')
    if F is None:
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=50,
            batch_size=32,
            verbose=1
        )
    else:
        print("f")
        F_tr, F_va = F[tr_idx], F[va_idx]
        history = model.fit(
            [X_tr, F_tr], y_tr,
            validation_data=([X_va, F_va], y_va),
            epochs=50,
            batch_size=32,
            verbose=1
        )

    if args.model_out:
        model_path = args.model_out
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    else:
        out_dir = os.path.join(sampled_dir, 'models')
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, 'mms_cnn.keras')
    model.save(model_path)
    print(f"[done] Model saved to {model_path}")


if __name__ == '__main__':
    main()
