import os
import glob
import argparse
import numpy as np
import pandas as pd
import cdflib
from typing import List, Tuple

import tensorflow as tf


def preprocess_distributions(X: np.ndarray) -> np.ndarray:
    """
    Same preprocessing used during training:
    - Replace zeros with minimum positive value per sample
    - Log10 transform
    - Min-max normalize per-sample across all voxels
    - Add channel dimension at the end
    """
    X = X.copy()
    for i in range(X.shape[0]):
        sample = X[i]
        pos = sample[sample > 0]
        if pos.size > 0:
            min_pos = np.min(pos)
            sample[sample == 0] = min_pos
            X[i] = sample
        else:
            X[i] = np.where(sample == 0, 1e-12, sample)

    X = np.log10(X)
    X_min = X.min(axis=(1, 2, 3), keepdims=True)
    X_max = X.max(axis=(1, 2, 3), keepdims=True)
    denom = (X_max - X_min)
    denom[denom == 0] = 1.0
    X = (X - X_min) / denom
    return X[..., np.newaxis]


def _match_epochs_and_indices(src_epochs, target_epochs) -> Tuple[List[int], List[int]]:
    """Return two lists: (indices_into_src, matched_target_indices).
    The order follows target_epochs so the matched target indices can be used to select
    labels and epochs from the sampled file.
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
    Load (X_raw, y) from minimal sampled CDFs and corresponding original CDFs.
    """
    sampled_paths = sorted(glob.glob(os.path.join(sampled_dir, '*.cdf')))
    if not sampled_paths:
        raise FileNotFoundError(f'No CDF files found under {sampled_dir}')

    parent_dir = os.path.dirname(sampled_dir.rstrip('\\/'))

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    epochs_list: List[np.ndarray] = []

    for spath in sampled_paths:
        fname = os.path.basename(spath)
        src_path = os.path.join(parent_dir, fname)
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

        # Read original file
        o_cdf = cdflib.CDF(src_path)
        try:
            o_epochs = o_cdf.varget('Epoch')
            dist = o_cdf.varget('mms1_dis_dist_fast')
        finally:
            try:
                o_cdf.close()
            except Exception:
                pass

        idx_src, idx_tgt = _match_epochs_and_indices(o_epochs, s_epochs)
        if not idx_src:
            print(f"[warn] No matching epochs in original for {fname}; skipping.")
            continue

        dist = np.asarray(dist)
        if dist.ndim != 4 or dist.shape[1:] != (32, 16, 32):
            print(f"[warn] Unexpected dist shape {dist.shape} in {fname}; expected (T,32,16,32). Skipping.")
            continue

        X_part = np.asarray(dist)[np.asarray(idx_src, dtype=int)]
        y_arr = np.asarray(s_labels, dtype=np.int32).ravel()
        epochs_arr = np.asarray(s_epochs).ravel()
        y_part = y_arr[np.asarray(idx_tgt, dtype=int)]
        epochs_part = epochs_arr[np.asarray(idx_tgt, dtype=int)]

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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description='Evaluate MMS 3D CNN on sampled CDFs')
    parser.add_argument('--sampled-dir', type=str,
                        default=r'D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2017\12\sampled',
                        help='Directory containing minimal sampled CDFs')
    parser.add_argument('--model-path', type=str,
                        default=r'D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2017\11\sampled\models\mms_cnn.keras',
                        help='Path to saved Keras model (.keras or .h5)')
    parser.add_argument('--features-csv', type=str, default=r"C:\Users\Meet Modi\Desktop\temporal_model\ootb\mms_features_for_training_201712.csv")
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    print('[stage] Loading dataset...')
    X_raw, y_true, epochs = load_dataset(args.sampled_dir)
    print(f"[stage] Loaded X_raw: {X_raw.shape}, y: {y_true.shape}, epochs: {epochs.shape}")

    print('[stage] Preprocessing...')
    X = preprocess_distributions(X_raw)

    # Optional: load engineered features and align to epochs
    features_csv = args.features_csv if args.features_csv and os.path.exists(args.features_csv) else None
    F = None
    feature_cols = ['beta', 'entropy', 'n_i', 'n_e', 'T_para_i', 'T_perp_i', 'T_para_e', 'T_perp_e']
    if features_csv:
        df_feat = pd.read_csv(features_csv)
        if 'epoch' not in df_feat.columns:
            raise RuntimeError("Features CSV must include an 'epoch' column for alignment. Regenerate CSV including epoch.")
        df_feat_indexed = df_feat.set_index('epoch')
        F = df_feat_indexed.reindex(epochs.astype(np.int64))[feature_cols].to_numpy(dtype=float)
        valid = ~np.isnan(F).any(axis=1)
        before = X.shape[0]
        X = X[valid]
        y_true = y_true[valid]
        F = F[valid]
        after = X.shape[0]
        print(f"[stage] Dropped {before - after} samples due to missing feature rows; kept {after}.")

    print(f"[stage] Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    print('[stage] Predicting...')
    # Basic sanity checks to avoid Keras progbar math errors when there are 0 samples
    n_samples = X.shape[0]
    if n_samples == 0:
        raise RuntimeError('[error] No samples available after preprocessing/alignment; aborting.')

    if F is not None:
        if F.shape[0] != n_samples:
            raise RuntimeError(f"[error] Feature matrix row count ({F.shape[0]}) does not match X samples ({n_samples}).")

    # Call predict. If dataset is small this avoids Keras progbar ValueError by ensuring target>0.
    if F is None:
        y_prob = model.predict(X, batch_size=args.batch_size, verbose=1)
    else:
        # model may expect 2 inputs
        y_prob = model.predict([X, F], batch_size=args.batch_size, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    acc = float(np.mean(y_pred == y_true)) if y_true.size else 0.0
    num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), num_classes)

    print('\n=== Evaluation Results ===')
    print(f'Accuracy: {acc:.4f}')
    print('Confusion matrix (rows=true, cols=pred):')
    for r in range(num_classes):
        print(' '.join(f'{int(v):5d}' for v in cm[r]))

    # Optional: per-class accuracy
    print('\nPer-class accuracy:')
    for c in range(num_classes):
        mask = (y_true == c)
        denom = int(np.sum(mask))
        if denom == 0:
            print(f'  class {c}: n=0')
            continue
        num = int(np.sum(y_pred[mask] == c))
        print(f'  class {c}: {num}/{denom} = {num/denom:.4f}')


if __name__ == '__main__':
    main()
