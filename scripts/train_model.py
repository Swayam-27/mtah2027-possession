"""Train the final model on every external match and save it to models/possession.txt.

No competition data is read here: the model is trained only on the open DFL matches, each
degraded several times with different random noise settings.

Written to run on a small machine: training rows go straight to a file on disk, one match at a
time, and only the compact binned copy LightGBM builds from it is held in memory.

    python scripts/train_model.py
"""

import gc
import glob
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")
import lightgbm as lgb  # noqa: E402

import cv  # noqa: E402

COPIES = 2  # differently degraded copies of each match
MODEL_PATH = os.path.join(ROOT, "models", "possession.txt")
SCRATCH = os.path.join(ROOT, "cache", "train_rows.npy")


if __name__ == '__main__':
    paths = sorted(glob.glob(os.path.join(ROOT, "cache", "DFL-*.npz")))
    if not paths:
        raise SystemExit("no cached external matches; run scripts/prepare_external.py first")

    started = time.time()
    store, labels, names, filled = None, [], None, 0
    jobs = [(copy, i, path) for copy in range(COPIES) for i, path in enumerate(paths)]
    for copy, i, path in jobs:
        x, y, names, _ = cv.training_rows(*cv.prepare(path, seed=100 * copy + i))
        if store is None:
            # Generous upper bound on rows; trimmed when training starts.
            store = np.lib.format.open_memmap(
                SCRATCH, mode="w+", dtype=np.float32,
                shape=(int(len(y) * 1.25) * len(jobs), x.shape[1]))
        store[filled:filled + len(y)] = x
        labels.append(y)
        filled += len(y)
        del x, y
        gc.collect()
        print("  prepared {} copy {} ({:.0f} s)".format(os.path.basename(path), copy,
                                                        time.time() - started))
    store.flush()
    y = np.concatenate(labels)
    print("training on {} rows, {} features from {} matches x {} copies".format(
        len(y), len(names), len(paths), COPIES))

    dataset = lgb.Dataset(store[:filled], y, feature_name=names, free_raw_data=True)
    dataset.construct()
    del store
    gc.collect()
    booster = lgb.train(cv.PARAMS, dataset, cv.ROUNDS)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    booster.save_model(MODEL_PATH)
    os.remove(SCRATCH)
    print("saved {} ({:.0f} s)".format(MODEL_PATH, time.time() - started))
