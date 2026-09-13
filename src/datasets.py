"""Torch dataset + fast in-memory augmentation (no albumentations / no worker spawn).

Key rule (fix #4): the horizontal mirror / flip and every other augmentation are
applied to TRAIN samples only. Validation and test samples are the plain
preprocessed image (resize -> 3ch -> ImageNet norm) -- no test-set mirror
inflation, no train/test correlation through flips.

3-stage augmentation (manuscript 3.2.2), train only:
  stage 1  geometric (rotate/translate/scale) + brightness/contrast, all classes
  stage 2  wider contrast + gamma for the minority (non-urgent) class
  stage 3  additive Gaussian noise, sub-sampled
The dataset is tiny (~230 train images) so everything is decoded once into RAM
and num_workers = 0; an epoch is GPU-bound (~5 s).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from src.common import imread_gray

_MEAN = np.array(C.IMAGENET_MEAN, np.float32)
_STD = np.array(C.IMAGENET_STD, np.float32)


@lru_cache(maxsize=512)
def _load_resized(path: str, size: int) -> np.ndarray:
    """decoded, resized grayscale uint8 -- cached across folds/trials."""
    g = imread_gray(path)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def _to_tensor(gray_u8: np.ndarray) -> torch.Tensor:
    x = np.repeat(gray_u8[:, :, None].astype(np.float32) / 255.0, 3, axis=2)
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1)).contiguous()


def _augment(g: np.ndarray, rng: np.random.Generator, *, minority: bool) -> np.ndarray:
    h, w = g.shape
    # -- stage 1: geometry -------------------------------------------------
    if rng.random() < 0.5:                                   # horizontal mirror
        g = g[:, ::-1]
    ang = rng.uniform(-5, 5)
    scale = rng.uniform(0.88, 1.12)
    tx, ty = rng.uniform(-0.08, 0.08, size=2) * [w, h]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[:, 2] += (tx, ty)
    g = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # -- stage 1: photometric -------------------------------------------------
    g = g.astype(np.float32)
    g = g * rng.uniform(0.94, 1.06) + rng.uniform(-6, 6)     # brightness
    mean = g.mean()
    g = (g - mean) * rng.uniform(0.94, 1.06) + mean          # contrast
    # -- stage 2: minority-only extras --------------------------------------
    if minority:
        if rng.random() < 0.5:
            mean = g.mean()
            g = (g - mean) * rng.uniform(0.8, 1.2) + mean
        if rng.random() < 0.3:
            gamma = rng.uniform(0.8, 1.2)
            g = 255.0 * np.clip(g / 255.0, 0, 1) ** gamma
    # -- stage 3: noise ----------------------------------------------------
    if rng.random() < (0.5 if minority else 0.3):
        g = g + rng.normal(0, rng.uniform(3, 10), size=g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


class UrgencyDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, patient_ids, *, train: bool, seed: int = 0):
        self.df = manifest.set_index("patient_id").loc[list(patient_ids)].reset_index()
        self.train = train
        self.size = C.MODEL_INPUT
        self._epoch = 0
        self._seed = seed

    def set_epoch(self, e: int) -> None:
        self._epoch = e

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> np.ndarray:
        return self.df["urgency_int"].to_numpy()

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        g = _load_resized(r["preproc_path"], self.size)
        if self.train:
            rng = np.random.default_rng((self._seed, self._epoch, i))
            g = _augment(g, rng, minority=(r["urgency"] == C.MINORITY_URGENCY))
        return _to_tensor(g), int(r["urgency_int"]), r["patient_id"]


def loaders(manifest, train_ids, eval_ids, batch_size: int, *, seed: int):
    from torch.utils.data import DataLoader, WeightedRandomSampler

    tr = UrgencyDataset(manifest, train_ids, train=True, seed=seed)
    ev = UrgencyDataset(manifest, eval_ids, train=False, seed=seed)

    y = tr.labels
    class_count = np.bincount(y, minlength=2).astype(np.float64)
    w = 1.0 / np.clip(class_count, 1, None)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(w[y], dtype=torch.double),
                                    num_samples=len(y), replacement=True, generator=g)
    dl_tr = DataLoader(tr, batch_size=batch_size, sampler=sampler, num_workers=0,
                       pin_memory=True, drop_last=len(tr) > batch_size)
    dl_ev = DataLoader(ev, batch_size=max(batch_size, 16), shuffle=False,
                       num_workers=0, pin_memory=True)
    return dl_tr, dl_ev, class_count
