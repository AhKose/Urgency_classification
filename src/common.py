"""Shared helpers: seeding, hashing, small utilities."""
from __future__ import annotations
import hashlib
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # benchmark=True (not fully deterministic on GPU convs) is a deliberate
        # speed choice for this feasibility study; data order & init are seeded.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def imread_gray(path: str | Path) -> "np.ndarray":
    """cv2.imread that tolerates non-ASCII Windows paths."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def imread_color(path: str | Path) -> "np.ndarray":
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(path)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def imwrite(path: str | Path, img: "np.ndarray") -> None:
    """cv2.imwrite that tolerates non-ASCII Windows paths."""
    import cv2

    path = Path(path)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"could not encode image for {path}")
    buf.tofile(str(path))


def list_pngs(folder: str | Path) -> list[str]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (f.name for f in folder.iterdir()
         if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}),
        key=lambda n: (len(n), n),
    )
