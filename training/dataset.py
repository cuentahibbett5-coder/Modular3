from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    input_mhd: Path
    target_mhd: Path
    pair_id: int
    level: str
    target_id: int


def _read_mhd(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    return arr.astype(np.float32, copy=False)


def _center_crop(volume: np.ndarray, patch: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = volume.shape
    pz, py, px = patch
    sz = max((z - pz) // 2, 0)
    sy = max((y - py) // 2, 0)
    sx = max((x - px) // 2, 0)
    return volume[sz:sz + pz, sy:sy + py, sx:sx + px]


def _random_crop_indices(
    shape: Tuple[int, int, int],
    patch: Tuple[int, int, int],
    rng: np.random.RandomState,
) -> Tuple[int, int, int]:
    z, y, x = shape
    pz, py, px = patch
    sz = rng.randint(0, max(z - pz + 1, 1))
    sy = rng.randint(0, max(y - py + 1, 1))
    sx = rng.randint(0, max(x - px + 1, 1))
    return sz, sy, sx


def _crop_at(volume: np.ndarray, start: Tuple[int, int, int], patch: Tuple[int, int, int]) -> np.ndarray:
    sz, sy, sx = start
    pz, py, px = patch
    return volume[sz:sz + pz, sy:sy + py, sx:sx + px]


def _parse_pair_id(pair_name: str) -> int:
    return int(pair_name.split("_")[-1])


def _target_for_pair(pair_id: int, n_targets: int) -> int:
    # 5 pares por target: 1-5 -> target_1, 6-10 -> target_2, etc.
    idx = ((pair_id - 1) // 5) + 1
    return min(max(idx, 1), n_targets)


class DosePairDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        levels: Sequence[str] = ("input_1M", "input_2M", "input_5M", "input_10M"),
        patch_size: Tuple[int, int, int] | None = (64, 64, 64),
        cache_size: int = 4,
        normalize: bool = True,
        seed: int = 1234,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.levels = list(levels)
        self.patch_size = patch_size
        self.normalize = normalize
        self.rng = np.random.RandomState(seed)

        self.targets = sorted(self.root_dir.glob("target_*"))
        if not self.targets:
            raise FileNotFoundError(f"No se encontraron targets en {self.root_dir}")
        self.n_targets = len(self.targets)

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"No existe split: {split_dir}")

        self.samples: List[Sample] = []
        for pair_dir in sorted(split_dir.glob("pair_*")):
            pair_id = _parse_pair_id(pair_dir.name)
            target_id = _target_for_pair(pair_id, self.n_targets)
            target_mhd = self.root_dir / f"target_{target_id}" / "dose_edep.mhd"
            if not target_mhd.exists():
                continue
            for level in self.levels:
                input_mhd = pair_dir / level / "dose_edep.mhd"
                if input_mhd.exists():
                    self.samples.append(Sample(input_mhd, target_mhd, pair_id, level, target_id))

        if not self.samples:
            raise FileNotFoundError("No se encontraron pares con dose_edep.mhd")

        self.cache_size = max(cache_size, 0)
        self._cache: Dict[Path, np.ndarray] = {}
        self._cache_keys: List[Path] = []

    def __len__(self) -> int:
        return len(self.samples)

    def _get_volume(self, path: Path) -> np.ndarray:
        if self.cache_size == 0:
            return _read_mhd(path)
        if path in self._cache:
            return self._cache[path]
        vol = _read_mhd(path)
        self._cache[path] = vol
        self._cache_keys.append(path)
        if len(self._cache_keys) > self.cache_size:
            oldest = self._cache_keys.pop(0)
            self._cache.pop(oldest, None)
        return vol

    def _normalize_pair(self, inp: np.ndarray, tgt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.normalize:
            return inp, tgt
        max_val = float(np.max(tgt))
        if max_val <= 0:
            max_val = 1.0
        return inp / max_val, tgt / max_val

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        inp = self._get_volume(sample.input_mhd)
        tgt = self._get_volume(sample.target_mhd)

        inp, tgt = self._normalize_pair(inp, tgt)

        if self.patch_size is not None:
            if self.split == "train":
                start = _random_crop_indices(inp.shape, self.patch_size, self.rng)
                inp = _crop_at(inp, start, self.patch_size)
                tgt = _crop_at(tgt, start, self.patch_size)
            else:
                inp = _center_crop(inp, self.patch_size)
                tgt = _center_crop(tgt, self.patch_size)

        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, Z, Y, X)
        tgt = torch.from_numpy(tgt).unsqueeze(0)

        return {
            "input": inp,
            "target": tgt,
            "pair_id": torch.tensor(sample.pair_id, dtype=torch.int64),
            "target_id": torch.tensor(sample.target_id, dtype=torch.int64),
        }
