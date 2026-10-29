"""Stage-I pose/expression dataset.

Compatible with the uploaded AE notebook's import/call signatures. Accepts
original full-sequence PKL records or directly packed BEAT2 NPZ clips.
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_record_index(path: str | Path, *, split: str = "train", kind: str = "clips") -> list[Any]:
    """Accept legacy PKL, one NPZ, an NPZ directory, or processed root/manifest."""
    p = Path(path)
    if p.suffix.lower() == ".pkl":
        with p.open("rb") as f:
            samples = pickle.load(f)
        if not isinstance(samples, list):
            raise TypeError(f"Expected a list of dictionaries in {p}")
        return samples
    if p.suffix.lower() == ".npz":
        return [p]
    if p.suffix.lower() == ".jsonl":
        root = p.parent.parent
        entries = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        key = "processed_path"
        if entries and key not in entries[0]:
            raise ValueError(f"Manifest {p} lacks {key}; run preprocessing pack step")
        return [root / e[key] for e in entries]
    if not p.is_dir():
        raise FileNotFoundError(f"Dataset not found: {p}")
    manifest_name = f"{split}_processed.jsonl" if kind == "clips" else f"{split}_sequence_processed.jsonl"
    manifest = p / "manifests" / manifest_name
    if manifest.exists():
        return load_record_index(manifest, split=split, kind=kind)
    for folder in [p / "clips" / split, p / "sequences" / split, p]:
        if folder.is_dir():
            paths = sorted(folder.glob("*.npz"))
            if paths:
                return paths
    raise FileNotFoundError(f"No {split} {kind} manifest or NPZ files found under {p}")


def _load_entry(entry: dict | Path) -> dict:
    if isinstance(entry, dict):
        return entry
    with np.load(entry, allow_pickle=False) as obj:
        return {k: obj[k] for k in obj.files}


def _pose_rot6d(record: dict) -> torch.Tensor:
    if "rot6d" in record:
        pose = torch.as_tensor(np.ascontiguousarray(_as_numpy(record["rot6d"])), dtype=torch.float32)
        if pose.ndim != 3 or pose.shape[1:] != (55, 6):
            raise ValueError(f"Expected rot6d [T,55,6], got {pose.shape}")
        return pose
    if "pose" in record and _as_numpy(record["pose"]).ndim == 3:
        pose = torch.as_tensor(np.ascontiguousarray(_as_numpy(record["pose"])), dtype=torch.float32)
        if pose.shape[1:] != (55, 6):
            raise ValueError(f"Expected pose [T,55,6], got {pose.shape}")
        return pose
    key = "poses" if "poses" in record else ("pose" if "pose" in record else None)
    if key is None:
        raise KeyError("Missing motion: expected 'poses' axis-angle or 'rot6d'")
    poses = _as_numpy(record[key]).astype(np.float32, copy=False)
    if poses.ndim != 2 or poses.shape[1] != 165:
        raise ValueError(f"Expected poses [T,165], got {poses.shape}")
    x = torch.as_tensor(np.ascontiguousarray(poses), dtype=torch.float32).reshape(-1, 55, 3)
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))


def _expression(record: dict) -> torch.Tensor:
    for key in ("expressions", "expression", "expr"):
        if key in record:
            x = torch.as_tensor(np.ascontiguousarray(_as_numpy(record[key])), dtype=torch.float32)
            if x.ndim != 2 or x.shape[-1] != 100:
                raise ValueError(f"Expected expression [T,100], got {x.shape}")
            return x
    raise KeyError("Missing expressions/expression/expr")


def _get_length(record: dict) -> int:
    if "poses" in record:
        n = len(record["poses"])
    elif "rot6d" in record:
        n = len(record["rot6d"])
    elif "pose" in record:
        n = len(record["pose"])
    else:
        return 0
    for key in ("expressions", "expression", "expr"):
        if key in record:
            return min(n, len(record[key]))
    return 0


def _crop_start(T: int, clip_len: Optional[int], crop_mode: str) -> tuple[int, int]:
    if clip_len is None or T <= clip_len:
        return 0, T
    if crop_mode == "random":
        s = random.randint(0, T - clip_len)
    elif crop_mode == "center":
        s = (T - clip_len) // 2
    elif crop_mode == "start":
        s = 0
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")
    return s, s + clip_len


class PoseExprLatentDataset(Dataset):
    """Motion-only loader returning `pose`, `expression`, `length` and optional metadata."""

    def __init__(
        self,
        pkl_path: str | Path,
        clip_len: Optional[int] = None,
        crop_mode: str = "random",
        poses_key: str = "poses",
        expr_key: str = "expressions",
        betas_key: str = "betas",
        trans_key: str = "trans",
        return_betas: bool = False,
        return_trans: bool = False,
        min_len: Optional[int] = None,
        split: str = "train",
    ):
        self.path = Path(pkl_path)
        self.crop_mode = crop_mode
        self.clip_len = clip_len
        self.poses_key = poses_key
        self.expr_key = expr_key
        self.betas_key = betas_key
        self.trans_key = trans_key
        self.return_betas = return_betas
        self.return_trans = return_trans
        entries = load_record_index(pkl_path, split=split)
        self.samples = []
        for entry in entries:
            try:
                if isinstance(entry, dict):
                    n = _get_length(entry)
                else:
                    with np.load(entry, allow_pickle=False) as d:
                        n = _get_length(d)
                if n > 0 and (min_len is None or n >= min_len):
                    self.samples.append(entry)
            except (OSError, ValueError, KeyError):
                continue
        if not self.samples:
            raise ValueError(f"No valid pose/expression samples in {pkl_path} (split={split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rec = _load_entry(self.samples[idx])
        pose = _pose_rot6d(rec)
        expression = _expression(rec)
        n = min(len(pose), len(expression))
        s, e = _crop_start(n, self.clip_len, self.crop_mode)
        out = {"pose": pose[s:e], "expression": expression[s:e], "length": e - s}
        if self.return_betas and self.betas_key in rec:
            b = torch.as_tensor(np.ascontiguousarray(_as_numpy(rec[self.betas_key])), dtype=torch.float32)
            if b.ndim == 2:
                b = b[0]
            out["betas"] = b.reshape(-1)
        trkey = self.trans_key if self.trans_key in rec else ("trans_ref_only" if "trans_ref_only" in rec else None)
        if self.return_trans and trkey is not None:
            trans = torch.as_tensor(np.ascontiguousarray(_as_numpy(rec[trkey])), dtype=torch.float32)
            out["trans"] = trans[s:e]
        return out


def pad_sequence_lastdim(seqs: list[torch.Tensor], pad_value: float = 0.0):
    if not seqs:
        raise ValueError("Cannot pad empty batch")
    tmax = max(s.shape[0] for s in seqs)
    padded = seqs[0].new_full((len(seqs), tmax) + tuple(seqs[0].shape[1:]), pad_value)
    mask = torch.zeros(len(seqs), tmax, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
        mask[i, :len(s)] = True
    return padded, mask


def pose_expr_collate_fn(batch: list[dict]) -> dict:
    pose, mask = pad_sequence_lastdim([b["pose"] for b in batch])
    expression, _ = pad_sequence_lastdim([b["expression"] for b in batch])
    out = {
        "pose": pose,
        "expression": expression,
        "length": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "seq_mask": mask,
        "valid_mask": mask,
    }
    for key in ("betas", "trans"):
        if all(key in b for b in batch):
            if key == "trans":
                out[key] = pad_sequence_lastdim([b[key] for b in batch])[0]
            else:
                out[key] = torch.stack([b[key] for b in batch])
    return out


def build_dataloader(
    pkl_path: str | Path,
    batch_size: int,
    shuffle: bool = True,
    clip_len: Optional[int] = 240,
    crop_mode: str = "random",
    return_betas: bool = False,
    return_trans: bool = False,
    use_padding_collate: bool = True,
    split: str = "train",
    num_workers: int = 0,
    drop_last: Optional[bool] = None,
):
    dataset = PoseExprLatentDataset(
        pkl_path=pkl_path, clip_len=clip_len, crop_mode=crop_mode,
        return_betas=return_betas, return_trans=return_trans,
        min_len=clip_len if clip_len is not None else None, split=split,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle if drop_last is None else drop_last,
        collate_fn=pose_expr_collate_fn if use_padding_collate else None,
    )
    return dataset, loader
