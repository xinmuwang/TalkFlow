"""Three-stream flow training/inference data loaders.

Accepts legacy full-sequence PKL records and the processed BEAT2 NPZ files.
Field names returned match the supplied Stage-II training and strict inference code.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import (
    _as_numpy, _crop_start, _expression, _load_entry, _pose_rot6d,
    load_record_index, pad_sequence_lastdim,
)

FPS = 30


def _native_features(record: dict, names: tuple[str, ...], dim: int) -> tuple[np.ndarray, float | None]:
    for name in names:
        if name in record:
            x = np.asarray(record[name], dtype=np.float32)
            if x.ndim != 2 or x.shape[1] != dim or not len(x):
                raise ValueError(f"{name}: expected nonempty [T,{dim}], got {x.shape}")
            fps_key = name + "_fps"
            freq = record.get(fps_key)
            if freq is None:
                if name == "hubert_seq":
                    freq = record.get("hubert_fps")
                elif name == "librosa_seq":
                    freq = record.get("librosa_fps")
            return x, (float(np.asarray(freq).item()) if freq is not None else None)
    raise KeyError(f"Missing feature from {names}")


def resample_by_time(seq: np.ndarray, target_len: int, src_fps: float, target_fps: float = FPS) -> np.ndarray:
    """Time-based linear interpolation consistent with the original long-test loader."""
    if not (src_fps > 0 and target_fps > 0 and target_len > 0):
        raise ValueError("Positive frame rates and target length required")
    if len(seq) == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)
    loc = np.clip(np.arange(target_len, dtype=np.float64) * src_fps / target_fps, 0, len(seq) - 1)
    left = np.floor(loc).astype(np.int64)
    right = np.minimum(left + 1, len(seq) - 1)
    w = (loc - left).astype(np.float32)[:, None]
    return ((1 - w) * seq[left] + w * seq[right]).astype(np.float32)


def _aligned_feature(record: dict, names: tuple[str, ...], dim: int, T: int, fps: int) -> torch.Tensor:
    native, src_fps = _native_features(record, names, dim)
    if src_fps is None:
        # For already-aligned packed clips: no resampling should be needed.
        if len(native) != T:
            raise ValueError(f"Already-aligned feature {names} has {len(native)} frames, expected {T}")
        arr = native
    else:
        arr = resample_by_time(native, T, src_fps=src_fps, target_fps=fps)
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)


def _frame_text(record: dict, T: int, fps: int):
    """Return [T,300], [T] mask, [384]; preserve word timing in PKL inputs."""
    if "fasttext" in record or "text_local_raw" in record:
        key = "fasttext" if "fasttext" in record else "text_local_raw"
        local = np.asarray(record[key], dtype=np.float32)
        if local.shape != (T, 300):
            raise ValueError(f"Expected {key} [{T},300], got {local.shape}")
        if "text_local_mask" in record:
            mask = np.asarray(record["text_local_mask"], dtype=bool)
            if mask.shape != (T,):
                raise ValueError(f"Expected text_local_mask [{T}], got {mask.shape}")
        else:
            # Existing processed clips did not store word-presence masks.
            # Nonzero proxy is used only for these older caches.
            mask = np.any(local != 0, axis=-1)
        global_feat = record.get("sbert", record.get("text_global", record.get("text_global_raw")))
    else:
        vecs = np.asarray(record["text_words_vecs"], dtype=np.float32)
        starts = np.asarray(record["text_words_starts"], dtype=np.float32)
        ends = np.asarray(record["text_words_ends"], dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[1] != 300 or starts.ndim != 1 or ends.ndim != 1:
            raise ValueError("Invalid word-level FastText shape")
        if not len(vecs) == len(starts) == len(ends):
            raise ValueError("Word embeddings and timestamps have different lengths")
        local = np.zeros((T,300), dtype=np.float32)
        mask = np.zeros(T, dtype=np.bool_)
        for vec, st, ed in zip(vecs, starts, ends):
            # Original training/inference notebook's floor/ceil frame assignment.
            a = int(np.floor(float(st) * fps))
            b = int(np.ceil(float(ed) * fps))
            a = max(0, min(T, a))
            b = min(T, max(a + 1, b)) if a < T else T
            if b > a:
                local[a:b] = vec
                mask[a:b] = True
        global_feat = record.get("text_global", record.get("sbert"))
    if global_feat is None:
        raise KeyError("Missing text_global / sbert")
    glob = np.asarray(global_feat, dtype=np.float32).reshape(-1)
    if glob.shape != (384,):
        raise ValueError(f"Expected text-global [384], got {glob.shape}")
    return (
        torch.from_numpy(np.ascontiguousarray(local)),
        torch.from_numpy(np.ascontiguousarray(mask)),
        torch.from_numpy(np.ascontiguousarray(glob)),
    )


def _raw_len(record: dict, *, load_gt: bool) -> int:
    if load_gt:
        pose = record.get("poses", record.get("rot6d", record.get("pose")))
        expr = record.get("expressions", record.get("expression", record.get("expr")))
        if pose is not None and expr is not None:
            T = min(len(pose), len(expr), int(np.asarray(record.get("T", len(pose))).item()))
            # Original aligned-PKL training loader cropped all modalities to
            # their common valid frame count. Keep that behavior for older data.
            for key in ("hubert", "librosa", "librosa35"):
                if key in record and (key + "_fps") not in record:
                    T = min(T, len(record[key]))
            return T
    if "T" in record:
        return int(np.asarray(record["T"]).item())
    for key in ("librosa", "librosa35", "librosa_seq", "hubert", "hubert_seq"):
        if key in record:
            return len(record[key])
    return 0


def _speech_sample(record: dict, *, fps: int, load_gt: bool, require_gt: bool) -> dict:
    T = _raw_len(record, load_gt=load_gt)
    if T <= 0:
        raise ValueError("Cannot infer sequence length")
    h = _aligned_feature(record, ("hubert", "hubert_seq"), 1024, T, fps)
    l = _aligned_feature(record, ("librosa", "librosa35", "librosa_seq"), 35, T, fps)
    words, text_mask, global_ = _frame_text(record, T, fps)
    seq_id = record.get("seq_id", record.get("clip_id", "sample"))
    if isinstance(seq_id, np.ndarray):
        seq_id = str(seq_id.item())
    out = {
        "seq_id": str(seq_id),
        "hubert": h,
        "librosa": l,
        "text_local_raw": words,
        "text_local_mask": text_mask,
        "text_global_raw": global_,
        "seq_mask": torch.ones(T, dtype=torch.bool),
        "valid_mask": torch.ones(T, dtype=torch.bool),
        "length": int(T),
    }
    if load_gt or require_gt:
        out["pose"] = _pose_rot6d(record)[:T]
        out["expression"] = _expression(record)[:T]
        if "trans" in record or "trans_ref_only" in record:
            t = record.get("trans", record.get("trans_ref_only"))
            out["trans"] = torch.tensor(np.asarray(t, dtype=np.float32)[:T].copy())
        if "betas" in record:
            b = np.asarray(record["betas"], dtype=np.float32)
            if b.ndim == 2:
                b = b[0]
            out["betas"] = torch.as_tensor(np.ascontiguousarray(b.reshape(-1)))
    return out


class SpeechMotionDataset(Dataset):
    """Stage-II training loader with frame-aligned motion, acoustic and text features."""
    def __init__(self, pkl_path: str | Path, clip_len: Optional[int] = 240,
                 crop_mode: str = "random", fps: int = FPS,
                 min_len: Optional[int] = None, split: str = "train"):
        self.entries = load_record_index(pkl_path, split=split, kind="clips")
        self.clip_len, self.crop_mode, self.fps = clip_len, crop_mode, fps
        self.samples = []
        for entry in self.entries:
            try:
                if isinstance(entry, Path):
                    with np.load(entry, allow_pickle=False) as r:
                        n = _raw_len(r, load_gt=True)
                else:
                    n = _raw_len(entry, load_gt=True)
                if n >= (min_len if min_len is not None else 1):
                    self.samples.append(entry)
            except (OSError, ValueError, KeyError):
                continue
        if not self.samples:
            raise ValueError(f"No valid speech/motion samples in {pkl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = _load_entry(self.samples[idx])
        out = _speech_sample(rec, fps=self.fps, load_gt=True, require_gt=True)
        s,e = _crop_start(out["length"], self.clip_len, self.crop_mode)
        for k in ("pose","expression","hubert","librosa","text_local_raw","text_local_mask","seq_mask","valid_mask","trans"):
            if k in out:
                out[k] = out[k][s:e]
        out["length"] = e-s
        return out


def speech_motion_collate_fn(batch: list[dict]) -> dict:
    temporal = ("pose","expression","hubert","librosa","text_local_raw","text_local_mask","seq_mask","valid_mask")
    result = {}
    for k in temporal:
        if all(k in b for b in batch):
            x, m = pad_sequence_lastdim([b[k] for b in batch])
            result[k] = x
    result["seq_mask"] = torch.arange(result["hubert"].shape[1])[None] < torch.tensor([b["length"] for b in batch])[:,None]
    result["valid_mask"] = result["seq_mask"]
    result["text_global_raw"] = torch.stack([b["text_global_raw"] for b in batch])
    result["length"] = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    result["seq_id"] = [b["seq_id"] for b in batch]
    return result


def build_speech_dataloader(pkl_path: str | Path, batch_size: int,
                            shuffle: bool = True, clip_len: Optional[int] = 240,
                            crop_mode: str = "random", fps: int = FPS,
                            use_padding_collate: bool = True, split: str = "train",
                            num_workers: int = 0):
    ds = SpeechMotionDataset(pkl_path, clip_len, crop_mode, fps,
                             min_len=clip_len, split=split)
    return ds, DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=speech_motion_collate_fn if use_padding_collate else None,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)


class SpeechMotionLongTestDataset(Dataset):
    """Full-length loader for strict 3-stream inference; no training clip cropping."""
    def __init__(self, pkl_path: str | Path, fps: int = FPS,
                 load_gt: bool = True, split: str = "test"):
        self.fps = fps
        self.load_gt = load_gt
        self.entries = load_record_index(pkl_path, split=split, kind="sequences")
        self.samples = []
        for entry in self.entries:
            try:
                if isinstance(entry, Path):
                    with np.load(entry, allow_pickle=False) as rec:
                        n = _raw_len(rec, load_gt=load_gt)
                else:
                    n = _raw_len(entry, load_gt=load_gt)
                if n > 0:
                    self.samples.append(entry)
            except (OSError, ValueError, KeyError):
                continue
        if not self.samples:
            raise ValueError(f"No valid full sequences in {pkl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = _load_entry(self.samples[idx])
        return _speech_sample(rec, fps=self.fps, load_gt=self.load_gt, require_gt=False)


def long_test_collate_fn(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("Long inference requires batch_size=1")
    b = batch[0]
    result = {"seq_id": b["seq_id"], "length": torch.tensor([b["length"]], dtype=torch.long)}
    for k in ("hubert","librosa","text_local_raw","text_local_mask","text_global_raw","seq_mask","valid_mask", "pose","expression","trans","betas"):
        if k in b:
            result[k] = b[k].unsqueeze(0)
    return result


def build_long_test_dataloader(pkl_path: str | Path, batch_size: int = 1,
                               fps: int = FPS, load_gt: bool = True,
                               split: str = "test", num_workers: int = 0):
    if batch_size != 1:
        raise ValueError("Strict long-sequence inference supports batch_size=1")
    ds = SpeechMotionLongTestDataset(pkl_path, fps, load_gt, split)
    return ds, DataLoader(ds, batch_size=1, shuffle=False,
                          collate_fn=long_test_collate_fn, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available())
