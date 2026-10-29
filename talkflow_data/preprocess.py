"""TalkFlow / BEAT2 preprocessing primitives.

Data layout is the *official downloaded BEAT2 English release*:
    beat_english_v2.0.0/{smplxflame_30,wave16k,textgrid,train_test_split.csv}

All paths stored in manifests are relative to --data-root or --out-dir.
Raw BEAT2 files are read but never modified.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

FPS = 30
AUDIO_SR = 16000
CLIP_FRAMES = 240
OVERLAP_FRAMES = 60
STRIDE_FRAMES = CLIP_FRAMES - OVERLAP_FRAMES
HUBERT_DIM = 1024
LIBROSA_DIM = 35
WORD_DIM = 300
SENT_DIM = 384

# Official BEAT2 / SMPL-X joint order in `poses`: 55 axis-angle rotations.
JOINT_SLICES = {
    "global_orient": slice(0, 1),
    "body": slice(1, 22),
    "jaw": slice(22, 23),
    "left_eye": slice(23, 24),
    "right_eye": slice(24, 25),
    "left_hand": slice(25, 40),
    "right_hand": slice(40, 55),
}

LOGGER = logging.getLogger(__name__)


def jsonl_write(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the previous preprocessing step first.")
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sequence_manifest_path(out_dir: Path, split: str) -> Path:
    return out_dir / 'manifests' / f'{split}_sequences.jsonl'


def clip_manifest_path(out_dir: Path, split: str) -> Path:
    return out_dir / 'manifests' / f'{split}_clips.jsonl'


def final_manifest_path(out_dir: Path, split: str) -> Path:
    return out_dir / 'manifests' / f'{split}_processed.jsonl'


def resolved_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _available_splits(out_dir: Path, requested: list[str]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for split in requested:
        yield split, jsonl_read(sequence_manifest_path(out_dir, split))


def build_sequence_manifests(
    data_root: Path,
    out_dir: Path,
    *,
    speaker_id: str | None = '2',
    missing_policy: str = 'skip',
) -> dict[str, Any]:
    """Use BEAT2 official CSV, preserving its train/val/test labels (no resplit)."""
    data_root = Path(data_root).resolve()
    out_dir = Path(out_dir).resolve()
    csv_file = data_root / 'train_test_split.csv'
    if not csv_file.is_file():
        raise FileNotFoundError(
            f'Cannot find {csv_file}. Set --data-root to the folder containing '
            'train_test_split.csv, smplxflame_30, wave16k, textgrid.'
        )
    if missing_policy not in ('skip', 'error'):
        raise ValueError('missing_policy must be skip or error')

    result: dict[str, list[dict[str, Any]]] = {s: [] for s in ('train', 'val', 'test')}
    stats = Counter()
    seen_ids: set[str] = set()
    with csv_file.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not {'id', 'type'} <= set(reader.fieldnames or []):
            raise ValueError(f'Unexpected CSV headers {reader.fieldnames}; expected id,type')
        for row in reader:
            seq_id = row['id'].strip()
            category = row['type'].strip().lower()
            stats['csv_entries'] += 1
            if category == 'additional':
                stats['ignored_additional'] += 1
                continue
            if category not in result:
                stats['ignored_unknown_split'] += 1
                continue
            if speaker_id not in (None, 'all') and seq_id.split('_', 1)[0] != str(speaker_id):
                stats['ignored_other_speaker'] += 1
                continue
            if seq_id in seen_ids:
                raise ValueError(f'Duplicate sequence ID in CSV: {seq_id}')
            seen_ids.add(seq_id)
            motion = Path('smplxflame_30') / f'{seq_id}.npz'
            audio = Path('wave16k') / f'{seq_id}.wav'
            textgrid = Path('textgrid') / f'{seq_id}.TextGrid'
            if not (data_root / motion).is_file() or not (data_root / audio).is_file():
                stats['missing_required'] += 1
                if missing_policy == 'error':
                    raise FileNotFoundError(f'Missing motion or audio for {seq_id}')
                LOGGER.warning('Skipping %s: missing motion or audio file', seq_id)
                continue
            if not (data_root / textgrid).is_file():
                stats['missing_textgrid'] += 1
            result[category].append({
                'seq_id': seq_id,
                'split': category,
                'speaker_id': seq_id.split('_', 1)[0],
                'motion_path': motion.as_posix(),
                'audio_path': audio.as_posix(),
                'textgrid_path': textgrid.as_posix() if (data_root / textgrid).is_file() else None,
                'fps': FPS,
                'sr': AUDIO_SR,
            })

    for split, values in result.items():
        jsonl_write(sequence_manifest_path(out_dir, split), values)
        stats[f'{split}_sequences'] = len(values)
    stats['speaker_id'] = speaker_id if speaker_id is not None else 'all'
    jsonl_write(out_dir / 'manifests' / 'index_stats.jsonl', [dict(stats)])
    return dict(stats)


def load_motion_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read official BEAT2 axis-angle motion: poses [T,165], expressions [T,100]."""
    with np.load(path, allow_pickle=False) as data:
        required = {'poses', 'expressions'}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f'{path} is missing required BEAT2 fields: {sorted(missing)}')
        poses = np.asarray(data['poses'], dtype=np.float32)
        expr = np.asarray(data['expressions'], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 165:
            raise ValueError(f'{path}: expected poses [T,165], got {poses.shape}')
        if expr.shape != (poses.shape[0], 100):
            raise ValueError(f'{path}: expected expressions [{len(poses)},100], got {expr.shape}')
        if not np.isfinite(poses).all() or not np.isfinite(expr).all():
            raise ValueError(f'Non-finite pose/expression values in {path}')
        if 'mocap_frame_rate' in data.files:
            fr = float(np.asarray(data['mocap_frame_rate']).reshape(-1)[0])
            if abs(fr-FPS) > 1e-3:
                raise ValueError(f'{path}: expected 30 FPS but found {fr}')
        ans = {'poses': poses, 'expressions': expr}
        if 'betas' in data.files:
            ans['betas'] = np.asarray(data['betas'], dtype=np.float32)
        if 'trans' in data.files:
            ans['trans'] = np.asarray(data['trans'], dtype=np.float32)
    return ans


def motion_to_rot6d(poses: np.ndarray) -> np.ndarray:
    """[T,165] SMPL-X axis-angle -> [T,55,6] (Zhou et al. convention)."""
    import torch
    from .rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 165:
        raise ValueError(f'Invalid pose input: {poses.shape}')
    with torch.no_grad():
        aa = torch.from_numpy(np.ascontiguousarray(poses)).reshape(-1, 3)
        rot6 = matrix_to_rotation_6d(axis_angle_to_matrix(aa)).reshape(-1, 55, 6)
    return rot6.numpy().astype(np.float32, copy=False)


def _wav_info(path: Path) -> tuple[int, int]:
    import soundfile as sf
    info = sf.info(path)
    return int(info.frames), int(info.samplerate)


def build_clip_manifests(
    data_root: Path,
    out_dir: Path,
    *,
    splits: list[str] = ['train', 'val', 'test'],
    clip_frames: int = CLIP_FRAMES,
    overlap_frames: int = OVERLAP_FRAMES,
) -> dict[str, dict[str, int]]:
    """Frame-exact 240/60 sliding windows; tail shorter than 240 is dropped."""
    if clip_frames <= 0 or overlap_frames < 0 or overlap_frames >= clip_frames:
        raise ValueError('Require clip_frames > overlap_frames >= 0')
    stride = clip_frames - overlap_frames
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    stats: dict[str, dict[str, int]] = {}
    for split, records in _available_splits(out_dir, splits):
        clips: list[dict[str, Any]] = []
        short = 0
        mismatches = 0
        for rec in records:
            motion = load_motion_arrays(resolved_path(data_root, rec['motion_path']))
            n_motion = motion['poses'].shape[0]
            audio_len, sample_rate = _wav_info(resolved_path(data_root, rec['audio_path']))
            n_audio_frames = int(round(audio_len * FPS / sample_rate))
            n_valid = min(n_motion, n_audio_frames)
            if abs(n_motion-n_audio_frames) > 1:
                mismatches += 1
                LOGGER.warning('%s: motion=%d frames, audio≈%d frames; clipping to %d',
                               rec['seq_id'], n_motion, n_audio_frames, n_valid)
            if n_valid < clip_frames:
                short += 1
                continue
            for i, f0 in enumerate(range(0, n_valid-clip_frames+1, stride)):
                f1 = f0+clip_frames
                s0 = int(round(f0 * sample_rate / FPS))
                s1 = min(audio_len, int(round(f1 * sample_rate / FPS)))
                clips.append({
                    **rec,
                    'clip_id': f"{rec['seq_id']}_{i:04d}",
                    'f0': f0,
                    'f1': f1,
                    's0': s0,
                    's1': s1,
                    't0': f0/FPS,
                    't1': f1/FPS,
                    'clip_frames': clip_frames,
                    'overlap_frames': overlap_frames,
                    'source_motion_frames': n_motion,
                    'source_audio_frames': n_audio_frames,
                    'n_valid_frames': n_valid,
                })
        jsonl_write(clip_manifest_path(out_dir, split), clips)
        stats[split] = {
            'sequences': len(records), 'clips': len(clips),
            'too_short': short, 'audio_motion_mismatch_gt_1frame': mismatches,
        }
    return stats


def interp_to_length(array: np.ndarray, length: int, *, nearest: bool = False) -> np.ndarray:
    """Match original notebook endpoint-aligned interpolation, including T=1 cases.

    This reproduces the original `np.linspace(0,1,...)` convention; it does NOT
    assert sub-frame accuracy for HuBERT at clip boundaries.
    """
    a = np.asarray(array, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or not a.shape[0] or length < 1:
        raise ValueError(f'Invalid interpolation shape {a.shape}, target={length}')
    if length == a.shape[0]:
        return a.astype(np.float32, copy=False)
    if a.shape[0] == 1:
        return np.repeat(a, length, axis=0)
    loc = np.linspace(0, a.shape[0]-1, length, dtype=np.float64)
    if nearest:
        return a[np.rint(loc).astype(np.int64)]
    left = np.floor(loc).astype(np.int64)
    right = np.minimum(left+1, a.shape[0]-1)
    w = (loc-left).astype(np.float32)[:, None]
    return ((1-w)*a[left] + w*a[right]).astype(np.float32)


def _norm01(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float32)
    return z / (float(np.max(np.abs(z))) + 1e-8)


def extract_librosa35(y_30720: np.ndarray, *, target_sr: int = 30720,
                      hop_length: int = 1024, n_fft: int = 4096) -> tuple[np.ndarray, dict]:
    """Original notebook's RMS1 + MFCC20 + chroma12 + onset1 + beat1 = 35D.

    Uses center=True and discards the final frame (T_ct-1), exactly as the
    submitted `data_process.ipynb`; no arbitrary z-score normalization.
    """
    import librosa
    y = np.asarray(y_30720, dtype=np.float32)
    env = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length,
                             center=True).reshape(-1, 1)
    env = _norm01(env)
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=20, n_fft=n_fft,
                                hop_length=hop_length, center=True).T
    chroma = librosa.feature.chroma_stft(y=y, sr=target_sr, n_fft=n_fft,
                                         hop_length=hop_length, center=True).T
    onset = librosa.onset.onset_strength(y=y, sr=target_sr, hop_length=hop_length,
                                        center=True, n_fft=n_fft)
    peaks = librosa.onset.onset_detect(onset_envelope=onset, units='frames')
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=target_sr,
                                            hop_length=hop_length, units='frames')
    t_ref = env.shape[0]
    t_target = max(1, t_ref-1)
    onset_oh = np.zeros((t_ref, 1), dtype=np.float32)
    beat_oh = np.zeros((t_ref, 1), dtype=np.float32)
    for frames, dest in ((peaks, onset_oh), (beat_frames, beat_oh)):
        valid = np.asarray(frames, dtype=np.int64)
        valid = valid[(0 <= valid) & (valid < t_ref)]
        dest[valid, 0] = 1.0
    if len(mfcc) != t_ref:
        mfcc = interp_to_length(mfcc, t_ref)
    if len(chroma) != t_ref:
        chroma = interp_to_length(chroma, t_ref)
    features = np.concatenate([env[:t_target], mfcc[:t_target], chroma[:t_target],
                               onset_oh[:t_target], beat_oh[:t_target]], axis=1).astype(np.float32)
    if features.shape[1] != LIBROSA_DIM:
        raise AssertionError(features.shape)
    return features, {
        'sr': target_sr, 'hop_length': hop_length, 'n_fft': n_fft,
        'center': True, 'native_fps': target_sr/hop_length,
        't_center': t_ref, 't_target': t_target,
        'columns': 'RMS1,MFCC20,chroma12,onset_peaks1,beats1',
    }


def extract_hubert_full(wav16k: 'torch.Tensor', model: Any, *, device: str,
                        mode: str = 'last4') -> dict[str, np.ndarray]:
    """Reproduce primary notebook: middle and last-layer averages, feat=high.

    'forward' reproduces the incompatible early `_1` notebook only when
    explicitly selected; no silent fallback is permitted.
    """
    import torch
    with torch.inference_mode():
        x = wav16k.to(device)
        if mode == 'last4':
            if not hasattr(model, 'extract_features'):
                raise RuntimeError('HuBERT model has no extract_features(); select --hubert-mode forward')
            layers, _ = model.extract_features(x)
            if len(layers) < 10:
                raise RuntimeError('Expected >=10 HuBERT layers for low/high mean extraction')
            low = torch.stack([z.squeeze(0) for z in layers[6:10]], dim=0).mean(0)
            high = torch.stack([z.squeeze(0) for z in layers[-4:]], dim=0).mean(0)
            outputs = {'feat': high, 'feat_low': low, 'feat_high': high}
        elif mode == 'forward':
            forward, _ = model(x)
            outputs = {'feat': forward.squeeze(0)}
        else:
            raise ValueError(f'Unknown hubert mode: {mode}')
    ans = {key: feat.detach().cpu().numpy().astype(np.float32)
           for key, feat in outputs.items()}
    for key, feat in ans.items():
        if feat.ndim != 2 or feat.shape[1] != HUBERT_DIM:
            raise ValueError(f'Unexpected HuBERT {key} shape: {feat.shape}')
    return ans


def extract_hubert(wav16k: 'torch.Tensor', model: Any, *, device: str,
                   mode: str = 'last4') -> np.ndarray:
    """Main HuBERT sequence `[Th,1024]`; optional low/high are in full output."""
    return extract_hubert_full(wav16k, model, device=device, mode=mode)['feat']


def extract_audio_caches(data_root: Path, out_dir: Path, *, splits: list[str],
                         device: str = 'cpu', mode: str = 'last4', overwrite: bool = False,
                         limit: int | None = None) -> dict[str, int]:
    """One audio cache per original sequence; no copies of raw speech in output."""
    import torch
    import torchaudio
    from tqdm.auto import tqdm
    if mode not in ('last4', 'forward'):
        raise ValueError('mode must be last4 or forward')
    all_recs = [rec for _, records in _available_splits(out_dir, splits) for rec in records]
    if limit is not None:
        all_recs = all_recs[:limit]
    hubert_dir = Path(out_dir) / 'audio' / 'hubert'
    librosa_dir = Path(out_dir) / 'audio' / 'librosa'
    hubert_dir.mkdir(parents=True, exist_ok=True)
    librosa_dir.mkdir(parents=True, exist_ok=True)
    def cache_valid(rec: dict[str, Any]) -> bool:
        """Do not silently mix the incompatible HuBERT extraction variants."""
        hp = hubert_dir / f"{rec['seq_id']}.npz"
        lp = librosa_dir / f"{rec['seq_id']}.npz"
        if not hp.is_file() or not lp.is_file():
            return False
        try:
            with np.load(hp, allow_pickle=False) as h:
                if 'mode' not in h.files or str(h['mode'].item()) != mode:
                    return False
                if h['feat'].ndim != 2 or h['feat'].shape[1] != HUBERT_DIM:
                    return False
            with np.load(lp, allow_pickle=False) as l:
                if l['feat'].ndim != 2 or l['feat'].shape[1] != LIBROSA_DIM:
                    return False
                if int(l['sr']) != 30720 or int(l['hop_length']) != 1024:
                    return False
            return True
        except (OSError, ValueError, KeyError, EOFError):
            return False

    remaining = [rec for rec in all_recs if overwrite or not cache_valid(rec)]
    if not remaining:
        return {'requested': len(all_recs), 'processed': 0, 'already_cached': len(all_recs)}
    model = torchaudio.pipelines.HUBERT_LARGE.get_model().to(device).eval()
    completed = 0
    for rec in tqdm(remaining, desc='Extracting HuBERT/Librosa'):
        seq_id = rec['seq_id']
        # Separate outputs are regenerated together to prevent mixed extraction settings.
        wave, sr = torchaudio.load(resolved_path(Path(data_root), rec['audio_path']))
        if wave.shape[0] > 1:
            wave = wave.mean(dim=0, keepdim=True)
        wave = wave.float()
        wav16 = torchaudio.functional.resample(wave, sr, AUDIO_SR) if sr != AUDIO_SR else wave
        wav30 = torchaudio.functional.resample(wave, sr, 30720) if sr != 30720 else wave
        dur = float(wave.shape[-1]) / float(sr)
        feat_h = extract_hubert_full(wav16, model, device=device, mode=mode)
        feat_l, info = extract_librosa35(wav30.squeeze(0).numpy())
        h_path = hubert_dir / f'{seq_id}.npz'
        l_path = librosa_dir / f'{seq_id}.npz'
        h_payload = {k: v.astype(np.float16) for k, v in feat_h.items()}
        h_payload.update(dur_sec=np.float32(dur),
                         fps=np.float32(len(feat_h['feat'])/dur), mode=np.str_(mode))
        np.savez_compressed(h_path, **h_payload)
        np.savez_compressed(l_path,
                            feat=feat_l.astype(np.float16), dur_sec=np.float32(dur),
                            fps=np.float32(info['native_fps']), sr=np.int32(info['sr']),
                            hop_length=np.int32(info['hop_length']), n_fft=np.int32(info['n_fft']),
                            center=np.bool_(True), t_target=np.int32(info['t_target']))
        completed += 1
    return {'requested': len(all_recs), 'processed': completed,
            'already_cached': len(all_recs)-len(remaining)}


_ITEM_START = re.compile(r'^\s*item\s*\[\d+\]\s*:\s*$')
_INTERVAL_START = re.compile(r'^\s*intervals\s*\[\d+\]\s*:\s*$')
_NAME = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')
_XMIN = re.compile(r'^\s*xmin\s*=\s*([-+\d.eE]+)\s*$')
_XMAX = re.compile(r'^\s*xmax\s*=\s*([-+\d.eE]+)\s*$')
_TEXT = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')


def parse_textgrid_words(path: Path, tier: str = 'words') -> list[tuple[float, float, str]]:
    """Parse long-format Praat TextGrid's named IntervalTier without third-party parser.

    Unlike the original notebook, stops at the next `item [n]` tier boundary.
    """
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    words: list[tuple[float, float, str]] = []
    current_tier = ''
    current: dict[str, Any] | None = None

    def commit():
        if current_tier == tier and current is not None and {'xmin','xmax','text'} <= current.keys():
            word = str(current['text']).replace('""', '"').strip()
            if word and current['xmax'] > current['xmin']:
                words.append((float(current['xmin']), float(current['xmax']), word))

    for line in lines:
        if _ITEM_START.match(line):
            commit()
            current_tier = ''
            current = None
            continue
        m = _NAME.match(line)
        if m:
            current_tier = m.group(1)
            continue
        if _INTERVAL_START.match(line):
            commit()
            current = {}
            continue
        if current is None or current_tier != tier:
            continue
        m = _XMIN.match(line)
        if m:
            current['xmin'] = float(m.group(1))
            continue
        m = _XMAX.match(line)
        if m:
            current['xmax'] = float(m.group(1))
            continue
        m = _TEXT.match(line)
        if m:
            current['text'] = m.group(1)
    commit()
    return words


def normalize_word(word: str) -> str:
    return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', word.strip().lower())


def clip_word_events(intervals: list[tuple[float, float, str]],
                     t0: float, t1: float) -> list[tuple[float, float, str, str]]:
    events = []
    for start, end, raw in intervals:
        if end <= t0 or start >= t1:
            continue
        a = max(t0, start)
        b = min(t1, end)
        norm = normalize_word(raw)
        if b > a and norm:
            events.append((a-t0, b-t0, norm, raw))
    return events


def _load_fasttext(vec_path: Path, cache_path: Path | None) -> Any:
    from gensim.models import KeyedVectors
    if cache_path is not None and cache_path.is_file():
        return KeyedVectors.load(str(cache_path), mmap='r')
    if not vec_path.is_file():
        raise FileNotFoundError(f'FastText .vec not found: {vec_path}')
    LOGGER.info('Loading FastText vectors into memory: %s', vec_path)
    vectors = KeyedVectors.load_word2vec_format(str(vec_path), binary=False)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vectors.save(str(cache_path))
    return vectors


def extract_text_caches(data_root: Path, out_dir: Path, *, splits: list[str],
                        fasttext_vec: Path, fasttext_cache: Path | None = None,
                        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                        overwrite: bool = False, limit: int | None = None) -> dict[str, int]:
    """Create per-clip word event FastText vectors and a clip-level SBERT vector."""
    from tqdm.auto import tqdm
    from sentence_transformers import SentenceTransformer
    out_dir = Path(out_dir)
    clips = [c for split in splits for c in jsonl_read(clip_manifest_path(out_dir, split))]
    if limit is not None:
        clips = clips[:limit]
    gdir = out_dir / 'text' / 'global'
    wdir = out_dir / 'text' / 'words'
    gdir.mkdir(parents=True, exist_ok=True)
    wdir.mkdir(parents=True, exist_ok=True)
    remaining = [c for c in clips if overwrite or not (
        (gdir / f"{c['clip_id']}.npy").exists() and (wdir / f"{c['clip_id']}.npz").exists())]
    # Also prepare full-sequence text for long-sequence inference.
    seq_text_dir = out_dir / 'text' / 'sequences'
    seq_text_dir.mkdir(parents=True, exist_ok=True)
    selected = {rec['seq_id']: rec for split in splits
                for rec in jsonl_read(sequence_manifest_path(out_dir, split))}
    if limit is not None:
        selected = dict(list(selected.items())[:limit])
    remaining_sequences = [rec for sid, rec in selected.items()
                           if overwrite or not (seq_text_dir / f'{sid}.npz').exists()]
    if not remaining and not remaining_sequences:
        return {'requested': len(clips), 'processed': 0, 'already_cached': len(clips),
                'sequence_text_processed': 0}

    wv = _load_fasttext(Path(fasttext_vec), Path(fasttext_cache) if fasttext_cache else None)
    if wv.vector_size != WORD_DIM:
        raise ValueError(f'Expected 300-D FastText vectors, got {wv.vector_size}')
    sentence_model = SentenceTransformer(sbert_model)
    if sentence_model.get_sentence_embedding_dimension() != SENT_DIM:
        raise ValueError('Expected 384-D SBERT embedding: use all-MiniLM-L6-v2')
    word_intervals: dict[str, list[tuple[float, float, str]]] = {}
    for clip in tqdm(remaining, desc='Extracting FastText/SBERT'):
        sid = clip['seq_id']
        if sid not in word_intervals:
            path = resolved_path(Path(data_root), clip.get('textgrid_path'))
            if path is not None:
                word_intervals[sid] = parse_textgrid_words(path)
            else:
                LOGGER.warning('%s: TextGrid missing; using empty text', sid)
                word_intervals[sid] = []
        events = clip_word_events(word_intervals[sid], clip['t0'], clip['t1'])
        transcript = ' '.join(e[3] for e in events)
        g = np.asarray(sentence_model.encode(transcript, normalize_embeddings=False), dtype=np.float32)
        vecs = np.stack([wv[e[2]] if e[2] in wv else np.zeros(WORD_DIM, dtype=np.float32)
                         for e in events], axis=0).astype(np.float32) if events else np.zeros((0, WORD_DIM), dtype=np.float32)
        np.save(gdir / f"{clip['clip_id']}.npy", g)
        np.savez_compressed(wdir / f"{clip['clip_id']}.npz",
                            starts=np.array([e[0] for e in events], dtype=np.float32),
                            ends=np.array([e[1] for e in events], dtype=np.float32),
                            words_norm=np.array([e[2] for e in events], dtype=np.str_),
                            words_raw=np.array([e[3] for e in events], dtype=np.str_),
                            vecs=vecs)
    for rec in tqdm(remaining_sequences, desc='Extracting full-sequence text'):
        sid = rec['seq_id']
        if sid not in word_intervals:
            tg = resolved_path(Path(data_root), rec.get('textgrid_path'))
            word_intervals[sid] = parse_textgrid_words(tg) if tg is not None else []
        motion = load_motion_arrays(resolved_path(Path(data_root), rec['motion_path']))
        wav_len, wav_sr = _wav_info(resolved_path(Path(data_root), rec['audio_path']))
        duration = min(len(motion['poses']) / FPS, wav_len / wav_sr)
        events = clip_word_events(word_intervals[sid], 0.0, duration)
        transcript = ' '.join(e[3] for e in events)
        vecs = np.stack([wv[e[2]] if e[2] in wv else np.zeros(WORD_DIM, dtype=np.float32)
                         for e in events]).astype(np.float32) if events else np.empty((0, WORD_DIM), dtype=np.float32)
        global_vec = np.asarray(sentence_model.encode(transcript, normalize_embeddings=False), dtype=np.float32)
        np.savez_compressed(seq_text_dir / f'{sid}.npz',
                            text_global=global_vec,
                            text_words_vecs=vecs,
                            text_words_starts=np.asarray([e[0] for e in events], dtype=np.float32),
                            text_words_ends=np.asarray([e[1] for e in events], dtype=np.float32))
    return {'requested': len(clips), 'processed': len(remaining),
            'already_cached': len(clips)-len(remaining),
            'sequence_text_processed': len(remaining_sequences)}


def word_events_to_frames(starts: np.ndarray, ends: np.ndarray, vecs: np.ndarray,
                          clip_frames: int, *, fps: int = FPS) -> np.ndarray:
    """Frame-center word assignment; missing/unaligned words are zero."""
    result = np.zeros((clip_frames, WORD_DIM), dtype=np.float32)
    if len(starts) != len(ends) or len(starts) != len(vecs):
        raise ValueError('Bad word event cache lengths')
    for start, end, vec in zip(starts, ends, vecs):
        # Mask frames whose centers belong to this interval.
        f0 = max(0, int(np.ceil(float(start)*fps - 0.5)))
        f1 = min(clip_frames, int(np.ceil(float(end)*fps - 0.5)))
        if f1 > f0:
            result[f0:f1] = vec
    return result


def pack_clips(data_root: Path, out_dir: Path, *, splits: list[str],
               skip_text: bool = False, overwrite: bool = False,
               limit: int | None = None) -> dict[str, dict[str, int]]:
    """Build per-clip 55x6 SMPL-X motion and time-aligned features.

    Pack uses *sequence-level* feature caches. HuBERT is interpolated to sequence
    length before frame slicing, as in the supplied original notebooks.
    """
    from tqdm.auto import tqdm
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    result: dict[str, dict[str, int]] = {}
    for split in splits:
        clips = jsonl_read(clip_manifest_path(out_dir, split))
        if limit is not None:
            clips = clips[:limit]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for c in clips:
            grouped[c['seq_id']].append(c)
        processed: list[dict[str, Any]] = []
        clip_folder = out_dir / 'clips' / split
        clip_folder.mkdir(parents=True, exist_ok=True)
        def packed_cache_valid(path: Path) -> bool:
            if not path.is_file():
                return False
            try:
                with np.load(path, allow_pickle=False) as d:
                    req = {'rot6d', 'expression', 'hubert', 'librosa35'}
                    if not skip_text:
                        req |= {'fasttext', 'sbert'}
                    if not req <= set(d.files):
                        return False
                    return (d['rot6d'].shape == (CLIP_FRAMES, 55, 6)
                            and d['expression'].shape == (CLIP_FRAMES, 100)
                            and d['hubert'].shape == (CLIP_FRAMES, HUBERT_DIM)
                            and d['librosa35'].shape == (CLIP_FRAMES, LIBROSA_DIM)
                            and (skip_text or (d['fasttext'].shape == (CLIP_FRAMES, WORD_DIM)
                                                  and d['sbert'].shape == (SENT_DIM,))))
            except (OSError, ValueError, KeyError, EOFError):
                return False

        for sid, cseq in tqdm(grouped.items(), desc=f'Packing {split}'):
            todo = [c for c in cseq if overwrite or not packed_cache_valid(
                clip_folder/f"{c['clip_id']}.npz")]
            if not todo:
                processed.extend({**c, 'processed_path': (Path('clips') / split / f"{c['clip_id']}.npz").as_posix()}
                                 for c in cseq)
                continue
            motion = load_motion_arrays(resolved_path(data_root, cseq[0]['motion_path']))
            rot6d = motion_to_rot6d(motion['poses'])
            length = cseq[0]['n_valid_frames']
            hpath = out_dir / 'audio' / 'hubert' / f'{sid}.npz'
            lpath = out_dir / 'audio' / 'librosa' / f'{sid}.npz'
            if not hpath.exists() or not lpath.exists():
                raise FileNotFoundError(f'Audio cache missing for {sid}; run `audio` step first')
            with np.load(hpath, allow_pickle=False) as h:
                hubert = interp_to_length(h['feat'], length)
                hubert_low = (interp_to_length(h['feat_low'], length)
                              if 'feat_low' in h.files else None)
            with np.load(lpath, allow_pickle=False) as l:
                native_l = np.asarray(l['feat'], dtype=np.float32)
                if native_l.shape[1] != LIBROSA_DIM:
                    raise ValueError(f'Librosa cache for {sid} has shape {native_l.shape}')
                librosa = interp_to_length(native_l, length)
            for c in todo:
                f0, f1 = int(c['f0']), int(c['f1'])
                n = f1-f0
                if n != CLIP_FRAMES:
                    # The model architecture is configured for 240-frame windows.
                    raise ValueError(f'Expected 240-frame model clips, got {n}')
                a = rot6d[f0:f1]
                exp = motion['expressions'][f0:f1]
                h = hubert[f0:f1]
                l = librosa[f0:f1]
                if a.shape != (n, 55, 6) or exp.shape != (n, 100) or h.shape != (n, 1024) or l.shape != (n, 35):
                    raise AssertionError(f'Corrupted alignment for {sid}: {a.shape}, {exp.shape}, {h.shape}, {l.shape}')
                payload: dict[str, np.ndarray] = {
                    'rot6d': a.astype(np.float32),
                    'expression': exp.astype(np.float32),
                    'hubert': h.astype(np.float16),
                    'librosa35': l.astype(np.float16),
                    'fps': np.int32(FPS),
                    'frame_start': np.int32(f0),
                    'frame_end': np.int32(f1),
                    'seq_id': np.str_(sid),
                    'clip_id': np.str_(c['clip_id']),
                }
                if hubert_low is not None:
                    # Original notebook saved this intermediate 6:10-layer mean.
                    # Auxiliary data only; the published model loader must decide
                    # whether it consumes this or only the default 'hubert'.
                    payload['hubert_low'] = hubert_low[f0:f1].astype(np.float16)
                if 'betas' in motion:
                    payload['betas'] = motion['betas']
                if 'trans' in motion:
                    payload['trans_ref_only'] = motion['trans'][f0:f1]  # Not a trained target.
                if not skip_text:
                    gp = out_dir / 'text' / 'global' / f"{c['clip_id']}.npy"
                    wp = out_dir / 'text' / 'words' / f"{c['clip_id']}.npz"
                    if not gp.exists() or not wp.exists():
                        raise FileNotFoundError(f'Text cache missing for {c["clip_id"]}; run `text` step first')
                    sent = np.load(gp, allow_pickle=False).astype(np.float32)
                    with np.load(wp, allow_pickle=False) as w:
                        words = word_events_to_frames(w['starts'], w['ends'], w['vecs'], n)
                        coverage = word_events_to_frames(w['starts'], w['ends'],
                                                         np.ones_like(w['vecs'], dtype=np.float32), n)
                    if sent.shape != (SENT_DIM,):
                        raise ValueError(f'Expected sentence embedding [384], got {sent.shape}')
                    payload['fasttext'] = words.astype(np.float16)
                    payload['text_local_mask'] = coverage.any(axis=-1)
                    payload['sbert'] = sent.astype(np.float16)
                np.savez_compressed(clip_folder / f"{c['clip_id']}.npz", **payload)
            processed.extend({**c, 'processed_path': (Path('clips') / split / f"{c['clip_id']}.npz").as_posix()}
                             for c in cseq)
        # Export only clips that are verified to exist (e.g. --limit).
        processed = [c for c in processed if (out_dir / c['processed_path']).is_file()]
        jsonl_write(final_manifest_path(out_dir, split), processed)
        result[split] = {'packed_clips': len(processed), 'sequences': len(grouped)}
    return result


def validate_processed_data(out_dir: Path, *, splits: list[str],
                            skip_text: bool = False,
                            limit: int | None = None) -> dict[str, dict[str, int]]:
    """Check release outputs without requiring the raw dataset or pretrained models."""
    out_dir = Path(out_dir)
    seq_set_by_split: dict[str, set[str]] = {}
    validated: dict[str, dict[str, int]] = {}
    for split in splits:
        raw_sequences = jsonl_read(sequence_manifest_path(out_dir, split))
        seq_ids = {r['seq_id'] for r in raw_sequences}
        if len(seq_ids) != len(raw_sequences):
            raise ValueError(f'{split}: duplicate source sequence IDs')
        seq_set_by_split[split] = seq_ids
        rows = jsonl_read(final_manifest_path(out_dir, split))
        if limit is not None:
            rows = rows[:limit]
        ids: set[str] = set()
        for row in rows:
            cid = row['clip_id']
            if cid in ids:
                raise ValueError(f'{split}: duplicate processed clip ID {cid}')
            ids.add(cid)
            if row['seq_id'] not in seq_ids:
                raise ValueError(f'{split}: unknown seq_id {row["seq_id"]}')
            if int(row['f1']) - int(row['f0']) != CLIP_FRAMES:
                raise ValueError(f'{cid}: expected {CLIP_FRAMES} frames')
            path = out_dir / row['processed_path']
            if not path.is_file():
                raise FileNotFoundError(f'{cid}: expected processed file at {path}')
            with np.load(path, allow_pickle=False) as data:
                expected = {'rot6d': (CLIP_FRAMES,55,6),
                            'expression': (CLIP_FRAMES,100),
                            'hubert': (CLIP_FRAMES,HUBERT_DIM),
                            'librosa35': (CLIP_FRAMES,LIBROSA_DIM)}
                if not skip_text:
                    expected.update(fasttext=(CLIP_FRAMES,WORD_DIM), sbert=(SENT_DIM,))
                for name, shape in expected.items():
                    if name not in data.files:
                        raise ValueError(f'{cid}: missing required {name}')
                    if data[name].shape != shape:
                        raise ValueError(f'{cid}: {name} shape {data[name].shape}, expected {shape}')
                    if not np.isfinite(data[name]).all():
                        raise ValueError(f'{cid}: nonfinite {name}')
                if int(data['frame_start']) != row['f0'] or int(data['frame_end']) != row['f1']:
                    raise ValueError(f'{cid}: metadata frame bounds differ from manifest')
        validated[split] = {'source_sequences': len(seq_ids),
                            'validated_clips': len(rows)}
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            overlap = seq_set_by_split[a] & seq_set_by_split[b]
            if overlap:
                raise ValueError(f'Split leakage between {a} and {b}: {sorted(overlap)[:5]}')
    return validated


def pack_sequences(data_root: Path, out_dir: Path, *, splits: list[str],
                   overwrite: bool = False, limit: int | None = None,
                   skip_text: bool = False) -> dict[str, dict[str, int]]:
    """Pack full sequences for strict long-sequence inference (no window crop).

    Uses already extracted sequence-level audio and text caches. No full-sequence
    features are silently reconstructed from overlapping clip embeddings.
    """
    data_root, out_dir = Path(data_root), Path(out_dir)
    stats = {}
    for split in splits:
        records = jsonl_read(sequence_manifest_path(out_dir, split))
        if limit is not None:
            records = records[:limit]
        rows = []
        seq_dir = out_dir / 'sequences' / split
        seq_dir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            sid = rec['seq_id']
            path = seq_dir / f'{sid}.npz'
            if not path.is_file() or overwrite:
                motion = load_motion_arrays(resolved_path(data_root, rec['motion_path']))
                h_path = out_dir / 'audio' / 'hubert' / f'{sid}.npz'
                l_path = out_dir / 'audio' / 'librosa' / f'{sid}.npz'
                if not h_path.is_file() or not l_path.is_file():
                    raise FileNotFoundError(f'Missing sequence-level audio caches for {sid}')
                # Align motion to audio without changing the original feature time grid.
                with np.load(h_path, allow_pickle=False) as h, np.load(l_path, allow_pickle=False) as l:
                    duration = min(float(h['dur_sec']), float(l['dur_sec']))
                    T = min(len(motion['poses']), int(round(duration * FPS)))
                    if T <= 0:
                        raise ValueError(f'Empty sequence: {sid}')
                    payload = dict(
                        seq_id=np.str_(sid), T=np.int32(T),
                        poses=motion['poses'][:T].astype(np.float32),
                        expressions=motion['expressions'][:T].astype(np.float32),
                        hubert_seq=np.asarray(h['feat'], dtype=np.float16),
                        hubert_fps=np.float32(h['fps']),
                        librosa_seq=np.asarray(l['feat'], dtype=np.float16),
                        librosa_fps=np.float32(l['fps']),
                    )
                for k in ('trans', 'betas'):
                    if k in motion:
                        payload[k] = motion[k][:T] if k == 'trans' else motion[k]
                if not skip_text:
                    tp = out_dir / 'text' / 'sequences' / f'{sid}.npz'
                    if not tp.is_file():
                        raise FileNotFoundError(f'Missing full-sequence text cache {tp}; run text stage')
                    with np.load(tp, allow_pickle=False) as t:
                        payload.update({k: np.asarray(t[k]) for k in (
                            'text_global','text_words_vecs','text_words_starts','text_words_ends')})
                np.savez_compressed(path, **payload)
            rows.append({'seq_id': sid,
                         'processed_path': (Path('sequences') / split / f'{sid}.npz').as_posix()})
        jsonl_write(out_dir / 'manifests' / f'{split}_sequence_processed.jsonl', rows)
        stats[split] = {'packed_sequences': len(rows)}
    return stats


def validate_sequence_packs(out_dir: Path, *, splits: list[str],
                            skip_text: bool = False,
                            limit: int | None = None) -> dict[str, dict[str, int]]:
    """Check full-sequence NPZ fields needed by the long-inference dataset."""
    out_dir = Path(out_dir)
    output = {}
    for split in splits:
        manifest = out_dir / 'manifests' / f'{split}_sequence_processed.jsonl'
        if not manifest.exists():
            continue
        rows = jsonl_read(manifest)
        if limit is not None:
            rows = rows[:limit]
        for rec in rows:
            path = out_dir / rec['processed_path']
            with np.load(path, allow_pickle=False) as d:
                T = int(d['T'])
                if T < 1 or d['poses'].shape != (T,165) or d['expressions'].shape != (T,100):
                    raise ValueError(f'Invalid sequence motion in {path}')
                if d['hubert_seq'].ndim != 2 or d['hubert_seq'].shape[1] != 1024:
                    raise ValueError(f'Invalid sequence HuBERT in {path}')
                if d['librosa_seq'].ndim != 2 or d['librosa_seq'].shape[1] != 35:
                    raise ValueError(f'Invalid sequence Librosa in {path}')
                if float(d['hubert_fps']) <= 0 or float(d['librosa_fps']) <= 0:
                    raise ValueError(f'Invalid audio feature fps in {path}')
                if not skip_text:
                    if d['text_global'].shape != (384,):
                        raise ValueError(f'Invalid full-sequence sentence feature in {path}')
                    if (d['text_words_vecs'].ndim != 2 or d['text_words_vecs'].shape[1] != 300
                        or len(d['text_words_vecs']) != len(d['text_words_starts'])
                        or len(d['text_words_vecs']) != len(d['text_words_ends'])):
                        raise ValueError(f'Invalid full-sequence word features in {path}')
        output[split] = {'validated_sequences': len(rows)}
    return output
