"""Synthetic BEAT2-format tests: no pretrained models or dataset download needed."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from talkflow_data.preprocess import (
    FPS, JOINT_SLICES, build_sequence_manifests, build_clip_manifests,
    clip_word_events, extract_librosa35, final_manifest_path,
    interp_to_length, jsonl_read, load_motion_arrays, motion_to_rot6d,
    pack_clips, parse_textgrid_words, word_events_to_frames,
    validate_processed_data, extract_hubert_full,
)


def create_fake_beat2(tmp_path: Path, nframes: int = 510):
    root = tmp_path / 'beat_english_v2.0.0'
    for sub in ('smplxflame_30', 'wave16k', 'textgrid'):
        (root / sub).mkdir(parents=True)
    csv_path = root / 'train_test_split.csv'
    csv_path.write_text('id,type\n2_scott_0_104_104,train\n2_scott_0_10_10,val\n'
                        '2_scott_0_1_1,test\n2_scott_1_1_1,additional\n'
                        '10_kieks_0_104_104,train\n', encoding='utf-8')
    for name in ('2_scott_0_104_104', '2_scott_0_10_10', '2_scott_0_1_1',
                 '10_kieks_0_104_104'):
        poses = np.zeros((nframes, 165), dtype=np.float32)
        expressions = np.ones((nframes, 100), dtype=np.float32)
        np.savez(root / 'smplxflame_30' / f'{name}.npz', poses=poses,
                 expressions=expressions, betas=np.zeros(300, dtype=np.float32),
                 trans=np.zeros((nframes, 3), dtype=np.float32),
                 mocap_frame_rate=np.int32(30))
        sr=16000
        y = np.zeros(int(nframes / FPS * sr), dtype=np.float32)
        sf.write(root / 'wave16k' / f'{name}.wav', y, sr)
        (root / 'textgrid' / f'{name}.TextGrid').write_text('''File type = "ooTextFile"\nObject class = "TextGrid"\nitem [1]:\n class = "IntervalTier"\n name = "words"\n intervals [1]:\n  xmin = 0\n  xmax = 1.0\n  text = "Hello"\n intervals [2]:\n  xmin = 1.0\n  xmax = 2.0\n  text = "!world!"\nitem [2]:\n class = "IntervalTier"\n name = "phones"\n intervals [1]:\n  xmin = 0\n  xmax = 1.0\n  text = "SHOULD_NOT_APPEAR"\n''', encoding='utf-8')
    return root


def test_manifest_and_clip_splits(tmp_path):
    root = create_fake_beat2(tmp_path)
    out = tmp_path/'processed'
    stats = build_sequence_manifests(root, out)
    assert stats['train_sequences'] == stats['val_sequences'] == stats['test_sequences'] == 1
    assert stats['ignored_other_speaker'] == stats['ignored_additional'] == 1
    windows = build_clip_manifests(root, out)
    assert windows['train']['clips'] == 2  # start = 0,180 for T=510, not tail-aligned.
    clips = jsonl_read(out/'manifests/train_clips.jsonl')
    assert [(c['f0'], c['f1']) for c in clips] == [(0, 240), (180, 420)]
    assert all(c['speaker_id'] == '2' for c in clips)
    assert all(c['f1']-c['f0'] == 240 for c in clips)
    assert clips[0]['motion_path'].startswith('smplxflame_30/')
    all_stats = build_sequence_manifests(root, tmp_path/'all', speaker_id='all')
    assert all_stats['train_sequences'] == 2


def test_motion_rot6d_shapes_and_order(tmp_path):
    root = create_fake_beat2(tmp_path, nframes=240)
    motion = load_motion_arrays(root/'smplxflame_30/2_scott_0_104_104.npz')
    rot6d = motion_to_rot6d(motion['poses'])
    assert rot6d.shape == (240, 55, 6)
    np.testing.assert_allclose(rot6d[0, 0], [1,0,0,0,1,0], atol=1e-5)
    assert JOINT_SLICES['left_hand'] == slice(25, 40)
    assert JOINT_SLICES['right_hand'] == slice(40, 55)


def test_textgrid_tier_boundary(tmp_path):
    root = create_fake_beat2(tmp_path)
    words = parse_textgrid_words(root/'textgrid/2_scott_0_104_104.TextGrid')
    assert words == [(0., 1., 'Hello'), (1., 2., '!world!')]
    events = clip_word_events(words, 0.5, 1.5)
    assert events[0] == (0., 0.5, 'hello', 'Hello')
    assert events[1] == (0.5, 1., 'world', '!world!')


def test_word_frame_conversion():
    word_vec = np.ones((2, 300), dtype=np.float32)
    aligned = word_events_to_frames(np.array([0., 1.]), np.array([1., 2.]), word_vec, 240)
    assert aligned.shape == (240, 300)
    assert (aligned[:60] == 1).all()
    assert (aligned[60:] == 0).all()


def test_librosa_dimensions_and_interpolation():
    sr = 30720
    x = np.sin(2*np.pi*220*np.arange(sr, dtype=np.float32)/sr)
    features, meta = extract_librosa35(x)
    assert features.shape == (30,35)
    assert meta['native_fps'] == 30.0
    assert np.isfinite(features).all()
    assert interp_to_length(features, 240).shape == (240,35)
    assert interp_to_length(np.zeros((1, 35), dtype=np.float32), 240).shape == (240,35)


def test_pack_model_ready(tmp_path):
    root = create_fake_beat2(tmp_path)
    out = tmp_path/'processed'
    build_sequence_manifests(root, out)
    build_clip_manifests(root, out)
    sid='2_scott_0_104_104'
    for folder in ('hubert', 'librosa'):
        (out/'audio'/folder).mkdir(parents=True)
    np.savez(out/'audio/hubert'/f'{sid}.npz', feat=np.ones((800,1024), dtype=np.float16))
    np.savez(out/'audio/librosa'/f'{sid}.npz', feat=np.ones((510,35), dtype=np.float16))
    for c in jsonl_read(out/'manifests/train_clips.jsonl'):
        for folder in ('global','words'):
            (out/'text'/folder).mkdir(parents=True, exist_ok=True)
        np.save(out/'text/global'/f"{c['clip_id']}.npy", np.ones(384, dtype=np.float32))
        np.savez(out/'text/words'/f"{c['clip_id']}.npz",
                 starts=np.array([0.]), ends=np.array([8.]),
                 vecs=np.ones((1,300), dtype=np.float32))
    stats = pack_clips(root, out, splits=['train'])
    assert stats['train']['packed_clips'] == 2
    validated = validate_processed_data(out, splits=['train'])
    assert validated['train']['validated_clips'] == 2
    row = jsonl_read(final_manifest_path(out, 'train'))[0]
    with np.load(out/row['processed_path'], allow_pickle=False) as d:
        assert d['rot6d'].shape == (240,55,6)
        assert d['expression'].shape == (240,100)
        assert d['hubert'].shape == (240,1024)
        assert d['librosa35'].shape == (240,35)
        assert d['fasttext'].shape == (240,300)
        assert d['sbert'].shape == (384,)
        assert (d['fasttext'] == 1).all()
        assert d['trans_ref_only'].shape == (240,3)


def test_hubert_modes_with_fake_model():
    import torch
    class FakeHuBERT:
        def extract_features(self, x):
            return [torch.full((1, 4, 1024), float(j)) for j in range(12)], None
        def __call__(self, x):
            return torch.full((1, 4, 1024), 42.0), None
    wav = torch.zeros((1, 1600))
    model = FakeHuBERT()
    features = extract_hubert_full(wav, model, device='cpu', mode='last4')
    assert set(features) == {'feat', 'feat_low', 'feat_high'}
    assert np.allclose(features['feat_low'], 7.5)
    assert np.allclose(features['feat_high'], 9.5)
    assert np.array_equal(features['feat'], features['feat_high'])
    forward = extract_hubert_full(wav, model, device='cpu', mode='forward')
    assert set(forward) == {'feat'} and np.allclose(forward['feat'], 42)


def test_full_sequence_packing_for_inference(tmp_path):
    from talkflow_data.preprocess import pack_sequences
    from utils.speech_dataset import build_long_test_dataloader
    root=create_fake_beat2(tmp_path, nframes=300)
    out=tmp_path/'processed'
    build_sequence_manifests(root,out)
    sid='2_scott_0_1_1'
    hp=out/'audio'/'hubert'; hp.mkdir(parents=True)
    lp=out/'audio'/'librosa'; lp.mkdir(parents=True)
    tp=out/'text'/'sequences'; tp.mkdir(parents=True)
    np.savez_compressed(hp/f'{sid}.npz',feat=np.zeros((150,1024),dtype=np.float16),
                        dur_sec=np.float32(10),fps=np.float32(15))
    np.savez_compressed(lp/f'{sid}.npz',feat=np.zeros((300,35),dtype=np.float16),
                        dur_sec=np.float32(10),fps=np.float32(30))
    np.savez_compressed(tp/f'{sid}.npz',text_global=np.zeros(384,dtype=np.float32),
                        text_words_vecs=np.ones((1,300),dtype=np.float32),
                        text_words_starts=np.asarray([1.],dtype=np.float32),
                        text_words_ends=np.asarray([2.],dtype=np.float32))
    result=pack_sequences(root,out,splits=['test'])
    assert result['test']['packed_sequences']==1
    _,loader=build_long_test_dataloader(out,split='test')
    batch=next(iter(loader))
    assert batch['hubert'].shape==(1,300,1024)
    assert batch['text_local_raw'].shape==(1,300,300)
    assert batch['text_local_mask'].sum()==30


def test_validate_sequence_packs(tmp_path):
    from talkflow_data.preprocess import pack_sequences,validate_sequence_packs
    root=create_fake_beat2(tmp_path,nframes=240)
    out=tmp_path/'processed'
    build_sequence_manifests(root,out)
    sid='2_scott_0_1_1'
    hp=out/'audio'/'hubert';hp.mkdir(parents=True)
    lp=out/'audio'/'librosa';lp.mkdir(parents=True)
    np.savez_compressed(hp/f'{sid}.npz',feat=np.zeros((150,1024),dtype=np.float16),
                        dur_sec=np.float32(8),fps=np.float32(18.75))
    np.savez_compressed(lp/f'{sid}.npz',feat=np.zeros((240,35),dtype=np.float16),
                        dur_sec=np.float32(8),fps=np.float32(30))
    pack_sequences(root,out,splits=['test'],skip_text=True)
    assert validate_sequence_packs(out,splits=['test'],skip_text=True)['test']['validated_sequences']==1
