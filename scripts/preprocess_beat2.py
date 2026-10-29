#!/usr/bin/env python3
"""CLI entry point for TalkFlow's BEAT2 preprocessing (run from repository root)."""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from talkflow_data.preprocess import (
    build_sequence_manifests, build_clip_manifests, extract_audio_caches,
    extract_text_caches, pack_clips, pack_sequences, validate_processed_data, validate_sequence_packs,
)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Preprocess directly downloaded official BEAT2-English for TalkFlow.')
    p.add_argument('stage', choices=['index', 'clips', 'audio', 'text', 'pack', 'pack-sequences', 'validate', 'all'],
                   help='Pipeline stage (all = index -> clips -> audio -> text -> pack -> pack-sequences -> validate)')
    p.add_argument('--data-root', type=Path, required=True,
                   help='Path containing raw smplxflame_30/, wave16k/, textgrid/, train_test_split.csv')
    p.add_argument('--out-dir', type=Path, default=Path('data/processed_beat2_speaker2'))
    p.add_argument('--speaker-id', default='2',
                   help='CSV ID prefix; 2 = Scott (paper), all = all available speakers')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                   choices=['train', 'val', 'test'])
    p.add_argument('--missing-policy', choices=['skip', 'error'], default='skip')
    p.add_argument('--device', default='cuda', help='HuBERT device (cuda or cpu)')
    p.add_argument('--hubert-mode', choices=['last4', 'forward'], default='last4',
                   help='last4 averages the four last HuBERT layers (primary notebook)')
    p.add_argument('--fasttext-vec', type=Path,
                   help='Path to the external 300D FastText crawl-300d-2M.vec')
    p.add_argument('--fasttext-cache', type=Path,
                   help='Optional persistent gensim .kv file')
    p.add_argument('--sbert-model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--skip-text', action='store_true',
                   help='For Stage-I motion-only smoke test; omit FastText and SBERT in pack')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, help='Process first N entries per stage/split for smoke tests')
    p.add_argument('--verbose', action='store_true')
    return p


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s: %(message)s')
    root, out_dir = args.data_root.resolve(), args.out_dir.resolve()
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be >=1')
    stages = ['index', 'clips', 'audio', 'text', 'pack', 'pack-sequences', 'validate'] if args.stage == 'all' else [args.stage]
    if args.skip_text:
        stages = [s for s in stages if s != 'text']
    if 'text' in stages and args.fasttext_vec is None and args.fasttext_cache is None:
        raise SystemExit('Text extraction requires --fasttext-vec or an existing --fasttext-cache.')
    for stage in stages:
        print(f'\n== {stage.upper()} ==', flush=True)
        if stage == 'index':
            out = build_sequence_manifests(root, out_dir, speaker_id=args.speaker_id,
                                           missing_policy=args.missing_policy)
        elif stage == 'clips':
            out = build_clip_manifests(root, out_dir, splits=args.splits)
        elif stage == 'audio':
            out = extract_audio_caches(root, out_dir, splits=args.splits,
                                       device=args.device, mode=args.hubert_mode,
                                       overwrite=args.overwrite, limit=args.limit)
        elif stage == 'text':
            out = extract_text_caches(root, out_dir, splits=args.splits,
                                      fasttext_vec=args.fasttext_vec or Path('__unused__.vec'),
                                      fasttext_cache=args.fasttext_cache,
                                      sbert_model=args.sbert_model, overwrite=args.overwrite,
                                      limit=args.limit)
        elif stage == 'pack':
            out = pack_clips(root, out_dir, splits=args.splits,
                             skip_text=args.skip_text, overwrite=args.overwrite,
                             limit=args.limit)
        elif stage == 'pack-sequences':
            out = pack_sequences(root, out_dir, splits=args.splits, skip_text=args.skip_text,
                                 overwrite=args.overwrite, limit=args.limit)
        elif stage == 'validate':
            out = validate_processed_data(out_dir, splits=args.splits,
                                          skip_text=args.skip_text, limit=args.limit)
            out['full_sequences'] = validate_sequence_packs(out_dir, splits=args.splits,
                                                            skip_text=args.skip_text, limit=args.limit)
        else:
            raise AssertionError(stage)
        print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f'\nProcessed data location: {out_dir}', flush=True)


if __name__ == '__main__':
    main()
