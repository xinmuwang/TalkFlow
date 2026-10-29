# TalkFlow

**TalkFlow: Structured Latent Flow Matching for Speech Driven Holistic Motion Generation**  
*SIGGRAPH Asia 2026 Conference Papers*

[Project Page](https://xinmuwang.github.io/TalkFlow/) · [Paper](https://doi.org/10.1145/3829340.3842366)

TalkFlow generates coordinated body motion, hand gestures, and facial expressions from speech. This repository provides the implementation for BEAT2 data preparation, model training, motion reconstruction evaluation, and inference.

## Installation

Create a Python environment with PyTorch installed for your platform, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, use a CUDA-compatible PyTorch installation.

## Data Preparation

Download the [official BEAT2 English dataset](https://huggingface.co/datasets/H-Liu1997/BEAT2/tree/main/beat_english_v2.0.0) and prepare the data using:

```bash
python scripts/preprocess_beat2.py all \
    --data-root /path/to/BEAT2/beat_english_v2.0.0 \
    --out-dir data/beat2_processed \
    --speaker-id 2 \
    --fasttext-vec /path/to/crawl-300d-2M.vec
```

The script processes the downloaded BEAT2 files into datasets used by the training and inference scripts. Existing compatible BEAT2 training files can also be used directly.

## Training

**Stage I — Motion Autoencoder**

```bash
python scripts/train_ae.py \
    --data data/beat2_processed \
    --val-data data/beat2_processed \
    --out-dir checkpoints/ae
```

**Stage II — Speech-Conditioned Flow**

```bash
python scripts/train_flow.py \
    --data data/beat2_processed \
    --ae-checkpoint /path/to/ae_checkpoint.pt \
    --stage 1 \
    --out-dir checkpoints/flow
```

Use the training script's `--help` option for available settings, resuming checkpoints, and subsequent training stages. Equivalent launch notebooks are provided in `notebooks/`.

## Evaluation and Inference

Evaluate Stage I motion reconstruction:

```bash
python scripts/eval_ae.py \
    --data data/beat2_processed \
    --checkpoint /path/to/ae_checkpoint.pt \
    --out-dir outputs/ae_eval \
    --save-examples 5
```

Generate SMPL-X motion files:

```bash
python scripts/infer.py \
    --data data/beat2_processed \
    --ae-checkpoint /path/to/ae_checkpoint.pt \
    --flow-checkpoint /path/to/flow_checkpoint.pt \
    --out-dir outputs/generated_npz
```

Generated motion is saved as NPZ files. To check the exported file structure before using external tools:

```bash
python scripts/check_emage_npz.py outputs/generated_npz
```

### Benchmark Evaluation and Rendering

Our benchmark evaluation and SMPL-X rendering follow the **official EMAGE implementation**. We use the original EMAGE tools for FGD, BCS, DIV, facial MSE, LVD, and rendering, rather than supplying a separate implementation of these benchmarks.

Please obtain the evaluation and rendering tools from the [EMAGE / PantoMatrix repository](https://github.com/PantoMatrix/PantoMatrix). These external tools are not included here.

The provided `scripts/eval_ae.py` is for **autoencoder reconstruction evaluation only** and does not replace the official EMAGE benchmark evaluation.

## Tests

Run the included data-format and interface checks:

```bash
python -m pytest -q tests
```

These tests use synthetic inputs. Full inference and result quality should be checked locally using BEAT2 and the corresponding pretrained checkpoints.

## Citation

```bibtex
@inproceedings{wang2026talkflow,
  title={TalkFlow: Structured Latent Flow Matching for Speech Driven Holistic Motion Generation},
  author={Wang, Xinmu and Honda, Fumi and Gao, Xiang and Gu, Xianfeng and Song, Xiyun and Yu, Heather},
  booktitle={SIGGRAPH Asia 2026 Conference Papers},
  year={2026},
  doi={10.1145/3829340.3842366}
}
```

## Acknowledgments

We thank the authors of [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2), [EMAGE](https://github.com/PantoMatrix/PantoMatrix), and the libraries used in this project. The rotation-conversion utilities retain their upstream PyTorch3D attribution. Please follow the license and citation terms of each upstream resource.
