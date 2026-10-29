#!/usr/bin/env python3
"""Strict 3-stream long-sequence inference with optional current smoothing."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from configs.models import AE_MODEL_KWARGS, CONDITION_KWARGS, FLOW_KWARGS
from inference.flow import (
    generate_long_sequence_240_shift_last_flow_3stream,
    postprocess_all_generated_sequences,
    save_all_gen_out_to_npz_safe,
)
from utils.speech_dataset import build_long_test_dataloader


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,help='Full-sequence test PKL or processed BEAT2 directory')
    p.add_argument('--split',default='test')
    p.add_argument('--ae-checkpoint',required=True)
    p.add_argument('--flow-checkpoint',required=True)
    p.add_argument('--out-dir',default='outputs/generated_npz')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-steps',type=int,default=100)
    p.add_argument('--max-sequences',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-smooth',action='store_true')
    p.add_argument('--smooth-kernel',type=int,default=3)
    p.add_argument('--blend-radius',type=int,default=4)
    p.add_argument('--no-gt',action='store_true',help='Infer without reference pose/expression')
    args=p.parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device)
    dataset,loader=build_long_test_dataloader(args.data,batch_size=1,
                                              load_gt=not args.no_gt,split=args.split)
    from modules.ae_3stream import PoseExprVAE
    from modules.condition_encoder_3stream import ConditionEncoder3Stream
    from modules.flow_matching_backbone_3stream import FlowMatchingTransformerBackbone3Stream
    ae=PoseExprVAE(**AE_MODEL_KWARGS).to(device)
    ck=torch.load(args.ae_checkpoint,map_location=device,weights_only=False)
    ae.load_state_dict(ck.get('model_state_dict',ck),strict=True)
    condition=ConditionEncoder3Stream(**CONDITION_KWARGS).to(device)
    flow=FlowMatchingTransformerBackbone3Stream(**FLOW_KWARGS).to(device)
    ck=torch.load(args.flow_checkpoint,map_location=device,weights_only=False)
    condition.load_state_dict(ck['condition_encoder_state_dict'],strict=True)
    flow.load_state_dict(ck['flow_model_state_dict'],strict=True)
    result=generate_long_sequence_240_shift_last_flow_3stream(
        ae_model=ae,condition_encoder=condition,flow_model=flow,loader=loader,device=device,
        clip_len=240,downsample_factor=4,num_steps=args.num_steps,
        max_sequences=None if args.max_sequences<=0 else args.max_sequences,
        init_type='noise',use_sequence_level_noise=True,
        upper_noise_type='smooth',lower_noise_type='smooth',expr_noise_type='zero',
        upper_noise_sigma=0.2,lower_noise_sigma=0.08,expr_noise_sigma=0.01,
        upper_noise_kernel=31,lower_noise_kernel=31,expr_noise_kernel=1,
        upper_style_weight=0.2,upper_smooth_weight=0.9,
        lower_style_weight=0.2,lower_smooth_weight=0.9,
    )
    if not args.no_smooth:
        result=postprocess_all_generated_sequences(result,
                      smooth_kernel_size=args.smooth_kernel,blend_radius=args.blend_radius)
    paths=save_all_gen_out_to_npz_safe(result,loader,save_dir=args.out_dir)
    print('Saved',len(paths),'generated motion files')


if __name__=='__main__':main()
