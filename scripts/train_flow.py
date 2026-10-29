#!/usr/bin/env python3
"""Train Stage-II 3-stream flow, preserving existing model/checkpoint interfaces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import trange

from configs.models import AE_MODEL_KWARGS, CONDITION_KWARGS, FLOW_KWARGS
from configs.flow_train import STAGE_CONFIGS
from training.flow import train_flow_matching_one_epoch
from utils.speech_dataset import build_speech_dataloader


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--split',default='train')
    p.add_argument('--ae-checkpoint',required=True)
    p.add_argument('--resume',default=None,help='Previous flow checkpoint, required to continue stages')
    p.add_argument('--stage',type=int,choices=[1,2,3],default=1)
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--out-dir',default='flow_matching_checkpoints_3stream')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-batches',type=int,default=0,help='Optional debugging-only subset')
    args=p.parse_args(argv)
    device=torch.device(args.device)
    ds,loader=build_speech_dataloader(args.data,batch_size=args.batch_size,shuffle=True,
                                     clip_len=240,crop_mode='start',split=args.split)
    if args.max_batches:
        from itertools import islice
        # Re-create the iterator each epoch, preserving shuffle.
        loader_source=loader
        class _Limited:
            def __iter__(self):return islice(iter(loader_source),args.max_batches)
            def __len__(self):return min(len(loader_source),args.max_batches)
        loader=_Limited()
    from modules.ae_3stream import PoseExprVAE
    from modules.condition_encoder_3stream import ConditionEncoder3Stream
    from modules.flow_matching_backbone_3stream import FlowMatchingTransformerBackbone3Stream
    ae=PoseExprVAE(**AE_MODEL_KWARGS).to(device)
    ck=torch.load(args.ae_checkpoint,map_location=device,weights_only=False)
    ae.load_state_dict(ck.get('model_state_dict',ck),strict=True)
    ae.eval()
    for param in ae.parameters():param.requires_grad_(False)
    condition=ConditionEncoder3Stream(**CONDITION_KWARGS).to(device)
    flow=FlowMatchingTransformerBackbone3Stream(**FLOW_KWARGS).to(device)
    optim=torch.optim.AdamW([{'params':condition.parameters(),'lr':1e-5},
                              {'params':flow.parameters(),'lr':5e-5}],
                             betas=(0.9,0.99),weight_decay=1e-2)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=600,eta_min=1e-7)
    start_epoch=0
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False)
        condition.load_state_dict(ck['condition_encoder_state_dict'],strict=True)
        flow.load_state_dict(ck['flow_model_state_dict'],strict=True)
        if ck.get('optimizer_state_dict') is not None:optim.load_state_dict(ck['optimizer_state_dict'])
        if ck.get('scheduler_state_dict') is not None:scheduler.load_state_dict(ck['scheduler_state_dict'])
        if ck.get('stage')==args.stage:start_epoch=int(ck.get('epoch',0))
    config=dict(STAGE_CONFIGS[args.stage])
    outdir=Path(args.out_dir);outdir.mkdir(parents=True,exist_ok=True)
    for epoch in trange(start_epoch+1,start_epoch+args.epochs+1,desc=f'Flow stage {args.stage}'):
        stats=train_flow_matching_one_epoch(
            ae_model=ae,condition_encoder=condition,flow_model=flow,
            loader=loader,optimizer=optim,device=device,**config)
        scheduler.step()
        print(json.dumps({'stage':args.stage,'epoch':epoch,'lr':optim.param_groups[0]['lr'],'stats':stats}))
        if epoch%args.save_every==0 or epoch==start_epoch+args.epochs:
            path=outdir/f'flow_matching_3stream_stage{args.stage}_epoch_{epoch}.pt'
            torch.save({'stage':args.stage,'epoch':epoch,
                        'condition_encoder_state_dict':condition.state_dict(),
                        'flow_model_state_dict':flow.state_dict(),
                        'optimizer_state_dict':optim.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'train_cfg':config,'stats':stats},path)
            print('saved',path)


if __name__=='__main__':main()
