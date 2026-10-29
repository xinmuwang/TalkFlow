#!/usr/bin/env python3
"""Evaluate Stage-I reconstruction using local weights, optionally export examples."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from configs.models import AE_MODEL_KWARGS
from training.ae import ae_losses
from utils.dataset import build_dataloader
from utils.rotation_conversion import matrix_to_axis_angle, rotation_6d_to_matrix


def _aa(pose):
    return matrix_to_axis_angle(rotation_6d_to_matrix(pose.reshape(-1,6))).reshape(len(pose),165).cpu().numpy().astype(np.float32)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',default='test')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out-dir',default='outputs/ae_eval')
    p.add_argument('--save-examples',type=int,default=0)
    p.add_argument('--max-batches',type=int,default=0)
    args=p.parse_args(argv)
    device=torch.device(args.device)
    ds,loader=build_dataloader(args.data,batch_size=args.batch_size,shuffle=False,
                               crop_mode='start',clip_len=240,split=args.split,drop_last=False,
                               return_betas=True,return_trans=True)
    from modules.ae_3stream import PoseExprVAE
    model=PoseExprVAE(**AE_MODEL_KWARGS).to(device)
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ck.get('model_state_dict',ck),strict=True)
    model.eval()
    dest=Path(args.out_dir);dest.mkdir(parents=True,exist_ok=True)
    totals={};count=0;saved=0
    with torch.no_grad():
        for batch_idx,batch in enumerate(loader):
            if args.max_batches and batch_idx>=args.max_batches:break
            pose,expr=batch['pose'].to(device),batch['expression'].to(device)
            out=model(pose,expr)
            metrics=ae_losses(out,pose,expr)
            for k,v in metrics.items(): totals[k]=totals.get(k,0.)+v.item()*len(pose)
            count+=len(pose)
            if saved < args.save_examples:
                for i in range(len(pose)):
                    if saved>=args.save_examples:break
                    pr=out['pose_recon'][i].detach().cpu()
                    gt=pose[i].detach().cpu()
                    ep=out['expr_recon'][i].detach().cpu().numpy()
                    eg=expr[i].detach().cpu().numpy()
                    betas=(batch['betas'][i].cpu().numpy().astype(np.float32)
                           if 'betas' in batch else np.zeros(300,dtype=np.float32))
                    trans=(batch['trans'][i].cpu().numpy().astype(np.float32)
                           if 'trans' in batch else np.zeros((len(pr),3),dtype=np.float32))
                    meta=dict(betas=betas,trans=trans,model=np.str_('smplx2020'),
                              gender=np.str_('neutral'),mocap_frame_rate=np.int32(30))
                    np.savez_compressed(dest/f'ae_recon_{saved:04d}.npz',
                        poses=_aa(pr),expressions=ep.astype(np.float32),
                        **meta)
                    np.savez_compressed(dest/f'ae_gt_{saved:04d}.npz',
                        poses=_aa(gt),expressions=eg.astype(np.float32),
                        **meta)
                    saved+=1
    if count==0:raise RuntimeError('No validation samples processed')
    summary={'samples':count,'metrics':{k:totals[k]/count for k in totals}}
    (dest/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
