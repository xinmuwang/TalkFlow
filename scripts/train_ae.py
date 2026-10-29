#!/usr/bin/env python3
"""Train the supplied 3-stream motion autoencoder."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm

from configs.models import AE_MODEL_KWARGS
from training.ae import ae_losses, evaluate_ae
from utils.dataset import build_dataloader


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Legacy PKL or processed NPZ directory/manifest')
    p.add_argument('--val-data', default=None)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--out-dir', default='checkpoints_3stream')
    p.add_argument('--resume', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--split', default='train')
    args = p.parse_args(argv)
    device = torch.device(args.device)
    from modules.ae_3stream import PoseExprVAE
    ds, loader = build_dataloader(args.data, batch_size=args.batch_size, shuffle=True,
                                  clip_len=240, crop_mode='start', split=args.split)
    model = PoseExprVAE(**AE_MODEL_KWARGS).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9,0.99),weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=500,eta_min=1e-7)
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'], strict=True)
        if 'optimizer_state_dict' in ck: optim.load_state_dict(ck['optimizer_state_dict'])
        if 'scheduler_state_dict' in ck: scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch = int(ck.get('epoch', 0))
    val_loader = None
    if args.val_data:
        _, val_loader = build_dataloader(args.val_data, batch_size=1, shuffle=False,
                                         clip_len=240, crop_mode='start', split='val', drop_last=False)
    outdir = Path(args.out_dir);outdir.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch+1,start_epoch+args.epochs+1):
        model.train()
        total=0.0;steps=0
        for batch in tqdm(loader,desc=f'AE epoch {epoch}',leave=False):
            pose=batch['pose'].to(device)
            expr=batch['expression'].to(device)
            optim.zero_grad(set_to_none=True)
            losses=ae_losses(model(pose, expr),pose,expr)
            losses['total'].backward()
            optim.step()
            total+=float(losses['total'].detach());steps+=1
        if steps==0: raise ValueError('Training loader is empty (check batch-size/drop-last)')
        scheduler.step()
        stats={'epoch':epoch,'train_total':total/steps}
        if val_loader is not None:
            stats['val']=evaluate_ae(model,val_loader,device)
        print(json.dumps(stats))
        if epoch % args.save_every==0 or epoch==start_epoch+args.epochs:
            ck={'epoch':epoch, 'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optim.state_dict(),'scheduler_state_dict':scheduler.state_dict()}
            path=outdir/f'ae_3stream_joint_epoch_{epoch:04d}.pt'
            torch.save(ck,path)
            print('saved',path)


if __name__=='__main__': main()
