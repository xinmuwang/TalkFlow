"""Stage-I AE losses and evaluation. Mirrors the supplied active AE training cell."""
from __future__ import annotations

import torch
import torch.nn.functional as F

AE_WEIGHTS = {
    'pose': 5.0,
    'expr': 2.0,
    'pose_vel': 0.5,
    'expr_vel': 0.5,
    'pose_acc': 0.05,
    'expr_acc': 0.05,
}


def temporal_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def ae_losses(output: dict, pose: torch.Tensor, expr: torch.Tensor, weights: dict | None = None) -> dict:
    weights = AE_WEIGHTS if weights is None else weights
    pred_pose, pred_expr = output['pose_recon'], output['expr_recon']
    if pred_pose.shape != pose.shape or pred_expr.shape != expr.shape:
        raise ValueError(f'AE output/input shape mismatch: {pred_pose.shape} vs {pose.shape}, {pred_expr.shape} vs {expr.shape}')
    pvel, gvel = temporal_diff(pred_pose), temporal_diff(pose)
    evel, gevel = temporal_diff(pred_expr), temporal_diff(expr)
    pacc, gacc = temporal_diff(pvel), temporal_diff(gvel)
    eacc, geacc = temporal_diff(evel), temporal_diff(gevel)
    result = {
        'pose': F.mse_loss(pred_pose, pose),
        'expr': F.smooth_l1_loss(pred_expr, expr),
        'pose_vel': F.smooth_l1_loss(pvel, gvel),
        'expr_vel': F.smooth_l1_loss(evel, gevel),
        'pose_acc': F.smooth_l1_loss(pacc, gacc),
        'expr_acc': F.smooth_l1_loss(eacc, geacc),
    }
    result['total'] = sum(result[k] * weights[k] for k in result)
    return result


@torch.no_grad()
def evaluate_ae(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {k: 0.0 for k in list(AE_WEIGHTS) + ['total']}
    examples = 0
    for batch in loader:
        pose = batch['pose'].to(device)
        expr = batch['expression'].to(device)
        out = model(pose, expr)
        losses = ae_losses(out, pose, expr)
        B = len(pose)
        for k,v in losses.items():
            totals[k] += float(v) * B
        examples += B
    if not examples:
        raise ValueError('Evaluation DataLoader has zero batches')
    return {k: v/examples for k,v in totals.items()}
