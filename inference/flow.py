"""Inference and smoothing from supplied strict 3-stream inference notebook.

Postprocess functions intentionally retain the additional motion smoothing.
"""
import os
import random
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle

def resize_latent_to_target_len(z, target_T):
    """z: [1/B, T, D] -> [1/B, target_T, D]."""
    if z.shape[1] == target_T:
        return z

    return F.interpolate(
        z.transpose(1, 2),
        size=target_T,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def smooth_latent_batch(z, kernel_size=9):
    """z: [B, T, D]."""
    if kernel_size <= 1:
        return z
    assert kernel_size % 2 == 1

    B, T, D = z.shape
    x = z.transpose(1, 2)
    pad = kernel_size // 2
    x = F.pad(x, (pad, pad), mode="replicate")

    weight = torch.ones(D, 1, kernel_size, device=z.device, dtype=z.dtype) / kernel_size
    x = F.conv1d(x, weight, groups=D)
    return x.transpose(1, 2)

import torch
import torch.nn.functional as F
import random


def _first_existing_key(d, keys, required=True, name="value"):
    for k in keys:
        if k in d:
            return d[k]
    if required:
        raise KeyError(f"Cannot find {name}. Tried keys: {keys}. Available keys: {list(d.keys())}")
    return None


@torch.no_grad()
def ae_encode_3stream(ae_model, pose, expr):
    """
    Strict wrapper for a real 3-stream AE encoder.

    Required output keys:
        z_upper, z_lower, z_expr
    """
    if hasattr(ae_model, "encode_to_latents"):
        enc_out = ae_model.encode_to_latents(pose=pose, expr=expr)
    elif hasattr(ae_model, "encode"):
        enc_out = ae_model.encode(pose=pose, expr=expr)
    else:
        raise AttributeError("3-stream AE must provide encode_to_latents(...) or encode(...).")

    if not isinstance(enc_out, dict):
        raise TypeError("3-stream AE encode output must be a dict with z_upper, z_lower, z_expr.")

    required = ["z_upper", "z_lower", "z_expr"]
    missing = [k for k in required if k not in enc_out]
    if missing:
        raise KeyError(f"3-stream AE encode missing keys {missing}. Available keys: {list(enc_out.keys())}")

    return {
        "z_upper": enc_out["z_upper"],
        "z_lower": enc_out["z_lower"],
        "z_expr": enc_out["z_expr"],
    }


@torch.no_grad()
def ae_decode_3stream(ae_model, z_upper, z_lower, z_expr):
    """
    Strict wrapper for a real 3-stream AE decoder.

    Required call convention:
        decode_from_latents(z_upper=z_upper, z_lower=z_lower, z_expr=z_expr)
    Required output keys:
        pose_recon, expr_recon
    """
    if not hasattr(ae_model, "decode_from_latents"):
        raise AttributeError("3-stream AE must provide decode_from_latents(...).")

    dec_out = ae_model.decode_from_latents(
        z_upper=z_upper,
        z_lower=z_lower,
        z_expr=z_expr,
    )

    if not isinstance(dec_out, dict):
        raise TypeError("3-stream AE decode output must be a dict with pose_recon and expr_recon.")

    required = ["pose_recon", "expr_recon"]
    missing = [k for k in required if k not in dec_out]
    if missing:
        raise KeyError(f"3-stream AE decode missing keys {missing}. Available keys: {list(dec_out.keys())}")

    return {"pose_recon": dec_out["pose_recon"], "expr_recon": dec_out["expr_recon"], **dec_out}


@torch.no_grad()
def extract_valid_240_latent_init_3stream(
    ae_model,
    valid_loader,
    device,
    clip_len=240,
    downsample_factor=4,
    upper_latent_dim=512,
    lower_latent_dim=128,
    expr_latent_dim=512,
    crop_mode="middle",      # "middle" / "first" / "random"
    max_batches=None,
):
    """
    Extract one validation clip and encode it into 3-stream latent init.

    return:
        z_upper_init: [1, clip_len/downsample_factor, upper_latent_dim]
        z_lower_init: [1, clip_len/downsample_factor, lower_latent_dim]
        z_expr_init:  [1, clip_len/downsample_factor, expr_latent_dim]
    """

    ae_model.eval()
    target_T_latent = clip_len // downsample_factor

    selected_batch = None
    selected_b = 0
    selected_batch_idx = None

    for batch_idx, batch in enumerate(valid_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        pose = batch["pose"]
        B, T = pose.shape[:2]
        if T >= clip_len:
            selected_batch = batch
            selected_batch_idx = batch_idx
            break

    if selected_batch is None:
        raise RuntimeError(f"No validation sample has length >= clip_len={clip_len}.")

    pose = selected_batch["pose"][selected_b:selected_b + 1].to(device)
    expr = selected_batch["expression"][selected_b:selected_b + 1].to(device)
    T = pose.shape[1]

    if crop_mode == "first":
        start = 0
    elif crop_mode == "middle":
        start = max((T - clip_len) // 2, 0)
    elif crop_mode == "random":
        start = random.randint(0, T - clip_len)
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")

    end = start + clip_len
    pose_clip = pose[:, start:end]
    expr_clip = expr[:, start:end]

    enc_out = ae_encode_3stream(ae_model, pose_clip, expr_clip)

    z_upper_init = resize_latent_to_target_len(enc_out["z_upper"].detach(), target_T_latent)
    z_lower_init = resize_latent_to_target_len(enc_out["z_lower"].detach(), target_T_latent)
    z_expr_init = resize_latent_to_target_len(enc_out["z_expr"].detach(), target_T_latent)

    assert z_upper_init.shape == (1, target_T_latent, upper_latent_dim), z_upper_init.shape
    assert z_lower_init.shape == (1, target_T_latent, lower_latent_dim), z_lower_init.shape
    assert z_expr_init.shape == (1, target_T_latent, expr_latent_dim), z_expr_init.shape

    seq_id = None
    for key in ["sequence_id", "seq_id", "id", "name", "seq_name", "motion_id"]:
        if key in selected_batch:
            value = selected_batch[key]
            seq_id = str(value[selected_b]) if isinstance(value, (list, tuple)) else str(value)
            break

    print("========== Valid 3-stream init clip ==========")
    print("batch_idx:", selected_batch_idx)
    print("seq_id:", seq_id)
    print("valid full T:", T)
    print("crop:", start, end)
    print("pose_clip:", pose_clip.shape)
    print("expr_clip:", expr_clip.shape)
    print("z_upper_init:", z_upper_init.shape, "mean/std:", z_upper_init.mean().item(), z_upper_init.std().item())
    print("z_lower_init:", z_lower_init.shape, "mean/std:", z_lower_init.mean().item(), z_lower_init.std().item())
    print("z_expr_init:", z_expr_init.shape, "mean/std:", z_expr_init.mean().item(), z_expr_init.std().item())

    return {
        "z_upper_init": z_upper_init,
        "z_lower_init": z_lower_init,
        "z_expr_init": z_expr_init,
        "pose_clip": pose_clip.detach(),
        "expr_clip": expr_clip.detach(),
        "seq_id": seq_id,
        "start": start,
        "end": end,
    }


import torch
import torch.nn.functional as F


@torch.no_grad()
def temporal_smooth_noise(x, kernel_size=9, normalize=True):
    """x: [B, T, D]."""
    if kernel_size <= 1:
        return x
    assert kernel_size % 2 == 1
    pad = kernel_size // 2

    h = x.transpose(1, 2)
    h = F.pad(h, (pad, pad), mode="replicate")
    h = F.avg_pool1d(h, kernel_size=kernel_size, stride=1)
    h = h.transpose(1, 2)

    if normalize:
        std = h.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        h = h / std
    return h


@torch.no_grad()
def resize_or_expand_init_latent(z_init, B, T_latent, D, device, dtype, name="z_init"):
    """Prepare externally provided latent initialization."""
    if z_init is None:
        return None

    z = z_init.to(device=device, dtype=dtype).detach().clone()
    if z.ndim != 3:
        raise ValueError(f"{name} should be [B,T,D] or [1,T,D], got {z.shape}")
    if z.shape[-1] != D:
        raise ValueError(f"{name} latent dim mismatch: expected D={D}, got {z.shape[-1]}")

    if z.shape[0] == 1 and B > 1:
        z = z.expand(B, -1, -1).clone()
    elif z.shape[0] != B:
        raise ValueError(f"{name} batch mismatch: expected B={B} or 1, got {z.shape[0]}")

    if z.shape[1] != T_latent:
        z = F.interpolate(
            z.transpose(1, 2),
            size=T_latent,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    assert z.shape == (B, T_latent, D), f"{name} final shape mismatch: {z.shape}"
    return z


@torch.no_grad()
def make_latent_source(
    B,
    T_latent,
    D,
    device,
    dtype=torch.float32,
    noise_type="style_smooth",
    sigma=0.5,
    smooth_kernel=9,
    style_weight=0.7,
    smooth_weight=0.3,
):
    """Create one stream source x0 for FM sampling."""
    if noise_type == "zero":
        x0 = torch.zeros(B, T_latent, D, device=device, dtype=dtype)
    elif noise_type == "white":
        x0 = torch.randn(B, T_latent, D, device=device, dtype=dtype)
    elif noise_type == "smooth":
        x0 = torch.randn(B, T_latent, D, device=device, dtype=dtype)
        x0 = temporal_smooth_noise(x0, kernel_size=smooth_kernel, normalize=True)
    elif noise_type == "style":
        x0 = torch.randn(B, 1, D, device=device, dtype=dtype).expand(B, T_latent, D)
        x0 = x0 / x0.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    elif noise_type == "style_smooth":
        style = torch.randn(B, 1, D, device=device, dtype=dtype).expand(B, T_latent, D)
        residual = torch.randn(B, T_latent, D, device=device, dtype=dtype)
        residual = temporal_smooth_noise(residual, kernel_size=smooth_kernel, normalize=True)
        x0 = style_weight * style + smooth_weight * residual
        x0 = x0 / x0.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return sigma * x0


def _split_condition_for_3stream(cond_out):
    """
    Strict 3-stream condition split.

    Required condition keys:
        cond_upper, cond_lower, cond_expr

    No 2-stream fallback is allowed here.
    """
    if not isinstance(cond_out, dict):
        raise TypeError("3-stream ConditionEncoder output must be a dict.")

    required = ["cond_upper", "cond_lower", "cond_expr"]
    missing = [k for k in required if k not in cond_out]
    if missing:
        raise KeyError(
            "ConditionEncoder is not producing strict 3-stream conditions. "
            f"Missing keys: {missing}. Available keys: {list(cond_out.keys())}"
        )

    cond_upper = cond_out["cond_upper"]
    cond_lower = cond_out["cond_lower"]
    cond_expr = cond_out["cond_expr"]
    seq_mask = cond_out.get("seq_mask", None)

    if cond_upper.shape[:2] != cond_lower.shape[:2] or cond_upper.shape[:2] != cond_expr.shape[:2]:
        raise ValueError(
            "Condition stream time shapes mismatch: "
            f"cond_upper={tuple(cond_upper.shape)}, "
            f"cond_lower={tuple(cond_lower.shape)}, "
            f"cond_expr={tuple(cond_expr.shape)}"
        )

    return cond_upper, cond_lower, cond_expr, seq_mask


def _call_flow_model_3stream(flow_model, z_upper, z_lower, z_expr, cond_upper, cond_lower, cond_expr, t, seq_mask):
    """
    Strict 3-stream flow-model call.

    Required forward convention:
        flow_model(
            x_upper_t=..., x_lower_t=..., x_expr_t=...,
            cond_upper=..., cond_lower=..., cond_expr=...,
            t=..., seq_mask=...
        )
    Required output keys:
        v_upper, v_lower, v_expr
    """
    kwargs = dict(
        x_upper_t=z_upper,
        x_lower_t=z_lower,
        x_expr_t=z_expr,
        cond_upper=cond_upper,
        cond_lower=cond_lower,
        cond_expr=cond_expr,
        t=t,
    )
    if seq_mask is not None:
        kwargs["seq_mask"] = seq_mask

    out = flow_model(**kwargs)
    if not isinstance(out, dict):
        raise TypeError("3-stream flow_model forward must return a dict with v_upper, v_lower, v_expr.")

    required = ["v_upper", "v_lower", "v_expr"]
    missing = [k for k in required if k not in out]
    if missing:
        raise KeyError(f"3-stream flow output missing keys {missing}. Available keys: {list(out.keys())}")

    return out["v_upper"], out["v_lower"], out["v_expr"]


@torch.no_grad()
def sample_flow_matching_euler_3stream(
    flow_model,
    cond_upper,
    cond_lower,
    cond_expr,
    seq_mask=None,
    upper_latent_dim=512,
    lower_latent_dim=128,
    expr_latent_dim=512,
    num_steps=20,
    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",
    upper_noise_sigma=0.5,
    lower_noise_sigma=0.5,
    expr_noise_sigma=1.0,
    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,
    upper_style_weight=0.7,
    upper_smooth_weight=0.3,
    lower_style_weight=0.7,
    lower_smooth_weight=0.3,
    z_upper_init=None,
    z_lower_init=None,
    z_expr_init=None,
):
    """Euler sampling for 3-stream Flow Matching."""
    flow_model.eval()

    B, T_latent, _ = cond_upper.shape
    device = cond_upper.device
    dtype = cond_upper.dtype

    if z_upper_init is not None:
        z_upper = resize_or_expand_init_latent(z_upper_init, B, T_latent, upper_latent_dim, device, dtype, "z_upper_init")
    else:
        z_upper = make_latent_source(B, T_latent, upper_latent_dim, device, dtype, upper_noise_type, upper_noise_sigma, upper_noise_kernel, upper_style_weight, upper_smooth_weight)

    if z_lower_init is not None:
        z_lower = resize_or_expand_init_latent(z_lower_init, B, T_latent, lower_latent_dim, device, dtype, "z_lower_init")
    else:
        z_lower = make_latent_source(B, T_latent, lower_latent_dim, device, dtype, lower_noise_type, lower_noise_sigma, lower_noise_kernel, lower_style_weight, lower_smooth_weight)

    if z_expr_init is not None:
        z_expr = resize_or_expand_init_latent(z_expr_init, B, T_latent, expr_latent_dim, device, dtype, "z_expr_init")
    else:
        z_expr = make_latent_source(B, T_latent, expr_latent_dim, device, dtype, expr_noise_type, expr_noise_sigma, expr_noise_kernel)

    if seq_mask is not None:
        m = seq_mask.unsqueeze(-1).to(device=device, dtype=dtype)
        z_upper = z_upper * m
        z_lower = z_lower * m
        z_expr = z_expr * m
    else:
        m = None

    dt = 1.0 / float(num_steps)
    for i in range(num_steps):
        t = torch.full((B,), i / float(num_steps), device=device, dtype=dtype)

        v_upper, v_lower, v_expr = _call_flow_model_3stream(
            flow_model=flow_model,
            z_upper=z_upper,
            z_lower=z_lower,
            z_expr=z_expr,
            cond_upper=cond_upper,
            cond_lower=cond_lower,
            cond_expr=cond_expr,
            t=t,
            seq_mask=seq_mask,
        )

        z_upper = z_upper + dt * v_upper
        z_lower = z_lower + dt * v_lower
        z_expr = z_expr + dt * v_expr

        if m is not None:
            z_upper = z_upper * m
            z_lower = z_lower * m
            z_expr = z_expr * m

    return z_upper, z_lower, z_expr


import torch
import torch.nn.functional as F


@torch.no_grad()
def build_sequence_source_latents_3stream(
    T_raw,
    clip_len,
    downsample_factor,
    windows,
    device,
    upper_latent_dim=512,
    lower_latent_dim=128,
    expr_latent_dim=512,

    init_type="noise",  # "noise" / "zero"

    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",
    upper_noise_sigma=0.5,
    lower_noise_sigma=0.5,
    expr_noise_sigma=1.0,
    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,
    upper_style_weight=0.7,
    upper_smooth_weight=0.3,
    lower_style_weight=0.7,
    lower_smooth_weight=0.3,
):
    """
    Build continuous sequence-level source latents for all 3 streams.
    This keeps long-sequence trend/style continuous across shifted 240-frame windows.
    """

    latent_ends = []
    for start, end in windows:
        cur_clip_len = end - start
        cur_T_latent = cur_clip_len // downsample_factor
        latent_start = start // downsample_factor
        latent_ends.append(latent_start + cur_T_latent)

    T_latent_total = max(latent_ends)

    if init_type == "zero":
        z_upper_source_seq = torch.zeros(1, T_latent_total, upper_latent_dim, device=device)
        z_lower_source_seq = torch.zeros(1, T_latent_total, lower_latent_dim, device=device)
        z_expr_source_seq = torch.zeros(1, T_latent_total, expr_latent_dim, device=device)

    elif init_type == "noise":
        z_upper_source_seq = make_latent_source(
            B=1,
            T_latent=T_latent_total,
            D=upper_latent_dim,
            device=device,
            dtype=torch.float32,
            noise_type=upper_noise_type,
            sigma=upper_noise_sigma,
            smooth_kernel=upper_noise_kernel,
            style_weight=upper_style_weight,
            smooth_weight=upper_smooth_weight,
        )
        z_lower_source_seq = make_latent_source(
            B=1,
            T_latent=T_latent_total,
            D=lower_latent_dim,
            device=device,
            dtype=torch.float32,
            noise_type=lower_noise_type,
            sigma=lower_noise_sigma,
            smooth_kernel=lower_noise_kernel,
            style_weight=lower_style_weight,
            smooth_weight=lower_smooth_weight,
        )
        z_expr_source_seq = make_latent_source(
            B=1,
            T_latent=T_latent_total,
            D=expr_latent_dim,
            device=device,
            dtype=torch.float32,
            noise_type=expr_noise_type,
            sigma=expr_noise_sigma,
            smooth_kernel=expr_noise_kernel,
        )

    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    return z_upper_source_seq, z_lower_source_seq, z_expr_source_seq


@torch.no_grad()
def slice_window_source_latents_3stream(
    z_upper_source_seq,
    z_lower_source_seq,
    z_expr_source_seq,
    start,
    end,
    downsample_factor,
):
    """Slice 3 stream source latents for one raw-frame window."""

    cur_clip_len = end - start
    T_window_latent = cur_clip_len // downsample_factor
    latent_start = start // downsample_factor
    latent_end = latent_start + T_window_latent

    z_upper_init = z_upper_source_seq[:, latent_start:latent_end]
    z_lower_init = z_lower_source_seq[:, latent_start:latent_end]
    z_expr_init = z_expr_source_seq[:, latent_start:latent_end]

    for name, z in [("z_upper_init", z_upper_init), ("z_lower_init", z_lower_init), ("z_expr_init", z_expr_init)]:
        if z.shape[1] != T_window_latent:
            raise ValueError(
                f"{name} length mismatch: expected {T_window_latent}, got {z.shape[1]}, "
                f"latent_start={latent_start}, latent_end={latent_end}"
            )

    return z_upper_init, z_lower_init, z_expr_init

@torch.no_grad()
def generate_one_chunk_flow_3stream(
    ae_model,
    condition_encoder,
    flow_model,
    batch,
    device,
    num_steps=20,
    upper_latent_dim=512,
    lower_latent_dim=128,
    expr_latent_dim=512,

    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",
    upper_noise_sigma=0.5,
    lower_noise_sigma=0.5,
    expr_noise_sigma=1.0,
    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,
    upper_style_weight=0.7,
    upper_smooth_weight=0.3,
    lower_style_weight=0.7,
    lower_smooth_weight=0.3,

    z_upper_init=None,
    z_lower_init=None,
    z_expr_init=None,
):
    """
    Generate one chunk with 3-stream Flow Matching.

    Returns:
        pose_recon: [B, T, 55, 6]
        expr_recon: [B, T, 100]
        z_upper:    [B, T_latent, upper_latent_dim]
        z_lower:    [B, T_latent, lower_latent_dim]
        z_expr:     [B, T_latent, expr_latent_dim]
    """

    ae_model.eval()
    condition_encoder.eval()
    flow_model.eval()

    hubert = batch["hubert"].to(device, non_blocking=True)
    librosa = batch["librosa"].to(device, non_blocking=True)
    text_local_raw = batch["text_local_raw"].to(device, non_blocking=True)
    text_local_mask = batch["text_local_mask"].to(device, non_blocking=True)
    text_global_raw = batch["text_global_raw"].to(device, non_blocking=True)

    valid_mask = batch.get("valid_mask", None)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device, non_blocking=True)

    B = hubert.shape[0]

    cond_out = condition_encoder(
        hubert=hubert,
        librosa=librosa,
        text_local_raw=text_local_raw,
        text_global_raw=text_global_raw,
        valid_mask=valid_mask,
        text_local_mask=text_local_mask,
    )

    cond_upper, cond_lower, cond_expr, seq_mask = _split_condition_for_3stream(cond_out)
    T_latent = cond_upper.shape[1]

    # External source latents are resized/expanded inside the sampler.
    z_upper, z_lower, z_expr = sample_flow_matching_euler_3stream(
        flow_model=flow_model,
        cond_upper=cond_upper,
        cond_lower=cond_lower,
        cond_expr=cond_expr,
        seq_mask=seq_mask,
        upper_latent_dim=upper_latent_dim,
        lower_latent_dim=lower_latent_dim,
        expr_latent_dim=expr_latent_dim,
        num_steps=num_steps,

        upper_noise_type=upper_noise_type,
        lower_noise_type=lower_noise_type,
        expr_noise_type=expr_noise_type,
        upper_noise_sigma=upper_noise_sigma,
        lower_noise_sigma=lower_noise_sigma,
        expr_noise_sigma=expr_noise_sigma,
        upper_noise_kernel=upper_noise_kernel,
        lower_noise_kernel=lower_noise_kernel,
        expr_noise_kernel=expr_noise_kernel,
        upper_style_weight=upper_style_weight,
        upper_smooth_weight=upper_smooth_weight,
        lower_style_weight=lower_style_weight,
        lower_smooth_weight=lower_smooth_weight,

        z_upper_init=z_upper_init,
        z_lower_init=z_lower_init,
        z_expr_init=z_expr_init,
    )

    dec_out = ae_decode_3stream(
        ae_model=ae_model,
        z_upper=z_upper,
        z_lower=z_lower,
        z_expr=z_expr,
    )

    return {
        "pose_recon": dec_out["pose_recon"],
        "expr_recon": dec_out["expr_recon"],
        "z_upper": z_upper,
        "z_lower": z_lower,
        "z_expr": z_expr,
        "seq_mask": seq_mask,
    }

def make_sliding_240_windows(T, clip_len=240):
    """
    Create non-overlap 240-frame windows.
    If the last segment is shorter than clip_len, slide the final window backward
    so that it ends exactly at T.

    Example:
        T=1704, clip_len=240
        -> [(0,240), ..., (1440,1680), (1464,1704)]
    """
    if T <= clip_len:
        return [(0, T)]

    windows = []
    start = 0

    while start + clip_len <= T:
        windows.append((start, start + clip_len))
        start += clip_len

    # If already exactly covered, return.
    if windows[-1][1] == T:
        return windows

    # Add final shifted window ending at T.
    final_start = T - clip_len
    final_end = T

    # Avoid duplicate if already same as last.
    if windows[-1] != (final_start, final_end):
        windows.append((final_start, final_end))

    return windows

def slice_batch_time(batch, start, end):
    """
    Slice time-dependent tensors in batch.

    Time-dependent:
        hubert, librosa, text_local_raw, text_local_mask, valid_mask,
        pose, expression

    Non-time-dependent:
        text_global_raw, speaker_id, etc.
    """
    sub = {}

    time_keys = {
        "hubert",
        "librosa",
        "text_local_raw",
        "text_local_mask",
        "valid_mask",
        "pose",
        "expression",
    }

    for k, v in batch.items():
        if torch.is_tensor(v) and k in time_keys:
            sub[k] = v[:, start:end].contiguous()
        else:
            sub[k] = v

    return sub

def append_nonoverlap_chunk(
    chunks,
    chunk,
    window_start,
    window_end,
    current_end,
):
    """
    Append only the non-overlapping part of chunk.

    Args:
        chunks: list of tensors
        chunk: [clip_len, ...]
        window_start, window_end: raw frame indices of this chunk
        current_end: already generated raw end index

    Returns:
        updated current_end
    """
    if current_end <= window_start:
        # No overlap.
        keep_start = 0
    else:
        # Overlap exists. Keep only new frames.
        keep_start = current_end - window_start

    keep_start = int(max(0, min(keep_start, chunk.shape[0])))

    if keep_start < chunk.shape[0]:
        chunks.append(chunk[keep_start:])
        current_end = window_end

    return current_end

def get_nonoverlap_keep_start(window_start, current_end, chunk_len):
    """
    Return the local index from which this chunk should be kept.

    window_start: global raw/latent start index
    current_end: already concatenated global end index
    chunk_len: local chunk length
    """
    if current_end <= window_start:
        keep_start = 0
    else:
        keep_start = current_end - window_start

    keep_start = int(max(0, min(keep_start, chunk_len)))
    return keep_start

@torch.no_grad()
def generate_long_sequence_240_shift_last_flow_3stream(
    ae_model,
    condition_encoder,
    flow_model,
    loader,
    device,
    clip_len=240,
    downsample_factor=4,
    num_steps=40,
    upper_latent_dim=512,
    lower_latent_dim=128,
    expr_latent_dim=512,
    max_sequences=None,

    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",
    upper_noise_sigma=0.5,
    lower_noise_sigma=0.5,
    expr_noise_sigma=1.0,
    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,
    upper_style_weight=0.7,
    upper_smooth_weight=0.3,
    lower_style_weight=0.7,
    lower_smooth_weight=0.3,

    init_type="noise",  # "noise" / "zero"
    use_sequence_level_noise=True,
):
    """
    Long sequence generation with 240-frame chunks and 3 latent streams.

    Sequence-level source is recommended because it makes the source latent
    continuous across shifted windows, reducing long-sequence discontinuities.
    """

    ae_model.eval()
    condition_encoder.eval()
    flow_model.eval()

    all_pose_sequences = []
    all_expr_sequences = []
    all_z_upper_sequences = []
    all_z_lower_sequences = []
    all_z_expr_sequences = []
    all_windows = []

    for seq_idx, batch in enumerate(loader):
        if max_sequences is not None and seq_idx >= max_sequences:
            break

        B, T_raw, _ = batch["hubert"].shape
        if B != 1:
            raise ValueError(
                f"This function assumes batch_size=1, got B={B}. "
                "Please set test_loader batch_size=1 for now."
            )

        windows = make_sliding_240_windows(T_raw, clip_len=clip_len)

        if use_sequence_level_noise:
            z_upper_source_seq, z_lower_source_seq, z_expr_source_seq = build_sequence_source_latents_3stream(
                T_raw=T_raw,
                clip_len=clip_len,
                downsample_factor=downsample_factor,
                windows=windows,
                device=device,
                upper_latent_dim=upper_latent_dim,
                lower_latent_dim=lower_latent_dim,
                expr_latent_dim=expr_latent_dim,
                init_type=init_type,
                upper_noise_type=upper_noise_type,
                lower_noise_type=lower_noise_type,
                expr_noise_type=expr_noise_type,
                upper_noise_sigma=upper_noise_sigma,
                lower_noise_sigma=lower_noise_sigma,
                expr_noise_sigma=expr_noise_sigma,
                upper_noise_kernel=upper_noise_kernel,
                lower_noise_kernel=lower_noise_kernel,
                expr_noise_kernel=expr_noise_kernel,
                upper_style_weight=upper_style_weight,
                upper_smooth_weight=upper_smooth_weight,
                lower_style_weight=lower_style_weight,
                lower_smooth_weight=lower_smooth_weight,
            )

            print(
                f"[seq {seq_idx}] sequence-level source:",
                "z_upper_source_seq", tuple(z_upper_source_seq.shape),
                "z_lower_source_seq", tuple(z_lower_source_seq.shape),
                "z_expr_source_seq", tuple(z_expr_source_seq.shape),
            )
        else:
            z_upper_source_seq = None
            z_lower_source_seq = None
            z_expr_source_seq = None

        pose_parts = []
        expr_parts = []
        z_upper_parts = []
        z_lower_parts = []
        z_expr_parts = []

        current_raw_end = 0
        current_latent_end = 0

        for start, end in windows:
            chunk_batch = slice_batch_time(batch, start, end)

            if use_sequence_level_noise:
                z_upper_init, z_lower_init, z_expr_init = slice_window_source_latents_3stream(
                    z_upper_source_seq=z_upper_source_seq,
                    z_lower_source_seq=z_lower_source_seq,
                    z_expr_source_seq=z_expr_source_seq,
                    start=start,
                    end=end,
                    downsample_factor=downsample_factor,
                )
            else:
                z_upper_init = None
                z_lower_init = None
                z_expr_init = None

            out = generate_one_chunk_flow_3stream(
                ae_model=ae_model,
                condition_encoder=condition_encoder,
                flow_model=flow_model,
                batch=chunk_batch,
                device=device,
                num_steps=num_steps,
                upper_latent_dim=upper_latent_dim,
                lower_latent_dim=lower_latent_dim,
                expr_latent_dim=expr_latent_dim,
                upper_noise_type=upper_noise_type,
                lower_noise_type=lower_noise_type,
                expr_noise_type=expr_noise_type,
                upper_noise_sigma=upper_noise_sigma,
                lower_noise_sigma=lower_noise_sigma,
                expr_noise_sigma=expr_noise_sigma,
                upper_noise_kernel=upper_noise_kernel,
                lower_noise_kernel=lower_noise_kernel,
                expr_noise_kernel=expr_noise_kernel,
                upper_style_weight=upper_style_weight,
                upper_smooth_weight=upper_smooth_weight,
                lower_style_weight=lower_style_weight,
                lower_smooth_weight=lower_smooth_weight,
                z_upper_init=z_upper_init,
                z_lower_init=z_lower_init,
                z_expr_init=z_expr_init,
            )

            pose_chunk = out["pose_recon"][0].detach().cpu()
            expr_chunk = out["expr_recon"][0].detach().cpu()
            # The temporal decoder may pad a final shorter, non-multiple-of-four
            # window. Keep the original frame count for consistent NPZ export.
            expected_frames = end - start
            if pose_chunk.shape[0] < expected_frames or expr_chunk.shape[0] < expected_frames:
                raise RuntimeError(f"Decoder produced fewer frames than input window: "
                                   f"pose={len(pose_chunk)}, expr={len(expr_chunk)}, expected={expected_frames}")
            pose_chunk = pose_chunk[:expected_frames]
            expr_chunk = expr_chunk[:expected_frames]
            z_upper_chunk = out["z_upper"][0].detach().cpu()
            z_lower_chunk = out["z_lower"][0].detach().cpu()
            z_expr_chunk = out["z_expr"][0].detach().cpu()

            raw_keep_start = get_nonoverlap_keep_start(
                window_start=start,
                current_end=current_raw_end,
                chunk_len=pose_chunk.shape[0],
            )

            if raw_keep_start < pose_chunk.shape[0]:
                pose_parts.append(pose_chunk[raw_keep_start:])
                expr_parts.append(expr_chunk[raw_keep_start:])
                current_raw_end = end

            latent_start = start // downsample_factor
            latent_end = latent_start + z_upper_chunk.shape[0]
            latent_keep_start = get_nonoverlap_keep_start(
                window_start=latent_start,
                current_end=current_latent_end,
                chunk_len=z_upper_chunk.shape[0],
            )

            if latent_keep_start < z_upper_chunk.shape[0]:
                z_upper_parts.append(z_upper_chunk[latent_keep_start:])
                z_lower_parts.append(z_lower_chunk[latent_keep_start:])
                z_expr_parts.append(z_expr_chunk[latent_keep_start:])
                current_latent_end = latent_end

        if len(pose_parts) == 0:
            raise RuntimeError(f"No pose chunks were generated for seq_idx={seq_idx}.")
        if len(expr_parts) == 0:
            raise RuntimeError(f"No expression chunks were generated for seq_idx={seq_idx}.")

        pose_long = torch.cat(pose_parts, dim=0)
        expr_long = torch.cat(expr_parts, dim=0)
        z_upper_long = torch.cat(z_upper_parts, dim=0)
        z_lower_long = torch.cat(z_lower_parts, dim=0)
        z_expr_long = torch.cat(z_expr_parts, dim=0)

        all_pose_sequences.append(pose_long)
        all_expr_sequences.append(expr_long)
        all_z_upper_sequences.append(z_upper_long)
        all_z_lower_sequences.append(z_lower_long)
        all_z_expr_sequences.append(z_expr_long)
        all_windows.append(windows)

        print(
            f"[seq {seq_idx}] T_raw={T_raw}, "
            f"pose_long={tuple(pose_long.shape)}, expr_long={tuple(expr_long.shape)}, "
            f"z_upper_long={tuple(z_upper_long.shape)}, z_lower_long={tuple(z_lower_long.shape)}, "
            f"z_expr_long={tuple(z_expr_long.shape)}"
        )

    return {
        "pose_sequences": all_pose_sequences,
        "expr_sequences": all_expr_sequences,
        "z_upper_sequences": all_z_upper_sequences,
        "z_lower_sequences": all_z_lower_sequences,
        "z_expr_sequences": all_z_expr_sequences,
        "windows": all_windows,
    }

import torch
import torch.nn.functional as F


def smooth_pose6d_temporal(
    pose_6d,
    kernel_size=5,
    joint_ids=None,
):
    """
    Simple temporal smoothing for generated pose.

    Args:
        pose_6d:
            [T, 55, 6]
        kernel_size:
            odd integer. 3/5/7 recommended.
        joint_ids:
            None: smooth all joints.
            list[int]: only smooth selected joints.

    Return:
        smoothed pose_6d: [T, 55, 6]
    """

    if kernel_size <= 1:
        return pose_6d

    assert kernel_size % 2 == 1, "kernel_size should be odd."

    if not isinstance(pose_6d, torch.Tensor):
        pose_6d = torch.tensor(pose_6d).float()

    x = pose_6d.detach().clone()

    T, J, C = x.shape
    assert J == 55 and C == 6, f"Expected [T,55,6], got {x.shape}"

    if joint_ids is None:
        joint_ids = list(range(J))

    y = x[:, joint_ids, :]  # [T, K, 6]
    K = y.shape[1]

    # [T, K, 6] -> [1, K*6, T]
    y_1d = y.permute(1, 2, 0).reshape(1, K * C, T)

    pad = kernel_size // 2
    y_pad = F.pad(y_1d, (pad, pad), mode="replicate")

    weight = torch.ones(
        K * C,
        1,
        kernel_size,
        device=y_1d.device,
        dtype=y_1d.dtype,
    ) / float(kernel_size)

    y_smooth = F.conv1d(
        y_pad,
        weight,
        groups=K * C,
    )  # [1, K*6, T]

    y_smooth = y_smooth.reshape(K, C, T).permute(2, 0, 1)  # [T, K, 6]

    x[:, joint_ids, :] = y_smooth

    return x

def get_concat_boundaries_from_windows(windows):
    """
    Recover actual hard-concat boundaries used by your non-overlap concat logic.

    Args:
        windows:
            list of (start, end)

    Return:
        boundaries:
            list of frame indices in final pose_long where a new chunk starts.
    """

    boundaries = []
    current_raw_end = 0

    for i, (start, end) in enumerate(windows):
        if i == 0:
            current_raw_end = end
            continue

        # The next appended part starts from current_raw_end in final raw timeline.
        boundary = current_raw_end

        if boundary > 0:
            boundaries.append(boundary)

        current_raw_end = end

    return boundaries

def blend_pose6d_at_boundaries(
    pose_6d,
    boundaries,
    blend_radius=8,
    joint_ids=None,
):
    """
    Local linear blending around hard-concat boundaries.

    Args:
        pose_6d:
            [T, 55, 6]
        boundaries:
            list of boundary frame indices.
        blend_radius:
            number of frames before/after boundary used for transition.
            4/8/12 recommended.
        joint_ids:
            None: blend all joints.
            list[int]: only blend selected joints.

    Return:
        blended pose_6d: [T, 55, 6]
    """

    if blend_radius <= 0:
        return pose_6d

    if not isinstance(pose_6d, torch.Tensor):
        pose_6d = torch.tensor(pose_6d).float()

    x = pose_6d.detach().clone()
    T, J, C = x.shape

    assert J == 55 and C == 6, f"Expected [T,55,6], got {x.shape}"

    if joint_ids is None:
        joint_ids = list(range(J))

    for b in boundaries:
        left = max(0, b - blend_radius)
        right = min(T - 1, b + blend_radius)

        if right <= left + 1:
            continue

        L = right - left + 1

        left_pose = x[left, joint_ids, :].clone()    # [K, 6]
        right_pose = x[right, joint_ids, :].clone()  # [K, 6]

        alpha = torch.linspace(
            0.0,
            1.0,
            L,
            device=x.device,
            dtype=x.dtype,
        ).view(L, 1, 1)

        interp = (1.0 - alpha) * left_pose.unsqueeze(0) + alpha * right_pose.unsqueeze(0)

        x[left:right + 1, joint_ids, :] = interp

    return x

def postprocess_generated_pose6d(
    pose_6d,
    windows,
    smooth_kernel_size=5,
    blend_radius=8,
    smooth_joint_ids=None,
    blend_joint_ids=None,
):
    """
    Post-process generated pose:
        1. simple temporal smoothing
        2. local boundary blending

    Args:
        pose_6d:
            [T, 55, 6]
        windows:
            gen_out["windows"][0]
        smooth_kernel_size:
            3/5/7 recommended.
        blend_radius:
            4/8/12 recommended.
        smooth_joint_ids:
            None means smooth all joints.
        blend_joint_ids:
            None means blend all joints.

    Return:
        pose_post: [T, 55, 6]
    """

    if not isinstance(pose_6d, torch.Tensor):
        pose_6d = torch.tensor(pose_6d).float()

    boundaries = get_concat_boundaries_from_windows(windows)

    print("concat boundaries:", boundaries)

    pose_post = smooth_pose6d_temporal(
        pose_6d,
        kernel_size=smooth_kernel_size,
        joint_ids=smooth_joint_ids,
    )

    pose_post = blend_pose6d_at_boundaries(
        pose_post,
        boundaries=boundaries,
        blend_radius=blend_radius,
        joint_ids=blend_joint_ids,
    )

    return pose_post

import os
import numpy as np
import torch

from utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pose6d_to_axis_angle_165(pose_6d: torch.Tensor):
    """
    pose_6d: [T, 55, 6]
    return:  [T, 165]
    """
    assert pose_6d.dim() == 3, f"Expected [T, 55, 6], got {pose_6d.shape}"
    T, J, C = pose_6d.shape

    assert J == 55, f"Expected 55 joints, got {J}"
    assert C == 6, f"Expected Rot6D dim 6, got {C}"

    pose_mat = rotation_6d_to_matrix(
        pose_6d.reshape(-1, 6)
    )  # [T * 55, 3, 3]

    pose_aa = matrix_to_axis_angle(
        pose_mat
    )  # [T * 55, 3]

    poses = pose_aa.reshape(T, J, 3).reshape(T, J * 3)  # [T, 165]

    return poses.detach().cpu().numpy().astype(np.float32)


def get_batch_by_index(loader, target_index):
    """
    适用于 test_loader batch_size=1。
    返回第 target_index 个 batch。
    """
    for i, batch in enumerate(loader):
        if i == target_index:
            return batch

    raise IndexError(f"Cannot find batch index {target_index} in loader.")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_seq_id_from_batch(batch, index=0):
    for key in ["sequence_id", "seq_id", "id", "name", "seq_name", "motion_id"]:
        if key in batch:
            value = batch[key]

            if isinstance(value, (list, tuple)):
                return str(value[index])

            if isinstance(value, np.ndarray):
                return str(value[index])

            if isinstance(value, torch.Tensor):
                item = value[index]
                if item.ndim == 0:
                    return str(item.detach().cpu().item())
                return str(item.detach().cpu().numpy())

            return str(value)

    return f"sample_{index:04d}"


def get_batch_item(x, index=0):
    """
    从 batch 中取第 index 个 sample。
    """
    if isinstance(x, torch.Tensor):
        return x[index].detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x[index]

    if isinstance(x, (list, tuple)):
        item = x[index]
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        return np.asarray(item)

    return np.asarray(x)


def save_one_sequence_to_npz_with_batch(
    pose_6d,
    expr,
    batch,
    save_dir,
    sample_index_in_batch=0,
    prefix="",
):
    """
    保存单条 generated sequence。

    Args:
        pose_6d:
            [T, 55, 6]
        expr:
            [T, 100]
        batch:
            与当前 generated sequence 对应的 test_loader batch。
            batch_size 通常为 1。

    保存格式:
        betas:             [300]
        poses:             [T, 165]
        expressions:       [T, 100]
        trans:             [T, 3]
        model:             "smplx2020"
        gender:            "neutral"
        mocap_frame_rate:  30
    """

    os.makedirs(save_dir, exist_ok=True)

    if not isinstance(pose_6d, torch.Tensor):
        pose_6d = torch.tensor(pose_6d).float()

    if not isinstance(expr, torch.Tensor):
        expr = torch.tensor(expr).float()

    pose_6d = pose_6d.detach().cpu()
    expr = expr.detach().cpu()

    T = pose_6d.shape[0]

    assert pose_6d.ndim == 3 and pose_6d.shape[1:] == (55, 6), \
        f"Expected pose_6d [T,55,6], got {pose_6d.shape}"

    assert expr.ndim == 2 and expr.shape[-1] == 100, \
        f"Expected expr [T,100], got {expr.shape}"

    assert expr.shape[0] == T, \
        f"expr T={expr.shape[0]} does not match pose T={T}"

    # --------------------------------------------------
    # pose 6D -> axis-angle [T, 165]
    # 这里使用你之前已经定义过的 pose6d_to_axis_angle_165
    # --------------------------------------------------
    poses_np = pose6d_to_axis_angle_165(pose_6d)  # [T, 165]
    expr_np = expr.numpy().astype(np.float32)

    # --------------------------------------------------
    # 从对应 batch 读取 betas / trans / seq_id
    # --------------------------------------------------
    seq_id = get_seq_id_from_batch(batch, sample_index_in_batch)
    seq_id = seq_id.replace("/", "_").replace("\\", "_")

    # Preserve reference trans/betas when present. For audio-only inference,
    # write neutral identity shape / zero global translation for SMPL-X rendering.
    # Translation is not predicted by TalkFlow.
    if "betas" in batch:
        betas = get_batch_item(batch["betas"], sample_index_in_batch).astype(np.float32)
    else:
        betas = np.zeros(300, dtype=np.float32)
    if "trans" in batch:
        trans = get_batch_item(batch["trans"], sample_index_in_batch).astype(np.float32)
    else:
        trans = np.zeros((T,3), dtype=np.float32)

    # betas sample.npz 目标格式是 [300]
    if betas.ndim == 2:
        betas = betas[0]

    assert betas.shape == (300,), f"Expected betas [300], got {betas.shape}"

    # trans 目标格式是 [T, 3]
    assert trans.ndim == 2 and trans.shape[-1] == 3, \
        f"Expected trans [T,3], got {trans.shape}"

    # --------------------------------------------------
    # 对齐 trans 长度
    # --------------------------------------------------
    if trans.shape[0] > T:
        trans = trans[:T]
    elif trans.shape[0] < T:
        # 理论上如果 batch 对齐正确，不应该出现。
        # 这里保守处理：裁剪 generated pose/expr 到 trans 长度，避免保存错误长度。
        print(
            f"[warning] trans shorter than generated sequence: "
            f"trans T={trans.shape[0]}, generated T={T}. "
            f"Cropping generated pose/expression to trans length."
        )

        T_new = trans.shape[0]
        poses_np = poses_np[:T_new]
        expr_np = expr_np[:T_new]
        T = T_new

    save_path = os.path.join(save_dir, f"{prefix}{seq_id}.npz")

    np.savez(
        save_path,
        betas=betas.astype(np.float32),
        poses=poses_np.astype(np.float32),
        expressions=expr_np.astype(np.float32),
        trans=trans.astype(np.float32),
        model=np.array("smplx2020"),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(30, dtype=np.int64),
    )

    print("saved:", save_path)
    print("  betas:", betas.shape)
    print("  poses:", poses_np.shape)
    print("  expressions:", expr_np.shape)
    print("  trans:", trans.shape)

    return save_path

def save_all_gen_out_to_npz_safe(
    gen_out,
    test_loader,
    save_dir="generated_npz_post_smooth_blend",
    prefix="",
):
    """
    保存 gen_out 中的全部 sequences。

    关键点：
        第 i 个 generated sequence
        对应 test_loader 中第 i 个 batch

    这样可以避免:
        sequence_index=1 但 trans 仍然来自第 0 个 sample
    """

    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []

    num_sequences = len(gen_out["pose_sequences"])

    for sequence_index in range(num_sequences):
        batch_i = get_batch_by_index(test_loader, sequence_index)

        pose_i = gen_out["pose_sequences"][sequence_index]
        expr_i = gen_out["expr_sequences"][sequence_index]

        save_path = save_one_sequence_to_npz_with_batch(
            pose_6d=pose_i,
            expr=expr_i,
            batch=batch_i,
            save_dir=save_dir,
            sample_index_in_batch=0,
            prefix=prefix,
        )

        saved_paths.append(save_path)

        print(f"[save {sequence_index + 1}/{num_sequences}] {save_path}")

    return saved_paths

import os
import copy
import torch


def postprocess_all_generated_sequences(gen_out, smooth_kernel_size=5, blend_radius=4, smooth_joint_ids=None, blend_joint_ids=None):
    """
    Pose smoothing + boundary blending for all generated sequences.
    Compatible with 3-stream gen_out keys.
    """
    gen_out_post = dict(gen_out)
    pose_sequences_post = []

    num_sequences = len(gen_out["pose_sequences"])
    for seq_idx in range(num_sequences):
        pose_long = gen_out["pose_sequences"][seq_idx]
        windows = gen_out["windows"][seq_idx]

        pose_long_post = postprocess_generated_pose6d(
            pose_6d=pose_long,
            windows=windows,
            smooth_kernel_size=smooth_kernel_size,
            blend_radius=blend_radius,
            smooth_joint_ids=smooth_joint_ids,
            blend_joint_ids=blend_joint_ids,
        )

        pose_sequences_post.append(pose_long_post.detach().cpu())
        print(f"[postprocess {seq_idx}] pose: {tuple(pose_long.shape)} -> {tuple(pose_long_post.shape)}")

    gen_out_post["pose_sequences"] = pose_sequences_post
    gen_out_post["expr_sequences"] = list(gen_out["expr_sequences"])
    gen_out_post["windows"] = list(gen_out["windows"])

    # Preserve strict 3-stream latent keys.
    for k in ["z_upper_sequences", "z_lower_sequences", "z_expr_sequences"]:
        if k in gen_out:
            gen_out_post[k] = list(gen_out[k])

    return gen_out_post


