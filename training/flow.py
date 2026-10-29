"""Flow-matching utilities retained from the supplied 3-stream training notebook."""
import torch
import torch.nn.functional as F
import torch.nn as nn

# =========================================================
# 1. Noise helpers
# =========================================================

def temporal_smooth_noise(
    x,
    kernel_size=9,
    normalize=True,
):
    """
    Smooth temporal noise for latent sequences.

    Args:
        x: [B, T, D]
        kernel_size: odd integer
        normalize: whether to normalize smoothed noise back to unit std

    Returns:
        h: [B, T, D]
    """
    if kernel_size <= 1:
        return x

    assert kernel_size % 2 == 1, "kernel_size must be odd."

    pad = kernel_size // 2

    h = x.transpose(1, 2)  # [B, D, T]
    h = F.pad(h, (pad, pad), mode="replicate")
    h = F.avg_pool1d(h, kernel_size=kernel_size, stride=1)
    h = h.transpose(1, 2)  # [B, T, D]

    if normalize:
        std = h.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        h = h / std

    return h


def sample_source_noise_like(
    z_gt,
    noise_type="white",
    sigma=1.0,
    smooth_kernel=9,
    style_weight=0.7,
    smooth_weight=0.3,
):
    """
    Sample source noise x0 with the same shape as z_gt.

    Args:
        z_gt: [B, T, D]
        noise_type:
            "white"        : frame-wise Gaussian noise
            "smooth"       : temporally smoothed Gaussian noise
            "style"        : sequence-level noise shared across time
            "style_smooth" : global style noise + smooth temporal residual
        sigma: overall source noise scale
    """
    B, T, D = z_gt.shape

    if noise_type == "white":
        x0 = torch.randn_like(z_gt)

    elif noise_type == "smooth":
        x0 = torch.randn_like(z_gt)
        x0 = temporal_smooth_noise(
            x0,
            kernel_size=smooth_kernel,
            normalize=True,
        )

    elif noise_type == "style":
        style = torch.randn(
            B,
            1,
            D,
            device=z_gt.device,
            dtype=z_gt.dtype,
        )
        x0 = style.expand(B, T, D)
        std = x0.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        x0 = x0 / std

    elif noise_type == "style_smooth":
        style = torch.randn(
            B,
            1,
            D,
            device=z_gt.device,
            dtype=z_gt.dtype,
        )
        style = style.expand(B, T, D)

        residual = torch.randn_like(z_gt)
        residual = temporal_smooth_noise(
            residual,
            kernel_size=smooth_kernel,
            normalize=True,
        )

        x0 = style_weight * style + smooth_weight * residual
        std = x0.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        x0 = x0 / std

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return sigma * x0


def sample_xt_and_velocity_branch(
    z_gt,
    t,
    noise_type="white",
    sigma=1.0,
    smooth_kernel=9,
    style_weight=0.7,
    smooth_weight=0.3,
):
    """
    Construct source x0, interpolated xt, and Flow Matching velocity target.

    Linear Flow Matching path:
        xt = (1 - t) * x0 + t * x1
        v_target = dxt / dt = x1 - x0

    Args:
        z_gt: [B, T, D], clean endpoint x1
        t:    [B], interpolation time

    Returns:
        x0:       [B, T, D]
        xt:       [B, T, D]
        v_target: [B, T, D]
    """
    x0 = sample_source_noise_like(
        z_gt=z_gt,
        noise_type=noise_type,
        sigma=sigma,
        smooth_kernel=smooth_kernel,
        style_weight=style_weight,
        smooth_weight=smooth_weight,
    )

    x1 = z_gt
    t_view = t.view(-1, 1, 1)

    xt = (1.0 - t_view) * x0 + t_view * x1
    v_target = x1 - x0

    return x0, xt, v_target


# =========================================================
# 2. Dynamics helpers
# =========================================================

def temporal_velocity(x):
    """
    Args:
        x: [B, T, ...]

    Returns:
        velocity: [B, T - 1, ...]
    """
    return x[:, 1:] - x[:, :-1]


def temporal_acceleration(x):
    """
    Args:
        x: [B, T, ...]

    Returns:
        acceleration: [B, T - 2, ...]
    """
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def smooth_l1_or_zero(pred, target):
    """
    Safe Smooth L1 loss. Returns zero if temporal length is too short.
    """
    if pred.numel() == 0 or target.numel() == 0:
        return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred, target)


def latent_velocity_loss(pred, target):
    return smooth_l1_or_zero(temporal_velocity(pred), temporal_velocity(target))


def latent_acceleration_loss(pred, target):
    return smooth_l1_or_zero(temporal_acceleration(pred), temporal_acceleration(target))


def decoded_velocity_loss(pred, target):
    return smooth_l1_or_zero(temporal_velocity(pred), temporal_velocity(target))


def decoded_acceleration_loss(pred, target):
    return smooth_l1_or_zero(temporal_acceleration(pred), temporal_acceleration(target))


# =========================================================
# 3. FM-consistent ODE rollout reconstruction helper
# =========================================================

def flow_matching_ode_rollout_3stream(
    flow_model,
    z_upper_gt,
    z_lower_gt,
    z_expr_gt,
    cond_upper,
    cond_lower,
    cond_expr,
    cond_mask=None,
    steps=4,

    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",

    upper_noise_sigma=0.5,
    lower_noise_sigma=0.3,
    expr_noise_sigma=1.0,

    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,

    upper_style_weight=0.7,
    upper_smooth_weight=0.3,

    lower_style_weight=0.8,
    lower_smooth_weight=0.2,

    expr_style_weight=0.0,
    expr_smooth_weight=1.0,
):
    """
    Differentiable Euler ODE rollout for 3-stream Flow Matching reconstruction.

    Streams:
        z_upper: upper-body pose latent
        z_lower: lower-body pose latent
        z_expr : expression latent

    ODE:
        x_{k+1} = x_k + dt * v_theta(x_k, t_k, cond)

    Important:
        Do NOT wrap this function in torch.no_grad(), because ODE recon loss
        must back-propagate through the rollout steps into flow_model and
        condition_encoder.
    """
    if steps <= 0:
        raise ValueError("ode rollout steps must be positive when ODE reconstruction is enabled.")

    B = z_upper_gt.shape[0]
    device = z_upper_gt.device
    dtype = z_upper_gt.dtype

    z_upper = sample_source_noise_like(
        z_gt=z_upper_gt,
        noise_type=upper_noise_type,
        sigma=upper_noise_sigma,
        smooth_kernel=upper_noise_kernel,
        style_weight=upper_style_weight,
        smooth_weight=upper_smooth_weight,
    )

    z_lower = sample_source_noise_like(
        z_gt=z_lower_gt,
        noise_type=lower_noise_type,
        sigma=lower_noise_sigma,
        smooth_kernel=lower_noise_kernel,
        style_weight=lower_style_weight,
        smooth_weight=lower_smooth_weight,
    )

    z_expr = sample_source_noise_like(
        z_gt=z_expr_gt,
        noise_type=expr_noise_type,
        sigma=expr_noise_sigma,
        smooth_kernel=expr_noise_kernel,
        style_weight=expr_style_weight,
        smooth_weight=expr_smooth_weight,
    )

    dt = 1.0 / float(steps)

    for k in range(steps):
        t_val = k / float(steps)
        t = torch.full((B,), t_val, device=device, dtype=dtype)

        out = flow_model(
            x_upper_t=z_upper,
            x_lower_t=z_lower,
            x_expr_t=z_expr,
            cond_upper=cond_upper,
            cond_lower=cond_lower,
            cond_expr=cond_expr,
            t=t,
            seq_mask=cond_mask,
        )

        z_upper = z_upper + dt * out["v_upper"]
        z_lower = z_lower + dt * out["v_lower"]
        z_expr = z_expr + dt * out["v_expr"]

    return z_upper, z_lower, z_expr


# backward-compatible alias if you want to replace the old function name
flow_matching_ode_rollout = flow_matching_ode_rollout_3stream


# =========================================================
# 4. Loss helper
# =========================================================

def compute_flow_matching_losses_3stream(
    flow_out,
    v_upper_target,
    v_lower_target,
    v_expr_target,

    z_upper_pred=None,
    z_lower_pred=None,
    z_expr_pred=None,

    z_upper_gt=None,
    z_lower_gt=None,
    z_expr_gt=None,

    pose_recon=None,
    expr_recon=None,
    pose_gt=None,
    expr_gt=None,

    w_flow_upper=1.0,
    w_flow_lower=0.5,
    w_flow_expr=1.0,

    w_pose_recon=0.0,
    w_expr_recon=0.0,

    w_pose_vel=0.0,
    w_expr_vel=0.0,

    w_pose_acc=0.0,
    w_expr_acc=0.0,

    w_upper_latent_vel=0.0,
    w_lower_latent_vel=0.0,
    w_expr_latent_vel=0.0,

    w_upper_latent_acc=0.0,
    w_lower_latent_acc=0.0,
    w_expr_latent_acc=0.0,
):
    """
    Compute 3-stream Flow Matching training losses.

    Core FM losses:
        v_upper_theta -> x1_upper - x0_upper
        v_lower_theta -> x1_lower - x0_lower
        v_expr_theta  -> x1_expr  - x0_expr

    Optional auxiliary losses:
        If z_*_pred are provided, they should come from an FM-consistent ODE rollout.
    """
    v_upper_pred = flow_out["v_upper"]
    v_lower_pred = flow_out["v_lower"]
    v_expr_pred = flow_out["v_expr"]

    loss_upper_flow = F.smooth_l1_loss(v_upper_pred, v_upper_target)
    loss_lower_flow = F.smooth_l1_loss(v_lower_pred, v_lower_target)
    loss_expr_flow = F.smooth_l1_loss(v_expr_pred, v_expr_target)

    zero = loss_upper_flow.new_tensor(0.0)

    loss_pose_recon = zero
    loss_expr_recon = zero

    loss_pose_vel = zero
    loss_expr_vel = zero
    loss_pose_acc = zero
    loss_expr_acc = zero

    loss_upper_latent_vel = zero
    loss_lower_latent_vel = zero
    loss_expr_latent_vel = zero

    loss_upper_latent_acc = zero
    loss_lower_latent_acc = zero
    loss_expr_latent_acc = zero

    has_decoded_recon = (
        pose_recon is not None and expr_recon is not None and
        pose_gt is not None and expr_gt is not None
    )

    has_latent_recon = (
        z_upper_pred is not None and z_lower_pred is not None and z_expr_pred is not None and
        z_upper_gt is not None and z_lower_gt is not None and z_expr_gt is not None
    )

    if has_decoded_recon:
        loss_pose_recon = F.mse_loss(pose_recon, pose_gt)
        loss_expr_recon = F.smooth_l1_loss(expr_recon, expr_gt)

        loss_pose_vel = decoded_velocity_loss(pose_recon, pose_gt)
        loss_expr_vel = decoded_velocity_loss(expr_recon, expr_gt)

        loss_pose_acc = decoded_acceleration_loss(pose_recon, pose_gt)
        loss_expr_acc = decoded_acceleration_loss(expr_recon, expr_gt)

    if has_latent_recon:
        loss_upper_latent_vel = latent_velocity_loss(z_upper_pred, z_upper_gt)
        loss_lower_latent_vel = latent_velocity_loss(z_lower_pred, z_lower_gt)
        loss_expr_latent_vel = latent_velocity_loss(z_expr_pred, z_expr_gt)

        loss_upper_latent_acc = latent_acceleration_loss(z_upper_pred, z_upper_gt)
        loss_lower_latent_acc = latent_acceleration_loss(z_lower_pred, z_lower_gt)
        loss_expr_latent_acc = latent_acceleration_loss(z_expr_pred, z_expr_gt)

    loss_total = (
        w_flow_upper * loss_upper_flow
        + w_flow_lower * loss_lower_flow
        + w_flow_expr * loss_expr_flow

        + w_pose_recon * loss_pose_recon
        + w_expr_recon * loss_expr_recon

        + w_pose_vel * loss_pose_vel
        + w_expr_vel * loss_expr_vel

        + w_pose_acc * loss_pose_acc
        + w_expr_acc * loss_expr_acc

        + w_upper_latent_vel * loss_upper_latent_vel
        + w_lower_latent_vel * loss_lower_latent_vel
        + w_expr_latent_vel * loss_expr_latent_vel

        + w_upper_latent_acc * loss_upper_latent_acc
        + w_lower_latent_acc * loss_lower_latent_acc
        + w_expr_latent_acc * loss_expr_latent_acc
    )

    return {
        "loss_total": loss_total,

        "loss_upper_flow": loss_upper_flow,
        "loss_lower_flow": loss_lower_flow,
        "loss_expr_flow": loss_expr_flow,

        "loss_pose_recon": loss_pose_recon,
        "loss_expr_recon": loss_expr_recon,

        "loss_pose_vel": loss_pose_vel,
        "loss_expr_vel": loss_expr_vel,

        "loss_pose_acc": loss_pose_acc,
        "loss_expr_acc": loss_expr_acc,

        "loss_upper_latent_vel": loss_upper_latent_vel,
        "loss_lower_latent_vel": loss_lower_latent_vel,
        "loss_expr_latent_vel": loss_expr_latent_vel,

        "loss_upper_latent_acc": loss_upper_latent_acc,
        "loss_lower_latent_acc": loss_lower_latent_acc,
        "loss_expr_latent_acc": loss_expr_latent_acc,
    }


# backward-compatible alias if you replace the old function name
compute_flow_matching_losses = compute_flow_matching_losses_3stream

def train_flow_matching_one_epoch(
    ae_model,
    condition_encoder,
    flow_model,
    loader,
    optimizer,
    device,

    w_flow_upper=1.0,
    w_flow_lower=0.5,
    w_flow_expr=1.0,

    # These auxiliary losses are now computed from FM ODE rollout output.
    w_pose_recon=0.0,
    w_expr_recon=0.0,

    w_pose_vel=0.0,
    w_expr_vel=0.0,

    w_pose_acc=0.0,
    w_expr_acc=0.0,

    w_upper_latent_vel=0.0,
    w_lower_latent_vel=0.0,
    w_expr_latent_vel=0.0,

    w_upper_latent_acc=0.0,
    w_lower_latent_acc=0.0,
    w_expr_latent_acc=0.0,

    # Source distribution for both FM training and ODE rollout reconstruction.
    upper_noise_type="style_smooth",
    lower_noise_type="style_smooth",
    expr_noise_type="white",

    upper_noise_sigma=0.5,
    lower_noise_sigma=0.3,
    expr_noise_sigma=1.0,

    upper_noise_kernel=9,
    lower_noise_kernel=9,
    expr_noise_kernel=1,

    upper_style_weight=0.7,
    upper_smooth_weight=0.3,

    lower_style_weight=0.8,
    lower_smooth_weight=0.2,

    expr_style_weight=0.0,
    expr_smooth_weight=1.0,

    # ODE recon is disabled when ode_recon_steps <= 0 or all auxiliary weights are zero.
    ode_recon_steps=0,
    ode_recon_every=1,

    grad_clip=1.0,
):
    ae_model.eval()
    condition_encoder.train()
    flow_model.train()

    for p in ae_model.parameters():
        p.requires_grad_(False)

    running = {
        "loss_total": 0.0,

        "loss_upper_flow": 0.0,
        "loss_lower_flow": 0.0,
        "loss_expr_flow": 0.0,

        "loss_pose_recon": 0.0,
        "loss_expr_recon": 0.0,

        "loss_pose_vel": 0.0,
        "loss_expr_vel": 0.0,

        "loss_pose_acc": 0.0,
        "loss_expr_acc": 0.0,

        "loss_upper_latent_vel": 0.0,
        "loss_lower_latent_vel": 0.0,
        "loss_expr_latent_vel": 0.0,

        "loss_upper_latent_acc": 0.0,
        "loss_lower_latent_acc": 0.0,
        "loss_expr_latent_acc": 0.0,
    }

    num_batches = 0

    aux_weight_sum = (
        w_pose_recon + w_expr_recon
        + w_pose_vel + w_expr_vel
        + w_pose_acc + w_expr_acc
        + w_upper_latent_vel + w_lower_latent_vel + w_expr_latent_vel
        + w_upper_latent_acc + w_lower_latent_acc + w_expr_latent_acc
    )

    for batch_idx, batch in enumerate(loader):
        pose = batch["pose"].to(device, non_blocking=True)
        expr = batch["expression"].to(device, non_blocking=True)

        hubert = batch["hubert"].to(device, non_blocking=True)
        librosa = batch["librosa"].to(device, non_blocking=True)

        text_local_raw = batch["text_local_raw"].to(device, non_blocking=True)
        text_local_mask = batch["text_local_mask"].to(device, non_blocking=True)
        text_global_raw = batch["text_global_raw"].to(device, non_blocking=True)

        valid_mask = batch.get("valid_mask", None)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------
        # Frozen 3-stream AE latent extraction
        # -------------------------------------------------
        with torch.no_grad():
            ae_enc = ae_model.encode_to_latents(pose, expr)

            z_upper_gt = ae_enc["z_upper"]
            z_lower_gt = ae_enc["z_lower"]
            z_expr_gt = ae_enc["z_expr"]

        # -------------------------------------------------
        # 3-stream condition encoding
        # -------------------------------------------------
        cond_out = condition_encoder(
            hubert=hubert,
            librosa=librosa,
            text_local_raw=text_local_raw,
            text_global_raw=text_global_raw,
            valid_mask=valid_mask,
            text_local_mask=text_local_mask,
        )

        cond_upper = cond_out["cond_upper"]
        cond_lower = cond_out["cond_lower"]
        cond_expr = cond_out["cond_expr"]
        cond_mask = cond_out["cond_mask"]

        T_upper = z_upper_gt.shape[1]
        T_lower = z_lower_gt.shape[1]
        T_expr = z_expr_gt.shape[1]

        T_cond_upper = cond_upper.shape[1]
        T_cond_lower = cond_lower.shape[1]
        T_cond_expr = cond_expr.shape[1]

        if not (
            T_upper == T_lower == T_expr
            == T_cond_upper == T_cond_lower == T_cond_expr
        ):
            raise ValueError(
                "Temporal length mismatch:\n"
                f"  z_upper_gt: {tuple(z_upper_gt.shape)}\n"
                f"  z_lower_gt: {tuple(z_lower_gt.shape)}\n"
                f"  z_expr_gt:  {tuple(z_expr_gt.shape)}\n"
                f"  cond_upper: {tuple(cond_upper.shape)}\n"
                f"  cond_lower: {tuple(cond_lower.shape)}\n"
                f"  cond_expr:  {tuple(cond_expr.shape)}\n"
                "Check AE latent downsample factor and ConditionEncoder downsample_factor."
            )

        if cond_mask is not None and cond_mask.shape[1] != T_upper:
            raise ValueError(
                "cond_mask length mismatch:\n"
                f"  cond_mask:  {tuple(cond_mask.shape)}\n"
                f"  z_upper_gt: {tuple(z_upper_gt.shape)}"
            )

        B = pose.shape[0]
        t = torch.rand(B, device=device)

        # -------------------------------------------------
        # Standard linear Flow Matching velocity targets
        # -------------------------------------------------
        x0_upper, xt_upper, v_upper_target = sample_xt_and_velocity_branch(
            z_gt=z_upper_gt,
            t=t,
            noise_type=upper_noise_type,
            sigma=upper_noise_sigma,
            smooth_kernel=upper_noise_kernel,
            style_weight=upper_style_weight,
            smooth_weight=upper_smooth_weight,
        )

        x0_lower, xt_lower, v_lower_target = sample_xt_and_velocity_branch(
            z_gt=z_lower_gt,
            t=t,
            noise_type=lower_noise_type,
            sigma=lower_noise_sigma,
            smooth_kernel=lower_noise_kernel,
            style_weight=lower_style_weight,
            smooth_weight=lower_smooth_weight,
        )

        x0_expr, xt_expr, v_expr_target = sample_xt_and_velocity_branch(
            z_gt=z_expr_gt,
            t=t,
            noise_type=expr_noise_type,
            sigma=expr_noise_sigma,
            smooth_kernel=expr_noise_kernel,
            style_weight=expr_style_weight,
            smooth_weight=expr_smooth_weight,
        )

        flow_out = flow_model(
            x_upper_t=xt_upper,
            x_lower_t=xt_lower,
            x_expr_t=xt_expr,
            cond_upper=cond_upper,
            cond_lower=cond_lower,
            cond_expr=cond_expr,
            t=t,
            seq_mask=cond_mask,
        )

        # -------------------------------------------------
        # Optional FM-consistent reconstruction via ODE rollout
        # -------------------------------------------------
        use_ode_recon = (
            ode_recon_steps is not None
            and ode_recon_steps > 0
            and aux_weight_sum > 0
            and (
                ode_recon_every is None
                or ode_recon_every <= 1
                or batch_idx % ode_recon_every == 0
            )
        )

        z_upper_pred = None
        z_lower_pred = None
        z_expr_pred = None

        pose_recon = None
        expr_recon = None

        if use_ode_recon:
            z_upper_pred, z_lower_pred, z_expr_pred = flow_matching_ode_rollout(
                flow_model=flow_model,

                z_upper_gt=z_upper_gt,
                z_lower_gt=z_lower_gt,
                z_expr_gt=z_expr_gt,

                cond_upper=cond_upper,
                cond_lower=cond_lower,
                cond_expr=cond_expr,
                cond_mask=cond_mask,

                steps=ode_recon_steps,

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

                expr_style_weight=expr_style_weight,
                expr_smooth_weight=expr_smooth_weight,
            )

            dec_out = ae_model.decode_from_latents(
                z_upper=z_upper_pred,
                z_lower=z_lower_pred,
                z_expr=z_expr_pred,
                target_pose_len=pose.shape[1],
                target_expr_len=expr.shape[1],
                apply_bottleneck_interaction=False,
            )

            pose_recon = dec_out["pose_recon"]
            expr_recon = dec_out["expr_recon"]

        loss_dict = compute_flow_matching_losses(
            flow_out=flow_out,

            v_upper_target=v_upper_target,
            v_lower_target=v_lower_target,
            v_expr_target=v_expr_target,

            z_upper_pred=z_upper_pred,
            z_lower_pred=z_lower_pred,
            z_expr_pred=z_expr_pred,

            z_upper_gt=z_upper_gt,
            z_lower_gt=z_lower_gt,
            z_expr_gt=z_expr_gt,

            pose_recon=pose_recon,
            expr_recon=expr_recon,
            pose_gt=pose,
            expr_gt=expr,

            w_flow_upper=w_flow_upper,
            w_flow_lower=w_flow_lower,
            w_flow_expr=w_flow_expr,

            w_pose_recon=w_pose_recon,
            w_expr_recon=w_expr_recon,

            w_pose_vel=w_pose_vel,
            w_expr_vel=w_expr_vel,

            w_pose_acc=w_pose_acc,
            w_expr_acc=w_expr_acc,

            w_upper_latent_vel=w_upper_latent_vel,
            w_lower_latent_vel=w_lower_latent_vel,
            w_expr_latent_vel=w_expr_latent_vel,

            w_upper_latent_acc=w_upper_latent_acc,
            w_lower_latent_acc=w_lower_latent_acc,
            w_expr_latent_acc=w_expr_latent_acc,
        )

        loss = loss_dict["loss_total"]
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(condition_encoder.parameters()) + list(flow_model.parameters()),
                max_norm=grad_clip,
            )

        optimizer.step()

        for k in running.keys():
            if k in loss_dict:
                running[k] += float(loss_dict[k].detach().item())

        num_batches += 1

    avg = {k: v / max(num_batches, 1) for k, v in running.items()}
    return avg

