"""
Per-Part Flow Matching for SmolVLA.

Decomposes the action expert's denoising into separate loops for
arm joints (12D) and wheel velocities (2D), sharing the frozen VLM
conditioning via cached KV.

Each part has its own projection layers (in_proj, out_proj, time_mlp)
but shares the expert transformer weights and the VLM KV cache.

Theory: If arm and wheel actions are conditionally independent given
observations, the joint velocity field decomposes as:
    v_t([a_arm, a_wheel]) = [v_t^arm(a_arm | obs), v_t^wheel(a_wheel | obs)]
This allows each part to be denoised independently.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.policies.smolvla.modeling_smolvla import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

logger = logging.getLogger(__name__)


class PerPartModule(nn.Module):
    """Per-part projection layers for arms and wheels.

    Each part gets its own input/output projections and time-action MLP,
    but shares the expert transformer from the base SmolVLA model.
    """

    def __init__(self, expert_hidden_size: int, arm_dim: int = 12, wheel_dim: int = 2):
        super().__init__()
        self.expert_hidden_size = expert_hidden_size
        self.arm_dim = arm_dim
        self.wheel_dim = wheel_dim

        # Arm head
        self.arm_in_proj = nn.Linear(arm_dim, expert_hidden_size)
        self.arm_out_proj = nn.Linear(expert_hidden_size, arm_dim)
        self.arm_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.arm_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)

        # Wheel head
        self.wheel_in_proj = nn.Linear(wheel_dim, expert_hidden_size)
        self.wheel_out_proj = nn.Linear(expert_hidden_size, wheel_dim)
        self.wheel_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.wheel_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)

        # Initialize
        for module in [self.arm_in_proj, self.arm_out_proj,
                       self.wheel_in_proj, self.wheel_out_proj,
                       self.arm_time_mlp_in, self.arm_time_mlp_out,
                       self.wheel_time_mlp_in, self.wheel_time_mlp_out]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"PerPartModule: arm_dim={arm_dim}, wheel_dim={wheel_dim}, "
                     f"expert_hidden={expert_hidden_size}, params={total_params:,}")


def _embed_suffix_part(noisy_actions, timestep, in_proj, time_mlp_in, time_mlp_out,
                       expert_hidden_size, chunk_size, config):
    """Embed noisy actions + timestep for one body part.

    Mirrors VLAFlowMatching.embed_suffix but uses per-part projections.
    """
    action_emb = in_proj(noisy_actions)  # (B, chunk, expert_hidden)
    device = action_emb.device
    bsize = action_emb.shape[0]
    dtype = action_emb.dtype

    time_emb = create_sinusoidal_pos_embedding(
        timestep, expert_hidden_size, config.min_period, config.max_period, device=device
    )
    time_emb = time_emb.type(dtype=dtype)
    time_emb = time_emb[:, None, :].expand_as(action_emb)

    action_time_emb = torch.cat([action_emb, time_emb], dim=2)
    action_time_emb = time_mlp_in(action_time_emb)
    action_time_emb = F.silu(action_time_emb)
    action_time_emb = time_mlp_out(action_time_emb)

    embs = action_time_emb
    action_time_dim = action_time_emb.shape[1]
    pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)

    # att_masks: 1 means prefix tokens don't attend to these (same as original)
    att_masks = torch.ones(1, chunk_size, dtype=embs.dtype, device=device)
    att_masks = att_masks.expand(bsize, -1)

    return embs, pad_masks, att_masks


def _denoise_step_part(base_model, noisy_actions, prefix_pad_masks, past_key_values,
                       timestep, in_proj, out_proj, time_mlp_in, time_mlp_out,
                       expert_hidden_size, chunk_size, config):
    """Single denoising step for one body part.

    Uses the shared expert transformer from base_model but per-part projections.
    Mirrors VLAFlowMatching.denoise_step exactly, replacing projection layers.
    """
    suffix_embs, suffix_pad_masks, suffix_att_masks = _embed_suffix_part(
        noisy_actions, timestep, in_proj, time_mlp_in, time_mlp_out,
        expert_hidden_size, chunk_size, config
    )

    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

    outputs_embeds, _ = base_model.vlm_with_expert.forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=config.use_cache,
        fill_kv_cache=False,
    )

    suffix_out = outputs_embeds[1]
    suffix_out = suffix_out[:, -chunk_size:]
    suffix_out = suffix_out.to(dtype=torch.float32)
    v_t = out_proj(suffix_out)
    return v_t


def per_part_forward(base_model, per_part, images, img_masks, lang_tokens, lang_masks,
                     state, actions, arm_chunk_size, wheel_chunk_size, noise=None, time=None):
    """Training forward pass with per-part flow matching.

    Args:
        base_model: VLAFlowMatching instance (the inner SmolVLA model)
        per_part: PerPartModule with per-part projection layers
        images, img_masks, lang_tokens, lang_masks, state: standard SmolVLA inputs
        actions: raw actions tensor, shape (B, chunk_size, action_dim) where
                 action_dim = arm_dim + wheel_dim (e.g. 14)
        arm_chunk_size: chunk size for arm actions (usually same as model chunk_size)
        wheel_chunk_size: chunk size for wheel actions

    Returns:
        (arm_loss, wheel_loss): per-element MSE losses for each part
            arm_loss: (B, arm_chunk_size, arm_dim)
            wheel_loss: (B, wheel_chunk_size, wheel_dim)
    """
    config = base_model.config
    arm_dim = per_part.arm_dim
    wheel_dim = per_part.wheel_dim
    expert_hidden = per_part.expert_hidden_size

    # Split actions into parts
    arm_actions = actions[:, :arm_chunk_size, :arm_dim]
    wheel_actions = actions[:, :wheel_chunk_size, arm_dim:arm_dim + wheel_dim]

    B = actions.shape[0]
    device = actions.device

    # Sample noise for each part independently
    if noise is None:
        arm_noise = torch.randn_like(arm_actions)
        wheel_noise = torch.randn_like(wheel_actions)
    else:
        arm_noise = noise[:, :arm_chunk_size, :arm_dim]
        wheel_noise = noise[:, :wheel_chunk_size, arm_dim:arm_dim + wheel_dim]

    # Shared time
    if time is None:
        time = base_model.sample_time(B, device)

    time_expanded = time[:, None, None]

    # Flow matching interpolation per part
    x_t_arm = time_expanded * arm_noise + (1 - time_expanded) * arm_actions
    u_t_arm = arm_noise - arm_actions

    x_t_wheel = time_expanded * wheel_noise + (1 - time_expanded) * wheel_actions
    u_t_wheel = wheel_noise - wheel_actions

    # Shared prefix (compute once)
    prefix_embs, prefix_pad_masks, prefix_att_masks = base_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # Cache VLM prefix (VLM is frozen, but state_proj gradients flow through)
    _, past_key_values = base_model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )

    # Arm flow matching
    v_t_arm = _denoise_step_part(
        base_model, x_t_arm, prefix_pad_masks, past_key_values, time,
        per_part.arm_in_proj, per_part.arm_out_proj,
        per_part.arm_time_mlp_in, per_part.arm_time_mlp_out,
        expert_hidden, arm_chunk_size, config
    )
    arm_loss = F.mse_loss(u_t_arm, v_t_arm, reduction="none")

    # Wheel flow matching
    v_t_wheel = _denoise_step_part(
        base_model, x_t_wheel, prefix_pad_masks, past_key_values, time,
        per_part.wheel_in_proj, per_part.wheel_out_proj,
        per_part.wheel_time_mlp_in, per_part.wheel_time_mlp_out,
        expert_hidden, wheel_chunk_size, config
    )
    wheel_loss = F.mse_loss(u_t_wheel, v_t_wheel, reduction="none")

    return arm_loss, wheel_loss


def per_part_sample_actions(base_model, per_part, images, img_masks, lang_tokens, lang_masks,
                            state, arm_chunk_size, wheel_chunk_size,
                            arm_num_steps, wheel_num_steps):
    """Inference with per-part denoising loops.

    Runs arm and wheel denoising independently with shared VLM KV cache.

    Returns:
        actions: (B, arm_chunk_size, arm_dim + wheel_dim) combined actions
    """
    config = base_model.config
    arm_dim = per_part.arm_dim
    wheel_dim = per_part.wheel_dim
    expert_hidden = per_part.expert_hidden_size

    B = state.shape[0]
    device = state.device

    # Shared prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks = base_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    _, past_key_values = base_model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )

    # === ARM DENOISING LOOP ===
    x_arm = torch.randn(B, arm_chunk_size, arm_dim, device=device)
    dt_arm = -1.0 / arm_num_steps

    for step in range(arm_num_steps):
        t = 1.0 + step * dt_arm
        timestep = torch.tensor(t, dtype=torch.float32, device=device).expand(B)
        v_t = _denoise_step_part(
            base_model, x_arm, prefix_pad_masks, past_key_values, timestep,
            per_part.arm_in_proj, per_part.arm_out_proj,
            per_part.arm_time_mlp_in, per_part.arm_time_mlp_out,
            expert_hidden, arm_chunk_size, config
        )
        x_arm = x_arm + dt_arm * v_t

    # === WHEEL DENOISING LOOP ===
    x_wheel = torch.randn(B, wheel_chunk_size, wheel_dim, device=device)
    dt_wheel = -1.0 / wheel_num_steps

    for step in range(wheel_num_steps):
        t = 1.0 + step * dt_wheel
        timestep = torch.tensor(t, dtype=torch.float32, device=device).expand(B)
        v_t = _denoise_step_part(
            base_model, x_wheel, prefix_pad_masks, past_key_values, timestep,
            per_part.wheel_in_proj, per_part.wheel_out_proj,
            per_part.wheel_time_mlp_in, per_part.wheel_time_mlp_out,
            expert_hidden, wheel_chunk_size, config
        )
        x_wheel = x_wheel + dt_wheel * v_t

    # Combine: if different chunk sizes, interpolate wheel to match arm
    if wheel_chunk_size != arm_chunk_size:
        x_wheel = F.interpolate(
            x_wheel.permute(0, 2, 1),
            size=arm_chunk_size,
            mode='linear',
            align_corners=True
        ).permute(0, 2, 1)

    actions = torch.cat([x_arm, x_wheel], dim=-1)
    return actions
