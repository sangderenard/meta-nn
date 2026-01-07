# ant_phase_colony_sketch.py
# PyTorch sketch: shared backbone + id-sliced personal overlays + genetics + personality
# + pheromone field = emitted axis states with phase dialects; speciation via phase alignment.

from __future__ import annotations
import colorsys
import math
from dataclasses import dataclass, field
from typing import Optional
import threading
import time

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: phase wrapping and axis rotation in (u,v) encoding
# -----------------------------

TAU = 2.0 * math.pi

def wrap_phase(x: torch.Tensor) -> torch.Tensor:
    # Wrap to [-pi, pi)
    return (x + math.pi) % TAU - math.pi

def rotate_uv(uv: torch.Tensor, phase: torch.Tensor, sign: float = +1.0) -> torch.Tensor:
    """
    uv: [..., K, 2] (u,v) pairs
    phase: [..., K] phase per axis
    sign: +1 rotate by +phase; -1 rotate by -phase
    Returns rotated uv with same shape.
    """
    # rotation: (u', v') = (u cos - v sin, u sin + v cos) for +phase
    # for -phase, sin flips sign.
    c = torch.cos(phase)
    s = torch.sin(phase) * sign
    u = uv[..., 0]
    v = uv[..., 1]
    up = u * c - v * s
    vp = u * s + v * c
    return torch.stack([up, vp], dim=-1)

def clamp_vector_norm(x: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Clamp last-dim vector norm to max_norm (works on uv-flattened vectors too).
    """
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    scale = (max_norm / n).clamp_max(1.0)
    return x * scale


# -----------------------------
# Speciation: alignment metric on torus
# -----------------------------

def phase_alignment(mu_i: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
    """
    mu_i, mu_j: [..., K] phases
    Returns alignment in [-1,1]: mean_k cos(mu_i - mu_j)
    """
    return torch.mean(torch.cos(mu_i - mu_j), dim=-1)

def phase_distance(mu_i: torch.Tensor, mu_j: torch.Tensor) -> torch.Tensor:
    # d = 1 - alignment in [0,2]
    return 1.0 - phase_alignment(mu_i, mu_j)


def phase_to_color(phase: float, saturation: float = 0.9, value: float = 0.85) -> tuple[int, int, int]:
    hue = (phase / TAU + 0.5) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def hsv_to_rgb_np(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h6 = (h * 6.0) % 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [v, q, p, p, t, v],
        default=0.0,
    )
    g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [t, v, v, q, p, p],
        default=0.0,
    )
    b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [p, p, t, v, v, q],
        default=0.0,
    )
    return r, g, b


@dataclass(frozen=True)
class MotivationHook:
    axis: int
    need_index: int  # 0: hydration, 1: hunger
    tile_type: str   # e.g., "water", "food"
    reward_scale: float = 1.0
    punish_scale: float = 0.2


class WorldTiles:
    def __init__(
        self,
        height: int,
        width: int,
        water_sites: list[tuple[int, int]],
        food_sites: list[tuple[int, int]],
        pleasure_sites: list[tuple[int, int]] | None = None,
        risk_sites: list[tuple[int, int]] | None = None,
    ):
        self.height = height
        self.width = width
        self.max_distance = math.hypot(height - 1, width - 1)
        pleasure_sites = pleasure_sites or []
        risk_sites = risk_sites or []
        ordered_sites = [
            ("water", water_sites),
            ("food", food_sites),
            ("pleasure", pleasure_sites),
            ("risk", risk_sites),
        ]
        self.tile_types = [name for name, _ in ordered_sites]
        site_counts = [len(coords) for _, coords in ordered_sites]
        max_sites = max(site_counts) if site_counts else 1
        coords_tensor = torch.zeros(len(self.tile_types), max_sites, 2, dtype=torch.float32)
        masks = torch.zeros(len(self.tile_types), max_sites, dtype=torch.bool)
        for idx, (_, sites) in enumerate(ordered_sites):
            length = len(sites)
            if length == 0:
                continue
            coords_tensor[idx, :length] = torch.tensor(sites, dtype=torch.float32)
            masks[idx, :length] = True
        self.tile_coords = coords_tensor
        self.tile_masks = masks
        self.tile_type_to_idx = {name: idx for idx, name in enumerate(self.tile_types)}
        self.tile_positions = {name: [tuple(pos) for pos in sites] for name, sites in ordered_sites}

    def tile_index(self, tile_type: str) -> int:
        return self.tile_type_to_idx.get(tile_type, 0)

    def closeness_per_types(self, pos: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        num_types, max_sites, _ = self.tile_coords.shape
        coords = self.tile_coords.to(pos.device)
        mask = self.tile_masks.to(pos.device)
        pos_exp = pos.unsqueeze(-2).unsqueeze(-2)  # [B,N,1,1,2]
        coords_exp = coords.unsqueeze(0).unsqueeze(0)  # [1,1,num_types,max_sites,2]
        diff = pos_exp - coords_exp
        dist = torch.linalg.norm(diff, dim=-1)  # [B,N,num_types,max_sites]
        mask_exp = mask.unsqueeze(0).unsqueeze(0)  # [1,1,num_types,max_sites]
        inf_dist = torch.full_like(dist, self.max_distance)
        dist = torch.where(mask_exp, dist, inf_dist)
        min_dist = dist.min(dim=-1).values  # [B,N,num_types]
        radius_safe = torch.clamp(radius.unsqueeze(-1), min=1e-3)
        closeness = torch.clamp(1.0 - min_dist / radius_safe, 0.0, 1.0)
        return closeness

    def direction_per_types(self, pos: torch.Tensor, radius: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        num_types, max_sites, _ = self.tile_coords.shape
        coords = self.tile_coords.to(pos.device)
        mask = self.tile_masks.to(pos.device)
        pos_exp = pos.unsqueeze(-2).unsqueeze(-2)  # [B,N,1,1,2]
        coords_exp = coords.unsqueeze(0).unsqueeze(0)  # [1,1,num_types,max_sites,2]
        vec = coords_exp - pos_exp
        dist = torch.linalg.norm(vec, dim=-1)
        radius_safe = torch.clamp(radius.unsqueeze(-1).unsqueeze(-1), min=eps)
        closeness = torch.clamp(1.0 - dist / radius_safe, 0.0, 1.0)
        closeness = closeness * mask.unsqueeze(0).unsqueeze(0)
        unit = vec / (dist.unsqueeze(-1) + eps)
        weighted = unit * closeness.unsqueeze(-1)
        sum_vec = weighted.sum(dim=-2)
        sum_weight = closeness.sum(dim=-1, keepdim=True)
        return sum_vec / (sum_weight + eps)


# -----------------------------
# Pheromone field: diffusion + decay + deposit
# -----------------------------

class PheromoneField2D(nn.Module):
    """
    P: [B, H, W, C] where C = 2K channels (u,v per axis).
    Diffusion via depthwise conv; decay multiplies.
    Deposits are scatter-add at integer cell positions for sketch simplicity.
    """
    def __init__(self, channels: int, diffusion: float = 0.15, decay: float = 0.02):
        super().__init__()
        self.channels = channels
        self.diffusion = diffusion
        self.decay = decay

        # 3x3 Laplacian-ish kernel (stable starter); applied per-channel (depthwise)
        k = torch.tensor([[0.0, 1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("lap", k[None, None, :, :])  # [1,1,3,3]

    def diffuse_decay(self, P: torch.Tensor) -> torch.Tensor:
        # P: [B,H,W,C] -> [B,C,H,W]
        B, H, W, C = P.shape
        x = P.permute(0, 3, 1, 2)  # BCHW

        # depthwise Laplacian
        lap = self.lap.expand(C, 1, 3, 3)  # [C,1,3,3]
        dx = F.conv2d(x, lap, padding=1, groups=C)

        # diffusion + decay
        x = x + self.diffusion * dx
        x = (1.0 - self.decay) * x

        return x.permute(0, 2, 3, 1)  # BHWC

    def deposit(self, P: torch.Tensor, pos_yx: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        """
        pos_yx: [B, N, 2] int64 positions (y,x)
        dep:    [B, N, C] deposit vectors
        """
        B, H, W, C = P.shape
        _, N, _ = pos_yx.shape
        y = pos_yx[..., 0].clamp(0, H - 1)
        x = pos_yx[..., 1].clamp(0, W - 1)

        # Flatten grid for scatter-add
        Pflat = P.view(B, H * W, C)
        idx = (y * W + x).unsqueeze(-1).expand(B, N, C)  # [B,N,C]

        # scatter_add on dim=1
        Pflat = Pflat.scatter_add(1, idx, dep)
        return Pflat.view(B, H, W, C)

    def forward(self, P: torch.Tensor, pos_yx: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        # Deposit then diffuse+decay (order can be swapped)
        P = self.deposit(P, pos_yx, dep)
        P = self.diffuse_decay(P)
        return P


# -----------------------------
# Shared backbone with personal FiLM overlay + genetic output offsets
# -----------------------------

class SharedBackbone(nn.Module):
    """
    Produces:
      delta_z_uv: [B,N,2K]  (proposed axis update in uv encoding, flattened)
      act_logits: [B,N,A]
      alpha_logit:[B,N,1]
    Uses FiLM modulation from personal overlay vector w_i.
    """
    def __init__(self, obs_dim: int, hidden: int, K: int, act_dim: int, w_dim: int):
        super().__init__()
        self.K = K
        self.act_dim = act_dim
        self.hidden = hidden

        self.fc1 = nn.Linear(obs_dim + 2*K + 2*K, hidden)  # obs + z_uv + field_uv_at_pos
        self.fc2 = nn.Linear(hidden, hidden)

        # FiLM from w_i
        self.film1 = nn.Linear(w_dim, 2 * hidden)  # gamma,beta
        self.film2 = nn.Linear(w_dim, 2 * hidden)

        self.out_z = nn.Linear(hidden, 2*K)
        self.out_a = nn.Linear(hidden, act_dim)
        self.out_alpha = nn.Linear(hidden, 1)

    def _apply_film(self, h: torch.Tensor, w: torch.Tensor, film: nn.Linear) -> torch.Tensor:
        """
        h: [B,N,H], w: [B,N,w_dim]
        """
        gb = film(w)  # [B,N,2H]
        gamma, beta = gb.chunk(2, dim=-1)
        # Stabilize: gamma near 1
        gamma = 1.0 + 0.1 * torch.tanh(gamma)
        return gamma * h + beta

    def forward(self, obs: torch.Tensor, z_uv: torch.Tensor, field_uv: torch.Tensor, w: torch.Tensor):
        x = torch.cat([obs, z_uv, field_uv], dim=-1)
        h = torch.tanh(self.fc1(x))
        h = self._apply_film(h, w, self.film1)

        h = torch.tanh(self.fc2(h))
        h = self._apply_film(h, w, self.film2)

        dz = self.out_z(h)
        act = self.out_a(h)
        alpha = self.out_alpha(h)
        return dz, act, alpha


class GeneticHeads(nn.Module):
    """
    Genetics g_i -> output offsets + species code mu_i (phase modifiers)
    """
    def __init__(self, g_dim: int, K: int, act_dim: int):
        super().__init__()
        self.K = K
        self.act_dim = act_dim
        self.bias_z = nn.Linear(g_dim, 2*K)
        self.bias_a = nn.Linear(g_dim, act_dim)
        self.bias_alpha = nn.Linear(g_dim, 1)
        self.mu_map = nn.Linear(g_dim, K)  # raw phases -> wrapped

    def forward(self, g: torch.Tensor):
        """
        g: [B,N,g_dim]
        Returns:
          b_z:     [B,N,2K]
          b_a:     [B,N,A]
          b_alpha: [B,N,1]
          mu:      [B,N,K] wrapped
        """
        bz = self.bias_z(g)
        ba = self.bias_a(g)
        balpha = self.bias_alpha(g)
        mu = wrap_phase(self.mu_map(g))
        return bz, ba, balpha, mu


# -----------------------------
# Simulation container: state, tables, and step()
# -----------------------------

@dataclass
class SimConfig:
    K: int = 8            # number of axes
    act_dim: int = 5      # action logits dim (placeholder)
    obs_dim: int = 12     # observation dim (placeholder)
    hidden: int = 128
    w_dim: int = 64       # personal overlay size
    g_dim: int = 32       # genetics vector size
    z_max_norm: float = 3.0
    lam: float = 0.35     # axis update blend
    kappa: float = 0.20   # coupling to interpreted pheromone
    pher_decay: float = 0.02
    pher_diff: float = 0.15
    need_decay: float = 0.01
    need_gain: float = 0.04
    stress_decay: float = 0.005
    perception_base_radius: float = 5.0
    perception_range: float = 10.0
    drive_steer: float = 0.35
    drive_eps: float = 1e-6
    pleasure_risk_axis: int = 2
    motivation_hooks: tuple[MotivationHook, ...] = field(
        default_factory=lambda: (
            MotivationHook(axis=0, need_index=0, tile_type="water", reward_scale=1.2),
            MotivationHook(axis=1, need_index=1, tile_type="food", reward_scale=1.1),
            MotivationHook(axis=2, need_index=2, tile_type="pleasure", reward_scale=1.0),
            MotivationHook(axis=3, need_index=3, tile_type="risk", reward_scale=0.9, punish_scale=0.35),
        )
    )


class AntColonySim(nn.Module):
    """
    Maintains shared params theta and global per-agent tables:
      W_table: personal overlays w_i
      B_table: personality phase biases beta_i
      G_table: genetics vectors g_i
    Runs batched simulation via id_map: [B,N] -> gather those rows.
    """
    def __init__(self, cfg: SimConfig, n_agents_global: int, grid_hw=(64, 64), world_tiles: WorldTiles | None = None):
        super().__init__()
        self.cfg = cfg
        self.n_agents_global = n_agents_global
        self.H, self.W = grid_hw
        C = 2 * cfg.K

        self.world_tiles = world_tiles

        self.backbone = SharedBackbone(cfg.obs_dim, cfg.hidden, cfg.K, cfg.act_dim, cfg.w_dim)
        self.gen_heads = GeneticHeads(cfg.g_dim, cfg.K, cfg.act_dim)
        self.field = PheromoneField2D(C, diffusion=cfg.pher_diff, decay=cfg.pher_decay)

        # Global per-agent tables
        self.W_table = nn.Parameter(torch.randn(n_agents_global, cfg.w_dim) * 0.02)
        self.B_table = nn.Parameter(torch.zeros(n_agents_global, cfg.K))  # personality beta_i (learned)
        self.G_table = nn.Parameter(torch.randn(n_agents_global, cfg.g_dim) * 0.02)  # genetics (evolved externally)

        hooks = cfg.motivation_hooks
        axes = torch.tensor([hook.axis for hook in hooks], dtype=torch.long)
        needs = torch.tensor([hook.need_index for hook in hooks], dtype=torch.long)
        rewards = torch.tensor([hook.reward_scale for hook in hooks], dtype=torch.float32)
        punishments = torch.tensor([hook.punish_scale for hook in hooks], dtype=torch.float32)
        tile_idx = torch.tensor([0] * len(hooks), dtype=torch.long)
        if world_tiles is not None:
            tile_idx = torch.tensor(
                [world_tiles.tile_index(hook.tile_type) for hook in hooks],
                dtype=torch.long,
            )

        self.need_dim = int(needs.max().item()) + 1 if len(hooks) > 0 else 0
        self.register_buffer("hook_axis", axes)
        self.register_buffer("hook_need", needs)
        self.register_buffer("hook_reward", rewards)
        self.register_buffer("hook_punish", punishments)
        self.register_buffer("hook_tile_idx", tile_idx)

        need_onehot = torch.zeros(len(hooks), self.need_dim, dtype=torch.float32)
        if len(hooks) > 0:
            need_onehot[torch.arange(len(hooks)), needs] = 1.0
        self.register_buffer("hook_need_onehot", need_onehot)

    def gather_by_id(self, id_map: torch.Tensor):
        """
        id_map: [B,N] int64
        Returns w,beta,g in batch shapes.
        """
        w = self.W_table[id_map]  # [B,N,w_dim]
        beta = self.B_table[id_map]  # [B,N,K]
        g = self.G_table[id_map]  # [B,N,g_dim]
        return w, beta, g

    def sample_field_at_pos(self, P: torch.Tensor, pos_yx: torch.Tensor) -> torch.Tensor:
        """
        P: [B,H,W,C], pos_yx: [B,N,2] int64
        Returns field_uv: [B,N,C]
        """
        B, H, W, C = P.shape
        y = pos_yx[..., 0].clamp(0, H - 1)
        x = pos_yx[..., 1].clamp(0, W - 1)
        # gather (batch arange must live on the same device as the field)
        batch_idx = torch.arange(B, device=P.device)[:, None]
        return P[batch_idx, y, x, :]  # [B,N,C]

    def compute_reasoning_scale(self, stress: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - stress, 0.0, 1.0)

    def compute_perception_radius(self, reasoning_scale: torch.Tensor) -> torch.Tensor:
        base = self.cfg.perception_base_radius
        rng = self.cfg.perception_range
        scale = reasoning_scale.squeeze(-1)
        return base + scale.clamp(0.0, 1.0) * rng

    def evaluate_hook_closeness(self, pos_yx: torch.Tensor, perception_radius: torch.Tensor) -> torch.Tensor:
        if self.world_tiles is None:
            raise RuntimeError("WorldTiles instance is required for motivation hooks.")
        closeness_per_type = self.world_tiles.closeness_per_types(pos_yx, perception_radius)
        hook_idx = self.hook_tile_idx.to(closeness_per_type.device)
        gather_idx = hook_idx.view(1, 1, -1).expand(closeness_per_type.shape[0], closeness_per_type.shape[1], -1)
        return torch.gather(closeness_per_type, -1, gather_idx)

    def compute_drive_direction(
        self,
        pos_yx: torch.Tensor,
        perception_radius: torch.Tensor,
        needs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.world_tiles is None:
            raise RuntimeError("WorldTiles instance is required for motivation hooks.")
        dir_per_type = self.world_tiles.direction_per_types(pos_yx, perception_radius, eps=self.cfg.drive_eps)
        hook_idx = self.hook_tile_idx.to(dir_per_type.device)
        gather_idx = hook_idx.view(1, 1, -1, 1).expand(
            dir_per_type.shape[0], dir_per_type.shape[1], -1, 2
        )
        hook_dirs = torch.gather(dir_per_type, 2, gather_idx)  # [B,N,num_hooks,2]
        need_level = needs[..., self.hook_need]
        weight = (1.0 - need_level) * self.hook_reward
        weight_sum = weight.sum(dim=-1, keepdim=True)
        drive = (hook_dirs * weight.unsqueeze(-1)).sum(dim=2) / (weight_sum + self.cfg.drive_eps)
        return drive, weight_sum

    def compute_motivation_drive(self, needs: torch.Tensor, closeness: torch.Tensor) -> torch.Tensor:
        """
        needs: [B,N,2] hydration + hunger levels in [0,1]
        closeness: [B,N,num_hooks]
        returns: [B,N,1] additive drive into the alpha logits
        """
        need_level = needs[..., self.hook_need]
        contribution = (1.0 - need_level) * closeness * self.hook_reward
        return contribution.sum(dim=-1, keepdim=True)

    def update_needs(self, needs: torch.Tensor, closeness: torch.Tensor) -> torch.Tensor:
        """
        Apply decay and proximity-based gain for each need axis.
        """
        needs_next = needs * (1.0 - self.cfg.need_decay)
        gain_per_need = (
            closeness.unsqueeze(-1) * self.hook_need_onehot
        ).sum(dim=2) * self.cfg.need_gain
        needs_after_gain = needs_next + gain_per_need * (1.0 - needs_next)
        return torch.clamp(needs_after_gain, 0.0, 1.0)

    def update_stress(self, stress: torch.Tensor, closeness: torch.Tensor, pref: torch.Tensor) -> torch.Tensor:
        punish = ((1.0 - closeness) * self.hook_punish).sum(dim=-1, keepdim=True)
        reward = (closeness * self.hook_reward).sum(dim=-1, keepdim=True)
        pref = torch.clamp(pref, -1.0, 1.0)
        next_stress = stress * (1.0 - self.cfg.stress_decay) + (-pref * punish) + (pref * reward)
        return torch.clamp(next_stress, 0.0, 1.0)

    def step(
        self,
        P: torch.Tensor,
        id_map: torch.Tensor,
        pos_yx: torch.Tensor,
        obs: torch.Tensor,
        z_uv: torch.Tensor,
        needs: torch.Tensor,
        stress: torch.Tensor,
    ):
        """
        One differentiable tick.

        Inputs:
          P:      [B,H,W,2K] pheromone field
          id_map: [B,N] global ids
          pos_yx: [B,N,2] integer grid positions
          obs:    [B,N,obs_dim]
          z_uv:   [B,N,2K] agent internal axis state (flattened uv)
          needs:  [B,N,4] current hydration/hunger/pleasure/risk in [0,1]
          stress: [B,N,1] current stress

        Returns:
          P_next, z_uv_next, act_logits, alpha, mu, beta, needs_next, stress_next, drive_loss
        """
        cfg = self.cfg
        B, N, _ = z_uv.shape
        K = cfg.K
        C = 2 * K

        # Gather individuality
        w, beta, g = self.gather_by_id(id_map)
        bz, ba, balpha, mu = self.gen_heads(g)  # genetics -> offsets + species mu

        # Effective dialect phase modifiers
        delta = wrap_phase(mu + beta)  # [B,N,K]

        # Read pheromone at positions and interpret (demodulate by -delta)
        field_uv = self.sample_field_at_pos(P, pos_yx)  # [B,N,2K]
        field_uv_k2 = field_uv.view(B, N, K, 2)
        interpreted = rotate_uv(field_uv_k2, delta, sign=-1.0).view(B, N, C)

        reasoning_scale = self.compute_reasoning_scale(stress)
        perception_radius = self.compute_perception_radius(reasoning_scale)
        hook_closeness = self.evaluate_hook_closeness(pos_yx, perception_radius)
        drive_dir, drive_weight = self.compute_drive_direction(pos_yx, perception_radius, needs)

        # Backbone proposals
        dz, act, alpha_logit = self.backbone(obs, z_uv, interpreted, w)

        # Add genetic output offsets
        dz = dz + bz
        dz_uv = dz.view(B, N, K, 2)
        need_level = needs[..., self.hook_need]
        axis_contrib = (1.0 - need_level) * hook_closeness * self.hook_reward
        axis_boost = torch.zeros(B, N, K, device=dz_uv.device)
        axis_idx = self.hook_axis.clamp(0, K - 1)
        axis_idx_expanded = axis_idx.view(1, 1, -1).expand(B, N, -1)
        axis_boost.scatter_add_(2, axis_idx_expanded, axis_contrib)
        dz_uv = dz_uv + axis_boost.unsqueeze(-1)
        dz = dz_uv.view(B, N, C)
        act = act + ba
        act = act.clone()
        act[..., :2] = act[..., :2] + cfg.drive_steer * drive_dir
        motivation_drive = self.compute_motivation_drive(needs, hook_closeness)
        alpha = torch.sigmoid(alpha_logit + balpha + motivation_drive)  # [B,N,1]

        # Update axis state (contractive + bounded)
        z_next = (1.0 - cfg.lam) * z_uv + cfg.lam * (z_uv + dz) + cfg.kappa * interpreted
        z_next = clamp_vector_norm(z_next, cfg.z_max_norm)

        # Emit rotated axis state (modulated by +delta) into the field
        z_emit = rotate_uv(z_next.view(B, N, K, 2), delta, sign=+1.0).view(B, N, C)
        dep = alpha * z_emit  # [B,N,2K]

        # Field update
        P_next = self.field(P, pos_yx, dep)

        z_view = z_next.view(B, N, K, 2)
        axis = cfg.pleasure_risk_axis
        z_phase = torch.atan2(z_view[..., axis, 1], z_view[..., axis, 0])
        pref = torch.sin(z_phase).unsqueeze(-1)
        needs_next = self.update_needs(needs, hook_closeness)
        stress_next = self.update_stress(stress, hook_closeness, pref)

        move_vec = act[..., :2]
        move_norm = torch.linalg.norm(move_vec, dim=-1, keepdim=True)
        move_dir = move_vec / (move_norm + cfg.drive_eps)
        drive_norm = torch.linalg.norm(drive_dir, dim=-1, keepdim=True)
        drive_dir_norm = drive_dir / (drive_norm + cfg.drive_eps)
        alignment = (move_dir * drive_dir_norm).sum(dim=-1, keepdim=True)
        active = (drive_weight > cfg.drive_eps).float()
        drive_loss = (1.0 - alignment) * active

        return P_next, z_next, act, alpha, mu, beta, needs_next, stress_next, drive_loss


class AntVisualizer:
    def __init__(
        self,
        height: int,
        width: int,
        K: int,
        cell_size: int = 8,
        caption: str = "Ant colony",
        world_tiles: WorldTiles | None = None,
    ):
        pygame.init()
        self.height = height
        self.width = width
        self.K = K
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((width * cell_size, height * cell_size))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.running = True
        self.world_tiles = world_tiles

    def close(self):
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        return self.running

    def draw_field(self, field: np.ndarray):
        rgb = self._field_to_rgb(field)
        image = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        image = pygame.transform.scale(image, (self.width * self.cell_size, self.height * self.cell_size))
        self.screen.blit(image, (0, 0))
        self._draw_world_overlay()

    def draw_ants(
        self,
        pos: np.ndarray,
        z_uv: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        orientation: np.ndarray,
        mobility: np.ndarray,
    ):
        positions = pos.astype(np.int32)
        z_view = z_uv.reshape(-1, self.K, 2)
        mu_view = mu
        beta_view = beta
        phase_state = np.arctan2(z_view[..., 1], z_view[..., 0])
        phase_mix = (phase_state + mu_view + beta_view) / 3.0

        for idx in range(positions.shape[0]):
            grid_y, grid_x = int(positions[idx, 0]), int(positions[idx, 1])
            self._draw_ant(
                grid_y,
                grid_x,
                phase_mix[idx],
                float(orientation[idx]),
                float(mobility[idx]),
            )

    def _draw_ant(
        self,
        grid_y: int,
        grid_x: int,
        phases: torch.Tensor,
        orientation: float,
        mobility: float,
    ):
        center_x = grid_x * self.cell_size + self.cell_size / 2
        center_y = grid_y * self.cell_size + self.cell_size / 2
        segment_radius = max(2, self.cell_size // 3)
        stride = segment_radius * 2
        start = center_x - stride * (self.K - 1) / 2.0

        for k in range(self.K):
            color = phase_to_color(float(phases[k].item()))
            cx = int(start + k * stride)
            pygame.draw.circle(self.screen, color, (cx, int(center_y)), segment_radius)

        end_x = int(center_x + math.cos(orientation) * segment_radius * 3)
        end_y = int(center_y + math.sin(orientation) * segment_radius * 3)
        head_color = phase_to_color((orientation % TAU))
        pygame.draw.line(
            self.screen,
            head_color,
            (int(center_x), int(center_y)),
            (end_x, end_y),
            width=max(1, segment_radius // 2),
        )

        rim_color = tuple(min(255, int(c * (0.5 + 0.5 * mobility))) for c in head_color)
        pygame.draw.circle(self.screen, rim_color, (int(center_x), int(center_y)), segment_radius, width=1)

    def _field_to_rgb(self, field: np.ndarray) -> np.ndarray:
        H, W, C = field.shape
        K = C // 2
        uv = field.reshape(H, W, K, 2)
        mag = np.linalg.norm(uv, axis=-1)
        phase = np.arctan2(uv[..., 1], uv[..., 0])
        axes = np.linspace(0.0, 1.0, K, endpoint=False, dtype=np.float32)
        hue = (axes[None, None, :] + phase / (2.0 * np.pi)) % 1.0
        sat = np.full_like(hue, 0.85, dtype=np.float32)
        val = np.tanh(mag).astype(np.float32)
        r, g, b = hsv_to_rgb_np(hue, sat, val)
        weight = mag + 1e-6
        norm = weight.sum(axis=-1, keepdims=True)[..., None]
        coeff = weight[..., None] / norm
        rgb_mix = (np.stack([r, g, b], axis=-1) * coeff).sum(axis=2)
        brightness = np.clip(np.tanh(norm[..., 0, 0]), 0.15, 1.0)
        rgb_final = np.clip(rgb_mix * brightness[..., None], 0.0, 1.0)
        return (rgb_final * 255.0).astype(np.uint8)

    def _draw_world_overlay(self):
        if self.world_tiles is None:
            return
        overlay = pygame.Surface(
            (self.width * self.cell_size, self.height * self.cell_size),
            pygame.SRCALPHA,
        )
        style = (
            ("water", (50, 140, 255, 90)),
            ("food", (220, 70, 70, 90)),
        )
        radius = max(2, self.cell_size // 2)
        for tile_type, color in style:
            for y, x in self.world_tiles.tile_positions.get(tile_type, []):
                center = (int(x * self.cell_size + self.cell_size / 2), int(y * self.cell_size + self.cell_size / 2))
                pygame.draw.circle(overlay, color, center, radius)
        self.screen.blit(overlay, (0, 0))

    def render(
        self,
        field: np.ndarray,
        pos: np.ndarray,
        z_uv: np.ndarray,
        mu: np.ndarray,
        beta: np.ndarray,
        orientation: np.ndarray,
        mobility: np.ndarray,
    ) -> bool:
        if not self.running:
            return False
        self.handle_events()
        self.draw_field(field)
        self.draw_ants(pos, z_uv, mu, beta, orientation, mobility)
        pygame.display.flip()
        self.clock.tick(60)
        return self.running


@dataclass
class RenderFrame:
    field: np.ndarray
    pos: np.ndarray
    z_uv: np.ndarray
    mu: np.ndarray
    beta: np.ndarray
    angle: np.ndarray
    mobility: np.ndarray


class FlipBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._buffers: list[Optional[RenderFrame]] = [None, None]
        self._write_idx = 0
        self._read_idx = 1

    def publish(self, frame: RenderFrame):
        with self._lock:
            self._buffers[self._write_idx] = frame
            self._write_idx, self._read_idx = self._read_idx, self._write_idx

    def read_latest(self) -> Optional[RenderFrame]:
        with self._lock:
            return self._buffers[self._read_idx]


# -----------------------------
# Minimal demo run
# -----------------------------

def demo(device="cpu", steps: Optional[int] = None, render=False, cell_size=8):
    torch.manual_seed(0)

    cfg = SimConfig(K=6, act_dim=5, obs_dim=12)
    n_global = 10_000
    B = 4      # batch of worlds
    N = 256    # agents per world
    H, W = 64, 64

    water_sites = [
        (int(0.2 * H), int(0.2 * W)),
        (int(0.25 * H), int(0.15 * W)),
        (int(0.15 * H), int(0.25 * W)),
    ]
    food_sites = [
        (int(0.8 * H), int(0.8 * W)),
        (int(0.75 * H), int(0.85 * W)),
        (int(0.85 * H), int(0.75 * W)),
    ]
    pleasure_sites = [
        (int(0.5 * H), int(0.2 * W)),
        (int(0.55 * H), int(0.25 * W)),
    ]
    risk_sites = [
        (int(0.5 * H), int(0.8 * W)),
        (int(0.45 * H), int(0.85 * W)),
    ]
    world_tiles = WorldTiles(H, W, water_sites, food_sites, pleasure_sites, risk_sites)
    sim = AntColonySim(cfg, n_agents_global=n_global, grid_hw=(H, W), world_tiles=world_tiles).to(device)

    # Initial field and agent state
    P = torch.zeros(B, H, W, 2 * cfg.K, device=device)
    z = torch.zeros(B, N, 2 * cfg.K, device=device)

    needs = torch.full((B, N, 4), 0.5, device=device)
    stress = torch.full((B, N, 1), 0.5, device=device)

    # Random id mapping and positions
    id_map = torch.randint(0, n_global, (B, N), device=device)
    pos = torch.stack([
        torch.randint(0, H, (B, N), device=device),
        torch.randint(0, W, (B, N), device=device),
    ], dim=-1)  # [B,N,2] (y,x)

    renderer = (
        AntVisualizer(H, W, cfg.K, cell_size=cell_size, world_tiles=world_tiles)
        if render
        else None
    )
    max_speed = 2.0

    t = 0
    stop_event = threading.Event()
    buffer = FlipBuffer()

    def _snapshot(
        field_t: torch.Tensor,
        pos_t: torch.Tensor,
        z_t: torch.Tensor,
        mu_t: torch.Tensor,
        beta_t: torch.Tensor,
        angle_t: torch.Tensor,
        mobility_t: torch.Tensor,
    ) -> RenderFrame:
        return RenderFrame(
            field=field_t.detach().cpu().numpy(),
            pos=pos_t.detach().cpu().numpy(),
            z_uv=z_t.detach().cpu().numpy(),
            mu=mu_t.detach().cpu().numpy(),
            beta=beta_t.detach().cpu().numpy(),
            angle=angle_t.detach().cpu().numpy(),
            mobility=mobility_t.detach().cpu().numpy(),
        )

    def _sim_loop():
        nonlocal P, z, pos, needs, stress, t
        try:
            while not stop_event.is_set() and (steps is None or t < steps):
                obs = torch.randn(B, N, cfg.obs_dim, device=device) * 0.1  # placeholder
                P, z, act, alpha, mu, beta, needs, stress, drive_loss = sim.step(
                    P, id_map, pos, obs, z, needs, stress
                )

                angle = torch.atan2(act[..., 1], act[..., 0])
                mobility = torch.sigmoid(act[..., 2])

                direction = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
                step_delta = direction * (mobility.unsqueeze(-1) * max_speed)
                move = step_delta.round().to(pos.dtype)
                pos = (pos + move).clamp(0, H - 1)

                z0 = z[0].view(N, cfg.K, 2)
                phi0 = torch.atan2(z0[:, :, 1], z0[:, :, 0])  # [N,K]
                order_axis0 = torch.abs(torch.mean(torch.exp(1j * phi0[:, 0].to(torch.complex64))))
                order_axis0_val = order_axis0.detach().item()
                d01 = phase_distance(mu[0, 0], mu[0, 1])
                hydration = needs[0, 0, 0].item()
                hunger = needs[0, 0, 1].item()
                pleasure = needs[0, 0, 2].item()
                risk = needs[0, 0, 3].item()
                current_stress = stress[0, 0, 0].item()
                reasoning_tensor = sim.compute_reasoning_scale(stress)
                perception_tensor = sim.compute_perception_radius(reasoning_tensor)
                reasoning_scale = reasoning_tensor[0, 0, 0].item()
                perception_radius = perception_tensor[0, 0].item()
                drive_loss_val = drive_loss.mean().item()
                print(
                    f"t={t}  mean|P|={P.abs().mean().item():.4f}  mean|z|={z.abs().mean().item():.4f}  "
                    f"order_axis0={order_axis0_val:.4f}  d(mu0,mu1)={d01.item():.4f}  "
                    f"mobility={mobility[0,0].item():.4f}  orientation={angle[0,0].item():.4f}  "
                    f"hydration={hydration:.3f}  hunger={hunger:.3f}  "
                    f"pleasure={pleasure:.3f}  risk={risk:.3f}  "
                    f"stress={current_stress:.3f}  reasoning={reasoning_scale:.3f}  "
                    f"perception={perception_radius:.2f}  drive_loss={drive_loss_val:.4f}"
                )

                buffer.publish(_snapshot(P[0], pos[0], z[0], mu[0], beta[0], angle[0], mobility[0]))
                t += 1
        finally:
            stop_event.set()

    sim_thread = threading.Thread(target=_sim_loop, name="sim-thread", daemon=True)
    sim_thread.start()

    try:
        while renderer and not stop_event.is_set():
            frame = buffer.read_latest()
            if frame is not None:
                if not renderer.render(
                    frame.field,
                    frame.pos,
                    frame.z_uv,
                    frame.mu,
                    frame.beta,
                    frame.angle,
                    frame.mobility,
                ):
                    stop_event.set()
            else:
                time.sleep(0.01)
        if not renderer:
            while not stop_event.is_set():
                time.sleep(0.05)
    finally:
        stop_event.set()
        sim_thread.join(timeout=1.0)
        if renderer:
            renderer.close()

    # Example colony-level alignment (order parameter) for axis 0 in batch 0:
    # decode phase from z_uv (u,v) pairs: phi = atan2(v,u)
    z0 = z[0].view(N, cfg.K, 2)
    phi0 = torch.atan2(z0[:, :, 1], z0[:, :, 0])  # [N,K]
    final_order_axis0 = torch.abs(torch.mean(torch.exp(1j * phi0[:, 0].to(torch.complex64))))
    print("Order parameter axis0 (batch0):", final_order_axis0.detach().item())

if __name__ == "__main__":
    demo("cpu", render=True)
