from __future__ import annotations

"""
Correct prototype implementation.

This version does the following:
- Uses real sentence-transformer embeddings for vocabulary phrases.
- Keeps dynamic trainable parameter rows separate from ST vectors.
- Takes an arbitrary active batch vocabulary directly from each sample.
- Assembles the active parameter rows into a dynamic layer for the forward pass.
- Produces a bag of N predicted vectors.
- Uses one-to-one assignment so each slot picks one item and each item gets one opportunity.
- Passes each produced vector into a mask head to produce exactly one mask per slot.
- Compares each chosen slot mask only to the mask of the single item it was assigned to.
- Tracks observed active vocabulary combinations in a hypergraph memory.
- Includes a synthetic PIL text dataset and a pygame viewer.
"""

import keyword
import math
import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install sentence-transformers: pip install sentence-transformers") from exc


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


_MATCHER_INVALID_COST = 1.0e4


def _sanitize_finite_tensor(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    fill_value = float(fill)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"_sanitize_finite_tensor expected torch.Tensor, got {type(x)!r}")
    dtype = x.dtype if bool(x.is_floating_point()) else torch.float32
    return torch.nan_to_num(x.to(dtype=dtype), nan=fill_value, posinf=fill_value, neginf=fill_value)


def _count_nonfinite(x: Optional[torch.Tensor]) -> int:
    if x is None:
        return 0
    return int((~torch.isfinite(x.detach())).sum().item())


# ============================================================
# Sentence-transformer vocabulary bank
# ============================================================


class SentenceTransformerVocabBank:
    def __init__(
        self,
        phrases: Sequence[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.phrases = list(phrases)
        if len(set(self.phrases)) != len(self.phrases):
            raise ValueError("Phrases must be unique.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)
        self._vectors_cpu: Dict[str, torch.Tensor] = {}
        self._encode_all()

    def _encode_all(self) -> None:
        vecs = self.model.encode(
            self.phrases,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=64,
        ).detach().cpu()
        for phrase, vec in zip(self.phrases, vecs):
            self._vectors_cpu[phrase] = vec.clone()

    @property
    def dim(self) -> int:
        return int(next(iter(self._vectors_cpu.values())).numel())

    def get_vector(self, phrase: str, device: Optional[torch.device] = None) -> torch.Tensor:
        vec = self._vectors_cpu[phrase]
        return vec if device is None else vec.to(device)


# ============================================================
# Dynamic parameter rows
# ============================================================


class DynamicRowBank(nn.Module):
    @staticmethod
    def _sanitize_key(phrase: str) -> str:
        # nn.ParameterDict stores keys as module attributes; avoid collisions
        # with reserved nn.Module names (train, eval, parameters, …)
        key = phrase.replace(" ", "_")
        if not key.isidentifier() or keyword.iskeyword(key) or hasattr(nn.Module, key):
            key = f"p_{key}"
        return key

    def __init__(self, phrases: Sequence[str], row_dim: int, init_scale: float = 0.02) -> None:
        super().__init__()
        self.row_dim = row_dim
        self.rows = nn.ParameterDict()
        self._key_map: Dict[str, str] = {}
        for phrase in phrases:
            key = self._sanitize_key(phrase)
            self._key_map[phrase] = key
            self.rows[key] = nn.Parameter(torch.randn(row_dim) * init_scale)

    def assemble(self, batch_vocab: List[List[str]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        max_m = max(len(v) for v in batch_vocab)
        bsz = len(batch_vocab)
        row_dtype = next(iter(self.rows.values())).dtype if len(self.rows) > 0 else torch.float32
        rows = torch.zeros(bsz, max_m, self.row_dim, device=device, dtype=row_dtype)
        valid = torch.zeros(bsz, max_m, dtype=torch.bool, device=device)
        for b, vocab in enumerate(batch_vocab):
            for i, phrase in enumerate(vocab):
                rows[b, i] = self.rows[self._key_map[phrase]]
                valid[b, i] = True
        return rows, valid


# ============================================================
# Observed hypergraph memory
# ============================================================


@dataclass
class HyperEdgeRecord:
    members: Tuple[str, ...]
    count: int = 0
    last_seen_step: int = -1


class ObservedHypergraph:
    def __init__(self) -> None:
        self.edges: Dict[Tuple[str, ...], HyperEdgeRecord] = {}
        self.node_counts: Dict[str, float] = {}
        self.pair_counts: Dict[Tuple[str, str], float] = {}
        self._stats_cache_key: Optional[Tuple[Tuple[str, ...], float]] = None
        self._stats_cache_value: Optional[Dict[str, torch.Tensor]] = None

    def observe(self, vocab: Sequence[str], step: int) -> None:
        key = tuple(sorted(set(vocab)))
        if not key:
            return
        if key not in self.edges:
            self.edges[key] = HyperEdgeRecord(members=key)
        self.edges[key].count += 1
        self.edges[key].last_seen_step = step
        for term in key:
            self.node_counts[term] = float(self.node_counts.get(term, 0.0) + 1.0)
        for left, right in combinations(key, 2):
            pair_key = (str(left), str(right))
            self.pair_counts[pair_key] = float(self.pair_counts.get(pair_key, 0.0) + 1.0)
        self._stats_cache_key = None
        self._stats_cache_value = None

    def search_outward(self, seed_vocab: Sequence[str], max_results: int = 10) -> List[Tuple[Tuple[str, ...], int, int]]:
        seed = set(seed_vocab)
        out: List[Tuple[Tuple[str, ...], int, int]] = []
        for key, rec in self.edges.items():
            overlap = len(seed.intersection(key))
            if overlap <= 0:
                continue
            score = overlap * 100000 + rec.count
            out.append((key, score, rec.count))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:max_results]

    def __len__(self) -> int:
        return len(self.edges)

    def total_observations(self) -> int:
        return int(sum(int(rec.count) for rec in self.edges.values()))

    def summary(self) -> Dict[str, int]:
        max_order = 0
        for key in self.edges.keys():
            max_order = max(max_order, len(key))
        return {
            "edge_count": int(len(self.edges)),
            "total_observations": int(self.total_observations()),
            "max_order": int(max_order),
        }

    def outward_report(self, seed_vocab: Sequence[str], max_results: int = 5) -> List[Dict[str, Any]]:
        seed = tuple(sorted({str(x).strip() for x in seed_vocab if str(x).strip()}))
        if not seed:
            return []
        seed_set = set(seed)
        report: List[Dict[str, Any]] = []
        for members, score, count in self.search_outward(seed, max_results=max_results):
            novelty = tuple(x for x in members if x not in seed_set)
            report.append(
                {
                    "members": tuple(members),
                    "score": int(score),
                    "count": int(count),
                    "overlap": int(len(seed_set.intersection(members))),
                    "novelty": novelty,
                }
            )
        return report

    def vocab_statistics(
        self,
        vocab: Sequence[str],
        *,
        alpha: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        vocab_key = tuple(str(x).strip() for x in vocab)
        cache_key = (vocab_key, float(alpha))
        if self._stats_cache_key == cache_key and isinstance(self._stats_cache_value, dict):
            cached = self._stats_cache_value
            return {
                "base_log_prior": cached["base_log_prior"].clone(),
                "pair_log_prior": cached["pair_log_prior"].clone(),
            }

        vocab_list = list(vocab_key)
        n_vocab = max(1, int(len(vocab_list)))
        total_obs = float(max(1.0, float(self.total_observations())))
        node = torch.zeros(n_vocab, dtype=torch.float32)
        pair = torch.zeros(n_vocab, n_vocab, dtype=torch.float32)

        for i, term in enumerate(vocab_list):
            node_count = float(self.node_counts.get(term, 0.0))
            node[i] = float(node_count)
            pair[i, i] = float(node_count)

        for i, left in enumerate(vocab_list):
            for j in range(i + 1, n_vocab):
                right = vocab_list[j]
                pair_count = float(self.pair_counts.get((str(left), str(right)), 0.0))
                pair[i, j] = float(pair_count)
                pair[j, i] = float(pair_count)

        alpha_t = float(max(1e-6, alpha))
        base_log_prior = torch.log((node + alpha_t) / (total_obs + (alpha_t * float(n_vocab))))
        pair_log_prior = torch.empty_like(pair)
        for i in range(n_vocab):
            denom = float(node[i].item()) + (alpha_t * float(n_vocab))
            pair_log_prior[i] = torch.log((pair[i] + alpha_t) / max(1e-6, denom))

        self._stats_cache_key = cache_key
        self._stats_cache_value = {
            "base_log_prior": base_log_prior.clone(),
            "pair_log_prior": pair_log_prior.clone(),
        }
        return {
            "base_log_prior": base_log_prior,
            "pair_log_prior": pair_log_prior,
        }


# ============================================================
# Synthetic dataset
# ============================================================


class SyntheticTextSupportDataset(Dataset):
    def __init__(
        self,
        vocab_bank: SentenceTransformerVocabBank,
        image_size: int = 192,
        min_items: int = 1,
        max_items: int = 4,
        samples: int = 5000,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_bank = vocab_bank
        self.image_size = image_size
        self.min_items = min_items
        self.max_items = max_items
        self.samples = samples
        self.rng = random.Random(seed)
        self.vocab = list(vocab_bank.phrases)
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except Exception:
            self.font = ImageFont.load_default(size=20)
        # Pre-stack all vocab vectors once so __getitem__ doesn't re-fetch per sample.
        self._all_vocab_vectors = torch.stack(
            [vocab_bank.get_vector(p) for p in self.vocab], dim=0
        )  # [V, D]

    def __len__(self) -> int:
        return self.samples

    def _render_phrase(self, phrase: str, color: Tuple[int, int, int], x: int, y: int, angle: float = 0.0) -> Tuple[Image.Image, Image.Image]:
        rgba = Image.new("RGBA", (self.image_size, self.image_size), (0, 0, 0, 0))
        mask = Image.new("L", (self.image_size, self.image_size), 0)
        d_rgba = ImageDraw.Draw(rgba)
        d_mask = ImageDraw.Draw(mask)
        d_rgba.text((x, y), phrase, font=self.font, fill=(*color, 255))
        d_mask.text((x, y), phrase, font=self.font, fill=255)
        if angle != 0.0:
            rgba = rgba.rotate(angle, resample=Image.BICUBIC, expand=False)
            mask = mask.rotate(angle, resample=Image.BICUBIC, expand=False)
        return rgba, mask

    def __getitem__(self, index: int) -> Dict[str, Any]:
        del index
        m = self.rng.randint(self.min_items, self.max_items)
        active = self.rng.sample(self.vocab, m)
        active_set = set(active)

        composite = Image.new("RGBA", (self.image_size, self.image_size), (0, 0, 0, 0))
        phrase_to_mask: Dict[str, torch.Tensor] = {}

        margin = max(8, self.image_size // 4)
        pos_lo = margin
        pos_hi = max(pos_lo + 1, self.image_size - margin)

        for phrase in active:
            color = (
                self.rng.randint(96, 255),
                self.rng.randint(96, 255),
                self.rng.randint(96, 255),
            )
            x = self.rng.randint(pos_lo, pos_hi)
            y = self.rng.randint(pos_lo, pos_hi)
            angle = self.rng.uniform(-180.0, 180.0)
            rgba_obj, mask_obj = self._render_phrase(phrase, color, x, y, angle)
            composite = Image.alpha_composite(composite, rgba_obj)
            phrase_to_mask[phrase] = torch.from_numpy(np.array(mask_obj, dtype=np.float32) / 255.0)

        # Targets cover ALL vocab items so slots compete across the full vocabulary.
        # Absent items receive zero masks; their ST vectors still participate as
        # candidates in the matcher, forcing the model to prefer present items.
        blank = torch.zeros(self.image_size, self.image_size)
        target_masks = torch.stack(
            [phrase_to_mask[p] if p in active_set else blank for p in self.vocab], dim=0
        )  # [V, H, W]
        target_vectors = self._all_vocab_vectors  # [V, D]
        present_mask = torch.tensor([p in active_set for p in self.vocab], dtype=torch.bool)  # [V]

        image = torch.from_numpy(np.array(composite, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return {
            "image": image,
            "batch_vocab": list(self.vocab),   # full vocab — model context
            "item_vocab": list(active),        # actually-present phrases
            "present_mask": present_mask,      # [V] bool
            "target_vectors": target_vectors,  # [V, D] — full vocab
            "target_masks": target_masks,      # [V, H, W] — full vocab, zeros for absent
        }



def collate_synthetic(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    batch_vocab = [list(b["batch_vocab"]) for b in batch]
    item_vocab  = [list(b["item_vocab"])  for b in batch]
    # All samples share the same vocab, so target shapes are fixed [V, ...].
    target_vectors = torch.stack([b["target_vectors"] for b in batch], dim=0)  # [B, V, D]
    target_masks   = torch.stack([b["target_masks"]   for b in batch], dim=0)  # [B, V, H, W]
    present_masks  = torch.stack([b["present_mask"]   for b in batch], dim=0)  # [B, V]
    # All vocab items are valid targets — slots compete across the full vocabulary.
    target_valid = torch.ones(len(batch), target_vectors.shape[1], dtype=torch.bool)  # [B, V]

    return {
        "image": images,
        "batch_vocab": batch_vocab,
        "item_vocab": item_vocab,
        "present_mask": present_masks,
        "target_vectors": target_vectors,
        "target_masks": target_masks,
        "target_valid": target_valid,
    }


# ============================================================
# Model
# ============================================================


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        quarter_channels = max(96, int(hidden_dim) // 2)
        self.output_channels: Dict[str, int] = {
            "full": 32,
            "half": 64,
            "quarter": quarter_channels,
            "eighth": int(hidden_dim),
        }

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.output_channels["full"], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.output_channels["full"], self.output_channels["full"], 3, padding=1),
            nn.GELU(),
        )
        self.full_refine = ResidualConvBlock(self.output_channels["full"])

        self.down_half = nn.Sequential(
            nn.Conv2d(self.output_channels["full"], self.output_channels["half"], 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.output_channels["half"], self.output_channels["half"], 3, padding=1),
            nn.GELU(),
        )
        self.half_refine = ResidualConvBlock(self.output_channels["half"])

        self.down_quarter = nn.Sequential(
            nn.Conv2d(self.output_channels["half"], self.output_channels["quarter"], 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.output_channels["quarter"], self.output_channels["quarter"], 3, padding=1),
            nn.GELU(),
        )
        self.quarter_refine = ResidualConvBlock(self.output_channels["quarter"])

        self.down_eighth = nn.Sequential(
            nn.Conv2d(self.output_channels["quarter"], self.output_channels["eighth"], 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(self.output_channels["eighth"], self.output_channels["eighth"], 3, padding=1),
            nn.GELU(),
        )
        self.eighth_refine = nn.Sequential(
            ResidualConvBlock(self.output_channels["eighth"]),
            ResidualConvBlock(self.output_channels["eighth"]),
        )

        self.alpha_proj_full = nn.Conv2d(1, self.output_channels["full"], 1)
        self.alpha_proj_half = nn.Conv2d(1, self.output_channels["half"], 1)
        self.alpha_proj_quarter = nn.Conv2d(1, self.output_channels["quarter"], 1)
        self.alpha_proj_eighth = nn.Conv2d(1, self.output_channels["eighth"], 1)

    def forward(self, rgba: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Support both RGB and RGBA by synthesizing an opaque alpha plane when missing.
        if rgba.shape[1] >= 4:
            alpha = rgba[:, 3:4]
        else:
            alpha = torch.ones(
                rgba.shape[0],
                1,
                rgba.shape[2],
                rgba.shape[3],
                device=rgba.device,
                dtype=rgba.dtype,
            )

        full = self.stem(rgba)
        full = full + self.alpha_proj_full(alpha)
        full = self.full_refine(full)

        half = self.down_half(full)
        alpha_half = F.interpolate(alpha, size=half.shape[-2:], mode="bilinear", align_corners=False)
        half = half + self.alpha_proj_half(alpha_half)
        half = self.half_refine(half)

        quarter = self.down_quarter(half)
        alpha_quarter = F.interpolate(alpha, size=quarter.shape[-2:], mode="bilinear", align_corners=False)
        quarter = quarter + self.alpha_proj_quarter(alpha_quarter)
        quarter = self.quarter_refine(quarter)

        eighth = self.down_eighth(quarter)
        alpha_eighth = F.interpolate(alpha, size=eighth.shape[-2:], mode="bilinear", align_corners=False)
        eighth = eighth + self.alpha_proj_eighth(alpha_eighth)
        eighth = self.eighth_refine(eighth)

        return {
            "full": full,
            "half": half,
            "quarter": quarter,
            "eighth": eighth,
        }


class DynamicLayerAssembler(nn.Module):
    def __init__(self, row_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.row_proj = nn.Sequential(
            nn.Linear(row_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mix = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, image_features: torch.Tensor, active_rows: torch.Tensor, active_valid: torch.Tensor) -> torch.Tensor:
        pooled = image_features.mean(dim=(2, 3))
        pooled = self.global_proj(pooled)
        row_hidden = self.row_proj(active_rows)
        pooled_expand = pooled.unsqueeze(1).expand_as(row_hidden)
        mixed = self.mix(torch.cat([row_hidden, pooled_expand], dim=-1))
        mixed = mixed * active_valid.unsqueeze(-1)
        return mixed


class SequentialMaskHead(nn.Module):
    def __init__(
        self,
        *,
        st_dim: int,
        hidden_dim: int,
        image_size: int,
        encoder_channels: Dict[str, int],
        spatial_memory_hw: Tuple[int, int] = (8, 8),
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.full_dim = max(64, int(hidden_dim) // 4)
        self.half_dim = max(96, int(hidden_dim) // 2)
        self.coarse_dim = int(hidden_dim)
        self.spatial_memory_hw = (
            max(2, int(spatial_memory_hw[0])),
            max(2, int(spatial_memory_hw[1])),
        )

        self.full_proj = nn.Sequential(
            nn.Conv2d(int(encoder_channels["full"]), self.full_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.full_dim, self.full_dim, 3, padding=1),
            nn.GELU(),
        )
        self.half_proj = nn.Sequential(
            nn.Conv2d(int(encoder_channels["half"]), self.half_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.GELU(),
        )
        self.quarter_proj = nn.Sequential(
            nn.Conv2d(int(encoder_channels["quarter"]), self.coarse_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 3, padding=1),
            nn.GELU(),
        )
        self.eighth_proj = nn.Sequential(
            nn.Conv2d(int(encoder_channels["eighth"]), self.coarse_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 3, padding=1),
            nn.GELU(),
        )
        self.coarse_fuse = nn.Sequential(
            nn.Conv2d(self.coarse_dim * 2, self.coarse_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 3, padding=1),
            nn.GELU(),
        )
        self.coarse_to_half = nn.Sequential(
            nn.Conv2d(self.coarse_dim, self.half_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.GELU(),
        )
        self.half_to_full = nn.Sequential(
            nn.Conv2d(self.half_dim, self.full_dim, 1),
            nn.GELU(),
            nn.Conv2d(self.full_dim, self.full_dim, 3, padding=1),
            nn.GELU(),
        )
        self.memory_refine = nn.Sequential(
            nn.Conv2d(self.coarse_dim, int(hidden_dim), 1),
            nn.GELU(),
            nn.Conv2d(int(hidden_dim), int(hidden_dim), 3, padding=1),
            nn.GELU(),
        )

        self.coarse_canvas_proj = nn.Sequential(
            nn.Conv2d(1, self.coarse_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 1),
        )
        self.half_canvas_proj = nn.Sequential(
            nn.Conv2d(1, self.half_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 1),
        )
        self.full_canvas_proj = nn.Sequential(
            nn.Conv2d(1, self.full_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.full_dim, self.full_dim, 1),
        )
        self.half_seed_proj = nn.Sequential(
            nn.Conv2d(1, self.half_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 1),
        )
        self.full_seed_proj = nn.Sequential(
            nn.Conv2d(1, self.full_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.full_dim, self.full_dim, 1),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim + st_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coarse_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, self.coarse_dim),
            nn.GELU(),
            nn.Linear(self.coarse_dim, self.coarse_dim),
        )
        self.half_token_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.half_dim),
            nn.GELU(),
            nn.Linear(self.half_dim, self.half_dim),
        )
        self.full_token_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.full_dim),
            nn.GELU(),
            nn.Linear(self.full_dim, self.full_dim),
        )

        self.coarse_refine = nn.Sequential(
            nn.Conv2d(self.coarse_dim * 2, self.coarse_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 3, padding=1),
            nn.GELU(),
        )
        self.half_refine = nn.Sequential(
            nn.Conv2d(self.half_dim * 4, self.half_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.half_dim, self.half_dim, 3, padding=1),
            nn.GELU(),
        )
        self.full_refine = nn.Sequential(
            nn.Conv2d(self.full_dim * 4, self.full_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.full_dim, self.full_dim, 3, padding=1),
            nn.GELU(),
        )

        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.occupancy_encoder = nn.Sequential(
            nn.Conv2d(self.coarse_dim, self.coarse_dim, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def prepare_features(self, encoder_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        full = self.full_proj(encoder_features["full"])
        half = self.half_proj(encoder_features["half"])
        quarter = self.quarter_proj(encoder_features["quarter"])
        eighth = self.eighth_proj(encoder_features["eighth"])
        coarse = self.coarse_fuse(
            torch.cat(
                [
                    quarter,
                    F.interpolate(eighth, size=quarter.shape[-2:], mode="bilinear", align_corners=False),
                ],
                dim=1,
            )
        )
        half_context = self.coarse_to_half(
            F.interpolate(coarse, size=half.shape[-2:], mode="bilinear", align_corners=False)
        )
        full_context = self.half_to_full(
            F.interpolate(half_context, size=full.shape[-2:], mode="bilinear", align_corners=False)
        )
        spatial_memory_map = F.adaptive_avg_pool2d(
            self.memory_refine(coarse),
            output_size=self.spatial_memory_hw,
        )
        spatial_memory = spatial_memory_map.flatten(2).transpose(1, 2).contiguous()
        return {
            "full": full,
            "half": half,
            "coarse": coarse,
            "half_context": half_context,
            "full_context": full_context,
            "spatial_memory": spatial_memory,
        }

    def initial_canvas(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, 1, self.image_size, self.image_size, device=device, dtype=dtype)

    def step(
        self,
        *,
        base_features: Dict[str, torch.Tensor],
        slot_hidden: torch.Tensor,
        pred_vector: torch.Tensor,
        canvas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_token = self.token_proj(torch.cat([slot_hidden, pred_vector], dim=-1))
        base_token = F.layer_norm(base_token, (int(base_token.shape[-1]),))

        coarse = base_features["coarse"]
        half = base_features["half"]
        full = base_features["full"]
        half_context = base_features["half_context"]
        full_context = base_features["full_context"]

        coarse_canvas = F.interpolate(canvas, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        coarse_feat = self.coarse_refine(
            torch.cat([coarse, self.coarse_canvas_proj(coarse_canvas)], dim=1)
        )
        coarse_token = F.normalize(self.coarse_token_proj(base_token), dim=-1)
        coarse_feat = F.normalize(coarse_feat, dim=1)
        coarse_logits = (
            torch.einsum("bh,bhxy->bxy", coarse_token, coarse_feat)
            * math.sqrt(float(self.coarse_dim))
        ).clamp(-20.0, 20.0)
        coarse_prob = torch.sigmoid(coarse_logits).unsqueeze(1)

        occ_hidden = self.occupancy_encoder(coarse_feat * coarse_prob).flatten(1)
        mask_context = self.context_proj(torch.cat([base_token, occ_hidden], dim=-1))
        mask_context = F.layer_norm(mask_context, (int(mask_context.shape[-1]),))

        half_canvas = F.interpolate(canvas, size=half.shape[-2:], mode="bilinear", align_corners=False)
        half_seed = F.interpolate(coarse_prob, size=half.shape[-2:], mode="bilinear", align_corners=False)
        half_feat = self.half_refine(
            torch.cat(
                [
                    half,
                    half_context,
                    self.half_canvas_proj(half_canvas),
                    self.half_seed_proj(half_seed),
                ],
                dim=1,
            )
        )
        half_token = F.normalize(self.half_token_proj(torch.cat([base_token, mask_context], dim=-1)), dim=-1)
        half_feat = F.normalize(half_feat, dim=1)
        half_logits = (
            torch.einsum("bh,bhxy->bxy", half_token, half_feat)
            * math.sqrt(float(self.half_dim))
        )
        half_logits = half_logits + F.interpolate(
            coarse_logits.unsqueeze(1),
            size=half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        half_logits = half_logits.clamp(-20.0, 20.0)
        half_prob = torch.sigmoid(half_logits).unsqueeze(1)

        full_seed = F.interpolate(half_prob, size=full.shape[-2:], mode="bilinear", align_corners=False)
        full_feat = self.full_refine(
            torch.cat(
                [
                    full,
                    full_context,
                    self.full_canvas_proj(canvas),
                    self.full_seed_proj(full_seed),
                ],
                dim=1,
            )
        )
        full_token = F.normalize(self.full_token_proj(torch.cat([base_token, mask_context], dim=-1)), dim=-1)
        full_feat = F.normalize(full_feat, dim=1)
        mask_logits = (
            torch.einsum("bh,bhxy->bxy", full_token, full_feat)
            * math.sqrt(float(self.full_dim))
        )
        mask_logits = mask_logits + F.interpolate(
            half_logits.unsqueeze(1),
            size=full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        mask_logits = mask_logits.clamp(-20.0, 20.0)

        mask_prob = torch.sigmoid(mask_logits).unsqueeze(1)
        next_canvas = torch.maximum(canvas, mask_prob)
        return mask_logits, next_canvas, mask_context


class ParameterizedHypergraphPrior(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        predictive_memory_momentum: float = 0.96,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.predictive_memory_momentum = float(min(0.999, max(0.0, predictive_memory_momentum)))
        self.term_embeddings = nn.Parameter(torch.randn(self.vocab_size, hidden_dim) * 0.02)
        self.feature_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bias_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("predictive_node_state", torch.zeros(self.vocab_size, dtype=torch.float32))
        self.register_buffer("predictive_pair_state", torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.float32))

    def forward(
        self,
        *,
        base_log_prior: torch.Tensor,
        pair_log_prior: torch.Tensor,
        selected_mass: torch.Tensor,
        selected_context: torch.Tensor,
    ) -> torch.Tensor:
        bsz = int(selected_mass.shape[0])
        total = selected_mass.sum(dim=1, keepdim=True).clamp_min(1.0)
        selected_dist = selected_mass / total

        predictive_node = self.predictive_node_state.to(
            device=selected_mass.device,
            dtype=selected_mass.dtype,
        ).unsqueeze(0).expand(bsz, -1)
        predictive_pair = self.predictive_pair_state.to(
            device=selected_mass.device,
            dtype=selected_mass.dtype,
        )
        observed_pair_signal = selected_dist @ pair_log_prior
        predictive_pair_signal = selected_dist @ predictive_pair
        base_prior = base_log_prior.unsqueeze(0).expand(bsz, -1)

        feature_stack = torch.stack(
            [
                base_prior,
                predictive_node,
                observed_pair_signal,
                predictive_pair_signal,
                selected_mass,
            ],
            dim=-1,
        )
        feature_hidden = self.feature_mlp(feature_stack)

        term_basis = self.term_embeddings.to(
            device=selected_mass.device,
            dtype=selected_mass.dtype,
        )
        term_embed = term_basis.unsqueeze(0).expand(bsz, -1, -1)
        predictive_context = selected_dist @ term_basis
        state_context = self.context_fuse(
            torch.cat([selected_context, predictive_context], dim=-1)
        ).unsqueeze(1).expand(-1, self.vocab_size, -1)

        bias = self.bias_head(
            torch.cat([term_embed, feature_hidden, state_context], dim=-1)
        ).squeeze(-1)
        return bias

    @torch.no_grad()
    def observe_predictions(
        self,
        *,
        selection_probs: torch.Tensor,
        confidence_probs: torch.Tensor,
    ) -> None:
        if int(selection_probs.ndim) != 3 or int(confidence_probs.ndim) != 2:
            return
        selection_probs = selection_probs.detach().to(
            device=self.predictive_node_state.device,
            dtype=self.predictive_node_state.dtype,
        )
        confidence_probs = confidence_probs.detach().to(
            device=self.predictive_node_state.device,
            dtype=self.predictive_node_state.dtype,
        ).clamp(0.0, 1.0)
        if int(selection_probs.shape[0]) <= 0 or int(selection_probs.shape[-1]) != int(self.vocab_size):
            return

        slot_weights = confidence_probs.unsqueeze(-1)
        weighted_mass = (selection_probs * slot_weights).sum(dim=1)
        weight_norm = slot_weights.sum(dim=1).clamp_min(1e-6)
        sample_mass = weighted_mass / weight_norm
        sample_mass = sample_mass.clamp(0.0, 1.0)
        node_update = sample_mass.mean(dim=0)
        pair_update = torch.einsum("bi,bj->ij", sample_mass, sample_mass) / float(max(1, int(sample_mass.shape[0])))
        pair_update = 0.5 * (pair_update + pair_update.transpose(0, 1))

        momentum = float(self.predictive_memory_momentum)
        self.predictive_node_state.mul_(momentum).add_((1.0 - momentum) * node_update)
        self.predictive_pair_state.mul_(momentum).add_((1.0 - momentum) * pair_update)


class SequentialSlotDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        st_dim: int,
        n_slots: int,
        vocab_size: int,
        hypergraph_prior_weight: float = 0.35,
        duplicate_penalty: float = 1.25,
        state_selection_temp: float = 6.0,
        predictive_memory_momentum: float = 0.96,
    ) -> None:
        super().__init__()
        self.n_slots = int(n_slots)
        self.hypergraph_prior_weight = float(max(0.0, hypergraph_prior_weight))
        self.duplicate_penalty = float(max(0.0, duplicate_penalty))
        self.state_selection_temp = float(max(1e-4, state_selection_temp))
        self.hypergraph_prior = ParameterizedHypergraphPrior(
            vocab_size=int(vocab_size),
            hidden_dim=hidden_dim,
            predictive_memory_momentum=predictive_memory_momentum,
        )

        self.slot_queries = nn.Parameter(torch.randn(self.n_slots, hidden_dim) * 0.02)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.slot_hidden_norm = nn.LayerNorm(hidden_dim)
        self.recurrent_norm = nn.LayerNorm(hidden_dim)
        self.selected_context_norm = nn.LayerNorm(hidden_dim)
        self.state_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.slot_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.vector_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, st_dim),
        )
        self.term_context_proj = nn.Sequential(
            nn.Linear(st_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        image_features: torch.Tensor,
        dynamic_layer: torch.Tensor,
        dynamic_valid: torch.Tensor,
        spatial_memory: torch.Tensor,
        spatial_valid: torch.Tensor,
        base_mask_features: Dict[str, torch.Tensor],
        mask_head: SequentialMaskHead,
        vocab_matrix: torch.Tensor,
        base_log_prior: torch.Tensor,
        pair_log_prior: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bsz = int(image_features.shape[0])
        vocab_matrix = F.normalize(vocab_matrix, dim=-1)
        pooled = image_features.mean(dim=(2, 3))
        recurrent = self.state_init(pooled)
        selected_context = torch.zeros_like(recurrent)
        mask_context = torch.zeros_like(recurrent)
        selected_mass = torch.zeros(
            bsz,
            int(vocab_matrix.shape[0]),
            device=image_features.device,
            dtype=image_features.dtype,
        )
        memory_tokens = torch.cat([dynamic_layer, spatial_memory], dim=1)
        memory_tokens = self.memory_norm(memory_tokens)
        memory_valid = torch.cat([dynamic_valid, spatial_valid], dim=1)
        canvas = mask_head.initial_canvas(
            batch_size=bsz,
            device=image_features.device,
            dtype=base_mask_features["full"].dtype,
        )

        pred_vectors: List[torch.Tensor] = []
        slot_hiddens: List[torch.Tensor] = []
        slot_masks: List[torch.Tensor] = []
        slot_confidence: List[torch.Tensor] = []
        slot_selection_logits: List[torch.Tensor] = []
        slot_selection_probs: List[torch.Tensor] = []
        slot_prior_bias: List[torch.Tensor] = []

        base_prior = base_log_prior.to(device=image_features.device, dtype=image_features.dtype)
        pair_prior = pair_log_prior.to(device=image_features.device, dtype=image_features.dtype)
        for slot_idx in range(self.n_slots):
            slot_query = self.slot_queries[slot_idx].unsqueeze(0).expand(bsz, -1)
            query = self.query_mlp(
                torch.cat(
                    [slot_query + recurrent, pooled, selected_context, mask_context],
                    dim=-1,
                )
            )
            query = self.query_norm(query)
            attn_out, _ = self.context_attn(
                query=query.unsqueeze(1),
                key=memory_tokens,
                value=memory_tokens,
                key_padding_mask=~memory_valid,
                need_weights=False,
            )
            attn_vec = attn_out.squeeze(1)
            slot_hidden = self.slot_fuse(
                torch.cat([query, attn_vec, selected_context, mask_context], dim=-1)
            )
            slot_hidden = self.slot_hidden_norm(slot_hidden)
            recurrent = self.state_cell(slot_hidden, recurrent)
            recurrent = self.recurrent_norm(recurrent)

            pred_vector = self.vector_head(slot_hidden)
            pred_vector_norm = F.normalize(pred_vector, dim=-1)
            raw_scores = torch.einsum("bd,vd->bv", pred_vector_norm, vocab_matrix)

            prior_bias = self.hypergraph_prior(
                base_log_prior=base_prior,
                pair_log_prior=pair_prior,
                selected_mass=selected_mass,
                selected_context=selected_context,
            )
            if float(self.duplicate_penalty) > 0.0:
                prior_bias = prior_bias - (float(self.duplicate_penalty) * selected_mass)
            prior_bias = prior_bias - prior_bias.mean(dim=-1, keepdim=True)
            prior_bias = 6.0 * torch.tanh(prior_bias / 6.0)
            selection_logits = raw_scores + (float(self.hypergraph_prior_weight) * prior_bias)
            selection_probs = torch.softmax(
                selection_logits * float(self.state_selection_temp),
                dim=-1,
            )
            selected_mass = selected_mass + selection_probs
            selected_embed = selection_probs @ vocab_matrix
            selected_context = self.term_context_proj(selected_embed)
            selected_context = self.selected_context_norm(selected_context)

            mask_logits, canvas, mask_context = mask_head.step(
                base_features=base_mask_features,
                slot_hidden=slot_hidden,
                pred_vector=pred_vector,
                canvas=canvas,
            )
            confidence_logits = self.confidence_head(
                torch.cat([slot_hidden, selected_context, mask_context], dim=-1)
            ).squeeze(-1).clamp(-20.0, 20.0)

            pred_vectors.append(pred_vector)
            slot_hiddens.append(slot_hidden)
            slot_masks.append(mask_logits)
            slot_confidence.append(confidence_logits)
            slot_selection_logits.append(selection_logits)
            slot_selection_probs.append(selection_probs)
            slot_prior_bias.append(prior_bias)

        return {
            "pred_vectors": torch.stack(pred_vectors, dim=1),
            "slot_hidden": torch.stack(slot_hiddens, dim=1),
            "pred_masks": torch.stack(slot_masks, dim=1),
            "confidence_logits": torch.stack(slot_confidence, dim=1),
            "slot_selection_logits": torch.stack(slot_selection_logits, dim=1),
            "slot_selection_probs": torch.stack(slot_selection_probs, dim=1),
            "slot_prior_bias": torch.stack(slot_prior_bias, dim=1),
            "mask_canvas": canvas,
        }

    @torch.no_grad()
    def observe_predictions(
        self,
        *,
        selection_probs: torch.Tensor,
        confidence_probs: torch.Tensor,
    ) -> None:
        self.hypergraph_prior.observe_predictions(
            selection_probs=selection_probs,
            confidence_probs=confidence_probs,
        )


try:
    from pipeline.network_api import NetworkOutput as _PipelineNetworkOutput
    from pipeline.network_api import coerce_network_output as _coerce_network_output
except ImportError:
    _PipelineNetworkOutput = None  # type: ignore[assignment,misc]
    _coerce_network_output = None  # type: ignore[assignment,misc]


class PrototypeAutoClassifier(nn.Module):
    def __init__(
        self,
        vocab_bank: SentenceTransformerVocabBank,
        row_bank: DynamicRowBank,
        hidden_dim: int = 256,
        n_slots: int = 8,
        image_size: int = 192,
        in_channels: int = 4,
        selection_temp: float = 6.0,
        hypergraph_prior_weight: float = 0.35,
        duplicate_penalty: float = 1.25,
        hypergraph_alpha: float = 0.5,
        predictive_hypergraph_momentum: float = 0.96,
    ) -> None:
        super().__init__()
        self.vocab_bank = vocab_bank
        self.row_bank = row_bank
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots
        self.image_size = image_size
        self.in_channels = in_channels
        self.selection_temp = float(max(1e-4, selection_temp))
        self.hypergraph_alpha = float(max(1e-6, hypergraph_alpha))

        self.image_encoder = ImageEncoder(in_channels=in_channels, hidden_dim=hidden_dim)
        self.dynamic_layer = DynamicLayerAssembler(row_dim=row_bank.row_dim, hidden_dim=hidden_dim)
        self.mask_head = SequentialMaskHead(
            encoder_channels=dict(self.image_encoder.output_channels),
            st_dim=vocab_bank.dim,
            hidden_dim=hidden_dim,
            image_size=image_size,
        )
        self.slot_decoder = SequentialSlotDecoder(
            hidden_dim=hidden_dim,
            st_dim=vocab_bank.dim,
            n_slots=n_slots,
            vocab_size=len(vocab_bank.phrases),
            hypergraph_prior_weight=hypergraph_prior_weight,
            duplicate_penalty=duplicate_penalty,
            state_selection_temp=self.selection_temp,
            predictive_memory_momentum=predictive_hypergraph_momentum,
        )
        self.register_buffer(
            "vocab_matrix",
            F.normalize(
                torch.stack([vocab_bank.get_vector(p) for p in vocab_bank.phrases], dim=0),
                dim=-1,
            ),
            persistent=False,
        )
        self.hypergraph = ObservedHypergraph()
        self.preview_output_panels = 2
        self.preview_confidence_floor = 0.0

    @property
    def vocab_phrases(self) -> List[str]:
        """NetworkContract — all vocabulary phrases this network was built for."""
        return list(self.vocab_bank.phrases)

    def set_preview_preferences(
        self,
        *,
        output_panels: Optional[int] = None,
        confidence_floor: Optional[float] = None,
    ) -> None:
        if output_panels is not None:
            self.preview_output_panels = 1 if int(output_panels) <= 1 else 2
        if confidence_floor is not None:
            self.preview_confidence_floor = float(max(0.0, min(1.0, confidence_floor)))

    def get_preview_preferences(self) -> Dict[str, Any]:
        return {
            "output_panel_span": int(1 if int(self.preview_output_panels) <= 1 else 2),
            "confidence_floor": float(max(0.0, min(1.0, self.preview_confidence_floor))),
        }

    def observe_terms(self, terms_rows: Sequence[Sequence[str]], step: int) -> None:
        for row in terms_rows:
            terms = [str(x).strip() for x in row if str(x).strip()]
            if terms:
                self.hypergraph.observe(terms, step=int(step))

    @torch.no_grad()
    def observe_predictions(
        self,
        slot_selection_probs: torch.Tensor,
        slot_confidence_logits: torch.Tensor,
    ) -> None:
        if not isinstance(slot_selection_probs, torch.Tensor) or not isinstance(slot_confidence_logits, torch.Tensor):
            return
        self.slot_decoder.observe_predictions(
            selection_probs=slot_selection_probs,
            confidence_probs=slot_confidence_logits.detach().sigmoid(),
        )

    def hypergraph_summary(self) -> Dict[str, int]:
        return self.hypergraph.summary()

    def hypergraph_query(self, seed_vocab: Sequence[str], max_results: int = 5) -> List[Dict[str, Any]]:
        return self.hypergraph.outward_report(seed_vocab, max_results=max_results)

    @staticmethod
    def _preview_output_field(output: Any, name: str, fallback: Optional[str] = None) -> Any:
        value = getattr(output, name, None)
        if value is not None:
            return value
        if isinstance(output, dict):
            if name in output:
                return output.get(name)
            if fallback is not None:
                return output.get(fallback)
        return None

    @staticmethod
    def _preview_output_aux(output: Any) -> Dict[str, Any]:
        aux = getattr(output, "aux", None)
        if isinstance(aux, dict):
            return aux
        if isinstance(output, dict):
            raw_aux = output.get("aux")
            if isinstance(raw_aux, dict):
                return raw_aux
            return {
                k: v
                for k, v in output.items()
                if k not in ("pred_vectors", "pred_masks", "confidence_logits", "slot_vectors", "slot_masks", "slot_confidence")
            }
        return {}

    def collect_preview_diagnostics(
        self,
        output: Any,
        batch_vocab: Sequence[Sequence[str]],
        *,
        terms_rows: Optional[Sequence[Sequence[str]]] = None,
    ) -> List[Dict[str, Any]]:
        aux = self._preview_output_aux(output)
        slot_conf = self._preview_output_field(output, "slot_confidence", fallback="confidence_logits")
        if not isinstance(slot_conf, torch.Tensor):
            return []
        conf_prob = slot_conf.detach().sigmoid()
        batch_size = int(conf_prob.shape[0])
        image_features = aux.get("image_features")
        active_rows = aux.get("active_rows")
        dynamic_layer = aux.get("dynamic_layer")
        slot_hidden = aux.get("slot_hidden")

        hg_summary = self.hypergraph_summary()
        diagnostics: List[Dict[str, Any]] = []
        for b in range(batch_size):
            targets_raw = terms_rows[b] if terms_rows is not None and b < len(terms_rows) else []
            seen: set = set()
            target_labels: List[str] = []
            for item in targets_raw:
                txt = str(item).strip()
                if txt and txt not in seen:
                    seen.add(txt)
                    target_labels.append(txt)

            vocab_count = len(batch_vocab[b]) if b < len(batch_vocab) else len(self.vocab_phrases)
            present_count = len(target_labels)
            rows: List[str] = [f"tgt={present_count} vocab={vocab_count} slots={int(self.n_slots)}"]

            if isinstance(image_features, torch.Tensor) and b < int(image_features.shape[0]):
                feat = image_features[b].detach()
                rows.append(
                    f"enc {int(feat.shape[-2])}x{int(feat.shape[-1])} "
                    f"mu={float(feat.mean().item()):.2f} sd={float(feat.std(unbiased=False).item()):.2f}"
                )

            if isinstance(active_rows, torch.Tensor) and b < int(active_rows.shape[0]):
                row_norm = float(active_rows[b].detach().norm(dim=-1).mean().item())
                rows.append(f"rows mean|r|={row_norm:.2f}")

            dyn_txt = None
            if isinstance(dynamic_layer, torch.Tensor) and b < int(dynamic_layer.shape[0]):
                dyn_txt = f"dyn|h|={float(dynamic_layer[b].detach().norm(dim=-1).mean().item()):.2f}"
            slot_txt = None
            if isinstance(slot_hidden, torch.Tensor) and b < int(slot_hidden.shape[0]):
                slot_txt = f"slot|h|={float(slot_hidden[b].detach().norm(dim=-1).mean().item()):.2f}"
            if dyn_txt is not None or slot_txt is not None:
                rows.append(" ".join(x for x in (dyn_txt, slot_txt) if x))

            conf_b = conf_prob[b].detach()
            rows.append(
                f"conf mean={float(conf_b.mean().item()):.2f} "
                f"hot={int((conf_b >= 0.5).sum().item())}/{int(conf_b.numel())}"
            )
            rows.append(
                f"hyper e={int(hg_summary.get('edge_count', 0))} "
                f"obs={int(hg_summary.get('total_observations', 0))} "
                f"max={int(hg_summary.get('max_order', 0))}"
            )

            clue_rows: List[Dict[str, Any]] = []
            if target_labels:
                clue_rows = self.hypergraph_query(target_labels, max_results=4)
            clue = None
            if clue_rows:
                clue = next((row for row in clue_rows if row.get("novelty")), clue_rows[0])
            if clue is None:
                rows.append("hg clue cold-start")
            else:
                novelty = list(clue.get("novelty") or [])
                members = [str(x) for x in list(clue.get("members") or [])[:3]]
                if novelty:
                    rows.append(
                        f"hg clue ov={int(clue.get('overlap', 0))} "
                        f"x={int(clue.get('count', 0))} "
                        f"new={','.join(str(x) for x in novelty[:2])}"
                    )
                else:
                    rows.append(
                        f"hg clue ov={int(clue.get('overlap', 0))} "
                        f"x={int(clue.get('count', 0))} "
                        f"path={'+'.join(members) if members else '-'}"
                    )

            diagnostics.append(
                {
                    "target_labels": target_labels or ["none"],
                    "decision_rows": rows,
                    "hypergraph_summary": dict(hg_summary),
                    "hypergraph_query": clue_rows,
                }
            )
        return diagnostics

    def forward(self, image_rgba: torch.Tensor, batch_vocab: List[List[str]]) -> Dict[str, torch.Tensor]:
        device = image_rgba.device
        encoder_features = self.image_encoder(image_rgba)
        image_features = encoder_features["eighth"]
        active_rows, active_valid = self.row_bank.assemble(batch_vocab, device=device)
        dynamic_layer = self.dynamic_layer(image_features, active_rows, active_valid)
        base_mask_features = self.mask_head.prepare_features(encoder_features)
        spatial_memory = base_mask_features["spatial_memory"]
        spatial_valid = torch.ones(
            spatial_memory.shape[0],
            spatial_memory.shape[1],
            device=device,
            dtype=torch.bool,
        )
        hg_stats = self.hypergraph.vocab_statistics(
            self.vocab_phrases,
            alpha=self.hypergraph_alpha,
        )
        decoded = self.slot_decoder(
            image_features=image_features,
            dynamic_layer=dynamic_layer,
            dynamic_valid=active_valid,
            spatial_memory=spatial_memory,
            spatial_valid=spatial_valid,
            base_mask_features=base_mask_features,
            mask_head=self.mask_head,
            vocab_matrix=self.vocab_matrix,
            base_log_prior=hg_stats["base_log_prior"],
            pair_log_prior=hg_stats["pair_log_prior"],
        )
        return {
            "image_features": image_features,
            "active_rows": active_rows,
            "active_valid": active_valid,
            "dynamic_layer": dynamic_layer,
            "spatial_memory": spatial_memory,
            "slot_hidden": decoded["slot_hidden"],
            "pred_vectors": decoded["pred_vectors"],
            "pred_masks": decoded["pred_masks"],
            "confidence_logits": decoded["confidence_logits"],
            "slot_selection_logits": decoded["slot_selection_logits"],
            "slot_selection_probs": decoded["slot_selection_probs"],
            "slot_prior_bias": decoded["slot_prior_bias"],
            "mask_canvas": decoded["mask_canvas"],
        }

    def forward_batch(self, image: torch.Tensor, vocab: List[List[str]]) -> Any:
        """NetworkContract entry-point with channel adaptation."""
        if int(image.shape[1]) < int(self.in_channels):
            pad_ch = int(self.in_channels) - int(image.shape[1])
            pad = torch.ones(
                image.shape[0],
                pad_ch,
                image.shape[2],
                image.shape[3],
                device=image.device,
                dtype=image.dtype,
            )
            image = torch.cat([image, pad], dim=1)
        elif int(image.shape[1]) > int(self.in_channels):
            image = image[:, : int(self.in_channels)]
        out = self.forward(image, vocab)
        if _PipelineNetworkOutput is not None:
            return _PipelineNetworkOutput(
                slot_vectors=out["pred_vectors"],
                slot_masks=out["pred_masks"],
                slot_confidence=out["confidence_logits"],
                assignments=[],  # filled by criterion during training
                aux={k: v for k, v in out.items()
                     if k not in ("pred_vectors", "pred_masks", "confidence_logits")},
            )
        return out


# ============================================================
# One-to-one assignment
# ============================================================


class OneToOneMatcher(nn.Module):
    def __init__(self, dustbin_cost: float = 1.25) -> None:
        super().__init__()
        self.dustbin_cost = dustbin_cost

    @staticmethod
    def _solve_rectangular_assignment(sample_cost: np.ndarray) -> List[Optional[int]]:
        """
        Exact one-to-one assignment for the tiny slot counts used here.
        We optimize over row bitmasks instead of bouncing into SciPy/NumPy LAP code.
        """
        n_slots, valid_m = sample_cost.shape
        if n_slots <= 0:
            return []
        if valid_m <= 0:
            return [None] * n_slots

        state_count = 1 << n_slots
        full_mask = state_count - 1
        slot_to_target: List[Optional[int]] = [None] * n_slots

        if valid_m <= n_slots:
            dp = np.full(state_count, np.inf, dtype=np.float64)
            dp[0] = 0.0
            prev_masks: List[np.ndarray] = []
            chosen_rows: List[np.ndarray] = []
            for target_idx in range(valid_m):
                new_dp = np.full(state_count, np.inf, dtype=np.float64)
                prev_mask = np.full(state_count, -1, dtype=np.int32)
                chosen_row = np.full(state_count, -1, dtype=np.int16)
                for mask in range(state_count):
                    current = float(dp[mask])
                    if not math.isfinite(current) or int(mask.bit_count()) != int(target_idx):
                        continue
                    free_rows = (~mask) & full_mask
                    while free_rows:
                        bit = free_rows & -free_rows
                        row_idx = int(bit.bit_length() - 1)
                        next_mask = int(mask | bit)
                        candidate = current + float(sample_cost[row_idx, target_idx])
                        if candidate < float(new_dp[next_mask]):
                            new_dp[next_mask] = candidate
                            prev_mask[next_mask] = mask
                            chosen_row[next_mask] = row_idx
                        free_rows ^= bit
                dp = new_dp
                prev_masks.append(prev_mask)
                chosen_rows.append(chosen_row)

            best_mask = -1
            best_cost = float("inf")
            for mask in range(state_count):
                if int(mask.bit_count()) != int(valid_m):
                    continue
                current = float(dp[mask])
                if current < best_cost:
                    best_cost = current
                    best_mask = mask
            if best_mask < 0:
                return slot_to_target

            mask = int(best_mask)
            for target_idx in range(valid_m - 1, -1, -1):
                row_idx = int(chosen_rows[target_idx][mask])
                prev_mask = int(prev_masks[target_idx][mask])
                if row_idx >= 0 and prev_mask >= 0:
                    slot_to_target[row_idx] = target_idx
                    mask = prev_mask
            return slot_to_target

        dp = np.full(state_count, np.inf, dtype=np.float64)
        dp[0] = 0.0
        prev_masks = []
        chosen_rows = []
        for target_idx in range(valid_m):
            new_dp = dp.copy()
            prev_mask = np.arange(state_count, dtype=np.int32)
            chosen_row = np.full(state_count, -1, dtype=np.int16)
            for mask in range(state_count):
                current = float(dp[mask])
                if not math.isfinite(current):
                    continue
                free_rows = (~mask) & full_mask
                while free_rows:
                    bit = free_rows & -free_rows
                    row_idx = int(bit.bit_length() - 1)
                    next_mask = int(mask | bit)
                    candidate = current + float(sample_cost[row_idx, target_idx])
                    if candidate < float(new_dp[next_mask]):
                        new_dp[next_mask] = candidate
                        prev_mask[next_mask] = mask
                        chosen_row[next_mask] = row_idx
                    free_rows ^= bit
            dp = new_dp
            prev_masks.append(prev_mask)
            chosen_rows.append(chosen_row)

        mask = int(full_mask)
        for target_idx in range(valid_m - 1, -1, -1):
            row_idx = int(chosen_rows[target_idx][mask])
            prev_mask = int(prev_masks[target_idx][mask])
            if row_idx >= 0:
                slot_to_target[row_idx] = target_idx
            mask = prev_mask
        return slot_to_target

    def forward(self, pred_vectors: torch.Tensor, target_vectors: torch.Tensor, target_valid: torch.Tensor) -> Dict[str, Any]:
        """
        Returns hard one-to-one assignment from predicted bag to target bag.
        Each slot gets at most one item.
        Each item gets at most one slot.
        Extra slots go to dustbin.
        """
        device = pred_vectors.device
        bsz, n_slots, _ = pred_vectors.shape
        _, max_m, _ = target_vectors.shape

        pred_n = l2_normalize(_sanitize_finite_tensor(pred_vectors), dim=-1)
        targ_n = l2_normalize(_sanitize_finite_tensor(target_vectors), dim=-1)
        pair_cost = ((pred_n.unsqueeze(2) - targ_n.unsqueeze(1)) ** 2).sum(dim=-1)  # [B, N, M]
        pair_cost = _sanitize_finite_tensor(pair_cost, fill=_MATCHER_INVALID_COST)
        pair_cost = pair_cost.masked_fill(~target_valid.unsqueeze(1), _MATCHER_INVALID_COST)

        assignments: List[Dict[str, Any]] = []
        assign_matrix = torch.zeros(bsz, n_slots, max_m, device=device, dtype=pair_cost.dtype)
        dustbin_mask = torch.zeros(bsz, n_slots, device=device, dtype=torch.bool)

        for b in range(bsz):
            valid_m = int(target_valid[b].sum().item())
            slot_to_target: List[Optional[int]] = [None] * n_slots
            if valid_m > 0:
                sample_cost = (
                    pair_cost[b, :, :valid_m]
                    .detach()
                    .cpu()
                    .to(dtype=torch.float64)
                    .numpy()
                )  # [N, valid_m]
                slot_to_target = self._solve_rectangular_assignment(sample_cost)

            for slot_idx, target_idx in enumerate(slot_to_target):
                if target_idx is None:
                    dustbin_mask[b, slot_idx] = True
                    continue
                assign_matrix[b, slot_idx, int(target_idx)] = 1.0

            assignments.append(
                {
                    "slot_to_target": slot_to_target,
                    "valid_target_count": valid_m,
                }
            )

        return {
            "pair_cost": pair_cost,
            "assign_matrix": assign_matrix,
            "dustbin_mask": dustbin_mask,
            "assignments": assignments,
        }


# ============================================================
# Loss
# ============================================================


class PrototypeLoss(nn.Module):
    def __init__(
        self,
        matcher: OneToOneMatcher,
        vocab_matrix: torch.Tensor,
        enable_vector_loss: bool = True,
        vector_weight: float = 1.0,
        enable_mask_loss: bool = True,
        mask_weight: float = 0.7,
        enable_selection_loss: bool = True,
        selection_weight: float = 0.5,
        selection_temp: float = 10.0,
        enable_confidence_loss: bool = True,
        confidence_weight: float = 0.5,
        enable_mask_order_loss: bool = True,
        mask_order_weight: float = 0.25,
        mask_order_margin: float = 0.02,
        mask_order_target_volume_weight: float = 1.0,
        mask_order_target_spread_weight: float = 0.75,
        mask_order_pred_volume_weight: float = 1.0,
        mask_order_pred_spread_weight: float = 0.75,
        enable_residual_mask_loss: bool = True,
        residual_mask_weight: float = 0.20,
        residual_mask_detach_canvas: bool = True,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        # [V, D] normalised ST vectors for the full vocabulary — kept on CPU,
        # moved to the right device lazily at forward time.
        self.register_buffer("vocab_matrix", vocab_matrix, persistent=False)
        self.enable_vector_loss = bool(enable_vector_loss)
        self.vector_weight = vector_weight
        self.enable_mask_loss = bool(enable_mask_loss)
        self.mask_weight = mask_weight
        self.enable_selection_loss = bool(enable_selection_loss)
        self.selection_weight = selection_weight
        self.selection_temp = selection_temp
        # confidence_weight: how strongly each slot is penalised for wrong
        # certainty.  0.0 = disabled; typical range 0.3–1.0.
        self.enable_confidence_loss = bool(enable_confidence_loss)
        self.confidence_weight = confidence_weight
        self.enable_mask_order_loss = bool(enable_mask_order_loss)
        self.mask_order_weight = float(mask_order_weight)
        self.mask_order_margin = float(mask_order_margin)
        self.mask_order_target_volume_weight = float(mask_order_target_volume_weight)
        self.mask_order_target_spread_weight = float(mask_order_target_spread_weight)
        self.mask_order_pred_volume_weight = float(mask_order_pred_volume_weight)
        self.mask_order_pred_spread_weight = float(mask_order_pred_spread_weight)
        self.enable_residual_mask_loss = bool(enable_residual_mask_loss)
        self.residual_mask_weight = float(residual_mask_weight)
        self.residual_mask_detach_canvas = bool(residual_mask_detach_canvas)
        self.loss_report_order = [
            "vector_loss",
            "mask_loss",
            "selection_loss",
            "confidence_loss",
            "mask_order_loss",
            "residual_mask_loss",
        ]
        self.loss_report_labels = {
            "vector_loss": "vec",
            "mask_loss": "mask",
            "selection_loss": "sel",
            "confidence_loss": "conf",
            "mask_order_loss": "ord",
            "residual_mask_loss": "resid",
        }

    @staticmethod
    def _spatial_entropy(mask: torch.Tensor) -> torch.Tensor:
        flat = mask.reshape(mask.shape[0], mask.shape[1], -1).clamp_min(0.0)
        total = flat.sum(dim=-1, keepdim=True)
        probs = flat / total.clamp_min(1e-6)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        norm = math.log(float(max(2, int(flat.shape[-1]))))
        entropy = entropy / max(1e-6, norm)
        return torch.where(total.squeeze(-1) > 1e-6, entropy, torch.zeros_like(entropy))

    def loss_report_items(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "vector_loss",
                "label": "vec",
                "enabled": bool(self.enable_vector_loss and float(self.vector_weight) > 0.0),
                "weight": float(self.vector_weight),
            },
            {
                "key": "mask_loss",
                "label": "mask",
                "enabled": bool(self.enable_mask_loss and float(self.mask_weight) > 0.0),
                "weight": float(self.mask_weight),
            },
            {
                "key": "selection_loss",
                "label": "sel",
                "enabled": bool(self.enable_selection_loss and float(self.selection_weight) > 0.0),
                "weight": float(self.selection_weight),
            },
            {
                "key": "confidence_loss",
                "label": "conf",
                "enabled": bool(self.enable_confidence_loss and float(self.confidence_weight) > 0.0),
                "weight": float(self.confidence_weight),
            },
            {
                "key": "mask_order_loss",
                "label": "ord",
                "enabled": bool(self.enable_mask_order_loss and float(self.mask_order_weight) > 0.0),
                "weight": float(self.mask_order_weight),
            },
            {
                "key": "residual_mask_loss",
                "label": "resid",
                "enabled": bool(self.enable_residual_mask_loss and float(self.residual_mask_weight) > 0.0),
                "weight": float(self.residual_mask_weight),
            },
        ]

    def forward(
        self,
        pred_vectors: torch.Tensor,
        pred_masks: torch.Tensor,
        target_vectors: torch.Tensor,
        target_masks: torch.Tensor,
        target_valid: torch.Tensor,
        present_mask: torch.Tensor,
        confidence_logits: Optional[torch.Tensor] = None,
        slot_selection_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        nonfinite_stats = {
            "pred_vectors": _count_nonfinite(pred_vectors),
            "pred_masks": _count_nonfinite(pred_masks),
            "target_vectors": _count_nonfinite(target_vectors),
            "target_masks": _count_nonfinite(target_masks),
            "confidence_logits": _count_nonfinite(confidence_logits),
            "slot_selection_logits": _count_nonfinite(slot_selection_logits),
        }

        pred_vectors = _sanitize_finite_tensor(pred_vectors)
        pred_masks = _sanitize_finite_tensor(pred_masks)
        target_vectors = _sanitize_finite_tensor(target_vectors)
        target_masks = _sanitize_finite_tensor(target_masks)
        if confidence_logits is not None:
            confidence_logits = _sanitize_finite_tensor(confidence_logits)
        if slot_selection_logits is not None:
            slot_selection_logits = _sanitize_finite_tensor(slot_selection_logits)

        B = pred_vectors.shape[0]
        device = pred_vectors.device

        # --- Present-only matching for vector_loss and mask_loss ---
        # Build padded tensors containing only the present items per sample so
        # the matcher never touches absent vocab items.  Whatever the full-vocab
        # scores look like, the binding for the reconstruction losses is decided
        # solely among real targets.
        counts = present_mask.sum(dim=1)           # [B]
        max_m = int(counts.max().item())
        _, _, H, W = target_masks.shape
        D = target_vectors.shape[-1]

        p_vecs  = torch.zeros(B, max_m, D,    device=device, dtype=target_vectors.dtype)
        p_masks = torch.zeros(B, max_m, H, W, device=device, dtype=target_masks.dtype)
        p_valid = torch.zeros(B, max_m,        device=device, dtype=torch.bool)
        for b in range(B):
            idx = present_mask[b].nonzero(as_tuple=True)[0]
            m = idx.shape[0]
            p_vecs[b,  :m] = target_vectors[b, idx]
            p_masks[b, :m] = target_masks[b,   idx]
            p_valid[b, :m] = True

        # Present-only match: slots compete only among real targets → clean reconstruction loss.
        pmatch       = self.matcher(pred_vectors, p_vecs, p_valid)
        p_pair_cost  = pmatch["pair_cost"]      # [B, N, max_m]
        p_assign_mat = pmatch["assign_matrix"]  # [B, N, max_m]
        sample_match_count = p_assign_mat.sum(dim=(1, 2)).clamp_min(1.0)
        match_count = sample_match_count.sum()
        active_slot_mask = (~pmatch["dustbin_mask"]).to(dtype=pred_vectors.dtype)

        vector_loss = ((p_assign_mat * p_pair_cost).sum(dim=(1, 2)) / sample_match_count).mean()

        # Full-vocab match: slots pick freely across all vocab → drives selection pressure
        # and is what the viewer shows as "committed" assignments.
        fmatch      = self.matcher(pred_vectors, target_vectors, target_valid)
        assignments = fmatch["assignments"]     # indices into full vocab (for display)
        assigned_target_masks = (
            torch.einsum("bnm,bmhw->bnhw", p_assign_mat, p_masks)
            if max_m > 0
            else pred_masks.new_zeros(B, int(pred_masks.shape[1]), H, W)
        )

        if max_m > 0:
            slot_mask_cost = F.binary_cross_entropy_with_logits(
                pred_masks,
                assigned_target_masks,
                reduction="none",
            ).mean(dim=(-1, -2))
            mask_pair_cost = slot_mask_cost
            mask_loss = ((slot_mask_cost * active_slot_mask).sum(dim=1) / sample_match_count).mean()
        else:
            mask_pair_cost = pred_vectors.new_zeros(B, int(pred_vectors.shape[1]))
            mask_loss = pred_vectors.new_tensor(0.0)

        pred_probs = pred_masks.sigmoid()

        # Selection loss: for each vocab term, aggregate cosine similarity across
        # all slots via logsumexp so that every slot receives a gradient
        # (not just the argmax winner).  logsumexp acts as a smooth-max; its
        # gradient w.r.t. each slot score is proportional to that slot's current
        # score, so well-aligned slots are pushed harder but no slot is frozen out.
        if slot_selection_logits is None:
            vocab_mat = _sanitize_finite_tensor(self.vocab_matrix.to(pred_vectors.device))  # [V, D]
            pred_norm = l2_normalize(pred_vectors, dim=-1)                 # [B, N, D]
            scores = torch.einsum("bnd,vd->bnv", pred_norm, vocab_mat)     # [B, N, V]
        else:
            scores = slot_selection_logits
        n_slots = max(1, int(scores.shape[1]))
        temp = float(max(1e-4, self.selection_temp))
        agg_scores = (
            torch.logsumexp(scores * temp, dim=1) - math.log(float(n_slots))
        ) / temp
        selection_loss = F.binary_cross_entropy_with_logits(
            agg_scores,
            present_mask.float(),
            reduction="mean",
        )

        # Confidence loss: slots assigned to real present items should be high
        # confidence (target=1); dustbin slots should be low (target=0).
        # pmatch["dustbin_mask"] is True for every slot that went to the dustbin
        # in the present-only match, so (~dustbin_mask).float() gives 1.0/0.0 targets.
        confidence_loss = pred_vectors.new_tensor(0.0)
        if confidence_logits is not None and float(self.confidence_weight) > 0.0:
            confidence_target = (~pmatch["dustbin_mask"]).float()   # [B, N]
            confidence_loss = F.binary_cross_entropy_with_logits(
                confidence_logits, confidence_target, reduction="mean"
            )

        target_volume = assigned_target_masks.mean(dim=(-1, -2))
        target_spread = self._spatial_entropy(assigned_target_masks)
        pred_volume = pred_probs.mean(dim=(-1, -2))
        pred_spread = self._spatial_entropy(pred_probs)

        target_priority = (
            (self.mask_order_target_volume_weight * target_volume)
            + (self.mask_order_target_spread_weight * target_spread)
        )
        pred_priority = (
            (self.mask_order_pred_volume_weight * pred_volume)
            + (self.mask_order_pred_spread_weight * pred_spread)
        )

        mask_order_loss = pred_vectors.new_tensor(0.0)
        if bool(self.enable_mask_order_loss) and float(self.mask_order_weight) > 0.0:
            target_i = target_priority.unsqueeze(2)
            target_j = target_priority.unsqueeze(1)
            pred_i = pred_priority.unsqueeze(2)
            pred_j = pred_priority.unsqueeze(1)
            active_i = active_slot_mask.unsqueeze(2)
            active_j = active_slot_mask.unsqueeze(1)
            pair_mask = (
                (active_i > 0.0)
                & (active_j > 0.0)
                & ((target_i + 1e-6) < target_j)
            ).to(dtype=pred_vectors.dtype)
            if confidence_logits is not None:
                conf_prob = confidence_logits.detach().sigmoid().to(dtype=pred_vectors.dtype)
                pair_weight = pair_mask * (conf_prob.unsqueeze(2) * conf_prob.unsqueeze(1))
            else:
                pair_weight = pair_mask
            pair_penalty = F.relu(pred_i - pred_j + float(max(0.0, self.mask_order_margin)))
            denom = pair_weight.sum().clamp_min(1.0)
            mask_order_loss = (pair_penalty * pair_weight).sum() / denom

        residual_mask_loss = pred_vectors.new_tensor(0.0)
        if bool(self.enable_residual_mask_loss) and float(self.residual_mask_weight) > 0.0:
            prev_canvas = pred_probs.new_zeros(B, 1, H, W)
            residual_parts: List[torch.Tensor] = []
            for slot_idx in range(int(pred_probs.shape[1])):
                pred_slot = pred_probs[:, slot_idx : slot_idx + 1]
                target_slot = assigned_target_masks[:, slot_idx : slot_idx + 1]
                ref_canvas = prev_canvas.detach() if self.residual_mask_detach_canvas else prev_canvas
                residual_target = (target_slot * (1.0 - ref_canvas)).clamp(0.0, 1.0)
                novel_pred = torch.relu(pred_slot - ref_canvas).clamp(0.0, 1.0)
                slot_mse = F.mse_loss(novel_pred, residual_target, reduction="none").mean(dim=(1, 2, 3))
                residual_parts.append(slot_mse * active_slot_mask[:, slot_idx])
                prev_canvas = torch.maximum(prev_canvas, pred_slot)
            if residual_parts:
                residual_stack = torch.stack(residual_parts, dim=1)
                residual_mask_loss = residual_stack.sum() / active_slot_mask.sum().clamp_min(1.0)

        weighted_terms = {
            "vector_loss": (
                vector_loss if bool(self.enable_vector_loss) and float(self.vector_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.vector_weight),
            "mask_loss": (
                mask_loss if bool(self.enable_mask_loss) and float(self.mask_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.mask_weight),
            "selection_loss": (
                selection_loss if bool(self.enable_selection_loss) and float(self.selection_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.selection_weight),
            "confidence_loss": (
                confidence_loss if bool(self.enable_confidence_loss) and float(self.confidence_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.confidence_weight),
            "mask_order_loss": (
                mask_order_loss if bool(self.enable_mask_order_loss) and float(self.mask_order_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.mask_order_weight),
            "residual_mask_loss": (
                residual_mask_loss if bool(self.enable_residual_mask_loss) and float(self.residual_mask_weight) > 0.0
                else pred_vectors.new_tensor(0.0)
            ) * float(self.residual_mask_weight),
        }
        total = pred_vectors.new_tensor(0.0)
        for term_value in weighted_terms.values():
            total = total + term_value
        return {
            "loss": total,
            "vector_loss": vector_loss.detach(),
            "mask_loss": mask_loss.detach(),
            "selection_loss": selection_loss.detach(),
            "confidence_loss": confidence_loss.detach(),
            "mask_order_loss": mask_order_loss.detach(),
            "residual_mask_loss": residual_mask_loss.detach(),
            "selection_logits": agg_scores.detach(),
            "match_count": match_count.detach(),
            "target_priority": target_priority.detach(),
            "pred_priority": pred_priority.detach(),
            "weighted_loss_terms": {k: v.detach() for k, v in weighted_terms.items()},
            "pair_cost": p_pair_cost.detach(),
            "assign_matrix": p_assign_mat.detach(),
            "mask_pair_cost": mask_pair_cost.detach(),
            "assignments": assignments,
            "nonfinite_stats": nonfinite_stats,
        }


# ============================================================
# Viewer
# ============================================================


class PrototypeViewer:
    def __init__(self, size: Tuple[int, int] = (1500, 950)) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 15)

    def close(self) -> None:
        pygame.quit()

    def pump(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _tensor_to_surface(self, x: torch.Tensor) -> pygame.Surface:
        x = x.detach().cpu().clamp(0, 1)
        if x.dim() == 2:
            x = x.unsqueeze(0).repeat(3, 1, 1)
        elif x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def draw(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor], losses: Dict[str, Any], hypergraph: ObservedHypergraph) -> None:
        self.screen.fill((18, 18, 22))
        image = batch["image"][0]
        full_vocab = batch["batch_vocab"][0]   # all vocab phrases — assignments index into this
        item_vocab = batch["item_vocab"][0]    # present phrases — for the "present:" display line
        pred_masks = outputs["pred_masks"][0].sigmoid()
        pred_conf = outputs.get("confidence_logits")
        if pred_conf is not None:
            pred_conf = pred_conf[0].sigmoid()   # [N]
        assignments = losses["assignments"][0]["slot_to_target"]

        rgb = pygame.transform.scale(self._tensor_to_surface(image[:3]), (256, 256))
        alpha = pygame.transform.scale(self._tensor_to_surface(image[3]), (256, 256))
        self.screen.blit(rgb, (16, 16))
        self.screen.blit(alpha, (288, 16))
        self.screen.blit(self.font.render("input RGB", True, (230, 230, 230)), (16, 278))
        self.screen.blit(self.font.render("alpha/support prior", True, (230, 230, 230)), (288, 278))

        start_x = 560
        start_y = 16
        tile = 140
        max_slots = min(pred_masks.shape[0], 8)
        for i in range(max_slots):
            surf = pygame.transform.scale(self._tensor_to_surface(pred_masks[i]), (tile, tile))
            x = start_x + (i % 4) * (tile + 12)
            y = start_y + (i // 4) * (tile + 40)
            self.screen.blit(surf, (x, y))
            picked = assignments[i]
            conf_str = f"{float(pred_conf[i]):.2f}" if pred_conf is not None else "?"
            if picked is None:
                label = f"slot {i} → dustbin ({conf_str})"
                col = (160, 100, 100)
            else:
                label = f"slot {i} → {full_vocab[picked]} ({conf_str})"
                col = (220, 220, 220)
            self.screen.blit(self.small.render(label, True, col), (x, y + tile + 4))

        text_y = 400
        self.screen.blit(self.font.render(f"loss={float(losses['loss']):.5f}", True, (180, 255, 180)), (16, text_y))
        loss_rows = [str(x) for x in list(losses.get("loss_rows") or []) if str(x).strip()]
        for row_idx, row in enumerate(loss_rows[:6]):
            self.screen.blit(self.small.render(row, True, (200, 220, 255)), (16, text_y + 28 + (row_idx * 20)))
        info_y = text_y + 28 + (max(1, len(loss_rows[:6])) * 20)
        self.screen.blit(self.font.render(f"present: {', '.join(item_vocab)}", True, (255, 220, 180)), (16, info_y))
        self.screen.blit(self.font.render(f"hyperedges stored: {len(hypergraph)}", True, (210, 210, 210)), (16, info_y + 24))

        text_y = info_y + 42
        self.screen.blit(self.font.render("resolved one-to-one picks", True, (255, 220, 120)), (16, text_y))
        text_y += 24
        for i, picked in enumerate(assignments[:8]):
            if picked is None:
                line = f"slot {i}: dustbin"
            else:
                line = f"slot {i}: {full_vocab[picked]}"
            self.screen.blit(self.small.render(line, True, (210, 210, 210)), (16, text_y))
            text_y += 20

        text_y += 10
        self.screen.blit(self.font.render("hypergraph outward search", True, (255, 220, 120)), (16, text_y))
        text_y += 24
        for edge, score, count in hypergraph.search_outward(item_vocab, max_results=8):
            self.screen.blit(self.small.render(f"{edge}  score={score} count={count}", True, (210, 210, 210)), (16, text_y))
            text_y += 20

        # Lower-right panel: mask thumbnails for non-dustbin slots, sorted alphabetically
        # by their assigned word so progress is visible as a stable ordered column.
        entries: List[Tuple[str, Any]] = []
        for slot_i, picked in enumerate(assignments):
            if picked is not None:
                entries.append((full_vocab[picked], pred_masks[slot_i]))
        entries.sort(key=lambda e: e[0])

        tile = 72
        lbl_h = self.small.get_linesize()
        pr = self.screen.get_width() - 20   # right edge
        wy = 390

        hdr_surf = self.small.render(f"sorted non-dustbin ({len(entries)})", True, (255, 220, 120))
        self.screen.blit(hdr_surf, (pr - hdr_surf.get_width(), wy))
        wy += lbl_h + 6

        for word, mask in entries:
            lbl_surf = self.small.render(word, True, (160, 230, 160))
            self.screen.blit(lbl_surf, (pr - lbl_surf.get_width(), wy))
            wy += lbl_h + 2
            mask_surf = pygame.transform.scale(
                self._tensor_to_surface(mask.unsqueeze(0)), (tile, tile)
            )
            self.screen.blit(mask_surf, (pr - tile, wy))
            wy += tile + 6
            if wy > self.screen.get_height() - 20:
                break

        pygame.display.flip()
        self.clock.tick(30)


# ============================================================
# Training harness
# ============================================================


@dataclass
class TrainConfig:
    phrases: List[str] = field(default_factory=lambda: list("abcdefghijklmnopqrstuvwxyz0123456789"))
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_size: int = 50
    row_dim: int = 128
    hidden_dim: int = 256
    n_slots: int = 8
    batch_size: int = 64
    train_samples: int = 20000
    learning_rate: float = 5e-4
    steps: int = 2000
    min_items: int = 1
    max_items: int = 8
    seed: int = 1234
    use_pygame: bool = True
    num_workers: int = 0


class PrototypeTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        set_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vocab_bank = SentenceTransformerVocabBank(
            phrases=config.phrases,
            model_name=config.sentence_transformer_model,
            device=str(self.device),
            normalize=True,
        )
        self.row_bank = DynamicRowBank(phrases=config.phrases, row_dim=config.row_dim)

        self.dataset = SyntheticTextSupportDataset(
            vocab_bank=self.vocab_bank,
            image_size=config.image_size,
            min_items=config.min_items,
            max_items=config.max_items,
            samples=config.train_samples,
            seed=config.seed + 99,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_synthetic,
        )

        self.model = PrototypeAutoClassifier(
            vocab_bank=self.vocab_bank,
            row_bank=self.row_bank,
            hidden_dim=config.hidden_dim,
            n_slots=config.n_slots,
            image_size=config.image_size,
        ).to(self.device)
        # Normalised ST vectors for all vocabulary phrases — used by the selection loss.
        vocab_matrix = F.normalize(
            torch.stack([self.vocab_bank.get_vector(p) for p in config.phrases], dim=0),
            dim=-1,
        )
        self.matcher = OneToOneMatcher(dustbin_cost=1.25)
        self.criterion = PrototypeLoss(
            self.matcher,
            vocab_matrix=vocab_matrix,
            vector_weight=1.0,
            mask_weight=0.7,
            selection_weight=0.5,
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.viewer = PrototypeViewer() if config.use_pygame else None

    def _forward_network_output(self, batch: Dict[str, Any]) -> Any:
        raw_out = (
            self.model.forward_batch(batch["image"], batch["batch_vocab"])
            if hasattr(self.model, "forward_batch") and callable(getattr(self.model, "forward_batch"))
            else self.model(batch["image"], batch["batch_vocab"])
        )
        if _coerce_network_output is not None:
            return _coerce_network_output(raw_out)
        return raw_out

    @staticmethod
    def _output_aux(output: Any) -> Dict[str, Any]:
        aux = getattr(output, "aux", None)
        if isinstance(aux, dict):
            return aux
        if isinstance(output, dict):
            raw_aux = output.get("aux")
            if isinstance(raw_aux, dict):
                return raw_aux
            return {
                k: v
                for k, v in output.items()
                if k not in ("pred_vectors", "pred_masks", "confidence_logits", "slot_vectors", "slot_masks", "slot_confidence")
            }
        return {}

    @staticmethod
    def _output_tensor(output: Any, name: str, fallback: Optional[str] = None) -> Optional[torch.Tensor]:
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(output, dict):
            raw = output.get(name)
            if isinstance(raw, torch.Tensor):
                return raw
            if fallback is not None:
                raw = output.get(fallback)
                if isinstance(raw, torch.Tensor):
                    return raw
        return None

    def _format_loss_rows(self, losses: Dict[str, Any], per_row: int = 3) -> List[str]:
        items = []
        if hasattr(self.criterion, "loss_report_items") and callable(getattr(self.criterion, "loss_report_items")):
            try:
                items = list(self.criterion.loss_report_items())
            except Exception:
                items = []
        if not items:
            items = [
                {"key": "vector_loss", "label": "vec", "enabled": True, "weight": 1.0},
                {"key": "mask_loss", "label": "mask", "enabled": True, "weight": 1.0},
                {"key": "selection_loss", "label": "sel", "enabled": True, "weight": 1.0},
                {"key": "confidence_loss", "label": "conf", "enabled": True, "weight": 1.0},
            ]
        tokens: List[str] = []
        for item in items:
            key = str(item.get("key", "")).strip()
            if not key:
                continue
            label = str(item.get("label", key))
            enabled = bool(item.get("enabled", True))
            weight = float(item.get("weight", 1.0))
            value = losses.get(key, 0.0)
            if isinstance(value, torch.Tensor):
                value = float(value.detach().item())
            if enabled:
                tokens.append(f"{label}={float(value):.5f}@{weight:.2f}")
            else:
                tokens.append(f"{label}=off@{weight:.2f}")
        rows: List[str] = []
        for start in range(0, len(tokens), max(1, int(per_row))):
            rows.append(" ".join(tokens[start : start + max(1, int(per_row))]))
        return rows

    def _move_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": batch["image"].to(self.device),
            "batch_vocab": batch["batch_vocab"],
            "item_vocab": batch["item_vocab"],
            "present_mask": batch["present_mask"].to(self.device),
            "target_vectors": batch["target_vectors"].to(self.device),
            "target_masks": batch["target_masks"].to(self.device),
            "target_valid": batch["target_valid"].to(self.device),
        }

    def train(self) -> None:
        loader_iter = iter(self.loader)
        step = 0
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.loader)
                batch = next(loader_iter)

            batch = self._move_batch(batch)
            outputs = self._forward_network_output(batch)
            aux = self._output_aux(outputs)
            slot_vectors = self._output_tensor(outputs, "slot_vectors", fallback="pred_vectors")
            slot_masks = self._output_tensor(outputs, "slot_masks", fallback="pred_masks")
            slot_confidence = self._output_tensor(outputs, "slot_confidence", fallback="confidence_logits")
            if slot_vectors is None or slot_masks is None or slot_confidence is None:
                raise RuntimeError("Prototype trainer requires slot_vectors, slot_masks, and slot_confidence from the active network.")
            losses = self.criterion(
                pred_vectors=slot_vectors,
                pred_masks=slot_masks,
                target_vectors=batch["target_vectors"],
                target_masks=batch["target_masks"],
                target_valid=batch["target_valid"],
                present_mask=batch["present_mask"],
                confidence_logits=slot_confidence,
                slot_selection_logits=aux.get("slot_selection_logits"),
            )

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            self.optimizer.step()
            self.model.observe_predictions(
                aux.get("slot_selection_probs"),
                slot_confidence,
            )
            if hasattr(self.model, "observe_terms") and callable(getattr(self.model, "observe_terms")):
                self.model.observe_terms(batch["item_vocab"], step=step + 1)
            diagnostics = (
                self.model.collect_preview_diagnostics(
                    outputs,
                    batch["batch_vocab"],
                    terms_rows=batch["item_vocab"],
                )
                if hasattr(self.model, "collect_preview_diagnostics") and callable(getattr(self.model, "collect_preview_diagnostics"))
                else []
            )
            loss_rows = self._format_loss_rows(losses)
            hypergraph_len = (
                len(getattr(self.model, "hypergraph"))
                if hasattr(self.model, "hypergraph")
                else 0
            )

            if step % 20 == 0:
                diag_rows = list(diagnostics[0].get("decision_rows") or []) if diagnostics else []
                diag_suffix = f" {' | '.join(diag_rows[:2])}" if diag_rows else ""
                print(
                    f"step={step:05d} "
                    f"loss={float(losses['loss']):.5f} "
                    f"{' | '.join(loss_rows)} "
                    f"hyperedges={int(hypergraph_len)}"
                    f"{diag_suffix}"
                )

            if self.viewer is not None and step % 2 == 0:
                if not self.viewer.pump():
                    break
                cpu_batch = {
                    "image": batch["image"].detach().cpu(),
                    "batch_vocab": batch["batch_vocab"],
                    "item_vocab": batch["item_vocab"],
                }
                cpu_outputs = {
                    "pred_masks": slot_masks.detach().cpu(),
                    "pred_vectors": slot_vectors.detach().cpu(),
                    "confidence_logits": slot_confidence.detach().cpu(),
                }
                cpu_losses = {
                    "loss": losses["loss"].detach().cpu(),
                    "vector_loss": losses["vector_loss"].detach().cpu(),
                    "mask_loss": losses["mask_loss"].detach().cpu(),
                    "selection_loss": losses["selection_loss"].detach().cpu(),
                    "confidence_loss": losses.get("confidence_loss", torch.zeros(1)).detach().cpu(),
                    "mask_order_loss": losses.get("mask_order_loss", torch.zeros(1)).detach().cpu(),
                    "residual_mask_loss": losses.get("residual_mask_loss", torch.zeros(1)).detach().cpu(),
                    "assignments": losses["assignments"],
                    "loss_rows": list(loss_rows),
                    "diagnostics": diagnostics,
                }
                self.viewer.draw(cpu_batch, cpu_outputs, cpu_losses, getattr(self.model, "hypergraph", ObservedHypergraph()))

            step += 1

        if self.viewer is not None:
            self.viewer.close()


# ============================================================
# Entry point
# ============================================================


def main() -> None:
    config = TrainConfig()
    trainer = PrototypeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
