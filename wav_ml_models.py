import copy
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wav_ml_core import RenderConfig, normalize_bit_window, render_mono_wave_to_tensor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(
    device: torch.device,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
    matmul_precision: str = "high",
):
    if device.type != "cuda":
        return
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    if str(matmul_precision).strip():
        try:
            torch.set_float32_matmul_precision(str(matmul_precision))
        except Exception:
            pass


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    key = str(amp_dtype).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}. Expected float16/fp16 or bfloat16/bf16.")


def _should_use_amp(device: torch.device, amp: bool) -> bool:
    return bool(amp and device.type == "cuda")


def _autocast_context(device: torch.device, use_amp: bool, amp_dtype_t: torch.dtype):
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=amp_dtype_t)
    return nullcontext()


def _maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def maybe_compile_module(module: nn.Module, enabled: bool = False, mode: str = "default") -> nn.Module:
    if not enabled:
        return module
    if not hasattr(torch, "compile"):
        return module
    try:
        return torch.compile(module, mode=str(mode))
    except Exception:
        return module


def _to_device_wave_batch(x_np: np.ndarray, device: torch.device, pin_memory: bool) -> torch.Tensor:
    x = torch.from_numpy(x_np)
    if pin_memory and device.type == "cuda":
        x = x.pin_memory()
    return x.to(device, non_blocking=True)


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _resolve_bit_loss_windows(sample_bits: int, cfg: RenderConfig):
    sb = max(2, int(sample_bits))
    cfg_low = int(getattr(cfg, "bitmask_low", 0))
    cfg_high = int(getattr(cfg, "bitmask_high", sb // 2))
    cfg_low, cfg_high, _ = normalize_bit_window(cfg_low, cfg_high, sb)
    low_hi = cfg_low
    high_lo = cfg_high

    if high_lo <= low_hi:
        split = max(1, min(sb - 1, high_lo))
        low_hi = max(0, split - 1)
        high_lo = split

    low_window = (0, int(low_hi))
    high_window = (int(high_lo), int(sb - 1))
    return high_window, low_window


def _wave_bit_window_ste(wave_mono: torch.Tensor, sample_bits: int, bit_low: int, bit_high: int):
    lo, hi, depth = normalize_bit_window(bit_low, bit_high, sample_bits)
    q_min = -(1 << (int(sample_bits) - 1))
    q_max = (1 << (int(sample_bits) - 1)) - 1
    q_scale = float(1 << (int(sample_bits) - 1))
    q_signed = torch.round(torch.clamp(wave_mono, -1.0, 1.0) * q_scale)
    q_signed = torch.clamp(q_signed, float(q_min), float(q_max)).to(torch.int64)

    full_mask = (1 << int(sample_bits)) - 1
    window_mask = (1 << int(depth)) - 1
    unsigned = torch.bitwise_and(q_signed, full_mask)
    window = torch.bitwise_and(torch.bitwise_right_shift(unsigned, int(lo)), window_mask)

    if depth == 1:
        hard = window.to(dtype=wave_mono.dtype)
    else:
        hard = window.to(dtype=wave_mono.dtype) / float(window_mask)
    hard = torch.clamp(hard, 0.0, 1.0)
    x01 = torch.clamp((wave_mono * 0.5) + 0.5, 0.0, 1.0)
    return hard + (x01 - x01.detach()), lo, hi, depth


def _binary_entropy_01(x: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return -(p * torch.log2(p)) - ((1.0 - p) * torch.log2(1.0 - p))


def _tensor_to_rgb_u8_image(x: torch.Tensor) -> np.ndarray:
    t = x.detach().to(device="cpu", dtype=torch.float32)
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected tensor image with ndim 2/3/4, got shape={tuple(t.shape)}")

    c = int(t.shape[0])
    if c <= 0:
        raise ValueError("Cannot visualize empty-channel tensor.")
    if c == 1:
        t = t.repeat(3, 1, 1)
    elif c == 2:
        t = torch.cat([t, t[:1]], dim=0)
    elif c > 3:
        t = t[:3]

    lo = float(t.min().item())
    hi = float(t.max().item())
    if hi <= lo + 1e-6:
        t = torch.zeros_like(t)
    elif lo < 0.0 or hi > 1.0:
        t = (t - lo) / (hi - lo)
    t = torch.clamp(t, 0.0, 1.0)
    img = (t.permute(1, 2, 0).contiguous().numpy() * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


class _TransformerStatusOpenGLViewer:
    def __init__(self, enabled: bool, image_hw: Tuple[int, int], scale: int = 3):
        self.enabled = bool(enabled)
        self.scale = max(1, int(scale))
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        self.window_w = max(320, (self.image_w * self.scale * 2) + (16 * self.scale))
        self.window_h = max(200, (self.image_h * self.scale) + (8 * self.scale))

        self._ready = False
        self._failed = False
        self._pygame = None
        self._gl = None
        self._textures = None

    def _init(self):
        if (not self.enabled) or self._ready or self._failed:
            return
        try:
            import pygame
            from OpenGL import GL

            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.set_mode((self.window_w, self.window_h), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("Transformer Status Viewer")

            GL.glViewport(0, 0, self.window_w, self.window_h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glClearColor(0.06, 0.06, 0.08, 1.0)

            tex = GL.glGenTextures(2)
            if isinstance(tex, int):
                tex = [tex, int(GL.glGenTextures(1))]
            self._textures = [int(tex[0]), int(tex[1])]
            for tid in self._textures:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

            self._pygame = pygame
            self._gl = GL
            self._ready = True
            print(
                f"[transformer-viz] pygame+OpenGL ready ({self.window_w}x{self.window_h})",
                flush=True,
            )
        except Exception as e:
            self._failed = True
            self.enabled = False
            print(f"[transformer-viz] disabled: could not initialize pygame OpenGL viewer ({e})", flush=True)

    def _upload_texture(self, tex_id: int, img_rgb: np.ndarray):
        gl = self._gl
        h, w, _ = img_rgb.shape
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            int(w),
            int(h),
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            img_rgb,
        )

    def _draw_texture(self, tex_id: int, x0: float, x1: float):
        gl = self._gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(float(x0), -1.0)
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(float(x1), -1.0)
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(float(x1), 1.0)
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(float(x0), 1.0)
        gl.glEnd()

    def update(self, input_img: torch.Tensor, output_img: torch.Tensor, caption: str):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return

        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                self.enabled = False
                return

        in_rgb = _tensor_to_rgb_u8_image(input_img)
        out_rgb = _tensor_to_rgb_u8_image(output_img)
        self._upload_texture(self._textures[0], in_rgb)
        self._upload_texture(self._textures[1], out_rgb)

        gl = self._gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self._draw_texture(self._textures[0], -1.0, -0.02)
        self._draw_texture(self._textures[1], 0.02, 1.0)
        pygame.display.set_caption(f"Transformer Status Viewer | {caption}")
        pygame.display.flip()

    def close(self):
        if self._ready:
            try:
                if self._gl is not None and self._textures is not None:
                    self._gl.glDeleteTextures(self._textures)
            except Exception:
                pass
            try:
                if self._pygame is not None:
                    self._pygame.display.quit()
                    self._pygame.quit()
            except Exception:
                pass
        self._ready = False
        self._textures = None
        self._pygame = None
        self._gl = None


_STFT_WINDOW_CACHE: Dict[Tuple[str, torch.dtype, int], torch.Tensor] = {}


def _cached_hann_window(win: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype, int(win))
    window = _STFT_WINDOW_CACHE.get(key)
    if window is None or window.device != device or window.dtype != dtype:
        window = torch.hann_window(int(win), device=device, dtype=dtype)
        _STFT_WINDOW_CACHE[key] = window
    return window


@dataclass
class SinusoidalLROptions:
    cycles: float = 1.0
    frequency: float = 0.0  # cycles per optimizer step; if >0 overrides cycles
    tail_fraction: float = 0.15
    min_scale: float = 0.0


def sinusoidal_lr_multiplier(
    step: int,
    total_steps: int,
    cycles: float = 1.0,
    frequency: float = 0.0,
    tail_fraction: float = 0.15,
    min_scale: float = 0.0,
) -> float:
    total_steps = max(1, int(total_steps))
    step = max(0, int(step))
    idx = min(step, total_steps - 1)
    denom = max(1, total_steps - 1)
    t = float(idx) / float(denom)

    if float(frequency) > 0.0:
        cycles_total = float(frequency) * float(denom)
    else:
        cycles_total = max(0.0, float(cycles))

    phase = (-0.5 * math.pi) + ((2.0 * math.pi) * cycles_total * t)
    mult = 0.5 * (math.sin(phase) + 1.0)

    tf = max(0.0, min(1.0, float(tail_fraction)))
    if tf > 0.0:
        tail_start = 1.0 - tf
        if t > tail_start:
            u = (t - tail_start) / max(1e-8, tf)
            coast = 0.5 * (1.0 + math.cos(math.pi * max(0.0, min(1.0, u))))
            mult *= coast

    min_scale = max(0.0, min(1.0, float(min_scale)))
    mult = min_scale + ((1.0 - min_scale) * mult)
    return float(max(0.0, min(1.0, mult)))


class SinusoidalLRController:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        options: Optional[SinusoidalLROptions] = None,
        start_step: int = 0,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.options = options if options is not None else SinusoidalLROptions()
        self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
        self.step_num = max(0, int(start_step))
        self.last_multiplier = 1.0
        self._apply(self.step_num)

    def _multiplier(self, step: int) -> float:
        return sinusoidal_lr_multiplier(
            step=step,
            total_steps=self.total_steps,
            cycles=self.options.cycles,
            frequency=self.options.frequency,
            tail_fraction=self.options.tail_fraction,
            min_scale=self.options.min_scale,
        )

    def _apply(self, step: int):
        mult = self._multiplier(step)
        for base_lr, g in zip(self.base_lrs, self.optimizer.param_groups):
            g["lr"] = float(base_lr) * float(mult)
        self.last_multiplier = float(mult)

    def set_base_lrs(self, lr: float):
        v = float(lr)
        self.base_lrs = [v for _ in self.optimizer.param_groups]
        self._apply(self.step_num)

    def set_step(self, step: int):
        self.step_num = max(0, int(step))
        self._apply(self.step_num)

    def step(self):
        self.step_num += 1
        self._apply(self.step_num)

    def state_dict(self):
        return {
            "step_num": int(self.step_num),
            "total_steps": int(self.total_steps),
            "base_lrs": [float(x) for x in self.base_lrs],
            "options": {
                "cycles": float(self.options.cycles),
                "frequency": float(self.options.frequency),
                "tail_fraction": float(self.options.tail_fraction),
                "min_scale": float(self.options.min_scale),
            },
        }

    def load_state_dict(self, state: Dict):
        if not isinstance(state, dict):
            return
        if isinstance(state.get("base_lrs"), list) and len(state["base_lrs"]) == len(self.base_lrs):
            self.base_lrs = [float(x) for x in state["base_lrs"]]
        self.step_num = max(0, int(state.get("step_num", self.step_num)))
        self._apply(self.step_num)


class TinyConvClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class WavePatchTransformer(nn.Module):
    def __init__(
        self,
        chunk_samples: int,
        patch_size: int = 16,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        ff_mult: int = 4,
        max_delta: float = 0.2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if chunk_samples % patch_size != 0:
            raise ValueError("chunk_samples must be divisible by patch_size")

        self.chunk_samples = int(chunk_samples)
        self.patch_size = int(patch_size)
        self.num_patches = self.chunk_samples // self.patch_size
        self.max_delta = float(max_delta)

        self.patch_in = nn.Linear(self.patch_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.patch_out = nn.Linear(d_model, self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected [B, T] waveform input")
        bsz, t = x.shape
        if t != self.chunk_samples:
            if t > self.chunk_samples:
                x = x[:, : self.chunk_samples]
            else:
                x = F.pad(x, (0, self.chunk_samples - t), value=0.0)

        patches = x.view(bsz, self.num_patches, self.patch_size)
        h = self.patch_in(patches) + self.pos
        h = self.encoder(h)
        delta = torch.tanh(self.patch_out(h)).view(bsz, self.chunk_samples) * self.max_delta
        return torch.clamp(x + delta, -1.0, 1.0)


def _iterate_indices(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i : i + batch_size]


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
) -> Dict[str, float]:
    model.eval()
    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    n = int(x.shape[0])
    total_loss = 0.0
    total_correct = 0
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size].to(device, non_blocking=True)
        xb = _maybe_channels_last(xb, enabled=channels_last)
        yb = y[i : i + batch_size].to(device, non_blocking=True)
        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * int(xb.shape[0])
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
    denom = max(1, n)
    return {"loss": total_loss / denom, "acc": total_correct / denom}


def train_classifier(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    lr_sine_cycles: float = 1.0,
    lr_sine_frequency: float = 0.0,
    lr_sine_tail_fraction: float = 0.15,
    lr_sine_min_scale: float = 0.0,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    cache_dataset_on_device: bool = False,
    seed: int = 0,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    use_scaler = bool(use_amp and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))

    x_train_buf, y_train_buf = x_train, y_train
    x_val_buf, y_val_buf = x_val, y_val
    if cache_dataset_on_device and device.type == "cuda":
        try:
            x_train_buf = x_train.to(device, non_blocking=True)
            y_train_buf = y_train.to(device, non_blocking=True)
            x_val_buf = x_val.to(device, non_blocking=True)
            y_val_buf = y_val.to(device, non_blocking=True)
        except RuntimeError:
            x_train_buf, y_train_buf, x_val_buf, y_val_buf = x_train, y_train, x_val, y_val
            torch.cuda.empty_cache()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_batches = int(math.ceil(float(max(1, int(x_train.shape[0]))) / float(max(1, int(batch_size)))))
    steps_per_epoch = int(math.ceil(float(max(1, num_batches)) / float(grad_accum_steps)))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * steps_per_epoch),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )
    history: List[Dict[str, float]] = []
    best_state = None
    best_val = -math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        for step_idx, idx in enumerate(
            _iterate_indices(int(x_train.shape[0]), batch_size=batch_size, shuffle=True, rng=rng),
            start=1,
        ):
            if x_train_buf.device.type == device.type:
                idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
                xb = x_train_buf.index_select(0, idx_t)
                yb = y_train_buf.index_select(0, idx_t)
            else:
                xb = x_train_buf[idx].to(device, non_blocking=True)
                yb = y_train_buf[idx].to(device, non_blocking=True)
            xb = _maybe_channels_last(xb, enabled=channels_last)
            with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = (step_idx % grad_accum_steps == 0) or (step_idx == num_batches)
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_ctl.step()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * int(xb.shape[0])
            n_seen += int(xb.shape[0])

        train_loss = total_loss / max(1, n_seen)
        val_stats = evaluate_classifier(
            model,
            x_val_buf,
            y_val_buf,
            batch_size=batch_size,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        if val_stats["acc"] > best_val:
            best_val = float(val_stats["acc"])
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def stft_mag_l1(x_hat: torch.Tensor, x_ref: torch.Tensor, n_fft: int = 512, hop: int = 128, win: int = 512):
    window = _cached_hann_window(win=int(win), device=x_hat.device, dtype=x_hat.dtype)
    spec_hat = torch.stft(x_hat, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
    spec_ref = torch.stft(x_ref, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
    mag_hat = torch.log1p(torch.abs(spec_hat))
    mag_ref = torch.log1p(torch.abs(spec_ref))
    return F.l1_loss(mag_hat, mag_ref)


def _sample_wave_batch(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    batch_size: int,
    chunk_samples: int,
    rng: np.random.Generator,
):
    xb = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    yb = np.zeros((batch_size,), dtype=np.int64)
    for i in range(batch_size):
        idx = int(rng.integers(0, len(streams)))
        yb[i] = int(labels[idx])
        s = streams[idx]
        if s.size == 0:
            continue
        if s.size >= chunk_samples:
            start = int(rng.integers(0, s.size - chunk_samples + 1))
            xb[i, :] = s[start : start + chunk_samples]
        else:
            xb[i, : s.size] = s
    return xb, yb


class PackedWaveBatchSampler:
    def __init__(
        self,
        streams: Sequence[np.ndarray],
        labels: Optional[Sequence[int]],
        chunk_samples: int,
        seed: int,
        device: torch.device,
        cache_on_device: bool,
        pin_memory: bool,
    ):
        if len(streams) == 0:
            raise ValueError("PackedWaveBatchSampler requires at least one stream.")
        self.chunk_samples = max(1, int(chunk_samples))
        self.device = device if (cache_on_device and device.type == "cuda") else torch.device("cpu")
        self.num_streams = int(len(streams))

        lengths_np = np.asarray([max(0, int(s.size)) for s in streams], dtype=np.int64)
        max_len = int(max(1, lengths_np.max(initial=0)))
        data = torch.zeros((self.num_streams, max_len), dtype=torch.float32)
        for i, s in enumerate(streams):
            if s.size > 0:
                v = torch.from_numpy(np.asarray(s, dtype=np.float32, order="C"))
                data[i, : int(v.shape[0])] = v

        lengths = torch.from_numpy(lengths_np)
        labels_t = None
        if labels is not None:
            labels_t = torch.as_tensor(labels, dtype=torch.long)

        if self.device.type == "cuda":
            data = data.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            if labels_t is not None:
                labels_t = labels_t.to(self.device, non_blocking=True)
        elif pin_memory and device.type == "cuda":
            data = data.pin_memory()
            lengths = lengths.pin_memory()
            if labels_t is not None:
                labels_t = labels_t.pin_memory()

        self.data = data
        self.lengths = lengths
        self.labels = labels_t
        self.base_device = device
        self._chunk_idx = torch.arange(self.chunk_samples, dtype=torch.long, device=self.device)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))

    def sample(self, batch_size: int):
        b = max(1, int(batch_size))
        idx = torch.randint(
            low=0,
            high=self.num_streams,
            size=(b,),
            generator=self.rng,
            device=self.device,
        )
        lengths = self.lengths.index_select(0, idx)
        picked = self.data.index_select(0, idx)

        max_start = torch.clamp(lengths - self.chunk_samples, min=0)
        u = torch.rand((b,), generator=self.rng, device=self.device)
        start = torch.floor(u * (max_start.to(torch.float32) + 1.0)).to(torch.long)
        pos = start.unsqueeze(1) + self._chunk_idx.unsqueeze(0)
        valid_len = torch.clamp(lengths, min=1)
        pos = torch.minimum(pos, valid_len.unsqueeze(1) - 1)
        xb = torch.gather(picked, dim=1, index=pos)
        mask = self._chunk_idx.unsqueeze(0) < lengths.unsqueeze(1)
        xb = xb * mask.to(dtype=xb.dtype)

        yb = None
        if self.labels is not None:
            yb = self.labels.index_select(0, idx)
        return xb, yb


def _build_packed_sampler(
    streams: Sequence[np.ndarray],
    labels: Optional[Sequence[int]],
    chunk_samples: int,
    seed: int,
    device: torch.device,
    cache_on_device: bool,
    pin_memory: bool,
) -> Optional[PackedWaveBatchSampler]:
    try:
        return PackedWaveBatchSampler(
            streams=streams,
            labels=labels,
            chunk_samples=chunk_samples,
            seed=seed,
            device=device,
            cache_on_device=cache_on_device,
            pin_memory=pin_memory,
        )
    except RuntimeError:
        if cache_on_device and device.type == "cuda":
            torch.cuda.empty_cache()
            try:
                return PackedWaveBatchSampler(
                    streams=streams,
                    labels=labels,
                    chunk_samples=chunk_samples,
                    seed=seed,
                    device=device,
                    cache_on_device=False,
                    pin_memory=pin_memory,
                )
            except Exception:
                return None
        return None
    except Exception:
        return None


def _sample_wave_batch_unlabeled(
    streams: Sequence[np.ndarray],
    batch_size: int,
    chunk_samples: int,
    rng: np.random.Generator,
):
    xb = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    for i in range(batch_size):
        idx = int(rng.integers(0, len(streams)))
        s = streams[idx]
        if s.size == 0:
            continue
        if s.size >= chunk_samples:
            start = int(rng.integers(0, s.size - chunk_samples + 1))
            xb[i, :] = s[start : start + chunk_samples]
        else:
            xb[i, : s.size] = s
    return xb


def _degrade_wave_batch(
    x_clean: torch.Tensor,
    strength: float,
    noise_std_min: float,
    noise_std_max: float,
    dropout_max: float,
    quant_bits_min: int,
    quant_bits_max: int,
):
    s = float(max(0.0, min(1.0, strength)))
    x = x_clean
    bsz, t = x.shape

    # Broadband additive noise.
    noise_std = float(noise_std_min) + (float(noise_std_max) - float(noise_std_min)) * s
    if noise_std > 0.0:
        x = x + (torch.randn_like(x) * float(noise_std))

    # Random gain jitter and partial dropout emulate capture/channel variation.
    gain_sigma = 0.03 + (0.20 * s)
    gain = torch.exp(torch.randn((bsz, 1), device=x.device, dtype=x.dtype) * gain_sigma)
    x = x * gain
    drop_p = max(0.0, min(0.95, float(dropout_max) * s))
    if drop_p > 0.0:
        keep = (torch.rand((bsz, t), device=x.device) > drop_p).to(x.dtype)
        x = x * keep

    # Short contiguous occlusion band.
    seg_len = int(max(1, round((0.005 + (0.045 * s)) * float(t))))
    if seg_len < t:
        starts = torch.randint(0, max(1, t - seg_len + 1), (bsz,), device=x.device)
        idx = torch.arange(t, device=x.device).unsqueeze(0)
        seg_keep = ((idx < starts.unsqueeze(1)) | (idx >= (starts + seg_len).unsqueeze(1))).to(x.dtype)
        x = x * seg_keep

    # Mild low-pass blur.
    blur_k = int(1 + (2 * int(round(1 + (4.0 * s)))))
    blur_k = max(3, min(15, blur_k | 1))
    x_blur = F.avg_pool1d(x.unsqueeze(1), kernel_size=blur_k, stride=1, padding=(blur_k // 2)).squeeze(1)
    blur_mix = 0.10 + (0.70 * s)
    x = ((1.0 - blur_mix) * x) + (blur_mix * x_blur)

    # Bit-depth reduction.
    qb_lo = max(2, int(quant_bits_min))
    qb_hi = max(qb_lo, int(quant_bits_max))
    q_bits_f = float(qb_hi) + (float(qb_lo - qb_hi) * s)
    q_bits = max(2, int(round(q_bits_f)))
    levels = float((1 << q_bits) - 1)
    x01 = torch.clamp((x * 0.5) + 0.5, 0.0, 1.0)
    xq = torch.round(x01 * levels) / levels
    xq = (xq * 2.0) - 1.0
    q_mix = 0.10 + (0.75 * s)
    x = ((1.0 - q_mix) * x) + (q_mix * xq)

    return torch.clamp(x, -1.0, 1.0)


def _build_eval_cache_labeled(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    chunk_samples: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    pin_memory: bool,
    stream_cache_on_device: bool,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    n_batches = max(0, int(max_batches))
    if n_batches <= 0:
        return out

    rng = np.random.default_rng(seed)
    sampler = _build_packed_sampler(
        streams=streams,
        labels=labels,
        chunk_samples=chunk_samples,
        seed=seed,
        device=device,
        cache_on_device=bool(stream_cache_on_device),
        pin_memory=pin_memory,
    )
    for _ in range(n_batches):
        if sampler is not None:
            xb, yb = sampler.sample(batch_size=batch_size)
            if yb is None:
                continue
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
        else:
            xb_np, yb_np = _sample_wave_batch(
                streams=streams,
                labels=labels,
                batch_size=batch_size,
                chunk_samples=chunk_samples,
                rng=rng,
            )
            xb = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True)
        out.append((xb.detach(), yb.detach()))
    return out


def _build_eval_cache_unlabeled(
    streams: Sequence[np.ndarray],
    chunk_samples: int,
    batch_size: int,
    max_batches: int,
    device: torch.device,
    pin_memory: bool,
    stream_cache_on_device: bool,
    seed: int,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    n_batches = max(0, int(max_batches))
    if n_batches <= 0:
        return out

    rng = np.random.default_rng(seed)
    sampler = _build_packed_sampler(
        streams=streams,
        labels=None,
        chunk_samples=chunk_samples,
        seed=seed,
        device=device,
        cache_on_device=bool(stream_cache_on_device),
        pin_memory=pin_memory,
    )
    for _ in range(n_batches):
        if sampler is not None:
            xb, _ = sampler.sample(batch_size=batch_size)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
        else:
            xb_np = _sample_wave_batch_unlabeled(
                streams=streams,
                batch_size=batch_size,
                chunk_samples=chunk_samples,
                rng=rng,
            )
            xb = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)
        out.append(xb.detach())
    return out


def multilabel_feature_score(
    logits: torch.Tensor,
    topk: int = 3,
    threshold: float = 0.35,
    coverage_temp: float = 0.20,
    w_topk: float = 0.60,
    w_cov: float = 0.30,
    w_mean: float = 0.10,
) -> Dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    k = max(1, min(int(topk), int(probs.shape[1])))
    topk_mean = torch.topk(probs, k=k, dim=1).values.mean()
    mean_prob = probs.mean()
    class_mean = probs.mean(dim=0)
    soft_cov = torch.sigmoid((class_mean - float(threshold)) / max(1e-4, float(coverage_temp))).mean()
    hard_cov = (class_mean > float(threshold)).float().mean()
    score = (float(w_topk) * topk_mean) + (float(w_cov) * soft_cov) + (float(w_mean) * mean_prob)
    return {
        "score": score,
        "topk_mean": topk_mean,
        "soft_coverage": soft_cov,
        "hard_coverage": hard_cov,
        "mean_prob": mean_prob,
    }


@torch.inference_mode()
def evaluate_transformer_accuracy(
    transformer: nn.Module,
    classifier: nn.Module,
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    chunk_samples: int,
    device: torch.device,
    max_batches: int = 20,
    batch_size: int = 16,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    pin_memory: bool = False,
    stream_cache_on_device: bool = False,
    cached_batches: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
):
    transformer.eval()
    classifier.eval()
    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    corr_before = torch.zeros((), dtype=torch.float32, device=device)
    corr_after = torch.zeros((), dtype=torch.float32, device=device)
    total = 0

    if cached_batches is None:
        rng = np.random.default_rng(12345)
        sampler = _build_packed_sampler(
            streams=streams,
            labels=labels,
            chunk_samples=chunk_samples,
            seed=12345,
            device=device,
            cache_on_device=bool(stream_cache_on_device),
            pin_memory=pin_memory,
        )
        n_batches = max(0, int(max_batches))
    else:
        sampler = None
        rng = None
        n_batches = min(max(0, int(max_batches)), len(cached_batches))

    for i in range(n_batches):
        if cached_batches is not None:
            xb, yb = cached_batches[i]
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
        elif sampler is not None:
            xb, yb = sampler.sample(batch_size=batch_size)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb is None:
                continue
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
        else:
            xb_np, yb_np = _sample_wave_batch(
                streams,
                labels,
                batch_size=batch_size,
                chunk_samples=chunk_samples,
                rng=rng,
            )
            xb = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=True)

        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            img_before = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
            img_before = _maybe_channels_last(img_before, enabled=channels_last)
            logits_before = classifier(img_before)
        pred_before = logits_before.argmax(dim=1)
        corr_before = corr_before + (pred_before == yb).to(torch.float32).sum()

        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            xh = transformer(xb)
            img_after = render_mono_wave_to_tensor(xh, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
            img_after = _maybe_channels_last(img_after, enabled=channels_last)
            logits_after = classifier(img_after)
        pred_after = logits_after.argmax(dim=1)
        corr_after = corr_after + (pred_after == yb).to(torch.float32).sum()

        total += int(yb.shape[0])

    denom = max(1, total)
    return {
        "acc_before": float((corr_before / float(denom)).item()),
        "acc_after": float((corr_after / float(denom)).item()),
    }


@torch.inference_mode()
def evaluate_feature_score_before_after(
    transformer: nn.Module,
    classifier: nn.Module,
    streams: Sequence[np.ndarray],
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    chunk_samples: int,
    device: torch.device,
    max_batches: int = 20,
    batch_size: int = 16,
    score_topk: int = 3,
    score_threshold: float = 0.35,
    score_w_topk: float = 0.60,
    score_w_cov: float = 0.30,
    score_w_mean: float = 0.10,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    pin_memory: bool = False,
    stream_cache_on_device: bool = False,
    cached_batches: Optional[Sequence[torch.Tensor]] = None,
):
    transformer.eval()
    classifier.eval()
    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    if cached_batches is None:
        rng = np.random.default_rng(12345)
        sampler = _build_packed_sampler(
            streams=streams,
            labels=None,
            chunk_samples=chunk_samples,
            seed=12345,
            device=device,
            cache_on_device=bool(stream_cache_on_device),
            pin_memory=pin_memory,
        )
        n_target_batches = max(0, int(max_batches))
    else:
        rng = None
        sampler = None
        n_target_batches = min(max(0, int(max_batches)), len(cached_batches))

    sum_before = torch.zeros((), dtype=torch.float32, device=device)
    sum_after = torch.zeros((), dtype=torch.float32, device=device)
    sum_cov_before = torch.zeros((), dtype=torch.float32, device=device)
    sum_cov_after = torch.zeros((), dtype=torch.float32, device=device)
    n_batches = 0

    for i in range(n_target_batches):
        if cached_batches is not None:
            xb = cached_batches[i]
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
        elif sampler is not None:
            xb, _ = sampler.sample(batch_size=batch_size)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
        else:
            xb_np = _sample_wave_batch_unlabeled(
                streams=streams,
                batch_size=batch_size,
                chunk_samples=chunk_samples,
                rng=rng,
            )
            xb = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)

        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            img_before = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
            img_before = _maybe_channels_last(img_before, enabled=channels_last)
            logits_before = classifier(img_before)
        score_b = multilabel_feature_score(
            logits_before,
            topk=score_topk,
            threshold=score_threshold,
            w_topk=score_w_topk,
            w_cov=score_w_cov,
            w_mean=score_w_mean,
        )

        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            xh = transformer(xb)
            img_after = render_mono_wave_to_tensor(xh, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
            img_after = _maybe_channels_last(img_after, enabled=channels_last)
            logits_after = classifier(img_after)
        score_a = multilabel_feature_score(
            logits_after,
            topk=score_topk,
            threshold=score_threshold,
            w_topk=score_w_topk,
            w_cov=score_w_cov,
            w_mean=score_w_mean,
        )

        sum_before = sum_before + score_b["score"].to(torch.float32)
        sum_after = sum_after + score_a["score"].to(torch.float32)
        sum_cov_before = sum_cov_before + score_b["hard_coverage"].to(torch.float32)
        sum_cov_after = sum_cov_after + score_a["hard_coverage"].to(torch.float32)
        n_batches += 1

    d = max(1, n_batches)
    return {
        "score_before": float((sum_before / float(d)).item()),
        "score_after": float((sum_after / float(d)).item()),
        "score_gain": float(((sum_after - sum_before) / float(d)).item()),
        "hard_coverage_before": float((sum_cov_before / float(d)).item()),
        "hard_coverage_after": float((sum_cov_after / float(d)).item()),
    }


def train_transformer_feature_metric(
    transformer: nn.Module,
    classifier: nn.Module,
    train_streams: Sequence[np.ndarray],
    val_streams: Sequence[np.ndarray],
    train_target_labels: Optional[Sequence[int]],
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    epochs: int = 6,
    steps_per_epoch: int = 80,
    batch_size: int = 16,
    chunk_samples: int = 32768,
    lr: float = 2e-4,
    lr_sine_cycles: float = 1.0,
    lr_sine_frequency: float = 0.0,
    lr_sine_tail_fraction: float = 0.15,
    lr_sine_min_scale: float = 0.0,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    pin_memory: bool = False,
    stream_cache_on_device: bool = False,
    score_topk: int = 3,
    score_threshold: float = 0.35,
    score_w_topk: float = 0.60,
    score_w_cov: float = 0.30,
    score_w_mean: float = 0.10,
    degrade_inputs: bool = True,
    degrade_min_strength: float = 0.25,
    degrade_max_strength: float = 0.95,
    degrade_noise_std_min: float = 0.01,
    degrade_noise_std_max: float = 0.14,
    degrade_dropout_max: float = 0.25,
    degrade_quant_bits_min: int = 3,
    degrade_quant_bits_max: int = 10,
    entropy_penalty_weight: float = 0.60,
    high_bit_penalty_weight: float = 0.50,
    low_bit_penalty_weight: float = 0.08,
    wave_l1_weight: float = 1.00,
    score_target_weight: float = 1.50,
    score_target_margin: float = 0.0,
    seed: int = 0,
    log_every_steps: int = 0,
    cache_eval_batches: bool = True,
    visualize_status: bool = False,
    visualize_scale: int = 3,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    transformer = transformer.to(device)
    classifier = classifier.to(device)
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    classifier_was_training = bool(classifier.training)
    classifier_requires_grad = [bool(p.requires_grad) for p in classifier.parameters()]
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    use_scaler = bool(use_amp and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))

    opt = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=1e-4)
    num_steps = max(1, int(steps_per_epoch))
    num_updates = int(math.ceil(float(num_steps) / float(grad_accum_steps)))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * max(1, num_updates)),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )
    history = []
    best_state = None
    best_gain = -math.inf
    log_every_steps = max(0, int(log_every_steps))
    status_viewer = _TransformerStatusOpenGLViewer(
        enabled=bool(visualize_status),
        image_hw=image_hw,
        scale=max(1, int(visualize_scale)),
    )
    entropy_penalty_weight = max(0.0, float(entropy_penalty_weight))
    high_bit_penalty_weight = max(0.0, float(high_bit_penalty_weight))
    low_bit_penalty_weight = max(0.0, float(low_bit_penalty_weight))
    wave_l1_weight = max(0.0, float(wave_l1_weight))
    score_target_weight = max(0.0, float(score_target_weight))
    score_target_margin = max(0.0, float(score_target_margin))
    target_labels = None
    target_supervision = False
    if train_target_labels is not None:
        if len(train_target_labels) == len(train_streams):
            target_labels = [int(x) for x in train_target_labels]
            target_supervision = True
        else:
            print(
                "[transformer-feature-config] target supervision disabled: "
                f"label_count={len(train_target_labels)} stream_count={len(train_streams)}",
                flush=True,
            )
    high_window, low_window = _resolve_bit_loss_windows(sample_bits=int(sample_bits), cfg=cfg)
    eval_max_batches = max(8, min(24, steps_per_epoch // 3))
    train_sampler = _build_packed_sampler(
        streams=train_streams,
        labels=target_labels,
        chunk_samples=chunk_samples,
        seed=seed + 29,
        device=device,
        cache_on_device=bool(stream_cache_on_device),
        pin_memory=pin_memory,
    )
    eval_cache = None
    if cache_eval_batches:
        eval_cache = _build_eval_cache_unlabeled(
            streams=val_streams,
            chunk_samples=chunk_samples,
            batch_size=batch_size,
            max_batches=eval_max_batches,
            device=device,
            pin_memory=pin_memory,
            stream_cache_on_device=stream_cache_on_device,
            seed=seed + 103,
        )
    print(
        "[transformer-feature-config] "
        "objective=signal_to_signal_category_score_gain_plus_entropy_hi_lo_bits_plus_wave_l1 "
        f"sample_bits={int(sample_bits)} cfg_bitmask={1 if bool(cfg.bitmask_enable) else 0}"
        f"[{int(cfg.bitmask_low)}..{int(cfg.bitmask_high)}] "
        f"loss_windows=hi[{high_window[0]}..{high_window[1]}] lo[{low_window[0]}..{low_window[1]}] "
        f"loss_w_score_target={score_target_weight:.3f} "
        f"loss_margin_score_target={score_target_margin:.3f} "
        f"loss_w_entropy={entropy_penalty_weight:.3f} loss_w_hi={high_bit_penalty_weight:.3f} "
        f"loss_w_wave={wave_l1_weight:.3f} loss_w_lo={low_bit_penalty_weight:.3f} "
        f"target_supervision={1 if target_supervision else 0} "
        f"target_streams={int(sum(1 for x in (target_labels or []) if int(x) > 0))} "
        f"score_topk={int(score_topk)} score_threshold={float(score_threshold):.4f} "
        f"score_w_topk={float(score_w_topk):.3f} score_w_cov={float(score_w_cov):.3f} score_w_mean={float(score_w_mean):.3f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        transformer.train()
        running_loss = torch.zeros((), dtype=torch.float32, device=device)
        running_denoise = torch.zeros((), dtype=torch.float32, device=device)
        running_hi_bits = torch.zeros((), dtype=torch.float32, device=device)
        running_lo_bits = torch.zeros((), dtype=torch.float32, device=device)
        running_entropy = torch.zeros((), dtype=torch.float32, device=device)
        running_score_target = torch.zeros((), dtype=torch.float32, device=device)
        running_score_after = torch.zeros((), dtype=torch.float32, device=device)
        running_score_gap = torch.zeros((), dtype=torch.float32, device=device)
        running_degrade = 0.0
        last_xb_preview = None
        last_xh_preview = None
        t_sample = 0.0
        t_forward = 0.0
        t_backward = 0.0
        epoch_t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        total_train_steps = max(1, int(epochs) * int(num_steps))

        for step_idx in range(num_steps):
            step_t0 = time.perf_counter()
            if train_sampler is not None:
                x_clean, y_target = train_sampler.sample(batch_size=batch_size)
                if x_clean.device != device:
                    x_clean = x_clean.to(device, non_blocking=True)
                if y_target is not None and y_target.device != device:
                    y_target = y_target.to(device, non_blocking=True)
            else:
                if target_labels is not None:
                    xb_np, yb_np = _sample_wave_batch(
                        streams=train_streams,
                        labels=target_labels,
                        batch_size=batch_size,
                        chunk_samples=chunk_samples,
                        rng=rng,
                    )
                    y_target = torch.from_numpy(yb_np).to(device, non_blocking=True)
                else:
                    xb_np = _sample_wave_batch_unlabeled(
                        streams=train_streams,
                        batch_size=batch_size,
                        chunk_samples=chunk_samples,
                        rng=rng,
                    )
                    y_target = None
                x_clean = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)
            global_step = ((epoch - 1) * num_steps) + step_idx
            progress = float(global_step) / float(max(1, total_train_steps - 1))
            degrade_strength = float(degrade_min_strength) + (
                float(degrade_max_strength) - float(degrade_min_strength)
            ) * progress
            if bool(degrade_inputs):
                xb = _degrade_wave_batch(
                    x_clean=x_clean,
                    strength=degrade_strength,
                    noise_std_min=float(degrade_noise_std_min),
                    noise_std_max=float(degrade_noise_std_max),
                    dropout_max=float(degrade_dropout_max),
                    quant_bits_min=int(degrade_quant_bits_min),
                    quant_bits_max=int(degrade_quant_bits_max),
                )
            else:
                xb = x_clean
            step_t1 = time.perf_counter()

            with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                xh = transformer(xb)
            xh_f = xh.float()
            x_clean_f = x_clean.float()
            denoise_l1 = F.l1_loss(xh_f, x_clean_f)

            xh_hi, _, _, _ = _wave_bit_window_ste(
                xh_f, sample_bits=int(sample_bits), bit_low=high_window[0], bit_high=high_window[1]
            )
            x_clean_hi, _, _, _ = _wave_bit_window_ste(
                x_clean_f, sample_bits=int(sample_bits), bit_low=high_window[0], bit_high=high_window[1]
            )
            xh_lo, _, _, _ = _wave_bit_window_ste(
                xh_f, sample_bits=int(sample_bits), bit_low=low_window[0], bit_high=low_window[1]
            )
            x_clean_lo, _, _, _ = _wave_bit_window_ste(
                x_clean_f, sample_bits=int(sample_bits), bit_low=low_window[0], bit_high=low_window[1]
            )
            hi_bit_l1 = F.l1_loss(xh_hi, x_clean_hi)
            lo_bit_l1 = F.l1_loss(xh_lo, x_clean_lo)
            ent_xh = _binary_entropy_01(xh_hi).mean()
            ent_ref = _binary_entropy_01(x_clean_hi).mean()
            entropy_pen = F.relu(ent_xh - ent_ref)
            with torch.no_grad():
                with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                    img_before = render_mono_wave_to_tensor(
                        xb.float(), cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                    )
                    img_before = _maybe_channels_last(img_before, enabled=channels_last)
                    logits_before = classifier(img_before)
            with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                img_after = render_mono_wave_to_tensor(
                    xh_f, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                )
                img_after = _maybe_channels_last(img_after, enabled=channels_last)
                logits_after = classifier(img_after)
            if y_target is not None:
                valid_mask = y_target > 0
            else:
                valid_mask = None

            if valid_mask is not None and bool(torch.any(valid_mask)):
                logits_before_v = logits_before[valid_mask]
                logits_after_v = logits_after[valid_mask]
                mask_bits = y_target[valid_mask].to(torch.long).unsqueeze(1)
                n_classes = int(logits_after_v.shape[1])
                class_bits = (1 << torch.arange(n_classes, device=device, dtype=torch.long)).unsqueeze(0)
                target_mask = (torch.bitwise_and(mask_bits, class_bits) != 0).to(torch.float32)
                target_count = torch.clamp(target_mask.sum(), min=1.0)
                probs_before_all = torch.sigmoid(logits_before_v).to(torch.float32)
                probs_after_all = torch.sigmoid(logits_after_v).to(torch.float32)
                score_target = (probs_before_all * target_mask).sum() / target_count
                score_after = (probs_after_all * target_mask).sum() / target_count
                score_gap = (
                    F.relu((probs_before_all + score_target_margin) - probs_after_all) * target_mask
                ).sum() / target_count
            else:
                score_target = multilabel_feature_score(
                    logits_before,
                    topk=score_topk,
                    threshold=score_threshold,
                    w_topk=score_w_topk,
                    w_cov=score_w_cov,
                    w_mean=score_w_mean,
                )["score"].to(torch.float32)
                score_after = multilabel_feature_score(
                    logits_after,
                    topk=score_topk,
                    threshold=score_threshold,
                    w_topk=score_w_topk,
                    w_cov=score_w_cov,
                    w_mean=score_w_mean,
                )["score"].to(torch.float32)
                score_gap = F.relu((score_target + score_target_margin) - score_after)

            loss = (
                (score_target_weight * score_gap)
                + (wave_l1_weight * denoise_l1)
                + (entropy_penalty_weight * entropy_pen)
                + (high_bit_penalty_weight * hi_bit_l1)
                + (low_bit_penalty_weight * lo_bit_l1)
            )
            if status_viewer.enabled and int(xb.shape[0]) > 0:
                preview_idx = int(rng.integers(0, int(xb.shape[0])))
                last_xb_preview = xb[preview_idx : preview_idx + 1].detach()
                last_xh_preview = xh[preview_idx : preview_idx + 1].detach()
            step_t2 = time.perf_counter()
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((step_idx + 1) % grad_accum_steps == 0) or ((step_idx + 1) == num_steps)
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_ctl.step()
                opt.zero_grad(set_to_none=True)
            step_t3 = time.perf_counter()

            running_loss = running_loss + loss.detach().to(torch.float32)
            running_denoise = running_denoise + denoise_l1.detach().to(torch.float32)
            running_hi_bits = running_hi_bits + hi_bit_l1.detach().to(torch.float32)
            running_lo_bits = running_lo_bits + lo_bit_l1.detach().to(torch.float32)
            running_entropy = running_entropy + entropy_pen.detach().to(torch.float32)
            running_score_target = running_score_target + score_target.detach().to(torch.float32)
            running_score_after = running_score_after + score_after.detach().to(torch.float32)
            running_score_gap = running_score_gap + score_gap.detach().to(torch.float32)
            running_degrade += float(degrade_strength)
            t_sample += step_t1 - step_t0
            t_forward += step_t2 - step_t1
            t_backward += step_t3 - step_t2

            if log_every_steps > 0 and (((step_idx + 1) % log_every_steps == 0) or ((step_idx + 1) == num_steps)):
                steps_done = step_idx + 1
                elapsed = max(1e-6, time.perf_counter() - epoch_t0)
                avg_loss = float((running_loss / float(steps_done)).item())
                avg_denoise = float((running_denoise / float(steps_done)).item())
                avg_hi_bits = float((running_hi_bits / float(steps_done)).item())
                avg_lo_bits = float((running_lo_bits / float(steps_done)).item())
                avg_ent = float((running_entropy / float(steps_done)).item())
                avg_score_tgt = float((running_score_target / float(steps_done)).item())
                avg_score_after = float((running_score_after / float(steps_done)).item())
                avg_score_gap = float((running_score_gap / float(steps_done)).item())
                avg_deg = float(running_degrade / float(steps_done))
                if status_viewer.enabled and (last_xb_preview is not None) and (last_xh_preview is not None):
                    with torch.no_grad():
                        img_in = render_mono_wave_to_tensor(
                            last_xb_preview, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                        )
                        img_out = render_mono_wave_to_tensor(
                            last_xh_preview, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                        )
                    status_viewer.update(
                        input_img=img_in[0],
                        output_img=img_out[0],
                        caption=(
                            f"epoch {epoch}/{epochs} step {steps_done}/{num_steps} "
                            f"loss {avg_loss:.4f} hi {avg_hi_bits:.4f} lo {avg_lo_bits:.4f}"
                        ),
                    )
                print(
                    f"[transformer-train] epoch={epoch}/{epochs} step={steps_done}/{num_steps} "
                    f"loss={avg_loss:.4f} denoise_l1={avg_denoise:.4f} "
                    f"hi_bits={avg_hi_bits:.4f} lo_bits={avg_lo_bits:.4f} entropy_excess={avg_ent:.4f} "
                    f"score_target={avg_score_tgt:.4f} score_after={avg_score_after:.4f} score_gap={avg_score_gap:.4f} "
                    f"degrade={avg_deg:.3f} "
                    f"samp_per_sec={(steps_done * int(batch_size)) / elapsed:.1f} "
                    f"sample_ms={(1000.0 * t_sample) / steps_done:.2f} "
                    f"forward_ms={(1000.0 * t_forward) / steps_done:.2f} "
                    f"backward_ms={(1000.0 * t_backward) / steps_done:.2f}",
                    flush=True,
                )

        s = 1.0 / max(1, num_steps)
        row_train = {
            "loss": float((running_loss * s).item()),
            "denoise_l1": float((running_denoise * s).item()),
            "high_bits_l1": float((running_hi_bits * s).item()),
            "low_bits_l1": float((running_lo_bits * s).item()),
            "entropy_excess": float((running_entropy * s).item()),
            "score_target": float((running_score_target * s).item()),
            "score_after": float((running_score_after * s).item()),
            "score_gap": float((running_score_gap * s).item()),
            "degrade_strength": float(running_degrade * s),
        }

        eval_t0 = time.perf_counter()
        eval_stats = evaluate_feature_score_before_after(
            transformer=transformer,
            classifier=classifier,
            streams=val_streams,
            cfg=cfg,
            sample_bits=sample_bits,
            image_hw=image_hw,
            chunk_samples=chunk_samples,
            device=device,
            max_batches=eval_max_batches,
            batch_size=batch_size,
            score_topk=score_topk,
            score_threshold=score_threshold,
            score_w_topk=score_w_topk,
            score_w_cov=score_w_cov,
            score_w_mean=score_w_mean,
            amp=amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            pin_memory=pin_memory,
            stream_cache_on_device=stream_cache_on_device,
            cached_batches=eval_cache,
        )
        eval_sec = time.perf_counter() - eval_t0
        gain = float(eval_stats["score_gain"])

        row = {
            "epoch": float(epoch),
            "train_loss": row_train["loss"],
            "train_denoise_l1": row_train["denoise_l1"],
            "train_high_bits_l1": row_train["high_bits_l1"],
            "train_low_bits_l1": row_train["low_bits_l1"],
            "train_entropy_excess": row_train["entropy_excess"],
            "train_score_target": row_train["score_target"],
            "train_score_after": row_train["score_after"],
            "train_score_gap": row_train["score_gap"],
            "val_score_before": float(eval_stats["score_before"]),
            "val_score_after": float(eval_stats["score_after"]),
            "val_score_gain": gain,
            "val_cov_before": float(eval_stats["hard_coverage_before"]),
            "val_cov_after": float(eval_stats["hard_coverage_after"]),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)
        epoch_sec = time.perf_counter() - epoch_t0
        print(
            f"[transformer-epoch] epoch={epoch}/{epochs} "
            f"train_loss={row_train['loss']:.4f} denoise_l1={row_train['denoise_l1']:.4f} "
            f"hi_bits={row_train['high_bits_l1']:.4f} lo_bits={row_train['low_bits_l1']:.4f} "
            f"entropy_excess={row_train['entropy_excess']:.4f} "
            f"score_target={row_train['score_target']:.4f} score_after={row_train['score_after']:.4f} "
            f"score_gap={row_train['score_gap']:.4f} "
            f"degrade={row_train['degrade_strength']:.3f} "
            f"val_gain={gain:.4f} "
            f"epoch_sec={epoch_sec:.2f} eval_sec={eval_sec:.2f} "
            f"sample_pct={(100.0 * t_sample / max(1e-6, epoch_sec)):.1f} "
            f"forward_pct={(100.0 * t_forward / max(1e-6, epoch_sec)):.1f} "
            f"backward_pct={(100.0 * t_backward / max(1e-6, epoch_sec)):.1f}",
            flush=True,
        )
        if status_viewer.enabled and (last_xb_preview is not None) and (last_xh_preview is not None):
            with torch.no_grad():
                img_in = render_mono_wave_to_tensor(
                    last_xb_preview, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                )
                img_out = render_mono_wave_to_tensor(
                    last_xh_preview, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits
                )
            status_viewer.update(
                input_img=img_in[0],
                output_img=img_out[0],
                caption=(
                    f"epoch {epoch}/{epochs} val_gain {gain:.4f} "
                    f"hi {row_train['high_bits_l1']:.4f} lo {row_train['low_bits_l1']:.4f}"
                ),
            )

        if gain > best_gain:
            best_gain = gain
            best_state = copy.deepcopy(transformer.state_dict())

    status_viewer.close()
    if best_state is not None:
        transformer.load_state_dict(best_state)
    for p, req in zip(classifier.parameters(), classifier_requires_grad):
        p.requires_grad_(bool(req))
    classifier.train(classifier_was_training)
    return transformer, history
