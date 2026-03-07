import copy
import math
import random
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wav_ml_core import COLOR_MODE_MAP, COLOR_MODES, RenderConfig, normalize_bit_window, render_mono_wave_to_tensor


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
    elif c > 4:
        t = t[:4]

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
    def __init__(
        self,
        enabled: bool,
        image_hw: Tuple[int, int],
        scale: int = 3,
        cycle_slots: int = 0,
    ):
        self.enabled = bool(enabled)
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        _ = max(1, int(scale))

        self.panel_w = max(8, min(256, int(self.image_w)))
        self.panel_h = max(8, min(256, int(self.image_h)))
        self.num_panels = 3
        self.top_bar_h = 56
        self.window_w = int(self.panel_w * self.num_panels)
        self.window_h = int(self.top_bar_h + (self.panel_h * 2))

        self._ready = False
        self._failed = False
        self._pygame = None
        self._gl = None
        self._textures = None
        self._stop_requested = False

        self._last_present_t = 0.0
        self._min_present_dt = 1.0 / 30.0
        self._pending_frame = None
        self._top_bar_dirty = True
        self._panel_text_dirty = True

        self._caption = ""
        self._panel_titles = ["target", "input", "output"]
        self._panel_rows = [[], [], []]
        self._panel_text_rgb = [
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
            np.full((self.panel_h, self.panel_w, 3), 14, dtype=np.uint8),
        ]
        self._top_bar_rgb = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)

        self._cycle_selected: List[bool] = []
        self._gate_override = False
        self._control_boxes: List[Tuple[str, int, Tuple[int, int, int, int]]] = []
        self.set_cycle_roster(total_cycles=int(cycle_slots))

    def set_cycle_roster(self, total_cycles: int, selected: Optional[Sequence[bool]] = None):
        n = max(0, int(total_cycles))
        prev = list(self._cycle_selected)
        if n <= 0:
            self._cycle_selected = []
        else:
            out = [True] * n
            for i in range(min(len(prev), n)):
                out[i] = bool(prev[i])
            if selected is not None:
                for i, v in enumerate(list(selected)[:n]):
                    out[i] = bool(v)
            self._cycle_selected = out
        self._top_bar_dirty = True

    def selected_cycle_ids(self) -> List[int]:
        return [int(i + 1) for i, v in enumerate(self._cycle_selected) if bool(v)]

    def is_cycle_selected(self, cycle_local: int) -> bool:
        idx = int(cycle_local) - 1
        if idx < 0 or idx >= len(self._cycle_selected):
            return True
        return bool(self._cycle_selected[idx])

    def gate_override_enabled(self) -> bool:
        return bool(self._gate_override)

    def _init(self):
        if (not self.enabled) or self._ready or self._failed:
            return
        try:
            import pygame
            from OpenGL import GL

            pygame.display.init()
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.set_mode((self.window_w, self.window_h), pygame.OPENGL | pygame.DOUBLEBUF)
            pygame.display.set_caption("Stage Status Viewer")

            GL.glViewport(0, 0, self.window_w, self.window_h)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glClearColor(0.06, 0.06, 0.08, 1.0)

            tex = GL.glGenTextures(7)
            if isinstance(tex, int):
                tex = [int(tex)]
                while len(tex) < 7:
                    tex.append(int(GL.glGenTextures(1)))
            else:
                tex = [int(t) for t in list(tex)]
                while len(tex) < 7:
                    tex.append(int(GL.glGenTextures(1)))
            self._textures = {
                "img": [int(tex[0]), int(tex[1]), int(tex[2])],
                "text": [int(tex[3]), int(tex[4]), int(tex[5])],
                "bar": int(tex[6]),
            }
            for tid in (
                self._textures["img"]
                + self._textures["text"]
                + [int(self._textures["bar"])]
            ):
                GL.glBindTexture(GL.GL_TEXTURE_2D, int(tid))
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

            self._pygame = pygame
            self._gl = GL
            self._ready = True
            self._top_bar_dirty = True
            self._panel_text_dirty = True
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
        channels = int(img_rgb.shape[2]) if int(img_rgb.ndim) == 3 else 0
        if int(channels) == 4:
            internal_format = gl.GL_RGBA
            src_format = gl.GL_RGBA
        else:
            internal_format = gl.GL_RGB
            src_format = gl.GL_RGB
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            internal_format,
            int(w),
            int(h),
            0,
            src_format,
            gl.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(img_rgb),
        )

    def _resize_rgb_to_panel(self, img_rgb: np.ndarray) -> np.ndarray:
        h = int(img_rgb.shape[0])
        w = int(img_rgb.shape[1])
        if (h == int(self.panel_h)) and (w == int(self.panel_w)):
            return np.ascontiguousarray(img_rgb)
        t = torch.from_numpy(img_rgb.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(int(self.panel_h), int(self.panel_w)), mode="nearest")
        out = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()
        return np.ascontiguousarray(out)

    def _normalize_panel_text(
        self,
        panel_titles: Optional[Sequence[str]],
        panel_rows: Optional[Sequence[Sequence[str]]],
    ) -> Tuple[List[str], List[List[str]]]:
        titles = []
        rows = []
        for i in range(3):
            if panel_titles is not None and i < len(panel_titles):
                titles.append(str(panel_titles[i]))
            else:
                titles.append(self._panel_titles[i] if i < len(self._panel_titles) else "")
            cur_rows: List[str] = []
            if panel_rows is not None and i < len(panel_rows):
                src_rows = panel_rows[i]
                if isinstance(src_rows, (list, tuple)):
                    for r in src_rows:
                        txt = str(r).strip()
                        if txt:
                            cur_rows.append(txt)
            rows.append(cur_rows)
        return titles, rows

    def _render_panel_text(self, title: str, rows: Sequence[str]) -> np.ndarray:
        out = np.full((self.panel_h, self.panel_w, 3), 16, dtype=np.uint8)
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out).convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            draw.rectangle([(0, 0), (self.panel_w - 1, self.panel_h - 1)], fill=(20, 24, 30))
            draw.rectangle([(0, 0), (self.panel_w - 1, 15)], fill=(34, 42, 52))
            draw.text((4, 2), str(title)[:48], fill=(255, 225, 70), font=font)
            draw.line([(0, 16), (self.panel_w - 1, 16)], fill=(70, 76, 88), width=1)
            y = 20
            for r in list(rows)[: max(1, (self.panel_h - 20) // 11)]:
                draw.text((4, y), str(r)[:72], fill=(230, 234, 240), font=font)
                y += 11
                if y >= (self.panel_h - 10):
                    break
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return out

    def _render_top_bar(self) -> np.ndarray:
        out = np.full((self.top_bar_h, self.window_w, 3), 18, dtype=np.uint8)
        self._control_boxes = []
        try:
            from PIL import Image, ImageDraw, ImageFont

            im = Image.fromarray(out).convert("RGB")
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()

            draw.rectangle([(0, 0), (self.window_w - 1, self.top_bar_h - 1)], fill=(18, 22, 28))
            draw.line(
                [(0, self.top_bar_h - 1), (self.window_w - 1, self.top_bar_h - 1)],
                fill=(70, 76, 88),
                width=1,
            )
            cap = str(self._caption).strip()
            if cap:
                draw.text((6, 4), cap[: max(16, (self.window_w // 6) - 4)], fill=(230, 234, 240), font=font)

            x = 8
            y = 24
            for i, is_on in enumerate(self._cycle_selected):
                token_w = 42
                if x + token_w >= (self.window_w - 170):
                    break
                box = (x, y, x + 11, y + 11)
                fill = (52, 120, 66) if bool(is_on) else (36, 40, 44)
                draw.rectangle([box[0], box[1], box[2], box[3]], outline=(186, 194, 204), fill=fill)
                if bool(is_on):
                    draw.line([(box[0] + 2, box[1] + 6), (box[0] + 5, box[1] + 9)], fill=(236, 244, 248), width=1)
                    draw.line([(box[0] + 5, box[1] + 9), (box[0] + 9, box[1] + 2)], fill=(236, 244, 248), width=1)
                draw.text((x + 15, y - 1), f"C{i + 1}", fill=(224, 230, 236), font=font)
                self._control_boxes.append(("cycle", int(i), box))
                x += token_w

            ox = max(x + 4, self.window_w - 166)
            oy = y
            o_box = (ox, oy, ox + 11, oy + 11)
            o_fill = (126, 84, 36) if bool(self._gate_override) else (36, 40, 44)
            draw.rectangle([o_box[0], o_box[1], o_box[2], o_box[3]], outline=(186, 194, 204), fill=o_fill)
            if bool(self._gate_override):
                draw.line([(o_box[0] + 2, o_box[1] + 6), (o_box[0] + 5, o_box[1] + 9)], fill=(236, 244, 248), width=1)
                draw.line([(o_box[0] + 5, o_box[1] + 9), (o_box[0] + 9, o_box[1] + 2)], fill=(236, 244, 248), width=1)
            draw.text((ox + 15, oy - 1), "Override gates", fill=(230, 208, 170), font=font)
            self._control_boxes.append(("override", -1, o_box))

            active = self.selected_cycle_ids()
            active_txt = ",".join(str(i) for i in active) if len(active) > 0 else "none"
            draw.text(
                (6, self.top_bar_h - 14),
                f"cycles={active_txt} gate_override={1 if self._gate_override else 0}",
                fill=(180, 188, 198),
                font=font,
            )
            return np.asarray(im, dtype=np.uint8)
        except Exception:
            return out

    def _draw_texture_px(self, tex_id: int, x0: int, y0: int, x1: int, y1: int):
        gl = self._gl
        xf0 = -1.0 + (2.0 * float(x0) / float(max(1, self.window_w)))
        xf1 = -1.0 + (2.0 * float(x1) / float(max(1, self.window_w)))
        yt = 1.0 - (2.0 * float(y0) / float(max(1, self.window_h)))
        yb = 1.0 - (2.0 * float(y1) / float(max(1, self.window_h)))
        gl.glBindTexture(gl.GL_TEXTURE_2D, int(tex_id))
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(float(xf0), float(yb))
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(float(xf1), float(yb))
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(float(xf1), float(yt))
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(float(xf0), float(yt))
        gl.glEnd()

    def _present(self, force: bool = False):
        if (not self._ready) or (not self.enabled) or self._stop_requested:
            return
        now = time.perf_counter()
        has_pending = isinstance(self._pending_frame, dict)
        dirty = bool(has_pending or self._top_bar_dirty or self._panel_text_dirty)
        if not dirty:
            return
        if (not force) and has_pending and (now - self._last_present_t) < self._min_present_dt:
            return

        if has_pending:
            frame = self._pending_frame
            self._pending_frame = None
            imgs = frame.get("images", None)
            if isinstance(imgs, list) and len(imgs) == 3:
                for i, tid in enumerate(self._textures["img"]):
                    self._upload_texture(int(tid), np.asarray(imgs[i], dtype=np.uint8))
            self._caption = str(frame.get("caption", ""))
            self._panel_titles = [str(x) for x in frame.get("titles", self._panel_titles)]
            self._panel_rows = [list(r) for r in frame.get("rows", self._panel_rows)]
            self._panel_text_dirty = True
            self._top_bar_dirty = True

        if self._panel_text_dirty:
            for i in range(3):
                title = self._panel_titles[i] if i < len(self._panel_titles) else ""
                rows = self._panel_rows[i] if i < len(self._panel_rows) else []
                self._panel_text_rgb[i] = self._render_panel_text(title=title, rows=rows)
                self._upload_texture(int(self._textures["text"][i]), self._panel_text_rgb[i])
            self._panel_text_dirty = False

        if self._top_bar_dirty:
            self._top_bar_rgb = self._render_top_bar()
            self._upload_texture(int(self._textures["bar"]), self._top_bar_rgb)
            self._top_bar_dirty = False

        gl = self._gl
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        self._draw_texture_px(
            int(self._textures["bar"]),
            0,
            0,
            self.window_w,
            self.top_bar_h,
        )
        for i in range(3):
            x0 = int(i * self.panel_w)
            x1 = int((i + 1) * self.panel_w)
            self._draw_texture_px(
                int(self._textures["text"][i]),
                x0,
                self.top_bar_h,
                x1,
                self.top_bar_h + self.panel_h,
            )
            self._draw_texture_px(
                int(self._textures["img"][i]),
                x0,
                self.top_bar_h + self.panel_h,
                x1,
                self.top_bar_h + (2 * self.panel_h),
            )

        gate_mode = "manual" if self._gate_override else "auto"
        self._pygame.display.set_caption(f"Stage Status Viewer | gate={gate_mode} | {self._caption}")
        self._pygame.display.flip()
        self._last_present_t = now

    def _handle_click(self, x: int, y: int) -> bool:
        xi = int(x)
        yi = int(y)
        if yi < 0 or yi >= int(self.top_bar_h):
            return False
        for kind, idx, box in self._control_boxes:
            if (xi >= int(box[0])) and (xi <= int(box[2])) and (yi >= int(box[1])) and (yi <= int(box[3])):
                if kind == "cycle":
                    if int(idx) >= 0 and int(idx) < len(self._cycle_selected):
                        self._cycle_selected[int(idx)] = not bool(self._cycle_selected[int(idx)])
                        self._top_bar_dirty = True
                        return True
                elif kind == "override":
                    self._gate_override = not bool(self._gate_override)
                    self._top_bar_dirty = True
                    return True
        return False

    def _poll_events(self):
        if (not self._ready) or (self._pygame is None):
            return
        try:
            for event in self._pygame.event.get():
                if event.type == self._pygame.QUIT:
                    self._stop_requested = True
                    self.enabled = False
                    self.close()
                    return
                if event.type == self._pygame.MOUSEBUTTONDOWN and int(getattr(event, "button", 0)) == 1:
                    pos = getattr(event, "pos", None)
                    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                        if self._handle_click(int(pos[0]), int(pos[1])):
                            self._present(force=True)
        except Exception:
            pass

    def pump(self):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return
        self._poll_events()
        self._present(force=False)

    def stop_requested(self) -> bool:
        return bool(self._stop_requested)

    def update(
        self,
        clean_img: torch.Tensor,
        input_img: torch.Tensor,
        output_img: torch.Tensor,
        caption: str,
        panel_titles: Optional[Sequence[str]] = None,
        panel_rows: Optional[Sequence[Sequence[str]]] = None,
    ):
        if not self.enabled:
            return
        self._init()
        if not self._ready:
            return
        self._poll_events()
        if self._stop_requested or (not self._ready):
            return

        clean_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(clean_img))
        in_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(input_img))
        out_rgb = self._resize_rgb_to_panel(_tensor_to_rgb_u8_image(output_img))
        titles, rows = self._normalize_panel_text(panel_titles=panel_titles, panel_rows=panel_rows)
        self._pending_frame = {
            "images": [clean_rgb, in_rgb, out_rgb],
            "caption": str(caption),
            "titles": titles,
            "rows": rows,
        }
        self._present(force=False)

    def close(self):
        if self._ready:
            try:
                if self._gl is not None and self._textures is not None:
                    tex_ids = (
                        list(self._textures.get("img", []))
                        + list(self._textures.get("text", []))
                        + [self._textures.get("bar", 0)]
                    )
                    tex_ids = [int(t) for t in tex_ids if int(t) > 0]
                    if len(tex_ids) > 0:
                        self._gl.glDeleteTextures(tex_ids)
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

    # Trivial off-switch: cycles<=0 and frequency<=0 keeps a constant base LR.
    if float(frequency) <= 0.0 and float(cycles) <= 0.0:
        return 1.0

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


class _ResidualContextBlock(nn.Module):
    def __init__(self, channels: int, dropout_p: float = 0.0):
        super().__init__()
        c = max(1, int(channels))
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(float(max(0.0, min(0.8, dropout_p)))) if float(dropout_p) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        return self.act(y + r)


class TinyConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        base_ch: int = 64,
        max_ch: int = 384,
        context_blocks: int = 8,
        context_dropout: float = 0.05,
        mask_decoder_channels: int = 0,
    ):
        super().__init__()
        c0 = max(16, int(base_ch))
        c1 = min(max(32, int(max_ch)), max(c0, int(round(c0 * 2.0))))
        c2 = min(max(48, int(max_ch)), max(c1, int(round(c0 * 3.0))))
        c3 = min(max(64, int(max_ch)), max(c2, int(round(c0 * 4.0))))
        n_ctx = max(0, int(context_blocks))
        d_ctx = float(max(0.0, min(0.8, context_dropout)))
        self.features = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, padding=1),
            nn.BatchNorm2d(c0),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c0, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            *[_ResidualContextBlock(c3, dropout_p=d_ctx) for _ in range(n_ctx)],
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )
        self.mask_decoder_channels = max(0, int(mask_decoder_channels))
        if int(self.mask_decoder_channels) > 0:
            m0 = max(16, int(self.mask_decoder_channels))
            m1 = max(12, int(round(m0 * 0.75)))
            m2 = max(8, int(round(m0 * 0.5)))
            self.mask_head = nn.Sequential(
                nn.Conv2d(c3, m0, kernel_size=3, padding=1),
                nn.BatchNorm2d(m0),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(m0, m0, kernel_size=3, padding=1),
                nn.BatchNorm2d(m0),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(m0, m1, kernel_size=3, padding=1),
                nn.BatchNorm2d(m1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(m1, m2, kernel_size=3, padding=1),
                nn.BatchNorm2d(m2),
                nn.GELU(),
                nn.Conv2d(m2, 1, kernel_size=1),
            )
        else:
            self.mask_head = None
        # Keep a wide semantic lane before projection to task/logit space.
        semantic_hidden = max(1024, int(c3) * 4)
        self.semantic_expand = nn.Sequential(
            nn.Linear(c3, semantic_hidden),
            nn.GELU(),
            nn.Linear(semantic_hidden, semantic_hidden),
            nn.GELU(),
        )
        self.embed_proj = nn.Linear(semantic_hidden, c3)
        self.embed_temperature = 10.0
        self.register_buffer("label_embed_bank", torch.zeros((0, 0), dtype=torch.float32), persistent=True)
        self.register_buffer("label_embed_enabled", torch.zeros((1,), dtype=torch.uint8), persistent=True)

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extract_feature_map(x)
        h = self.head[0](h)
        h = self.head[1](h)
        return h

    def set_label_embedding_bank(self, bank: torch.Tensor, temperature: float = 10.0):
        if bank.ndim != 2:
            raise ValueError("Label embedding bank must be shape [num_classes, embed_dim].")
        bank = bank.detach().to(dtype=torch.float32)
        bank = F.normalize(bank, dim=1, eps=1e-6)
        emb_dim = int(bank.shape[1])
        if emb_dim <= 0:
            raise ValueError("Label embedding dimension must be > 0.")
        if int(self.embed_proj.out_features) != emb_dim:
            old = self.embed_proj
            new_proj = nn.Linear(int(old.in_features), int(emb_dim))
            new_proj = new_proj.to(device=old.weight.device, dtype=old.weight.dtype)
            with torch.no_grad():
                new_proj.weight.zero_()
                d = min(int(old.out_features), int(emb_dim))
                new_proj.weight[:d, : int(old.in_features)] = old.weight[:d, : int(old.in_features)]
                if old.bias is not None and new_proj.bias is not None:
                    new_proj.bias.zero_()
                    new_proj.bias[:d] = old.bias[:d]
            self.embed_proj = new_proj
        self.label_embed_bank = bank
        self.label_embed_enabled.fill_(1)
        self.embed_temperature = max(0.1, float(temperature))

    def disable_label_embedding_bank(self):
        self.label_embed_bank = torch.zeros((0, 0), dtype=torch.float32, device=self.label_embed_bank.device)
        self.label_embed_enabled.fill_(0)

    def encode_semantic_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 2:
            raise ValueError(f"Expected pooled feature tensor [B,D], got shape={tuple(feat.shape)}")
        z = self.semantic_expand(feat)
        z = self.embed_proj(z)
        return F.normalize(z, dim=1, eps=1e-6)

    def semantic_logits_from_features(
        self,
        feat: torch.Tensor,
        bank: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        z = self.encode_semantic_from_features(feat)
        if bank is None:
            if not bool(int(self.label_embed_enabled.item())) or int(self.label_embed_bank.shape[0]) <= 0:
                raise RuntimeError("No active label embedding bank available for semantic logits.")
            bank_t = self.label_embed_bank.to(device=z.device, dtype=z.dtype)
        else:
            bank_t = bank.to(device=z.device, dtype=z.dtype)
        if bank_t.ndim != 2:
            raise ValueError(f"Expected semantic bank [N,D], got shape={tuple(bank_t.shape)}")
        if int(bank_t.shape[1]) != int(z.shape[1]):
            raise ValueError(
                f"Semantic bank dim mismatch: bank_dim={int(bank_t.shape[1])} embed_dim={int(z.shape[1])}"
            )
        bank_t = F.normalize(bank_t, dim=1, eps=1e-6)
        t = float(self.embed_temperature if temperature is None else temperature)
        t = max(1e-4, t)
        return (z @ bank_t.transpose(0, 1)) * t

    def semantic_logits(
        self,
        x: torch.Tensor,
        bank: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.semantic_logits_from_features(feat=feat, bank=bank, temperature=temperature)

    def mask_logits_from_feature_map(
        self,
        feature_map: torch.Tensor,
        output_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if self.mask_head is None:
            raise RuntimeError("Mask head is not enabled for this TinyConvClassifier.")
        y = self.mask_head(feature_map)
        if output_hw is not None and tuple(y.shape[-2:]) != tuple(output_hw):
            y = F.interpolate(y, size=output_hw, mode="bilinear", align_corners=False)
        return y

    def forward_with_aux(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature_map = self.extract_feature_map(x)
        feat = self.head[0](feature_map)
        feat = self.head[1](feat)
        if bool(int(self.label_embed_enabled.item())) and int(self.label_embed_bank.shape[0]) > 0:
            logits = self.semantic_logits_from_features(feat=feat)
        else:
            logits = self.head[-1](feat)
        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "pooled_features": feat,
        }
        if self.mask_head is not None:
            out["mask_logits"] = self.mask_logits_from_feature_map(
                feature_map=feature_map,
                output_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)["logits"]


class _LoRALinearSlot(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        r = max(1, int(rank))
        self.down = nn.Linear(int(in_features), int(r), bias=False)
        self.up = nn.Linear(int(r), int(out_features), bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class _LoRAConv1x1Slot(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rank: int):
        super().__init__()
        r = max(1, int(rank))
        self.down = nn.Conv2d(int(in_channels), int(r), kernel_size=1, bias=False)
        self.up = nn.Conv2d(int(r), int(out_channels), kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear requires nn.Linear, got {type(base).__name__}")
        self.base = base
        self.rank = max(1, int(rank))
        self.alpha = float(max(1.0, float(alpha)))
        self.scale = float(self.alpha / float(self.rank))
        self.slots = nn.ModuleDict()
        self.active_slot = ""

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def ensure_slot(self, slot_name: str):
        key = str(slot_name).strip()
        if not key:
            raise ValueError("LoRA slot name must be non-empty.")
        if key not in self.slots:
            self.slots[key] = _LoRALinearSlot(
                in_features=int(self.base.in_features),
                out_features=int(self.base.out_features),
                rank=int(self.rank),
            )

    def slot_names(self) -> List[str]:
        return [str(k) for k in self.slots.keys()]

    def set_active_slot(self, slot_name: str):
        self.active_slot = str(slot_name).strip()

    def set_trainable_state(self, slot_name: str, lora_only: bool):
        key = str(slot_name).strip()
        for p in self.base.parameters():
            p.requires_grad_(not bool(lora_only))
        for name, slot in self.slots.items():
            req = bool(lora_only and str(name) == key)
            for p in slot.parameters():
                p.requires_grad_(bool(req))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        key = str(self.active_slot).strip()
        if key and key in self.slots:
            y = y + (self.scale * self.slots[key](x))
        return y


class LoRAConv2d1x1(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError(f"LoRAConv2d1x1 requires nn.Conv2d, got {type(base).__name__}")
        if tuple(base.kernel_size) != (1, 1):
            raise ValueError(f"LoRAConv2d1x1 only supports 1x1 convolutions, got kernel={tuple(base.kernel_size)}")
        self.base = base
        self.rank = max(1, int(rank))
        self.alpha = float(max(1.0, float(alpha)))
        self.scale = float(self.alpha / float(self.rank))
        self.slots = nn.ModuleDict()
        self.active_slot = ""

    @property
    def in_channels(self) -> int:
        return int(self.base.in_channels)

    @property
    def out_channels(self) -> int:
        return int(self.base.out_channels)

    @property
    def kernel_size(self) -> Tuple[int, int]:
        return tuple(self.base.kernel_size)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def ensure_slot(self, slot_name: str):
        key = str(slot_name).strip()
        if not key:
            raise ValueError("LoRA slot name must be non-empty.")
        if key not in self.slots:
            self.slots[key] = _LoRAConv1x1Slot(
                in_channels=int(self.base.in_channels),
                out_channels=int(self.base.out_channels),
                rank=int(self.rank),
            )

    def slot_names(self) -> List[str]:
        return [str(k) for k in self.slots.keys()]

    def set_active_slot(self, slot_name: str):
        self.active_slot = str(slot_name).strip()

    def set_trainable_state(self, slot_name: str, lora_only: bool):
        key = str(slot_name).strip()
        for p in self.base.parameters():
            p.requires_grad_(not bool(lora_only))
        for name, slot in self.slots.items():
            req = bool(lora_only and str(name) == key)
            for p in slot.parameters():
                p.requires_grad_(bool(req))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        key = str(self.active_slot).strip()
        if key and key in self.slots:
            y = y + (self.scale * self.slots[key](x))
        return y


def install_tiny_classifier_lora(model: TinyConvClassifier, rank: int = 8, alpha: float = 16.0) -> TinyConvClassifier:
    if not isinstance(model, TinyConvClassifier):
        raise TypeError(f"LoRA install requires TinyConvClassifier, got {type(model).__name__}")
    if isinstance(model.semantic_expand[0], nn.Linear):
        model.semantic_expand[0] = LoRALinear(model.semantic_expand[0], rank=int(rank), alpha=float(alpha))
    if isinstance(model.semantic_expand[2], nn.Linear):
        model.semantic_expand[2] = LoRALinear(model.semantic_expand[2], rank=int(rank), alpha=float(alpha))
    if isinstance(model.embed_proj, nn.Linear):
        model.embed_proj = LoRALinear(model.embed_proj, rank=int(rank), alpha=float(alpha))
    if isinstance(model.head[-1], nn.Linear):
        model.head[-1] = LoRALinear(model.head[-1], rank=int(rank), alpha=float(alpha))
    if model.mask_head is not None and isinstance(model.mask_head[-1], nn.Conv2d) and tuple(model.mask_head[-1].kernel_size) == (1, 1):
        model.mask_head[-1] = LoRAConv2d1x1(model.mask_head[-1], rank=int(rank), alpha=float(alpha))
    return model


def tiny_classifier_lora_modules(model: TinyConvClassifier) -> List[nn.Module]:
    if not isinstance(model, TinyConvClassifier):
        return []
    out: List[nn.Module] = []
    candidates = [
        model.semantic_expand[0],
        model.semantic_expand[2],
        model.embed_proj,
        model.head[-1],
        (model.mask_head[-1] if model.mask_head is not None and int(len(model.mask_head)) > 0 else None),
    ]
    for mod in candidates:
        if isinstance(mod, (LoRALinear, LoRAConv2d1x1)):
            out.append(mod)
    return out


def ensure_tiny_classifier_lora_slot(model: TinyConvClassifier, slot_name: str):
    key = str(slot_name).strip()
    if not key:
        raise ValueError("LoRA slot name must be non-empty.")
    for mod in tiny_classifier_lora_modules(model):
        mod.ensure_slot(key)


def set_tiny_classifier_lora_state(model: TinyConvClassifier, slot_name: str = "", lora_only: bool = False):
    key = str(slot_name).strip()
    if bool(lora_only):
        if not key:
            raise ValueError("LoRA-only mode requires a non-empty slot name.")
        ensure_tiny_classifier_lora_slot(model, key)
    for p in model.parameters():
        p.requires_grad_(not bool(lora_only))
    for mod in tiny_classifier_lora_modules(model):
        mod.set_active_slot(key if bool(lora_only) else "")
        mod.set_trainable_state(slot_name=key, lora_only=bool(lora_only))


def tiny_classifier_lora_snapshot(model: TinyConvClassifier) -> Dict[str, Any]:
    if not isinstance(model, TinyConvClassifier):
        return {"installed": False, "slot_names": []}
    mods = tiny_classifier_lora_modules(model)
    if int(len(mods)) <= 0:
        return {"installed": False, "slot_names": []}
    first = mods[0]
    slot_names: List[str] = []
    seen: set = set()
    for mod in mods:
        for slot_name in mod.slot_names():
            key = str(slot_name).strip()
            if (not key) or (key in seen):
                continue
            seen.add(key)
            slot_names.append(key)
    active_slot = ""
    try:
        active_slot = str(getattr(first, "active_slot", "")).strip()
    except Exception:
        active_slot = ""
    lora_only = False
    try:
        lora_only = bool(active_slot) and all((not bool(p.requires_grad)) for p in model.parameters())
    except Exception:
        lora_only = False
    return {
        "installed": True,
        "rank": int(getattr(first, "rank", 0)),
        "alpha": float(getattr(first, "alpha", 0.0)),
        "slot_names": list(slot_names),
        "active_slot": str(active_slot),
        "lora_only": bool(lora_only),
        "module_count": int(len(mods)),
    }


def restore_tiny_classifier_lora_snapshot(model: TinyConvClassifier, snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(model, TinyConvClassifier):
        return {"used": False, "installed": False, "reason": f"unsupported_model:{type(model).__name__}"}
    if not isinstance(snapshot, dict):
        return {"used": False, "installed": False, "reason": "snapshot_missing"}
    if not bool(snapshot.get("installed", False)):
        return {"used": False, "installed": False, "reason": "not_installed"}
    rank = max(1, int(snapshot.get("rank", 8)))
    alpha = float(max(1.0, float(snapshot.get("alpha", 16.0))))
    if int(len(tiny_classifier_lora_modules(model))) <= 0:
        install_tiny_classifier_lora(model, rank=int(rank), alpha=float(alpha))
    slot_names = snapshot.get("slot_names", [])
    restored_slots = 0
    if isinstance(slot_names, (list, tuple, set)):
        for slot_name in slot_names:
            key = str(slot_name).strip()
            if not key:
                continue
            ensure_tiny_classifier_lora_slot(model, key)
            restored_slots += 1
    active_slot = str(snapshot.get("active_slot", "")).strip()
    lora_only = bool(snapshot.get("lora_only", False)) and bool(active_slot)
    set_tiny_classifier_lora_state(model, slot_name=active_slot if bool(lora_only) else "", lora_only=bool(lora_only))
    return {
        "used": True,
        "installed": True,
        "rank": int(rank),
        "alpha": float(alpha),
        "slots": int(restored_slots),
        "active_slot": str(active_slot),
        "lora_only": bool(lora_only),
        "reason": "restored",
    }


class DeskewFilterBundle(nn.Module):
    def __init__(self, d_model: int, max_skew: float = 0.25):
        super().__init__()
        self.max_skew = max(0.0, float(max_skew))
        hidden = max(16, int(d_model) // 4)
        self.d_model = int(d_model)
        self.prefilter_norm = nn.LayerNorm(int(d_model))
        self.prefilter_fc1 = nn.Linear(int(d_model), int(hidden))
        self.prefilter_fc2 = nn.Linear(int(hidden), 2)  # [skew, confidence_logit]
        self.prefilter_token_fc = nn.Linear(int(hidden), int(d_model))
        self.post_residual_norm = nn.LayerNorm(int(d_model))
        self.post_residual_fc1 = nn.Linear(int(d_model), int(hidden))
        self.post_residual_fc2 = nn.Linear(int(hidden), 1)  # residual skew estimate
        self.post_residual_token_fc = nn.Linear(int(hidden), int(d_model))
        with torch.no_grad():
            # Identity-ish startup so legacy checkpoints don't require full retrain.
            self.prefilter_fc2.weight.zero_()
            self.prefilter_fc2.bias.zero_()
            self.post_residual_fc2.weight.zero_()
            self.post_residual_fc2.bias.zero_()
            self.prefilter_token_fc.weight.zero_()
            self.prefilter_token_fc.bias.zero_()
            self.post_residual_token_fc.weight.zero_()
            self.post_residual_token_fc.bias.zero_()

    def apply_prefilter(
        self,
        x_in: torch.Tensor,
        pre_tokens: torch.Tensor,
        task_inputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz = int(x_in.shape[0])
        if int(pre_tokens.ndim) != 3:
            raise ValueError(f"Deskew prefilter expects [B,N,D] tokens, got {tuple(pre_tokens.shape)}")
        z = pre_tokens.mean(dim=1)
        z = self.prefilter_norm(z)
        z = F.gelu(self.prefilter_fc1(z))
        ctrl = self.prefilter_fc2(z)
        pred_skew = torch.tanh(ctrl[:, :1]) * float(self.max_skew)
        confidence = torch.sigmoid(ctrl[:, 1:2])
        pre_token_mix = float(task_inputs.get("pre_token_mix", 0.08)) if isinstance(task_inputs, dict) else 0.08
        pre_token_mix = max(0.0, min(0.50, float(pre_token_mix)))
        token_residual = torch.tanh(self.prefilter_token_fc(z)) * (confidence * pre_token_mix)
        applied_skew = -(pred_skew * confidence)
        if float(self.max_skew) <= 0.0:
            applied_skew = torch.zeros((bsz, 1), device=x_in.device, dtype=x_in.dtype)
            x_out = x_in
        else:
            x_out = _apply_stride_skew_explicit_batch(x_in, applied_skew)
        aux = {
            "pred_skew": pred_skew,
            "confidence": confidence,
            "applied_skew": applied_skew,
            "token_residual": token_residual,
        }
        return x_out, aux

    def observe_post(
        self,
        enc_tokens: torch.Tensor,
        task_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        if int(enc_tokens.ndim) != 3:
            raise ValueError(f"Deskew post observer expects [B,N,D] tokens, got {tuple(enc_tokens.shape)}")
        z = enc_tokens.mean(dim=1)
        z = self.post_residual_norm(z)
        z = F.gelu(self.post_residual_fc1(z))
        residual_skew = torch.tanh(self.post_residual_fc2(z)) * float(self.max_skew)
        post_token_mix = float(task_inputs.get("post_token_mix", 0.08)) if isinstance(task_inputs, dict) else 0.08
        post_token_mix = max(0.0, min(0.50, float(post_token_mix)))
        token_gate = torch.clamp(
            torch.abs(residual_skew) / max(1e-4, float(self.max_skew)),
            min=0.0,
            max=1.0,
        )
        token_residual = torch.tanh(self.post_residual_token_fc(z)) * (token_gate * post_token_mix)
        return {
            "residual_skew": residual_skew,
            "token_residual": token_residual,
            "token_gate": token_gate,
        }

    def compute_losses(
        self,
        pre_aux: Dict[str, torch.Tensor],
        post_aux: Dict[str, torch.Tensor],
        task_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        task_inputs = task_inputs or {}
        target_skew = task_inputs.get("target_skew", None)
        token_residual_weight = max(0.0, float(task_inputs.get("token_residual_weight", 0.01)))
        if not torch.is_tensor(target_skew):
            z = pre_aux["pred_skew"].new_zeros(())
            return {
                "prefilter_loss": z,
                "post_residual_loss": z,
                "token_residual_loss": z,
                "prefilter_loss_raw": z,
                "post_residual_loss_raw": z,
                "token_residual_loss_raw": z,
                "skew_mae": z,
            }
        target_skew = target_skew.to(device=pre_aux["pred_skew"].device, dtype=torch.float32).reshape(-1, 1)
        max_skew = float(max(float(task_inputs.get("max_skew", self.max_skew)), 1e-4))
        w_pre = max(0.0, float(task_inputs.get("prefilter_weight", 1.0)))
        w_post = max(0.0, float(task_inputs.get("post_weight", 1.0)))
        pred_skew = pre_aux["pred_skew"].to(torch.float32)
        applied_skew = pre_aux["applied_skew"].to(torch.float32)
        residual_obs = post_aux.get("residual_skew", torch.zeros_like(pred_skew)).to(torch.float32)

        pre_raw = F.smooth_l1_loss(pred_skew / max_skew, target_skew / max_skew)
        # After applying correction, residual should be target + applied (near 0 if corrected well).
        residual_target = target_skew + applied_skew
        post_raw = F.smooth_l1_loss(residual_obs / max_skew, residual_target / max_skew)
        pre_token = pre_aux.get("token_residual", None)
        post_token = post_aux.get("token_residual", None)
        token_raw = pre_raw.new_zeros(())
        if torch.is_tensor(pre_token):
            token_raw = token_raw + torch.mean(pre_token.to(torch.float32) ** 2)
        if torch.is_tensor(post_token):
            token_raw = token_raw + torch.mean(post_token.to(torch.float32) ** 2)
        mae = torch.mean(torch.abs(pred_skew - target_skew))
        return {
            "prefilter_loss": pre_raw * w_pre,
            "post_residual_loss": post_raw * w_post,
            "token_residual_loss": token_raw * token_residual_weight,
            "prefilter_loss_raw": pre_raw,
            "post_residual_loss_raw": post_raw,
            "token_residual_loss_raw": token_raw,
            "skew_mae": mae,
        }


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
        prefilter_max_skew: float = 0.25,
        prefilter_enabled: bool = True,
        aux_filter_bundle_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        if chunk_samples % patch_size != 0:
            raise ValueError("chunk_samples must be divisible by patch_size")
        if int(d_model) <= 0:
            raise ValueError("d_model must be > 0")
        if int(nhead) <= 0:
            raise ValueError("nhead must be > 0")
        if int(d_model) % int(nhead) != 0:
            raise ValueError(f"d_model ({int(d_model)}) must be divisible by nhead ({int(nhead)}).")

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
        self.aux_filter_bundles = nn.ModuleDict()
        bundle_names = list(aux_filter_bundle_names) if aux_filter_bundle_names is not None else ["deskew"]
        for name in bundle_names:
            key = str(name).strip().lower()
            if (not key) or (key in ("none", "off", "0")):
                continue
            if key == "deskew":
                if bool(prefilter_enabled):
                    self.register_aux_filter_bundle(
                        key,
                        DeskewFilterBundle(
                            d_model=int(d_model),
                            max_skew=float(prefilter_max_skew),
                        ),
                    )
                continue
            raise ValueError(
                f"Unknown transformer aux filter bundle '{name}'. "
                "Add its module and register it in WavePatchTransformer.__init__."
            )

    def register_aux_filter_bundle(self, name: str, bundle: nn.Module):
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Aux filter bundle name must be non-empty.")
        self.aux_filter_bundles[key] = bundle

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
        filter_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if x.ndim != 2:
            raise ValueError("Expected [B, T] waveform input")
        bsz, t = x.shape
        if t != self.chunk_samples:
            if t > self.chunk_samples:
                x = x[:, : self.chunk_samples]
            else:
                x = F.pad(x, (0, self.chunk_samples - t), value=0.0)

        filter_inputs = filter_inputs if isinstance(filter_inputs, dict) else {}
        bundle_aux: Dict[str, Dict[str, Any]] = {}

        def _align_token_residual(delta: Any, ref_tokens: torch.Tensor) -> Optional[torch.Tensor]:
            if not torch.is_tensor(delta):
                return None
            d = delta
            if int(d.ndim) == 2:
                if int(d.shape[0]) != int(ref_tokens.shape[0]) or int(d.shape[1]) != int(ref_tokens.shape[2]):
                    return None
                d = d.unsqueeze(1).expand(int(ref_tokens.shape[0]), int(ref_tokens.shape[1]), int(ref_tokens.shape[2]))
            elif int(d.ndim) == 3:
                if int(d.shape[0]) != int(ref_tokens.shape[0]) or int(d.shape[2]) != int(ref_tokens.shape[2]):
                    return None
                if int(d.shape[1]) == 1 and int(ref_tokens.shape[1]) > 1:
                    d = d.expand(int(ref_tokens.shape[0]), int(ref_tokens.shape[1]), int(ref_tokens.shape[2]))
                elif int(d.shape[1]) != int(ref_tokens.shape[1]):
                    return None
            else:
                return None
            return d.to(device=ref_tokens.device, dtype=ref_tokens.dtype)

        patches_seed = x.reshape(bsz, self.num_patches, self.patch_size)
        h_seed = self.patch_in(patches_seed) + self.pos
        pre_token_delta = torch.zeros_like(h_seed)
        x_in = x
        for name, bundle in self.aux_filter_bundles.items():
            task_in = filter_inputs.get(str(name), {})
            x_in, pre_aux = bundle.apply_prefilter(x_in=x_in, pre_tokens=h_seed, task_inputs=task_in)
            token_delta = _align_token_residual(pre_aux.get("token_residual", None), h_seed)
            if torch.is_tensor(token_delta):
                pre_token_delta = pre_token_delta + token_delta
                pre_aux["token_residual_applied"] = token_delta
            bundle_aux[str(name)] = {"pre": pre_aux}

        patches = x_in.reshape(bsz, self.num_patches, self.patch_size)
        h = self.patch_in(patches) + self.pos
        h = h + pre_token_delta
        h = self.encoder(h)
        h_out = h
        for name, bundle in self.aux_filter_bundles.items():
            task_in = filter_inputs.get(str(name), {})
            post_aux = bundle.observe_post(enc_tokens=h_out, task_inputs=task_in)
            token_delta = _align_token_residual(post_aux.get("token_residual", None), h_out)
            if torch.is_tensor(token_delta):
                h_out = h_out + token_delta
                post_aux["token_residual_applied"] = token_delta
            row = bundle_aux.setdefault(str(name), {})
            row["post"] = post_aux
            if hasattr(bundle, "compute_losses"):
                try:
                    row["losses"] = bundle.compute_losses(
                        pre_aux=dict(row.get("pre", {})),
                        post_aux=dict(post_aux),
                        task_inputs=task_in,
                    )
                except Exception:
                    row["losses"] = {}
        delta = torch.tanh(self.patch_out(h_out)).reshape(bsz, self.chunk_samples) * self.max_delta
        out = torch.clamp(x_in + delta, -1.0, 1.0)
        if not bool(return_aux):
            return out
        aux: Dict[str, Any] = {"filter_bundles": bundle_aux}
        # Back-compat: expose deskew fields at top-level if present.
        deskew_row = bundle_aux.get("deskew", {})
        pre_row = deskew_row.get("pre", {}) if isinstance(deskew_row, dict) else {}
        post_row = deskew_row.get("post", {}) if isinstance(deskew_row, dict) else {}
        if isinstance(pre_row, dict):
            aux["prefilter_skew"] = pre_row.get("pred_skew", None)
            aux["prefilter_confidence"] = pre_row.get("confidence", None)
            aux["prefilter_applied_skew"] = pre_row.get("applied_skew", None)
        if isinstance(post_row, dict):
            aux["post_residual_skew"] = post_row.get("residual_skew", None)
        return out, aux


class ConditionalBitPlaneGenerator(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_hw: Tuple[int, int],
        z_dim: int = 128,
        base_ch: int = 64,
        depth: int = 4,
        min_ch: int = 12,
        mask_decoder_channels: int = 0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.z_dim = int(z_dim)
        self.image_h = max(8, int(image_hw[0]))
        self.image_w = max(8, int(image_hw[1]))
        self.depth = max(2, int(depth))
        self.min_ch = max(8, int(min_ch))
        self.mask_decoder_channels = max(0, int(mask_decoder_channels))
        # Keep a high-capacity latent image seed for 256x256 synthesis.
        self.seed_h = min(int(self.image_h), 32)
        self.seed_w = min(int(self.image_w), 32)
        c0 = max(32, int(base_ch))
        channels: List[int] = []
        c = int(c0)
        for _ in range(self.depth):
            channels.append(int(c))
            c = max(int(self.min_ch), int(c // 2))
        cond_expand = max(1024, int(self.num_classes) * 4, int(self.z_dim) * 4)
        cond_latent = max(256, int(self.num_classes), int(self.z_dim) * 2)
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.num_classes, cond_expand),
            nn.GELU(),
            nn.Linear(cond_expand, cond_latent),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + cond_latent, c0 * self.seed_h * self.seed_w),
            nn.GELU(),
        )
        blocks = []
        prev_c = int(channels[0])
        for i in range(self.depth):
            out_c = int(channels[i])
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_c, out_c, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.GELU(),
                )
            )
            prev_c = out_c
        self.up_blocks = nn.ModuleList(blocks)
        self.out_head = nn.Sequential(
            nn.Conv2d(int(channels[-1]), int(channels[-1]), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(channels[-1]), 3, kernel_size=1),
        )
        if int(self.mask_decoder_channels) > 0:
            m0 = max(16, int(self.mask_decoder_channels))
            m1 = max(12, int(round(float(m0) * 0.75)))
            self.mask_head = nn.Sequential(
                nn.Conv2d(int(channels[-1]), m0, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(m0, m1, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(m1, 1, kernel_size=1),
            )
        else:
            self.mask_head = None

    def _decode_sizes(self) -> List[Tuple[int, int]]:
        # Grow spatial map aggressively, then refine at target.
        sizes: List[Tuple[int, int]] = []
        cur_h = int(self.seed_h)
        cur_w = int(self.seed_w)
        for i in range(self.depth):
            if i < (self.depth - 1):
                cur_h = min(int(self.image_h), int(cur_h) * 2)
                cur_w = min(int(self.image_w), int(cur_w) * 2)
            else:
                cur_h = int(self.image_h)
                cur_w = int(self.image_w)
            sizes.append((int(cur_h), int(cur_w)))
        return sizes

    def forward_with_aux(self, z: torch.Tensor, cond: torch.Tensor) -> Dict[str, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("Generator expects z shape [B, Z].")
        if cond.ndim != 2:
            raise ValueError("Generator expects cond shape [B, C].")
        if int(z.shape[0]) != int(cond.shape[0]):
            raise ValueError("Generator batch mismatch between z and cond.")
        if int(cond.shape[1]) != int(self.num_classes):
            raise ValueError(
                f"Generator condition width mismatch: got={int(cond.shape[1])} expected={int(self.num_classes)}"
            )
        cond_latent = self.cond_encoder(cond.to(dtype=z.dtype))
        x = torch.cat([z, cond_latent], dim=1)
        h = self.fc(x)
        h = h.view(int(z.shape[0]), -1, self.seed_h, self.seed_w)
        for block, (th, tw) in zip(self.up_blocks, self._decode_sizes()):
            if int(h.shape[2]) != int(th) or int(h.shape[3]) != int(tw):
                h = F.interpolate(h, size=(int(th), int(tw)), mode="nearest")
            h = block(h)
        out = self.out_head(h)
        if int(out.shape[2]) != int(self.image_h) or int(out.shape[3]) != int(self.image_w):
            out = F.interpolate(out, size=(int(self.image_h), int(self.image_w)), mode="nearest")
        out_img = torch.sigmoid(out)
        aux: Dict[str, torch.Tensor] = {"image": out_img}
        if self.mask_head is not None:
            mask_logits = self.mask_head(h)
            if int(mask_logits.shape[2]) != int(self.image_h) or int(mask_logits.shape[3]) != int(self.image_w):
                mask_logits = F.interpolate(mask_logits, size=(int(self.image_h), int(self.image_w)), mode="bilinear", align_corners=False)
            aux["mask_logits"] = mask_logits
        return aux

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(z, cond)["image"]


class ConditionalBitPlaneDiscriminator(nn.Module):
    def __init__(self, num_classes: int, base_ch: int = 48, depth: int = 3, max_ch: int = 512):
        super().__init__()
        self.num_classes = int(num_classes)
        c0 = max(16, int(base_ch))
        n = max(2, int(depth))
        cmax = max(c0, int(max_ch))
        layers: List[nn.Module] = []
        in_c = 4
        out_c = c0
        for _ in range(n):
            out_c = min(int(cmax), int(out_c))
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c
            out_c = min(int(cmax), int(out_c * 2))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        feat_dim = int(in_c)
        cond_expand = max(1024, int(self.num_classes) * 4, int(feat_dim) * 4)
        self.cond_encoder = nn.Sequential(
            nn.Linear(self.num_classes, cond_expand),
            nn.GELU(),
            nn.Linear(cond_expand, feat_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(feat_dim * 2, 2)

    def forward_with_aux(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Discriminator expects x shape [B, 3, H, W].")
        if cond.ndim != 2:
            raise ValueError("Discriminator expects cond shape [B, C].")
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError("Discriminator expects mask shape [B, 1, H, W].")
        if int(mask.shape[0]) != int(x.shape[0]):
            raise ValueError("Discriminator batch mismatch between x and mask.")
        if int(cond.shape[1]) != int(self.num_classes):
            raise ValueError(
                f"Discriminator condition width mismatch: got={int(cond.shape[1])} expected={int(self.num_classes)}"
            )
        if tuple(mask.shape[-2:]) != tuple(x.shape[-2:]):
            mask = F.interpolate(mask.to(dtype=x.dtype), size=tuple(x.shape[-2:]), mode="nearest")
        feat = self.features(torch.cat([x, mask.to(dtype=x.dtype)], dim=1))
        cfeat = self.cond_encoder(cond.to(dtype=feat.dtype))
        logits = self.head(torch.cat([feat, cfeat], dim=1))
        return {
            "subject_logits": logits[:, 0],
            "mask_logits": logits[:, 1],
        }

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x=x, cond=cond, mask=mask)["subject_logits"]


def _payload_condition_bank(
    payload_targets: Sequence[Any],
    num_classes: int,
) -> torch.Tensor:
    n = int(len(payload_targets))
    c = max(1, int(num_classes))
    if n <= 0:
        return torch.zeros((0, c), dtype=torch.float32)

    rows = np.zeros((n, c), dtype=np.float32)
    for i in range(n):
        x = payload_targets[i]
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        else:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if int(arr.size) <= 0:
            raise RuntimeError(f"Semantic condition row {i} is empty.")
        if int(arr.size) != int(c):
            raise RuntimeError(
                f"Semantic condition width mismatch at row {i}: got={int(arr.size)} expected={int(c)}"
            )
        rows[i, :] = arr
    return torch.from_numpy(rows)


def _payload_mask_bank(
    payload_masks: Sequence[Any],
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    n = int(len(payload_masks))
    h = max(1, int(image_hw[0]))
    w = max(1, int(image_hw[1]))
    if n <= 0:
        return torch.zeros((0, 1, h, w), dtype=torch.float32)
    rows = np.zeros((n, 1, h, w), dtype=np.float32)
    for i in range(n):
        x = payload_masks[i]
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(x, dtype=np.float32)
        if int(arr.ndim) == 3 and int(arr.shape[0]) == 1:
            arr = arr[0]
        elif int(arr.ndim) == 3:
            arr = np.mean(arr, axis=0).astype(np.float32, copy=False)
        elif int(arr.ndim) != 2:
            raise RuntimeError(f"Semantic mask row {i} must be [H,W] or [1,H,W], got {tuple(arr.shape)}")
        arr_t = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, None, ...]
        if tuple(arr.shape) != (int(h), int(w)):
            arr_t = F.interpolate(arr_t, size=(int(h), int(w)), mode="nearest")
        rows[int(i), 0, :, :] = np.clip(arr_t[0, 0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)
    return torch.from_numpy(rows)


def _payload_image_bank(
    payload_images: Sequence[Any],
    image_hw: Tuple[int, int],
) -> torch.Tensor:
    n = int(len(payload_images))
    h = max(1, int(image_hw[0]))
    w = max(1, int(image_hw[1]))
    if n <= 0:
        return torch.zeros((0, 3, h, w), dtype=torch.float32)
    rows = np.zeros((n, 3, h, w), dtype=np.float32)
    for i in range(n):
        x = payload_images[i]
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(x, dtype=np.float32)
        if int(arr.ndim) == 2:
            arr = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32, copy=False)
        elif int(arr.ndim) == 3 and int(arr.shape[0]) == 1:
            arr = np.repeat(arr, 3, axis=0).astype(np.float32, copy=False)
        elif int(arr.ndim) == 3 and int(arr.shape[0]) == 3:
            pass
        elif int(arr.ndim) == 3 and int(arr.shape[-1]) == 3:
            arr = np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
        else:
            raise RuntimeError(f"Semantic payload image row {i} must be [3,H,W] or [H,W,3], got {tuple(arr.shape)}")
        arr_t = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None, ...]
        if tuple(arr_t.shape[-2:]) != (int(h), int(w)):
            arr_t = F.interpolate(arr_t, size=(int(h), int(w)), mode="nearest")
        rows[int(i), :, :, :] = np.clip(arr_t[0].cpu().numpy().astype(np.float32, copy=False), 0.0, 1.0)
    return torch.from_numpy(rows)


def _payload_batch_bank(
    payload_images: Sequence[Any],
    payload_conditions: Sequence[Any],
    payload_masks: Sequence[Any],
    indices: Sequence[int],
    num_classes: int,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    picks = [int(i) for i in list(indices)]
    image_rows = [payload_images[int(i)] for i in picks]
    cond_rows = [payload_conditions[int(i)] for i in picks]
    mask_rows = [payload_masks[int(i)] for i in picks]
    return (
        _payload_image_bank(payload_images=image_rows, image_hw=image_hw),
        _payload_condition_bank(payload_targets=cond_rows, num_classes=int(num_classes)),
        _payload_mask_bank(payload_masks=mask_rows, image_hw=image_hw),
    )


def _payload_condition_mask_batch(
    payload_conditions: Sequence[Any],
    payload_masks: Sequence[Any],
    indices: Sequence[int],
    num_classes: int,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    picks = [int(i) for i in list(indices)]
    cond_rows = [payload_conditions[int(i)] for i in picks]
    mask_rows = [payload_masks[int(i)] for i in picks]
    return (
        _payload_condition_bank(payload_targets=cond_rows, num_classes=int(num_classes)),
        _payload_mask_bank(payload_masks=mask_rows, image_hw=image_hw),
    )


def _align_probs_with_condition(probs: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs [B,C], got shape={tuple(probs.shape)}")
    if cond.ndim != 2:
        raise ValueError(f"Expected cond [B,C], got shape={tuple(cond.shape)}")
    if int(probs.shape[0]) != int(cond.shape[0]):
        raise ValueError(
            f"Batch mismatch between probs and cond: probs={tuple(probs.shape)} cond={tuple(cond.shape)}"
        )
    if int(probs.shape[1]) != int(cond.shape[1]):
        raise ValueError(
            f"Class-width mismatch between probs and cond: probs={tuple(probs.shape)} cond={tuple(cond.shape)}"
        )
    return probs, cond


@torch.no_grad()
def evaluate_conditional_generator(
    generator: nn.Module,
    classifier: nn.Module,
    payload_conditions: Sequence[Any],
    payload_masks: Sequence[Any],
    num_classes: int,
    image_hw: Tuple[int, int],
    z_dim: int,
    device: torch.device,
    steps: int = 16,
    batch_size: int = 64,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    fake_class_idx: int = -1,
    discriminator: Optional[nn.Module] = None,
    mask_threshold: float = 0.5,
    seed: int = 0,
) -> Dict[str, float]:
    if len(payload_conditions) <= 0 or len(payload_masks) <= 0:
        return {
            "target_prob": 0.0,
            "mean_prob": 0.0,
            "coverage": 0.0,
            "fake_prob": 0.0,
            "disc_pass_rate": 0.0,
            "disc_mask_pass_rate": 0.0,
            "mask_iou": 0.0,
            "mask_dice": 0.0,
        }
    n_payload = min(int(len(payload_conditions)), int(len(payload_masks)))
    if int(n_payload) <= 0:
        return {
            "target_prob": 0.0,
            "mean_prob": 0.0,
            "coverage": 0.0,
            "fake_prob": 0.0,
            "disc_pass_rate": 0.0,
            "disc_mask_pass_rate": 0.0,
            "mask_iou": 0.0,
            "mask_dice": 0.0,
        }
    rng = np.random.default_rng(seed)
    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    generator.eval()
    classifier.eval()
    disc_was_training = False
    if discriminator is not None:
        discriminator = discriminator.to(device)
        disc_was_training = bool(discriminator.training)
        discriminator.eval()
    sum_target = torch.zeros((), device=device, dtype=torch.float32)
    sum_mean = torch.zeros((), device=device, dtype=torch.float32)
    sum_cov = torch.zeros((), device=device, dtype=torch.float32)
    sum_fake = torch.zeros((), device=device, dtype=torch.float32)
    sum_disc_pass = torch.zeros((), device=device, dtype=torch.float32)
    sum_disc_mask_pass = torch.zeros((), device=device, dtype=torch.float32)
    sum_mask_iou = torch.zeros((), device=device, dtype=torch.float32)
    sum_mask_dice = torch.zeros((), device=device, dtype=torch.float32)
    n = 0
    for _ in range(max(1, int(steps))):
        idx = rng.integers(0, int(n_payload), size=max(1, int(batch_size)))
        cond_cpu, mask_cpu = _payload_condition_mask_batch(
            payload_conditions=payload_conditions,
            payload_masks=payload_masks,
            indices=idx.tolist(),
            num_classes=int(num_classes),
            image_hw=image_hw,
        )
        cond = cond_cpu.to(device=device, dtype=torch.float32)
        mask_target = mask_cpu.to(device=device, dtype=torch.float32)
        z = torch.randn((int(cond.shape[0]), int(z_dim)), device=device)
        with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
            gen_out = generator.forward_with_aux(z, cond) if hasattr(generator, "forward_with_aux") else {"image": generator(z, cond)}
            fake = gen_out["image"].to(torch.float32)
            if channels_last:
                fake = fake.contiguous(memory_format=torch.channels_last)
            logits = classifier(fake)
            probs = torch.sigmoid(logits).to(torch.float32)
            fake_mask_logits = gen_out.get("mask_logits")
            if not isinstance(fake_mask_logits, torch.Tensor):
                raise RuntimeError("Conditional generator evaluation requires generator mask logits.")
            if tuple(fake_mask_logits.shape[-2:]) != tuple(mask_target.shape[-2:]):
                fake_mask_logits = F.interpolate(fake_mask_logits, size=tuple(mask_target.shape[-2:]), mode="bilinear", align_corners=False)
            fake_mask_probs = torch.sigmoid(fake_mask_logits).to(torch.float32)
        probs_target, cond_target = _align_probs_with_condition(probs=probs, cond=cond)
        target_count = torch.clamp(cond_target.sum(dim=1), min=1.0)
        target_prob = ((probs_target * cond_target).sum(dim=1) / target_count).mean()
        mean_prob = probs_target.mean()
        class_mean = probs_target.mean(dim=0)
        cov = (class_mean > 0.5).to(torch.float32).mean()
        if int(fake_class_idx) >= 0 and int(fake_class_idx) < int(probs.shape[1]):
            fake_prob = probs[:, int(fake_class_idx)].mean()
        else:
            fake_prob = torch.zeros((), device=device, dtype=torch.float32)
        fake_mask_bin = (fake_mask_probs >= float(max(0.0, min(1.0, float(mask_threshold))))).to(torch.float32)
        target_mask_bin = (mask_target >= 0.5).to(torch.float32)
        inter = (fake_mask_bin * target_mask_bin).sum(dim=(1, 2, 3))
        union = ((fake_mask_bin + target_mask_bin) > 0.0).to(torch.float32).sum(dim=(1, 2, 3))
        pred_mass = fake_mask_bin.sum(dim=(1, 2, 3))
        tgt_mass = target_mask_bin.sum(dim=(1, 2, 3))
        mask_iou = torch.mean(inter / (union + 1e-8))
        mask_dice = torch.mean((2.0 * inter) / (pred_mass + tgt_mass + 1e-8))
        if discriminator is not None:
            d_fake = discriminator.forward_with_aux(fake, cond, fake_mask_probs)
            disc_pass = (d_fake["subject_logits"].to(torch.float32) > 0.0).to(torch.float32).mean()
            disc_mask_pass = (d_fake["mask_logits"].to(torch.float32) > 0.0).to(torch.float32).mean()
        else:
            disc_pass = torch.zeros((), device=device, dtype=torch.float32)
            disc_mask_pass = torch.zeros((), device=device, dtype=torch.float32)
        sum_target += target_prob
        sum_mean += mean_prob
        sum_cov += cov
        sum_fake += fake_prob
        sum_disc_pass += disc_pass
        sum_disc_mask_pass += disc_mask_pass
        sum_mask_iou += mask_iou
        sum_mask_dice += mask_dice
        n += 1
    d = float(max(1, n))
    out = {
        "target_prob": float((sum_target / d).item()),
        "mean_prob": float((sum_mean / d).item()),
        "coverage": float((sum_cov / d).item()),
        "fake_prob": float((sum_fake / d).item()),
        "disc_pass_rate": float((sum_disc_pass / d).item()),
        "disc_mask_pass_rate": float((sum_disc_mask_pass / d).item()),
        "mask_iou": float((sum_mask_iou / d).item()),
        "mask_dice": float((sum_mask_dice / d).item()),
    }
    if discriminator is not None:
        discriminator.train(disc_was_training)
    return out


def train_conditional_generator_discriminator(
    generator: nn.Module,
    discriminator: nn.Module,
    classifier: nn.Module,
    payload_images: Sequence[np.ndarray],
    payload_conditions: Sequence[Any],
    payload_masks: Sequence[Any],
    num_classes: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    epochs: int = 1,
    steps_per_epoch: int = 120,
    batch_size: int = 64,
    disc_steps_per_gen_step: int = 1,
    z_dim: int = 128,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    w_adv: float = 0.50,
    w_cls: float = 1.50,
    w_mask: float = 0.0,
    w_disc_mask: float = 0.0,
    w_outside_mask: float = 0.0,
    w_wave: float = 0.0,
    wave_margin: float = 0.02,
    wave_denoise_weight: float = 0.25,
    wave_carrier_blend: float = 0.35,
    wave_strength: float = 0.35,
    wave_stride_skew_max: float = 0.18,
    wave_degrade_mode: str = "blur_stride",
    transformer_for_wave: Optional[nn.Module] = None,
    wave_cfg: Optional[RenderConfig] = None,
    wave_sample_bits: int = 16,
    wave_chunk_samples: int = 32768,
    wave_reference_streams: Optional[Sequence[np.ndarray]] = None,
    log_every_steps: int = 0,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    amp: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    classifier_forward_batch_cap: int = 0,
    seed: int = 0,
) -> Tuple[nn.Module, nn.Module, List[Dict[str, float]]]:
    if len(payload_images) <= 0 or len(payload_conditions) <= 0 or len(payload_masks) <= 0:
        raise RuntimeError("Generator/discriminator training requires non-empty payload bank.")
    if len(payload_images) != len(payload_conditions):
        raise RuntimeError(
            f"Payload image/condition mismatch: images={len(payload_images)} conditions={len(payload_conditions)}"
        )
    if len(payload_images) != len(payload_masks):
        raise RuntimeError(
            f"Payload image/mask mismatch: images={len(payload_images)} masks={len(payload_masks)}"
        )

    use_amp = _should_use_amp(device=device, amp=amp)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if use_amp else torch.float16
    use_scaler = bool(use_amp and amp_dtype_t == torch.float16)
    scaler_g = _make_grad_scaler(enabled=use_scaler)
    scaler_d = _make_grad_scaler(enabled=use_scaler)
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    classifier = classifier.to(device)
    use_wave_loss = (
        (float(w_wave) > 0.0)
        and (transformer_for_wave is not None)
        and (wave_cfg is not None)
        and bool(getattr(wave_cfg, "bitmask_enable", False))
    )
    wave_shape_info = None
    wave_canvas_hw = (0, 0)
    wave_empty_fill = 0.0
    wave_chunk_samples = max(1, int(wave_chunk_samples))
    wave_index = None
    ref_sampler = None
    transformer_was_train = False
    transformer_requires_grad: Optional[List[bool]] = None
    if use_wave_loss:
        transformer_for_wave = transformer_for_wave.to(device)
        transformer_was_train = bool(transformer_for_wave.training)
        transformer_requires_grad = [bool(p.requires_grad) for p in transformer_for_wave.parameters()]
        transformer_for_wave.eval()
        for p in transformer_for_wave.parameters():
            p.requires_grad_(False)
        wave_shape_info = _bitwindow_shape_for_wave_len(wave_len=wave_chunk_samples, cfg=wave_cfg)
        wave_canvas_hw = (int(wave_shape_info["height"]), int(wave_shape_info["width"]))
        wave_empty_fill = float(int(wave_cfg.empty_fill) & 0xFF) / 255.0
        downsample = max(1, int(wave_shape_info["downsample"]))
        base_len = int(wave_shape_info["base_len"])
        idx = (torch.arange(base_len, device=device, dtype=torch.long) * int(downsample)).to(torch.long)
        idx = idx[idx < int(wave_chunk_samples)]
        if int(idx.numel()) <= 0:
            use_wave_loss = False
        else:
            wave_index = idx
            if wave_reference_streams is not None and len(wave_reference_streams) > 0:
                ref_sampler = _build_packed_sampler(
                    streams=wave_reference_streams,
                    labels=None,
                    chunk_samples=wave_chunk_samples,
                    seed=int(seed) + 971,
                    device=device,
                    cache_on_device=False,
                    pin_memory=False,
                )
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    g_opt = torch.optim.AdamW(generator.parameters(), lr=float(lr_g), betas=(0.5, 0.999), weight_decay=1e-4)
    d_opt = torch.optim.AdamW(discriminator.parameters(), lr=float(lr_d), betas=(0.5, 0.999), weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    disc_steps_per_gen_step = max(1, int(disc_steps_per_gen_step))
    grad_accum_steps = max(1, int(grad_accum_steps))
    classifier_forward_batch_cap = max(0, int(classifier_forward_batch_cap))

    def _chunk_ranges(total: int, chunks: int) -> List[Tuple[int, int]]:
        n = max(1, int(total))
        c = max(1, min(int(chunks), n))
        bs = int(math.ceil(float(n) / float(c)))
        out: List[Tuple[int, int]] = []
        for st in range(0, n, bs):
            ed = min(n, st + bs)
            if ed > st:
                out.append((int(st), int(ed)))
        return out

    def _classifier_forward_chunked(x: torch.Tensor) -> torch.Tensor:
        cap = int(classifier_forward_batch_cap)
        if cap <= 0 or int(x.shape[0]) <= cap:
            return classifier(x)
        parts: List[torch.Tensor] = []
        for st in range(0, int(x.shape[0]), cap):
            parts.append(classifier(x[st : st + cap]))
        return torch.cat(parts, dim=0)

    n_payload = min(int(len(payload_images)), int(len(payload_conditions)), int(len(payload_masks)))
    if int(n_payload) <= 0:
        raise RuntimeError("Generator/discriminator training requires non-empty payload bank.")
    history: List[Dict[str, float]] = []

    stop_now = False
    for epoch in range(1, max(1, int(epochs)) + 1):
        generator.train()
        discriminator.train()
        run_d = 0.0
        run_g = 0.0
        run_mask = 0.0
        run_outside = 0.0
        run_tgt = 0.0
        run_adv = 0.0
        run_wave = 0.0
        run_disc_mask_fake_pass = 0.0
        run_disc_mask_fake_fail = 0.0
        run_mask_iou = 0.0
        run_mask_dice = 0.0
        run_disc_fake_pass = 0.0
        run_disc_fake_fail = 0.0
        n_steps = max(1, int(steps_per_epoch))
        steps_done = 0
        for step_idx in range(1, n_steps + 1):
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass
            idx = rng.integers(0, int(n_payload), size=max(1, int(batch_size)))
            real_cpu, cond_cpu, mask_cpu = _payload_batch_bank(
                payload_images=payload_images,
                payload_conditions=payload_conditions,
                payload_masks=payload_masks,
                indices=idx.tolist(),
                num_classes=int(num_classes),
                image_hw=image_hw,
            )
            real = real_cpu.to(device=device, non_blocking=True)
            cond = cond_cpu.to(device=device, non_blocking=True, dtype=torch.float32)
            real_mask = mask_cpu.to(device=device, non_blocking=True, dtype=torch.float32)
            if channels_last:
                real = real.contiguous(memory_format=torch.channels_last)
            d_loss_accum = 0.0
            for d_sub in range(disc_steps_per_gen_step):
                if d_sub == 0:
                    real_d = real
                    cond_d = cond
                    real_mask_d = real_mask
                else:
                    idx_d = rng.integers(0, int(n_payload), size=max(1, int(batch_size)))
                    real_d_cpu, cond_d_cpu, mask_d_cpu = _payload_batch_bank(
                        payload_images=payload_images,
                        payload_conditions=payload_conditions,
                        payload_masks=payload_masks,
                        indices=idx_d.tolist(),
                        num_classes=int(num_classes),
                        image_hw=image_hw,
                    )
                    real_d = real_d_cpu.to(device=device, non_blocking=True)
                    cond_d = cond_d_cpu.to(device=device, non_blocking=True, dtype=torch.float32)
                    real_mask_d = mask_d_cpu.to(device=device, non_blocking=True, dtype=torch.float32)
                    if channels_last:
                        real_d = real_d.contiguous(memory_format=torch.channels_last)

                d_opt.zero_grad(set_to_none=True)
                d_sub_loss = 0.0
                d_ranges = _chunk_ranges(total=int(real_d.shape[0]), chunks=grad_accum_steps)
                for st, ed in d_ranges:
                    real_m = real_d[st:ed]
                    cond_m = cond_d[st:ed]
                    real_mask_m = real_mask_d[st:ed]
                    z = torch.randn((int(real_m.shape[0]), int(z_dim)), device=device)
                    with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                        gen_out = generator.forward_with_aux(z, cond_m) if hasattr(generator, "forward_with_aux") else {"image": generator(z, cond_m)}
                        fake = gen_out["image"]
                        fake_mask_logits = gen_out.get("mask_logits")
                        if not isinstance(fake_mask_logits, torch.Tensor):
                            raise RuntimeError("Conditional generator training requires generator mask logits.")
                        if tuple(fake_mask_logits.shape[-2:]) != tuple(real_mask_m.shape[-2:]):
                            fake_mask_logits = F.interpolate(fake_mask_logits, size=tuple(real_mask_m.shape[-2:]), mode="bilinear", align_corners=False)
                        fake_mask_probs = torch.sigmoid(fake_mask_logits)
                        if channels_last:
                            fake = fake.contiguous(memory_format=torch.channels_last)
                        d_real = discriminator.forward_with_aux(real_m, cond_m, real_mask_m)
                        d_fake = discriminator.forward_with_aux(fake.detach(), cond_m, fake_mask_probs.detach())
                        d_loss = 0.5 * (
                            F.softplus(-d_real["subject_logits"]).mean()
                            + F.softplus(d_fake["subject_logits"]).mean()
                            + F.softplus(-d_real["mask_logits"]).mean()
                            + F.softplus(d_fake["mask_logits"]).mean()
                        )
                    d_loss_back = d_loss / float(max(1, len(d_ranges)))
                    if use_scaler:
                        scaler_d.scale(d_loss_back).backward()
                    else:
                        d_loss_back.backward()
                    d_sub_loss += float(d_loss.detach().item()) * (float(ed - st) / float(max(1, int(real_d.shape[0]))))
                if use_scaler:
                    scaler_d.step(d_opt)
                    scaler_d.update()
                else:
                    d_opt.step()
                d_loss_accum += float(d_sub_loss)
            d_loss_step = float(d_loss_accum / float(max(1, disc_steps_per_gen_step)))

            g_opt.zero_grad(set_to_none=True)
            g_loss_step = 0.0
            adv_loss_step = 0.0
            target_prob_step = 0.0
            mask_loss_step = 0.0
            outside_loss_step = 0.0
            wave_loss_step = 0.0
            disc_fake_pass_step = 0.0
            disc_mask_fake_pass_step = 0.0
            mask_iou_step = 0.0
            mask_dice_step = 0.0
            preview_payload = None
            g_ranges = _chunk_ranges(total=int(real.shape[0]), chunks=grad_accum_steps)
            for st, ed in g_ranges:
                cond_g = cond[st:ed]
                real_g = real[st:ed]
                real_mask_g = real_mask[st:ed]
                z2 = torch.randn((int(real_g.shape[0]), int(z_dim)), device=device)
                with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                    gen_out = generator.forward_with_aux(z2, cond_g) if hasattr(generator, "forward_with_aux") else {"image": generator(z2, cond_g)}
                    fake2 = gen_out["image"]
                    fake_mask_logits = gen_out.get("mask_logits")
                    if not isinstance(fake_mask_logits, torch.Tensor):
                        raise RuntimeError("Conditional generator training requires generator mask logits.")
                    if tuple(fake_mask_logits.shape[-2:]) != tuple(real_mask_g.shape[-2:]):
                        fake_mask_logits = F.interpolate(fake_mask_logits, size=tuple(real_mask_g.shape[-2:]), mode="bilinear", align_corners=False)
                    fake_mask_probs = torch.sigmoid(fake_mask_logits).to(torch.float32)
                    if channels_last:
                        fake2 = fake2.contiguous(memory_format=torch.channels_last)
                    d_fake2 = discriminator.forward_with_aux(fake2, cond_g, fake_mask_probs)
                    adv_loss = F.softplus(-d_fake2["subject_logits"]).mean()
                    disc_mask_loss = F.softplus(-d_fake2["mask_logits"]).mean()
                    logits = _classifier_forward_chunked(fake2)
                    probs = torch.sigmoid(logits).to(torch.float32)
                    probs_target, cond_target = _align_probs_with_condition(probs=probs, cond=cond_g)
                    target_count = torch.clamp(cond_target.sum(dim=1), min=1.0)
                    target_prob = (probs_target * cond_target).sum(dim=1) / target_count
                    cls_loss = (1.0 - target_prob).mean()
                    mask_loss = F.binary_cross_entropy_with_logits(fake_mask_logits.to(torch.float32), real_mask_g.to(torch.float32))
                    outside_weight = torch.clamp(1.0 - real_mask_g.to(torch.float32), 0.0, 1.0)
                    outside_denom = torch.clamp(outside_weight.sum(), min=1.0)
                    outside_loss = ((torch.abs(fake2.to(torch.float32) - real_g.to(torch.float32)) * outside_weight).sum() / outside_denom)
                    g_loss = (
                        (float(w_adv) * adv_loss)
                        + (float(w_cls) * cls_loss)
                        + (float(w_disc_mask) * disc_mask_loss)
                        + (float(w_mask) * mask_loss)
                        + (float(w_outside_mask) * outside_loss)
                    )
                    wave_loss = torch.zeros((), device=device, dtype=torch.float32)
                    if use_wave_loss and wave_shape_info is not None and wave_index is not None:
                        # Wave-coupled supervision is expensive at high resolutions.
                        # Cap coupling batch size to keep memory bounded on 12GB GPUs.
                        wave_cap = min(int(fake2.shape[0]), 8)
                        if wave_cap < int(fake2.shape[0]):
                            sel = torch.randperm(int(fake2.shape[0]), device=device)[: int(wave_cap)]
                            fake2_wave = fake2.index_select(0, sel)
                            cond_wave = cond_g.index_select(0, sel)
                            target_count_wave = target_count.index_select(0, sel)
                        else:
                            fake2_wave = fake2
                            cond_wave = cond_g
                            target_count_wave = target_count

                        fake_canvas = fake2_wave
                        if int(fake_canvas.shape[2]) != int(wave_canvas_hw[0]) or int(fake_canvas.shape[3]) != int(wave_canvas_hw[1]):
                            fake_canvas = F.interpolate(fake_canvas, size=wave_canvas_hw, mode="nearest")
                        base_seq = _bitwindow_canvas_to_base_sequence_batch(
                            canvas=fake_canvas.to(torch.float32),
                            shape_info=wave_shape_info,
                            empty_fill=float(wave_empty_fill),
                        )
                        n_use = min(int(wave_index.numel()), int(base_seq.shape[1]))
                        if n_use > 0:
                            bsz = int(fake2_wave.shape[0])
                            idx_use = wave_index[:n_use]
                            idx_valid = (idx_use >= 0) & (idx_use < int(wave_chunk_samples))
                            if bool(torch.any(idx_valid)):
                                idx_use = idx_use[idx_valid]
                            else:
                                idx_use = None
                            n_eff = 0 if idx_use is None else min(int(idx_use.numel()), int(base_seq.shape[1]))
                            if n_eff <= 0:
                                idx_use = None
                            if idx_use is not None:
                                seq01 = torch.clamp(base_seq[:, :n_eff], 0.0, 1.0).to(dtype=fake2.dtype)
                                payload_wave = torch.zeros((bsz, int(wave_chunk_samples)), device=device, dtype=fake2.dtype)
                                src = ((seq01 * 2.0) - 1.0).to(dtype=fake2.dtype)
                                payload_wave.scatter_(1, idx_use.unsqueeze(0).expand(bsz, -1), src)
                            else:
                                payload_wave = torch.zeros((bsz, int(wave_chunk_samples)), device=device, dtype=fake2.dtype)
                            if ref_sampler is not None:
                                carrier_wave, _ = ref_sampler.sample(batch_size=bsz)
                                if carrier_wave.device != device:
                                    carrier_wave = carrier_wave.to(device, non_blocking=True)
                                carrier_wave = carrier_wave.to(dtype=fake2.dtype)
                            elif wave_reference_streams is not None and len(wave_reference_streams) > 0:
                                carrier_np = _sample_wave_batch_unlabeled(
                                    streams=wave_reference_streams,
                                    batch_size=bsz,
                                    chunk_samples=int(wave_chunk_samples),
                                    rng=rng,
                                )
                                carrier_wave = _to_device_wave_batch(
                                    carrier_np,
                                    device=device,
                                    pin_memory=False,
                                ).to(dtype=fake2.dtype)
                            else:
                                carrier_wave = torch.zeros_like(payload_wave)
                            blend = float(max(0.0, min(1.0, float(wave_carrier_blend))))
                            clean_wave = torch.clamp(
                                ((1.0 - blend) * carrier_wave) + (blend * payload_wave),
                                -1.0,
                                1.0,
                            )
                            s_vec = torch.full((bsz, 1), float(max(0.0, min(1.0, float(wave_strength)))), device=device, dtype=fake2.dtype)
                            dirty_wave = _degrade_wave_batch(
                                x_clean=clean_wave,
                                strength=s_vec,
                                noise_std_min=0.0,
                                noise_std_max=0.0,
                                dropout_max=0.0,
                                quant_bits_min=16,
                                quant_bits_max=16,
                                mode=str(wave_degrade_mode),
                                stride_skew_max=float(max(0.0, float(wave_stride_skew_max))),
                            )
                            wave_out = transformer_for_wave(dirty_wave)
                            with torch.no_grad():
                                img_wave_before = render_mono_wave_to_tensor(
                                    clean_wave.to(torch.float32),
                                    cfg=wave_cfg,
                                    image_hw=image_hw,
                                    sample_bits=int(wave_sample_bits),
                                )
                            img_wave_after = render_mono_wave_to_tensor(
                                wave_out.to(torch.float32),
                                cfg=wave_cfg,
                                image_hw=image_hw,
                                sample_bits=int(wave_sample_bits),
                            )
                            if channels_last:
                                img_wave_before = img_wave_before.contiguous(memory_format=torch.channels_last)
                                img_wave_after = img_wave_after.contiguous(memory_format=torch.channels_last)
                            with torch.no_grad():
                                logits_wave_before = _classifier_forward_chunked(img_wave_before)
                            logits_wave_after = _classifier_forward_chunked(img_wave_after)
                            probs_wave_before = torch.sigmoid(logits_wave_before).to(torch.float32)
                            probs_wave_after = torch.sigmoid(logits_wave_after).to(torch.float32)
                            probs_wave_before_t, cond_wave_t = _align_probs_with_condition(
                                probs=probs_wave_before,
                                cond=cond_wave,
                            )
                            probs_wave_after_t, _ = _align_probs_with_condition(
                                probs=probs_wave_after,
                                cond=cond_wave_t,
                            )
                            target_count_wave_t = torch.clamp(cond_wave_t.sum(dim=1), min=1.0)
                            target_prob_before = (probs_wave_before_t * cond_wave_t).sum(dim=1) / target_count_wave_t
                            target_prob_after = (probs_wave_after_t * cond_wave_t).sum(dim=1) / target_count_wave_t
                            score_gap = F.relu((target_prob_before + float(max(0.0, float(wave_margin)))) - target_prob_after).mean()
                            denoise_term = F.l1_loss(wave_out.to(torch.float32), clean_wave.to(torch.float32))
                            wave_loss = score_gap + (float(max(0.0, float(wave_denoise_weight))) * denoise_term)
                            g_loss = g_loss + (float(w_wave) * wave_loss)
                g_loss_back = g_loss / float(max(1, len(g_ranges)))
                if use_scaler:
                    scaler_g.scale(g_loss_back).backward()
                else:
                    g_loss_back.backward()

                w = float(ed - st) / float(max(1, int(real.shape[0])))
                g_loss_step += float(g_loss.detach().item()) * w
                adv_loss_step += float(adv_loss.detach().item()) * w
                target_prob_step += float(target_prob.detach().mean().item()) * w
                mask_loss_step += float(mask_loss.detach().item()) * w
                outside_loss_step += float(outside_loss.detach().item()) * w
                wave_loss_step += float(wave_loss.detach().item()) * w
                disc_fake_pass = float((d_fake2["subject_logits"].detach().to(torch.float32) > 0.0).to(torch.float32).mean().item())
                disc_mask_fake_pass = float((d_fake2["mask_logits"].detach().to(torch.float32) > 0.0).to(torch.float32).mean().item())
                fake_mask_bin = (fake_mask_probs >= 0.5).to(torch.float32)
                target_mask_bin = (real_mask_g >= 0.5).to(torch.float32)
                inter = (fake_mask_bin * target_mask_bin).sum(dim=(1, 2, 3))
                union = ((fake_mask_bin + target_mask_bin) > 0.0).to(torch.float32).sum(dim=(1, 2, 3))
                pred_mass = fake_mask_bin.sum(dim=(1, 2, 3))
                tgt_mass = target_mask_bin.sum(dim=(1, 2, 3))
                mask_iou = torch.mean(inter / (union + 1e-8))
                mask_dice = torch.mean((2.0 * inter) / (pred_mass + tgt_mass + 1e-8))
                disc_fake_pass_step += float(disc_fake_pass) * w
                disc_mask_fake_pass_step += float(disc_mask_fake_pass) * w
                mask_iou_step += float(mask_iou.detach().item()) * w
                mask_dice_step += float(mask_dice.detach().item()) * w
                if preview_payload is None:
                    preview_payload = {
                        "target_img": real_g[0].detach().to(torch.float32),
                        "fake_img": fake2[0].detach().to(torch.float32),
                        "target_mask": real_mask_g[0].detach().to(torch.float32),
                        "fake_mask": fake_mask_probs[0].detach().to(torch.float32),
                        "probs": probs[0].detach().to(torch.float32),
                        "target_condition": cond_g[0].detach().to(torch.float32).cpu(),
                        "target_prob": float(target_prob[0].detach().to(torch.float32).item()),
                    }
            if use_scaler:
                scaler_g.step(g_opt)
                scaler_g.update()
            else:
                g_opt.step()

            run_d += float(d_loss_step)
            run_g += float(g_loss_step)
            run_adv += float(adv_loss_step)
            run_tgt += float(target_prob_step)
            run_mask += float(mask_loss_step)
            run_outside += float(outside_loss_step)
            run_wave += float(wave_loss_step)
            run_disc_fake_pass += float(disc_fake_pass_step)
            run_disc_fake_fail += (1.0 - float(disc_fake_pass_step))
            run_disc_mask_fake_pass += float(disc_mask_fake_pass_step)
            run_disc_mask_fake_fail += (1.0 - float(disc_mask_fake_pass_step))
            run_mask_iou += float(mask_iou_step)
            run_mask_dice += float(mask_dice_step)
            steps_done = int(step_idx)

            if step_preview_callback is not None:
                emit = False
                if int(log_every_steps) > 0:
                    emit = ((int(step_idx) % int(log_every_steps) == 0) or (int(step_idx) == int(n_steps)))
                else:
                    emit = (int(step_idx) == int(n_steps))
                if emit and isinstance(preview_payload, dict):
                    try:
                        step_preview_callback(
                            {
                                "epoch": int(epoch),
                                "step": int(step_idx),
                                "steps_per_epoch": int(n_steps),
                                "target_img": preview_payload["target_img"],
                                "fake_img": preview_payload["fake_img"],
                                "target_mask": preview_payload["target_mask"],
                                "fake_mask": preview_payload["fake_mask"],
                                "probs": preview_payload["probs"],
                                "target_condition": preview_payload["target_condition"],
                                "target_prob": float(preview_payload["target_prob"]),
                                "g_loss": float(g_loss_step),
                                "d_loss": float(d_loss_step),
                                "adv_loss": float(adv_loss_step),
                                "mask_loss": float(mask_loss_step),
                                "outside_loss": float(outside_loss_step),
                                "wave_loss": float(wave_loss_step),
                                "target_prob_avg": float(run_tgt / float(max(1, int(step_idx)))),
                                "g_loss_avg": float(run_g / float(max(1, int(step_idx)))),
                                "d_loss_avg": float(run_d / float(max(1, int(step_idx)))),
                            }
                        )
                    except Exception:
                        pass

            if int(log_every_steps) > 0 and ((int(step_idx) % int(log_every_steps) == 0) or (int(step_idx) == int(n_steps))):
                _g = float(run_g / float(step_idx))
                _d = float(run_d / float(step_idx))
                _a = float(run_adv / float(step_idx))
                _t = float(run_tgt / float(step_idx))
                _m = float(run_mask / float(step_idx))
                _o = float(run_outside / float(step_idx))
                _w = float(run_wave / float(step_idx))
                _dp = float(run_disc_fake_pass / float(step_idx))
                _df = float(run_disc_fake_fail / float(step_idx))
                _dmp = float(run_disc_mask_fake_pass / float(step_idx))
                _dmf = float(run_disc_mask_fake_fail / float(step_idx))
                _mi = float(run_mask_iou / float(step_idx))
                _md = float(run_mask_dice / float(step_idx))
                print(
                    f"[generator-train] epoch={epoch}/{max(1, int(epochs))} "
                    f"step={step_idx}/{n_steps} g_loss={_g:.4f} d_loss={_d:.4f} "
                    f"adv={_a:.4f} target_prob={_t:.4f} mask={_m:.4f} outside={_o:.4f} wave={_w:.4f} "
                    f"disc_fake_pass={_dp:.4f} disc_fake_fail={_df:.4f} "
                    f"disc_mask_pass={_dmp:.4f} disc_mask_fail={_dmf:.4f} "
                    f"mask_iou={_mi:.4f} mask_dice={_md:.4f} "
                    f"d_steps={int(disc_steps_per_gen_step)}",
                    flush=True,
                )

        if int(steps_done) <= 0:
            break
        denom = float(max(1, int(steps_done)))
        history.append(
            {
                "epoch": float(epoch),
                "d_loss": float(run_d / denom),
                "g_loss": float(run_g / denom),
                "g_adv_loss": float(run_adv / denom),
                "g_target_prob": float(run_tgt / denom),
                "g_mask_loss": float(run_mask / denom),
                "g_outside_loss": float(run_outside / denom),
                "g_wave_loss": float(run_wave / denom),
                "disc_fake_pass_rate": float(run_disc_fake_pass / denom),
                "disc_fake_fail_rate": float(run_disc_fake_fail / denom),
                "disc_mask_pass_rate": float(run_disc_mask_fake_pass / denom),
                "disc_mask_fail_rate": float(run_disc_mask_fake_fail / denom),
                "mask_iou": float(run_mask_iou / denom),
                "mask_dice": float(run_mask_dice / denom),
                "disc_steps_per_gen_step": float(disc_steps_per_gen_step),
            }
        )
        if stop_now:
            break
    if use_wave_loss and transformer_for_wave is not None:
        if transformer_requires_grad is not None:
            for p, req in zip(transformer_for_wave.parameters(), transformer_requires_grad):
                p.requires_grad_(bool(req))
        transformer_for_wave.train(transformer_was_train)
    return generator, discriminator, history


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
    log_every_steps: int = 0,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    semantic_mix_noise_prob: float = 0.0,
    semantic_mix_noise_std: float = 0.0,
    semantic_mix_blend_min: float = 0.10,
    semantic_mix_blend_max: float = 0.35,
    semantic_mix_class_index: int = -1,
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
    mix_prob = max(0.0, min(1.0, float(semantic_mix_noise_prob)))
    mix_noise_std = max(0.0, float(semantic_mix_noise_std))
    mix_blend_lo = max(0.0, min(1.0, float(semantic_mix_blend_min)))
    mix_blend_hi = max(mix_blend_lo, min(1.0, float(semantic_mix_blend_max)))
    mix_class_idx = int(semantic_mix_class_index)
    try:
        max_label_train = int(y_train.max().item()) if int(y_train.numel()) > 0 else -1
    except Exception:
        max_label_train = -1
    try:
        max_label_val = int(y_val.max().item()) if int(y_val.numel()) > 0 else -1
    except Exception:
        max_label_val = -1
    max_label = max(int(max_label_train), int(max_label_val))
    if int(mix_class_idx) < 0 or int(mix_class_idx) > int(max_label):
        mix_class_idx = -1

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

    stop_now = False
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        opt.zero_grad(set_to_none=True)
        for step_idx, idx in enumerate(
            _iterate_indices(int(x_train.shape[0]), batch_size=batch_size, shuffle=True, rng=rng),
            start=1,
        ):
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass
            if x_train_buf.device.type == device.type:
                idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)
                xb = x_train_buf.index_select(0, idx_t)
                yb = y_train_buf.index_select(0, idx_t)
            else:
                xb = x_train_buf[idx].to(device, non_blocking=True)
                yb = y_train_buf[idx].to(device, non_blocking=True)
            if float(mix_prob) > 0.0 and int(xb.shape[0]) > 0:
                mask = torch.rand((int(xb.shape[0]),), device=xb.device) < float(mix_prob)
                if bool(mask.any().item()):
                    mix_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
                    xb_f = xb.to(torch.float32)
                    src = xb_f.index_select(0, mix_idx)
                    partner_idx = torch.randint(
                        low=0,
                        high=max(1, int(xb_f.shape[0])),
                        size=(int(mix_idx.numel()),),
                        device=xb.device,
                        dtype=torch.long,
                    )
                    partner = xb_f.index_select(0, partner_idx)
                    if float(mix_blend_hi) <= float(mix_blend_lo):
                        alpha = torch.full(
                            (int(mix_idx.numel()), 1, 1, 1),
                            float(mix_blend_lo),
                            device=xb.device,
                            dtype=xb_f.dtype,
                        )
                    else:
                        alpha = torch.empty(
                            (int(mix_idx.numel()), 1, 1, 1),
                            device=xb.device,
                            dtype=xb_f.dtype,
                        ).uniform_(float(mix_blend_lo), float(mix_blend_hi))
                    mixed = ((1.0 - alpha) * src) + (alpha * partner)
                    if float(mix_noise_std) > 0.0:
                        mixed = mixed + (torch.randn_like(mixed) * float(mix_noise_std))
                    mixed = torch.clamp(mixed, 0.0, 1.0)
                    xb_f = xb_f.clone()
                    xb_f.index_copy_(0, mix_idx, mixed)
                    xb = xb_f.to(dtype=xb.dtype)
                    if int(mix_class_idx) >= 0:
                        yb = yb.clone()
                        yb.index_fill_(0, mix_idx, int(mix_class_idx))
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
            if step_preview_callback is not None:
                emit = False
                if int(log_every_steps) > 0:
                    emit = ((int(step_idx) % int(log_every_steps) == 0) or (int(step_idx) == int(num_batches)))
                else:
                    emit = (int(step_idx) == int(num_batches))
                if emit and int(xb.shape[0]) > 0:
                    try:
                        probs0 = torch.softmax(logits[0].detach().to(torch.float32), dim=0)
                        step_preview_callback(
                            {
                                "epoch": int(epoch),
                                "step": int(step_idx),
                                "steps_per_epoch": int(num_batches),
                                "img": xb[0].detach().to(torch.float32),
                                "probs": probs0.detach().to(torch.float32),
                                "target_class": int(yb[0].detach().item()),
                                "batch_loss": float(loss.detach().to(torch.float32).item()),
                                "train_loss": float(total_loss / float(max(1, n_seen))),
                            }
                        )
                    except Exception:
                        pass

        if stop_now:
            break
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


def _eligible_stream_indices_by_chunk_length(streams: Sequence[np.ndarray], chunk_samples: int) -> np.ndarray:
    n_need = max(1, int(chunk_samples))
    keep = [int(i) for i, s in enumerate(streams) if int(np.asarray(s).size) >= n_need]
    return np.asarray(keep, dtype=np.int64)


def _sample_wave_batch(
    streams: Sequence[np.ndarray],
    labels: Sequence[Any],
    batch_size: int,
    chunk_samples: int,
    rng: np.random.Generator,
):
    xb = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    eligible = _eligible_stream_indices_by_chunk_length(streams=streams, chunk_samples=chunk_samples)
    if int(eligible.size) <= 0:
        raise RuntimeError(
            f"No streams are long enough for chunk_samples={int(chunk_samples)} "
            f"(stream_count={len(streams)})."
        )
    first = labels[int(eligible[0])]
    first_arr = np.asarray(first)
    scalar_mode = bool(np.asarray(first_arr).ndim == 0)
    if scalar_mode:
        yb = np.zeros((batch_size,), dtype=np.int64)
        label_dim = 0
    else:
        label_dim = int(np.asarray(first_arr, dtype=np.float32).reshape(-1).size)
        if label_dim <= 0:
            raise RuntimeError("Wave batch labels must have non-empty semantic vectors.")
        yb = np.zeros((batch_size, label_dim), dtype=np.float32)
    for i in range(batch_size):
        pick = int(rng.integers(0, int(eligible.size)))
        idx = int(eligible[pick])
        if scalar_mode:
            yb[i] = int(labels[idx])
        else:
            arr = np.asarray(labels[idx], dtype=np.float32).reshape(-1)
            if int(arr.size) != int(label_dim):
                raise RuntimeError(
                    f"Wave batch label width mismatch: got={int(arr.size)} expected={int(label_dim)}"
                )
            yb[i, :] = arr
        s = streams[idx]
        start = int(rng.integers(0, s.size - chunk_samples + 1))
        xb[i, :] = s[start : start + chunk_samples]
    return xb, yb


class PackedWaveBatchSampler:
    def __init__(
        self,
        streams: Sequence[np.ndarray],
        labels: Optional[Sequence[Any]],
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
        eligible = _eligible_stream_indices_by_chunk_length(streams=streams, chunk_samples=self.chunk_samples)
        if int(eligible.size) <= 0:
            raise ValueError(
                f"PackedWaveBatchSampler requires streams with length >= chunk_samples={self.chunk_samples}; "
                f"got stream_count={len(streams)} eligible=0."
            )
        filtered_streams = [streams[int(i)] for i in eligible.tolist()]
        filtered_labels = None
        if labels is not None:
            filtered_labels = [labels[int(i)] for i in eligible.tolist()]
        self.num_streams = int(len(filtered_streams))

        lengths_np = np.asarray([max(0, int(s.size)) for s in filtered_streams], dtype=np.int64)
        max_len = int(max(1, lengths_np.max(initial=0)))
        data = torch.zeros((self.num_streams, max_len), dtype=torch.float32)
        for i, s in enumerate(filtered_streams):
            if s.size > 0:
                v = torch.from_numpy(np.asarray(s, dtype=np.float32, order="C"))
                data[i, : int(v.shape[0])] = v

        lengths = torch.from_numpy(lengths_np)
        labels_t = None
        if filtered_labels is not None and len(filtered_labels) > 0:
            first_arr = np.asarray(filtered_labels[0])
            scalar_mode = bool(np.asarray(first_arr).ndim == 0)
            if scalar_mode:
                labels_t = torch.as_tensor([int(x) for x in filtered_labels], dtype=torch.long)
            else:
                label_dim = int(np.asarray(first_arr, dtype=np.float32).reshape(-1).size)
                if label_dim <= 0:
                    raise RuntimeError("Packed labels require non-empty semantic vectors.")
                bank = np.zeros((len(filtered_labels), label_dim), dtype=np.float32)
                for i, row in enumerate(filtered_labels):
                    arr = np.asarray(row, dtype=np.float32).reshape(-1)
                    if int(arr.size) != int(label_dim):
                        raise RuntimeError(
                            f"Packed label width mismatch at row {i}: got={int(arr.size)} expected={int(label_dim)}"
                        )
                    bank[i, :] = arr
                labels_t = torch.from_numpy(bank)

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
        xb = torch.gather(picked, dim=1, index=pos)

        yb = None
        if self.labels is not None:
            yb = self.labels.index_select(0, idx)
        return xb, yb


def _build_packed_sampler(
    streams: Sequence[np.ndarray],
    labels: Optional[Sequence[Any]],
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
    eligible = _eligible_stream_indices_by_chunk_length(streams=streams, chunk_samples=chunk_samples)
    if int(eligible.size) <= 0:
        raise RuntimeError(
            f"No streams are long enough for chunk_samples={int(chunk_samples)} "
            f"(stream_count={len(streams)})."
        )
    for i in range(batch_size):
        pick = int(rng.integers(0, int(eligible.size)))
        idx = int(eligible[pick])
        s = streams[idx]
        start = int(rng.integers(0, s.size - chunk_samples + 1))
        xb[i, :] = s[start : start + chunk_samples]
    return xb


def _apply_stride_skew_explicit_batch(x: torch.Tensor, skew_vec: torch.Tensor) -> torch.Tensor:
    bsz, t = x.shape
    if int(t) <= 1:
        return x
    # Coordinate math in fp16/bf16 at long sequence lengths can quantize to
    # out-of-range indices (e.g., 32768 for a 32768-long signal). Keep index
    # calculations in fp32 and clamp indices explicitly.
    coord_dtype = torch.float32
    base = torch.arange(int(t), device=x.device, dtype=coord_dtype).unsqueeze(0).expand(int(bsz), -1)
    center = float(int(t) - 1) * 0.5
    skew = skew_vec.to(device=x.device, dtype=coord_dtype).reshape(-1, 1)
    if int(skew.shape[0]) == 1 and int(bsz) > 1:
        skew = skew.expand(int(bsz), 1)
    elif int(skew.shape[0]) != int(bsz):
        raise ValueError(
            f"skew tensor batch mismatch: got {int(skew.shape[0])}, expected {int(bsz)}"
        )
    scale = torch.clamp(1.0 + skew, 0.5, 1.5)
    src = ((base - center) * scale) + center
    src = torch.clamp(src, 0.0, float(int(t) - 1))
    src0 = torch.floor(src).to(torch.long)
    src0 = torch.clamp(src0, min=0, max=int(t) - 1)
    src1 = torch.clamp(src0 + 1, max=int(t) - 1)
    alpha = (src - src0.to(dtype=coord_dtype)).to(dtype=x.dtype)
    x0 = torch.gather(x, dim=1, index=src0)
    x1 = torch.gather(x, dim=1, index=src1)
    return ((1.0 - alpha) * x0) + (alpha * x1)


def _apply_stride_skew_batch(
    x: torch.Tensor,
    s_vec: torch.Tensor,
    max_skew: float,
    return_skew: bool = False,
):
    bsz, _ = x.shape
    if float(max_skew) <= 0.0:
        if bool(return_skew):
            z = torch.zeros((int(bsz), 1), device=x.device, dtype=x.dtype)
            return x, z
        return x
    s_loc = s_vec.to(device=x.device, dtype=torch.float32)
    skew = ((torch.rand((int(bsz), 1), device=x.device, dtype=torch.float32) * 2.0) - 1.0) * (
        float(max_skew) * s_loc
    )
    out = _apply_stride_skew_explicit_batch(x=x, skew_vec=skew)
    if bool(return_skew):
        return out, skew.to(dtype=x.dtype)
    return out


def _bitwindow_shape_for_wave_len(wave_len: int, cfg: RenderConfig):
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    base_len = max(1, (max(1, int(wave_len)) + downsample - 1) // downsample)
    if int(cfg.max_points) > 0:
        base_len = min(base_len, int(cfg.max_points))
    _, mapping = COLOR_MODE_MAP.get(cfg.color_mode, COLOR_MODE_MAP[COLOR_MODES[0][0]])
    special = bool(mapping.get("__special__") == "single_source_rgb_stride")
    max_len = (base_len + 2) // 3 if special else base_len
    total = int(math.ceil(float(max_len) / float(width)) * width)
    height = max(1, total // width)
    return {
        "downsample": int(downsample),
        "width": int(width),
        "base_len": int(base_len),
        "max_len": int(max_len),
        "total": int(total),
        "height": int(height),
        "mapping": mapping,
        "special": special,
    }


def _bitwindow_canvas_to_base_sequence_batch(canvas: torch.Tensor, shape_info: Dict, empty_fill: float) -> torch.Tensor:
    mapping = shape_info["mapping"]
    special = bool(shape_info["special"])
    base_len = int(shape_info["base_len"])
    max_len = int(shape_info["max_len"])
    bsz = int(canvas.shape[0])
    flat_ch = canvas.reshape(bsz, int(canvas.shape[1]), -1)

    if not special:
        streams = []
        if mapping.get("R") is not None:
            streams.append(flat_ch[:, 0, :])
        if mapping.get("G") is not None:
            streams.append(flat_ch[:, 1, :])
        if mapping.get("B") is not None:
            streams.append(flat_ch[:, 2, :])
        if len(streams) <= 0:
            flat = torch.full(
                (bsz, int(shape_info["total"])),
                float(empty_fill),
                device=canvas.device,
                dtype=canvas.dtype,
            )
        elif len(streams) == 1:
            flat = streams[0]
        else:
            flat = torch.stack(streams, dim=0).mean(dim=0)
        if int(flat.shape[1]) < int(max_len):
            pad_n = int(max_len) - int(flat.shape[1])
            flat = F.pad(flat, (0, int(pad_n)), value=float(empty_fill))
        return flat[:, : int(base_len)]

    r = flat_ch[:, 0, :]
    g = flat_ch[:, 1, :]
    b = flat_ch[:, 2, :]
    if int(r.shape[1]) < int(max_len):
        r = F.pad(r, (0, int(max_len - int(r.shape[1]))), value=float(empty_fill))
    if int(g.shape[1]) < int(max_len):
        g = F.pad(g, (0, int(max_len - int(g.shape[1]))), value=float(empty_fill))
    if int(b.shape[1]) < int(max_len):
        b = F.pad(b, (0, int(max_len - int(b.shape[1]))), value=float(empty_fill))
    idx = torch.arange(int(base_len), device=canvas.device, dtype=torch.long)
    src_idx = torch.div(idx, 3, rounding_mode="floor")
    mod = torch.remainder(idx, 3)
    gather_idx = src_idx.unsqueeze(0).expand(bsz, -1)
    rv = torch.gather(r, dim=1, index=gather_idx)
    gv = torch.gather(g, dim=1, index=gather_idx)
    bv = torch.gather(b, dim=1, index=gather_idx)
    return torch.where(mod.unsqueeze(0) == 0, rv, torch.where(mod.unsqueeze(0) == 1, gv, bv))


def _degrade_bitwindow_image_plane(canvas: torch.Tensor, s_vec: torch.Tensor, stride_skew_max: float) -> torch.Tensor:
    x = canvas
    bsz = int(x.shape[0])
    # Image-plane skew (horizontal shear by row) only.
    if float(stride_skew_max) > 0.0:
        shear = ((torch.rand((bsz, 1), device=x.device, dtype=x.dtype) * 2.0) - 1.0) * (
            float(stride_skew_max) * s_vec
        )
        theta = torch.zeros((bsz, 2, 3), device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 0, 1] = shear.squeeze(1)
        theta[:, 1, 1] = 1.0
        grid = F.affine_grid(theta, size=x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Image-plane blur.
    blur_k_choices = (1, 3, 5, 7)
    blur_rank = torch.round(3.0 * s_vec.squeeze(1).to(torch.float32)).to(torch.long)
    blur_idx = torch.clamp(blur_rank, min=0, max=len(blur_k_choices) - 1)
    blur_bank = []
    for k in blur_k_choices:
        if int(k) <= 1:
            blur_bank.append(x)
        else:
            blur_bank.append(F.avg_pool2d(x, kernel_size=int(k), stride=1, padding=(int(k) // 2)))
    blur_stack = torch.stack(blur_bank, dim=1)
    pick = torch.arange(bsz, device=x.device, dtype=torch.long)
    x_blur = blur_stack[pick, blur_idx, :, :, :]
    blur_mix = 0.05 + (0.80 * s_vec.view(bsz, 1, 1, 1))
    x = ((1.0 - blur_mix) * x) + (blur_mix * x_blur)
    return torch.clamp(x, 0.0, 1.0)


def _degrade_wave_batch_target_bits(
    x_clean: torch.Tensor,
    strength,
    noise_std_min: float,
    noise_std_max: float,
    dropout_max: float,
    quant_bits_min: int,
    quant_bits_max: int,
    cfg: RenderConfig,
    sample_bits: int,
    bit_low: int,
    bit_high: int,
    mode: str = "full",
    stride_skew_max: float = 0.0,
    return_meta: bool = False,
):
    mode_key = str(mode).strip().lower()
    if mode_key not in ("full", "blur_stride"):
        mode_key = "full"

    if torch.is_tensor(strength):
        s_vec = strength.detach().to(device=x_clean.device, dtype=torch.float32).reshape(-1, 1)
        if int(s_vec.shape[0]) == 1 and int(x_clean.shape[0]) > 1:
            s_vec = s_vec.expand(int(x_clean.shape[0]), 1)
        elif int(s_vec.shape[0]) != int(x_clean.shape[0]):
            raise ValueError(
                f"strength tensor batch mismatch: got {int(s_vec.shape[0])}, expected {int(x_clean.shape[0])}"
            )
    else:
        s_scalar = float(max(0.0, min(1.0, float(strength))))
        s_vec = torch.full((int(x_clean.shape[0]), 1), s_scalar, device=x_clean.device, dtype=torch.float32)
    s_vec = torch.clamp(s_vec, 0.0, 1.0).to(dtype=x_clean.dtype)

    lo, hi, depth = normalize_bit_window(int(bit_low), int(bit_high), int(sample_bits))
    full_mask = (1 << int(sample_bits)) - 1
    window_mask = (1 << int(depth)) - 1
    q_min = -(1 << (int(sample_bits) - 1))
    q_max = (1 << (int(sample_bits) - 1)) - 1
    q_scale = float(1 << (int(sample_bits) - 1))

    x = torch.clamp(x_clean, -1.0, 1.0)
    q_signed = torch.round(x * q_scale)
    q_signed = torch.clamp(q_signed, float(q_min), float(q_max)).to(torch.int64)
    unsigned = torch.bitwise_and(q_signed, full_mask)
    window = torch.bitwise_and(torch.bitwise_right_shift(unsigned, int(lo)), int(window_mask)).to(torch.float32)
    if int(depth) == 1:
        window01 = window
    else:
        window01 = window / float(window_mask)

    # Apply degradation directly in target-bit sequence domain so stride skew is
    # temporal (real resampling skew), not image-plane affine shear.
    window_wave = torch.clamp((window01 * 2.0) - 1.0, -1.0, 1.0).to(dtype=x_clean.dtype)
    degraded_pack = _degrade_wave_batch(
        x_clean=window_wave,
        strength=s_vec,
        noise_std_min=float(noise_std_min),
        noise_std_max=float(noise_std_max),
        dropout_max=float(dropout_max),
        quant_bits_min=int(quant_bits_min),
        quant_bits_max=int(quant_bits_max),
        mode=str(mode_key),
        stride_skew_max=float(stride_skew_max),
        return_meta=bool(return_meta),
    )
    if isinstance(degraded_pack, tuple):
        degraded_wave, degrade_meta = degraded_pack
    else:
        degraded_wave, degrade_meta = degraded_pack, None
    degraded01 = torch.clamp((degraded_wave.to(torch.float32) * 0.5) + 0.5, 0.0, 1.0)

    downsample = max(1, int(cfg.downsample))
    base_len = max(1, (max(1, int(x_clean.shape[1])) + downsample - 1) // downsample)
    if int(cfg.max_points) > 0:
        base_len = min(base_len, int(cfg.max_points))
    n_use = min(int(base_len), int(degraded01.shape[1]))
    if n_use <= 0:
        if bool(return_meta):
            z = torch.zeros((int(x_clean.shape[0]), 1), device=x_clean.device, dtype=x_clean.dtype)
            return x_clean, {"skew_applied": z}
        return x_clean
    idx = (torch.arange(n_use, device=x_clean.device, dtype=torch.long) * int(downsample)).to(torch.long)
    idx = idx[idx < int(x_clean.shape[1])]
    n_use = int(idx.shape[0])
    if n_use <= 0:
        if bool(return_meta):
            z = torch.zeros((int(x_clean.shape[0]), 1), device=x_clean.device, dtype=x_clean.dtype)
            return x_clean, {"skew_applied": z}
        return x_clean
    desired01 = torch.clamp(degraded01[:, :n_use], 0.0, 1.0)

    if int(depth) == 1:
        desired = (desired01 >= 0.5).to(torch.int64)
    else:
        desired = torch.round(desired01 * float(window_mask)).to(torch.int64)
        desired = torch.clamp(desired, min=0, max=int(window_mask))
    desired_full = torch.bitwise_left_shift(desired, int(lo))

    unsigned_sel = torch.gather(unsigned, dim=1, index=idx.unsqueeze(0).expand(int(x_clean.shape[0]), -1))
    clear_mask = torch.tensor(
        int(full_mask ^ (int(window_mask) << int(lo))),
        device=x_clean.device,
        dtype=torch.int64,
    )
    unsigned_new = torch.bitwise_and(unsigned_sel, clear_mask)
    unsigned_new = torch.bitwise_or(unsigned_new, desired_full)
    unsigned_out = unsigned.clone()
    unsigned_out.scatter_(1, idx.unsqueeze(0).expand(int(x_clean.shape[0]), -1), unsigned_new)
    sign_cut = 1 << (int(sample_bits) - 1)
    signed_out = torch.where(unsigned_out >= sign_cut, unsigned_out - (1 << int(sample_bits)), unsigned_out)
    out = torch.clamp(signed_out.to(dtype=x_clean.dtype) / q_scale, -1.0, 1.0)
    if not bool(return_meta):
        return out
    skew_applied = None
    if isinstance(degrade_meta, dict):
        skew_applied = degrade_meta.get("skew_applied", None)
    if not torch.is_tensor(skew_applied):
        skew_applied = torch.zeros((int(x_clean.shape[0]), 1), device=x_clean.device, dtype=x_clean.dtype)
    else:
        skew_applied = skew_applied.to(device=x_clean.device, dtype=x_clean.dtype).reshape(int(x_clean.shape[0]), 1)
    return out, {"skew_applied": skew_applied}


def _degrade_wave_batch(
    x_clean: torch.Tensor,
    strength,
    noise_std_min: float,
    noise_std_max: float,
    dropout_max: float,
    quant_bits_min: int,
    quant_bits_max: int,
    mode: str = "full",
    stride_skew_max: float = 0.0,
    return_meta: bool = False,
):
    x = x_clean
    bsz, t = x.shape
    if torch.is_tensor(strength):
        s_vec = strength.detach().to(device=x.device, dtype=torch.float32).reshape(-1, 1)
        if int(s_vec.shape[0]) == 1 and int(bsz) > 1:
            s_vec = s_vec.expand(int(bsz), 1)
        elif int(s_vec.shape[0]) != int(bsz):
            raise ValueError(
                f"strength tensor batch mismatch: got {int(s_vec.shape[0])}, expected {int(bsz)}"
            )
    else:
        s_scalar = float(max(0.0, min(1.0, float(strength))))
        s_vec = torch.full((int(bsz), 1), s_scalar, device=x.device, dtype=torch.float32)
    s_vec = torch.clamp(s_vec, 0.0, 1.0).to(dtype=x.dtype)
    mode_key = str(mode).strip().lower()
    if mode_key not in ("full", "blur_stride"):
        mode_key = "full"
    skew_applied = torch.zeros((int(bsz), 1), device=x.device, dtype=x.dtype)

    if mode_key == "blur_stride":
        x, skew_applied = _apply_stride_skew_batch(
            x=x,
            s_vec=s_vec,
            max_skew=float(stride_skew_max),
            return_skew=True,
        )
        blur_k_choices = (3, 5, 7, 9, 11)
        blur_rank = torch.round(1.0 + (4.0 * s_vec.squeeze(1).to(torch.float32))).to(torch.long)
        blur_idx = torch.clamp(blur_rank, min=1, max=len(blur_k_choices)) - 1
        x_b = x.unsqueeze(1)
        blur_bank = []
        for k in blur_k_choices:
            blur_bank.append(F.avg_pool1d(x_b, kernel_size=int(k), stride=1, padding=(int(k) // 2)).squeeze(1))
        blur_stack = torch.stack(blur_bank, dim=1)
        pick = torch.arange(bsz, device=x.device, dtype=torch.long)
        x_blur = blur_stack[pick, blur_idx, :]
        blur_mix = 0.10 + (0.70 * s_vec)
        x = ((1.0 - blur_mix) * x) + (blur_mix * x_blur)
        out = torch.clamp(x, -1.0, 1.0)
        if bool(return_meta):
            return out, {"skew_applied": skew_applied}
        return out

    # Broadband additive noise.
    noise_std = float(noise_std_min) + ((float(noise_std_max) - float(noise_std_min)) * s_vec)
    x = x + (torch.randn_like(x) * noise_std)

    # Random gain jitter and partial dropout emulate capture/channel variation.
    gain_sigma = 0.03 + (0.20 * s_vec)
    gain = torch.exp(torch.randn((bsz, 1), device=x.device, dtype=x.dtype) * gain_sigma)
    x = x * gain
    drop_p = torch.clamp(float(dropout_max) * s_vec, 0.0, 0.95)
    keep = (torch.rand((bsz, t), device=x.device, dtype=x.dtype) > drop_p).to(x.dtype)
    x = x * keep

    # Short contiguous occlusion band with per-sample lengths.
    seg_len = torch.round((0.005 + (0.045 * s_vec.squeeze(1))) * float(t)).to(torch.long)
    seg_len = torch.clamp(seg_len, min=1, max=max(1, int(t)))
    if int(t) > 1:
        max_start = torch.clamp(int(t) - seg_len + 1, min=1)
        starts = torch.floor(
            torch.rand((bsz,), device=x.device, dtype=x.dtype) * max_start.to(dtype=x.dtype)
        ).to(torch.long)
        idx = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0)
        seg_keep = ((idx < starts.unsqueeze(1)) | (idx >= (starts + seg_len).unsqueeze(1))).to(x.dtype)
        x = x * seg_keep

    # Mild low-pass blur, vectorized with per-sample kernel picks.
    blur_k_choices = (3, 5, 7, 9, 11)
    blur_rank = torch.round(1.0 + (4.0 * s_vec.squeeze(1).to(torch.float32))).to(torch.long)
    blur_idx = torch.clamp(blur_rank, min=1, max=len(blur_k_choices)) - 1
    x_b = x.unsqueeze(1)
    blur_bank = []
    for k in blur_k_choices:
        blur_bank.append(F.avg_pool1d(x_b, kernel_size=int(k), stride=1, padding=(int(k) // 2)).squeeze(1))
    blur_stack = torch.stack(blur_bank, dim=1)
    pick = torch.arange(bsz, device=x.device, dtype=torch.long)
    x_blur = blur_stack[pick, blur_idx, :]
    blur_mix = 0.10 + (0.70 * s_vec)
    x = ((1.0 - blur_mix) * x) + (blur_mix * x_blur)

    # Bit-depth reduction with per-sample quantization.
    qb_lo = max(2, int(quant_bits_min))
    qb_hi = max(qb_lo, int(quant_bits_max))
    q_bits_f = float(qb_hi) + (float(qb_lo - qb_hi) * s_vec.to(torch.float32))
    q_bits = torch.clamp(torch.round(q_bits_f), min=2.0)
    levels = torch.exp2(q_bits) - 1.0
    levels = levels.to(dtype=x.dtype)
    x01 = torch.clamp((x * 0.5) + 0.5, 0.0, 1.0)
    xq = torch.round(x01 * levels) / levels
    xq = (xq * 2.0) - 1.0
    q_mix = 0.10 + (0.75 * s_vec)
    x = ((1.0 - q_mix) * x) + (q_mix * xq)

    if float(stride_skew_max) > 0.0:
        x, skew_applied = _apply_stride_skew_batch(
            x=x,
            s_vec=s_vec,
            max_skew=float(stride_skew_max),
            return_skew=True,
        )

    out = torch.clamp(x, -1.0, 1.0)
    if bool(return_meta):
        return out, {"skew_applied": skew_applied}
    return out


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
    train_target_labels: Optional[Sequence[Any]],
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
    degrade_vectorized_window: int = 4,
    degrade_mode: str = "full",
    degrade_stride_skew_max: float = 0.0,
    degrade_domain: str = "waveform",
    deskew_prefilter_weight: float = 0.20,
    deskew_residual_weight: float = 0.10,
    deskew_pre_token_mix: float = 0.08,
    deskew_post_token_mix: float = 0.08,
    deskew_token_residual_weight: float = 0.01,
    score_rank_weight: float = 0.40,
    score_rank_margin: float = 0.02,
    score_spurious_weight: float = 0.35,
    score_spurious_threshold: float = 0.35,
    score_spurious_temp: float = 0.20,
    score_spurious_margin: float = 0.00,
    entropy_penalty_weight: float = 0.60,
    high_bit_penalty_weight: float = 0.50,
    low_bit_penalty_weight: float = 0.08,
    wave_l1_weight: float = 1.00,
    score_target_weight: float = 1.50,
    score_target_margin: float = 0.0,
    seed: int = 0,
    log_every_steps: int = 0,
    step_preview_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_requested: Optional[Callable[[], bool]] = None,
    aux_filter_input_builder: Optional[Callable[[Dict[str, Any]], Dict[str, Dict[str, Any]]]] = None,
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
    if bool(visualize_status):
        print(
            "[transformer-viz] standalone transformer preview removed; "
            "using shared stage preview callbacks only",
            flush=True,
        )
    deskew_prefilter_weight = max(0.0, float(deskew_prefilter_weight))
    deskew_residual_weight = max(0.0, float(deskew_residual_weight))
    deskew_pre_token_mix = max(0.0, min(0.50, float(deskew_pre_token_mix)))
    deskew_post_token_mix = max(0.0, min(0.50, float(deskew_post_token_mix)))
    deskew_token_residual_weight = max(0.0, float(deskew_token_residual_weight))
    score_rank_weight = max(0.0, float(score_rank_weight))
    score_rank_margin = max(0.0, float(score_rank_margin))
    score_spurious_weight = max(0.0, float(score_spurious_weight))
    score_spurious_threshold = max(0.0, min(1.0, float(score_spurious_threshold)))
    score_spurious_temp = max(1e-4, float(score_spurious_temp))
    score_spurious_margin = max(0.0, float(score_spurious_margin))
    entropy_penalty_weight = max(0.0, float(entropy_penalty_weight))
    high_bit_penalty_weight = max(0.0, float(high_bit_penalty_weight))
    low_bit_penalty_weight = max(0.0, float(low_bit_penalty_weight))
    wave_l1_weight = max(0.0, float(wave_l1_weight))
    score_target_weight = max(0.0, float(score_target_weight))
    score_target_margin = max(0.0, float(score_target_margin))
    degrade_vectorized_window = max(1, int(degrade_vectorized_window))
    degrade_mode = str(degrade_mode).strip().lower()
    if degrade_mode not in ("full", "blur_stride"):
        degrade_mode = "full"
    degrade_stride_skew_max = max(0.0, float(degrade_stride_skew_max))
    degrade_domain = str(degrade_domain).strip().lower()
    if degrade_domain not in ("waveform", "bitwindow"):
        degrade_domain = "waveform"
    if degrade_domain == "bitwindow" and (not bool(cfg.bitmask_enable)):
        print(
            "[transformer-feature-config] degrade_domain=bitwindow requested but cfg.bitmask_enable=0; "
            "falling back to waveform",
            flush=True,
        )
        degrade_domain = "waveform"
    aux_bundle_module = transformer
    if (not hasattr(aux_bundle_module, "aux_filter_bundles")) and hasattr(transformer, "module"):
        aux_bundle_module = getattr(transformer, "module")
    has_deskew_bundle = bool(
        hasattr(aux_bundle_module, "aux_filter_bundles")
        and isinstance(getattr(aux_bundle_module, "aux_filter_bundles"), nn.ModuleDict)
        and ("deskew" in getattr(aux_bundle_module, "aux_filter_bundles"))
    )
    deskew_aux_active = bool(
        bool(degrade_inputs)
        and (str(degrade_mode) == "blur_stride")
        and (float(degrade_stride_skew_max) > 0.0)
        and bool(has_deskew_bundle)
    )
    if not bool(deskew_aux_active):
        if (
            ((float(deskew_prefilter_weight) > 0.0) or (float(deskew_residual_weight) > 0.0))
            and (not bool(has_deskew_bundle))
        ):
            print(
                "[transformer-feature-config] deskew loss weights requested but deskew bundle is not active; "
                "deskew aux losses disabled.",
                flush=True,
            )
        deskew_prefilter_weight = 0.0
        deskew_residual_weight = 0.0
        deskew_pre_token_mix = 0.0
        deskew_post_token_mix = 0.0
        deskew_token_residual_weight = 0.0
    target_labels = None
    target_supervision = False
    if train_target_labels is not None:
        if len(train_target_labels) == len(train_streams):
            target_dim = 0
            for row in train_target_labels:
                if row is None:
                    continue
                arr = np.asarray(row, dtype=np.float32).reshape(-1)
                if int(arr.size) > 0:
                    target_dim = int(arr.size)
                    break
            if int(target_dim) > 0:
                target_rows: List[np.ndarray] = []
                for i, row in enumerate(train_target_labels):
                    if row is None:
                        target_rows.append(np.zeros((int(target_dim),), dtype=np.float32))
                        continue
                    arr = np.asarray(row, dtype=np.float32).reshape(-1)
                    if int(arr.size) != int(target_dim):
                        raise RuntimeError(
                            f"Transformer target label width mismatch at stream {i}: "
                            f"got={int(arr.size)} expected={int(target_dim)}"
                        )
                    target_rows.append(arr.astype(np.float32, copy=False))
                target_labels = target_rows
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
        f"target_streams={int(sum(1 for x in (target_labels or []) if float(np.asarray(x, dtype=np.float32).sum()) > 0.0))} "
        f"score_topk={int(score_topk)} score_threshold={float(score_threshold):.4f} "
        f"score_w_topk={float(score_w_topk):.3f} score_w_cov={float(score_w_cov):.3f} score_w_mean={float(score_w_mean):.3f} "
        f"signal_rank_w={score_rank_weight:.3f} signal_rank_margin={score_rank_margin:.3f} "
        f"signal_spurious_w={score_spurious_weight:.3f} signal_spurious_thr={score_spurious_threshold:.3f} "
        f"signal_spurious_temp={score_spurious_temp:.3f} signal_spurious_margin={score_spurious_margin:.3f} "
        f"degrade_domain={degrade_domain} degrade_vec_window={int(degrade_vectorized_window)} "
        f"degrade_mode={degrade_mode} degrade_stride_skew_max={degrade_stride_skew_max:.3f} "
        f"deskew_aux=({deskew_prefilter_weight:.3f},{deskew_residual_weight:.3f}) "
        f"deskew_token_mix=({deskew_pre_token_mix:.3f},{deskew_post_token_mix:.3f}) "
        f"deskew_token_l2={deskew_token_residual_weight:.4f}",
        flush=True,
    )
    if bool(degrade_inputs) and (degrade_domain == "bitwindow") and (degrade_mode == "blur_stride"):
        print(
            "[transformer-feature-config] bitwindow sequence-domain degradation active: "
            "target bit-window sequence is transformed (true stride-skew + blur), non-target bits are preserved.",
            flush=True,
        )

    stop_now = False
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
        running_score_rank_gap = torch.zeros((), dtype=torch.float32, device=device)
        running_score_spurious_gap = torch.zeros((), dtype=torch.float32, device=device)
        running_signal_gap = torch.zeros((), dtype=torch.float32, device=device)
        running_aux_bundle_loss = torch.zeros((), dtype=torch.float32, device=device)
        running_prefilter_loss = torch.zeros((), dtype=torch.float32, device=device)
        running_post_residual_loss = torch.zeros((), dtype=torch.float32, device=device)
        running_skew_mae = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_target_abs = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_pred_abs = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_applied_abs = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_remaining_abs = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_post_abs = torch.zeros((), dtype=torch.float32, device=device)
        running_deskew_conf = torch.zeros((), dtype=torch.float32, device=device)
        running_degrade = 0.0
        last_xclean_preview = None
        last_xb_preview = None
        last_xh_preview = None
        last_target_condition_preview = None
        last_deskew_preview = None
        t_sample = 0.0
        t_forward = 0.0
        t_backward = 0.0
        epoch_t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        total_train_steps = max(1, int(epochs) * int(num_steps))
        epoch_step_offset = (epoch - 1) * num_steps

        def _sample_clean_targets(total_batch: int):
            total_batch = max(1, int(total_batch))
            if train_sampler is not None:
                x_local, y_local = train_sampler.sample(batch_size=total_batch)
                if x_local.device != device:
                    x_local = x_local.to(device, non_blocking=True)
                if y_local is not None and y_local.device != device:
                    y_local = y_local.to(device, non_blocking=True)
                return x_local, y_local
            if target_labels is not None:
                xb_np, yb_np = _sample_wave_batch(
                    streams=train_streams,
                    labels=target_labels,
                    batch_size=total_batch,
                    chunk_samples=chunk_samples,
                    rng=rng,
                )
                y_local = torch.from_numpy(yb_np).to(device, non_blocking=True)
            else:
                xb_np = _sample_wave_batch_unlabeled(
                    streams=train_streams,
                    batch_size=total_batch,
                    chunk_samples=chunk_samples,
                    rng=rng,
                )
                y_local = None
            x_local = _to_device_wave_batch(xb_np, device=device, pin_memory=pin_memory)
            return x_local, y_local

        def _prepare_vectorized_block(step_start: int, n_steps_block: int):
            n_steps_block = max(0, int(n_steps_block))
            if n_steps_block <= 0:
                return []
            total_batch = int(n_steps_block) * int(batch_size)
            x_clean_block, y_target_block = _sample_clean_targets(total_batch=total_batch)
            global_steps = torch.arange(
                int(epoch_step_offset + step_start),
                int(epoch_step_offset + step_start + n_steps_block),
                device=device,
                dtype=torch.float32,
            )
            progress = global_steps / float(max(1, total_train_steps - 1))
            strengths = float(degrade_min_strength) + (
                float(degrade_max_strength) - float(degrade_min_strength)
            ) * progress
            strengths = torch.clamp(strengths, 0.0, 1.0)
            strengths_batch = strengths.repeat_interleave(int(batch_size))
            degrade_meta_block: Dict[str, Any] = {}
            if bool(degrade_inputs):
                if str(degrade_domain) == "bitwindow":
                    degrade_pack = _degrade_wave_batch_target_bits(
                        x_clean=x_clean_block,
                        strength=strengths_batch,
                        noise_std_min=float(degrade_noise_std_min),
                        noise_std_max=float(degrade_noise_std_max),
                        dropout_max=float(degrade_dropout_max),
                        quant_bits_min=int(degrade_quant_bits_min),
                        quant_bits_max=int(degrade_quant_bits_max),
                        cfg=cfg,
                        sample_bits=int(sample_bits),
                        bit_low=int(cfg.bitmask_low),
                        bit_high=int(cfg.bitmask_high),
                        mode=str(degrade_mode),
                        stride_skew_max=float(degrade_stride_skew_max),
                        return_meta=True,
                    )
                else:
                    degrade_pack = _degrade_wave_batch(
                        x_clean=x_clean_block,
                        strength=strengths_batch,
                        noise_std_min=float(degrade_noise_std_min),
                        noise_std_max=float(degrade_noise_std_max),
                        dropout_max=float(degrade_dropout_max),
                        quant_bits_min=int(degrade_quant_bits_min),
                        quant_bits_max=int(degrade_quant_bits_max),
                        mode=str(degrade_mode),
                        stride_skew_max=float(degrade_stride_skew_max),
                        return_meta=True,
                    )
                if isinstance(degrade_pack, tuple):
                    xb_block, degrade_meta_block = degrade_pack
                else:
                    xb_block = degrade_pack
            else:
                xb_block = x_clean_block
            out = []
            for j in range(n_steps_block):
                a = int(j) * int(batch_size)
                b = a + int(batch_size)
                y_j = None if y_target_block is None else y_target_block[a:b]
                meta_j: Dict[str, Any] = {}
                skew_block = degrade_meta_block.get("skew_applied", None)
                if torch.is_tensor(skew_block):
                    meta_j["skew_applied"] = skew_block[a:b].detach()
                out.append((x_clean_block[a:b], xb_block[a:b], y_j, float(strengths[j].item()), meta_j))
            return out

        prep_queue = deque()
        first_take = min(int(num_steps), int(degrade_vectorized_window))
        for item in _prepare_vectorized_block(step_start=0, n_steps_block=first_take):
            prep_queue.append(item)
        next_step_to_prepare = int(first_take)

        for step_idx in range(num_steps):
            if stop_requested is not None:
                try:
                    if bool(stop_requested()):
                        stop_now = True
                        break
                except Exception:
                    pass
            step_t0 = time.perf_counter()
            if len(prep_queue) <= 0:
                for item in _prepare_vectorized_block(step_start=step_idx, n_steps_block=1):
                    prep_queue.append(item)
            x_clean, xb, y_target, degrade_strength, degrade_meta = prep_queue.popleft()
            if len(prep_queue) <= 0 and next_step_to_prepare < int(num_steps):
                take = min(int(degrade_vectorized_window), int(num_steps) - int(next_step_to_prepare))
                for item in _prepare_vectorized_block(step_start=next_step_to_prepare, n_steps_block=take):
                    prep_queue.append(item)
                next_step_to_prepare += int(take)
            step_t1 = time.perf_counter()

            filter_inputs_step: Dict[str, Dict[str, Any]] = {}
            target_skew_step = None
            if bool(deskew_aux_active):
                target_skew = None
                if isinstance(degrade_meta, dict):
                    target_skew = degrade_meta.get("skew_applied", None)
                if torch.is_tensor(target_skew):
                    target_skew = target_skew.to(device=xb.device, dtype=torch.float32).reshape(-1, 1)
                else:
                    target_skew = torch.zeros((int(xb.shape[0]), 1), dtype=torch.float32, device=xb.device)
                target_skew_step = target_skew
                filter_inputs_step["deskew"] = {
                    "target_skew": target_skew,
                    "max_skew": float(degrade_stride_skew_max),
                    "prefilter_weight": float(deskew_prefilter_weight),
                    "post_weight": float(deskew_residual_weight),
                    "pre_token_mix": float(deskew_pre_token_mix),
                    "post_token_mix": float(deskew_post_token_mix),
                    "token_residual_weight": float(deskew_token_residual_weight),
                    "degrade_strength": torch.full(
                        (int(xb.shape[0]), 1),
                        float(degrade_strength),
                        dtype=torch.float32,
                        device=xb.device,
                    ),
                }
            if aux_filter_input_builder is not None:
                extra_inputs = None
                try:
                    extra_inputs = aux_filter_input_builder(
                        {
                            "epoch": int(epoch),
                            "step": int(step_idx + 1),
                            "steps_per_epoch": int(num_steps),
                            "x_clean": x_clean,
                            "x_in": xb,
                            "target_condition": y_target,
                            "degrade_strength": float(degrade_strength),
                            "degrade_meta": degrade_meta,
                            "deskew_aux_active": bool(deskew_aux_active),
                        }
                    )
                except Exception:
                    extra_inputs = None
                if isinstance(extra_inputs, dict):
                    for bundle_name, task_in in extra_inputs.items():
                        if not isinstance(task_in, dict):
                            continue
                        key = str(bundle_name).strip().lower()
                        row = filter_inputs_step.setdefault(key, {})
                        row.update(task_in)

            transformer_aux: Dict[str, Any] = {}
            with _autocast_context(device=device, use_amp=use_amp, amp_dtype_t=amp_dtype_t):
                try:
                    xh_out = transformer(xb, return_aux=True, filter_inputs=filter_inputs_step)
                except TypeError as e:
                    err_msg = str(e)
                    if ("return_aux" not in err_msg) and ("filter_inputs" not in err_msg):
                        raise
                    xh_out = transformer(xb)
            if isinstance(xh_out, tuple):
                xh = xh_out[0]
                if len(xh_out) >= 2 and isinstance(xh_out[1], dict):
                    transformer_aux = dict(xh_out[1])
            else:
                xh = xh_out
            xh_f = xh.float()
            x_clean_f = x_clean.float()
            denoise_l1 = F.l1_loss(xh_f, x_clean_f)
            aux_bundle_loss = torch.zeros((), dtype=torch.float32, device=device)
            prefilter_loss = torch.zeros((), dtype=torch.float32, device=device)
            post_residual_loss = torch.zeros((), dtype=torch.float32, device=device)
            skew_mae = torch.zeros((), dtype=torch.float32, device=device)
            deskew_target_abs = torch.zeros((), dtype=torch.float32, device=device)
            deskew_pred_abs = torch.zeros((), dtype=torch.float32, device=device)
            deskew_applied_abs = torch.zeros((), dtype=torch.float32, device=device)
            deskew_remaining_abs = torch.zeros((), dtype=torch.float32, device=device)
            deskew_post_abs = torch.zeros((), dtype=torch.float32, device=device)
            deskew_conf = torch.zeros((), dtype=torch.float32, device=device)
            deskew_pred_skew_t = None
            deskew_applied_skew_t = None
            deskew_confidence_t = None
            deskew_post_residual_t = None
            if isinstance(transformer_aux, dict):
                bundle_rows = transformer_aux.get("filter_bundles", {})
                if isinstance(bundle_rows, dict):
                    for bundle_name, bundle_row in bundle_rows.items():
                        if not isinstance(bundle_row, dict):
                            continue
                        losses = bundle_row.get("losses", {})
                        if not isinstance(losses, dict):
                            continue
                        for loss_name, loss_value in losses.items():
                            if not torch.is_tensor(loss_value):
                                continue
                            v = loss_value.to(device=device, dtype=torch.float32)
                            if str(loss_name).endswith("_loss") and (not str(loss_name).endswith("_loss_raw")):
                                aux_bundle_loss = aux_bundle_loss + v
                        if str(bundle_name).strip().lower() == "deskew":
                            v = losses.get("prefilter_loss", None)
                            if torch.is_tensor(v):
                                prefilter_loss = v.to(device=device, dtype=torch.float32)
                            v = losses.get("post_residual_loss", None)
                            if torch.is_tensor(v):
                                post_residual_loss = v.to(device=device, dtype=torch.float32)
                            v = losses.get("skew_mae", None)
                            if torch.is_tensor(v):
                                skew_mae = v.to(device=device, dtype=torch.float32)
                            pre_row = bundle_row.get("pre", {})
                            if isinstance(pre_row, dict):
                                v = pre_row.get("pred_skew", None)
                                if torch.is_tensor(v):
                                    deskew_pred_skew_t = v.to(device=device, dtype=torch.float32).reshape(-1, 1)
                                v = pre_row.get("applied_skew", None)
                                if torch.is_tensor(v):
                                    deskew_applied_skew_t = v.to(device=device, dtype=torch.float32).reshape(-1, 1)
                                v = pre_row.get("confidence", None)
                                if torch.is_tensor(v):
                                    deskew_confidence_t = v.to(device=device, dtype=torch.float32).reshape(-1, 1)
                            post_row = bundle_row.get("post", {})
                            if isinstance(post_row, dict):
                                v = post_row.get("residual_skew", None)
                                if torch.is_tensor(v):
                                    deskew_post_residual_t = v.to(device=device, dtype=torch.float32).reshape(-1, 1)
            if torch.is_tensor(target_skew_step):
                deskew_target_abs = torch.mean(torch.abs(target_skew_step.to(device=device, dtype=torch.float32)))
            if torch.is_tensor(deskew_pred_skew_t):
                deskew_pred_abs = torch.mean(torch.abs(deskew_pred_skew_t))
            if torch.is_tensor(deskew_applied_skew_t):
                deskew_applied_abs = torch.mean(torch.abs(deskew_applied_skew_t))
                if torch.is_tensor(target_skew_step):
                    tgt = target_skew_step.to(device=device, dtype=torch.float32).reshape(-1, 1)
                    deskew_remaining_abs = torch.mean(torch.abs(tgt + deskew_applied_skew_t))
            if torch.is_tensor(deskew_post_residual_t):
                deskew_post_abs = torch.mean(torch.abs(deskew_post_residual_t))
            if torch.is_tensor(deskew_confidence_t):
                deskew_conf = torch.mean(torch.clamp(deskew_confidence_t, 0.0, 1.0))

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
            target_cond = None
            if y_target is not None:
                if int(y_target.ndim) == 1:
                    raise RuntimeError(
                        "Transformer target supervision now requires semantic condition vectors [B,C], "
                        f"but got scalar labels with shape={tuple(y_target.shape)}"
                    )
                if int(y_target.ndim) != 2:
                    raise RuntimeError(f"Unexpected transformer target shape: {tuple(y_target.shape)}")
                target_cond = torch.clamp(y_target.to(torch.float32), 0.0, 1.0)
                valid_mask = target_cond.sum(dim=1) > 0.0
            else:
                valid_mask = None

            if valid_mask is not None and bool(torch.any(valid_mask)):
                logits_before_v = logits_before[valid_mask]
                logits_after_v = logits_after[valid_mask]
                target_vec = target_cond[valid_mask]
                if int(target_vec.shape[1]) != int(logits_after_v.shape[1]):
                    raise RuntimeError(
                        "Transformer target/logit width mismatch: "
                        f"target={tuple(target_vec.shape)} logits={tuple(logits_after_v.shape)}"
                    )
                target_count = torch.clamp(target_vec.sum(dim=1), min=1.0)
                probs_before_all = torch.sigmoid(logits_before_v).to(torch.float32)
                probs_after_all = torch.sigmoid(logits_after_v).to(torch.float32)
                score_target = ((probs_before_all * target_vec).sum(dim=1) / target_count).mean()
                score_after = ((probs_after_all * target_vec).sum(dim=1) / target_count).mean()
                score_gap = (
                    (F.relu((probs_before_all + score_target_margin) - probs_after_all) * target_vec).sum(dim=1)
                    / target_count
                ).mean()
                non_target_vec = torch.clamp(1.0 - target_vec, 0.0, 1.0)
                non_target_count = non_target_vec.sum(dim=1)
                has_non_target = non_target_count > 0.0
                if bool(torch.any(has_non_target)):
                    non_target_den = torch.clamp(non_target_count, min=1.0)
                    target_score_before = (probs_before_all * target_vec).sum(dim=1) / target_count
                    target_score_after = (probs_after_all * target_vec).sum(dim=1) / target_count

                    nt_before_masked = probs_before_all.masked_fill(non_target_vec <= 0.0, -1e9)
                    nt_after_masked = probs_after_all.masked_fill(non_target_vec <= 0.0, -1e9)
                    nt_max_before = nt_before_masked.max(dim=1).values
                    nt_max_after = nt_after_masked.max(dim=1).values
                    rank_violation_before = F.relu(nt_max_before + float(score_rank_margin) - target_score_before)
                    rank_violation_after = F.relu(nt_max_after + float(score_rank_margin) - target_score_after)
                    rank_improve_ref = F.relu(rank_violation_before - float(score_target_margin))
                    rank_gap_vec = F.relu(rank_violation_after - rank_improve_ref)

                    spurious_prob_before = (probs_before_all * non_target_vec).sum(dim=1) / non_target_den
                    spurious_prob_after = (probs_after_all * non_target_vec).sum(dim=1) / non_target_den
                    soft_hits_before = (
                        torch.sigmoid((probs_before_all - float(score_spurious_threshold)) / float(score_spurious_temp))
                        * non_target_vec
                    ).sum(dim=1) / non_target_den
                    soft_hits_after = (
                        torch.sigmoid((probs_after_all - float(score_spurious_threshold)) / float(score_spurious_temp))
                        * non_target_vec
                    ).sum(dim=1) / non_target_den
                    spurious_before = 0.5 * (spurious_prob_before + soft_hits_before)
                    spurious_after = 0.5 * (spurious_prob_after + soft_hits_after)
                    spurious_improve_ref = torch.clamp(spurious_before - float(score_spurious_margin), min=0.0)
                    spurious_gap_vec = F.relu(spurious_after - spurious_improve_ref)

                    score_rank_gap = rank_gap_vec[has_non_target].mean()
                    score_spurious_gap = spurious_gap_vec[has_non_target].mean()
                else:
                    score_rank_gap = score_gap.new_zeros(())
                    score_spurious_gap = score_gap.new_zeros(())
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
                score_rank_gap = score_gap.new_zeros(())
                score_spurious_gap = score_gap.new_zeros(())

            signal_gap = score_gap + (float(score_rank_weight) * score_rank_gap) + (
                float(score_spurious_weight) * score_spurious_gap
            )

            loss = (
                (score_target_weight * signal_gap)
                + (wave_l1_weight * denoise_l1)
                + (entropy_penalty_weight * entropy_pen)
                + (high_bit_penalty_weight * hi_bit_l1)
                + (low_bit_penalty_weight * lo_bit_l1)
                + aux_bundle_loss
            )
            if (step_preview_callback is not None) and int(xb.shape[0]) > 0:
                preview_idx = int(rng.integers(0, int(xb.shape[0])))
                if y_target is not None:
                    # Select a batch row that has at least one supervised target bit.
                    if int(y_target.ndim) == 2:
                        target_rows = torch.nonzero((y_target > 0).any(dim=1), as_tuple=False).squeeze(1)
                    else:
                        target_rows = torch.nonzero(y_target > 0, as_tuple=False).reshape(-1)
                    if int(target_rows.numel()) > 0:
                        pick = int(rng.integers(0, int(target_rows.numel())))
                        pick = max(0, min(int(target_rows.numel()) - 1, pick))
                        preview_idx = int(target_rows[pick].item())
                preview_idx = max(0, min(int(xb.shape[0]) - 1, int(preview_idx)))
                last_xclean_preview = x_clean[preview_idx : preview_idx + 1].detach()
                last_xb_preview = xb[preview_idx : preview_idx + 1].detach()
                last_xh_preview = xh[preview_idx : preview_idx + 1].detach()
                if target_cond is not None and int(preview_idx) < int(target_cond.shape[0]):
                    last_target_condition_preview = target_cond[preview_idx].detach().to(torch.float32)
                else:
                    last_target_condition_preview = None
                last_deskew_preview = None
                try:
                    if torch.is_tensor(target_skew_step) and int(preview_idx) < int(target_skew_step.shape[0]):
                        target_v = float(target_skew_step[preview_idx].detach().to(torch.float32).item())
                    else:
                        target_v = 0.0
                    pred_v = float(
                        deskew_pred_skew_t[preview_idx].detach().to(torch.float32).item()
                    ) if torch.is_tensor(deskew_pred_skew_t) and int(preview_idx) < int(deskew_pred_skew_t.shape[0]) else 0.0
                    applied_v = float(
                        deskew_applied_skew_t[preview_idx].detach().to(torch.float32).item()
                    ) if torch.is_tensor(deskew_applied_skew_t) and int(preview_idx) < int(deskew_applied_skew_t.shape[0]) else 0.0
                    conf_v = float(
                        torch.clamp(deskew_confidence_t[preview_idx].detach().to(torch.float32), 0.0, 1.0).item()
                    ) if torch.is_tensor(deskew_confidence_t) and int(preview_idx) < int(deskew_confidence_t.shape[0]) else 0.0
                    post_v = float(
                        deskew_post_residual_t[preview_idx].detach().to(torch.float32).item()
                    ) if torch.is_tensor(deskew_post_residual_t) and int(preview_idx) < int(deskew_post_residual_t.shape[0]) else 0.0
                    remain_v = float(target_v + applied_v)
                    last_deskew_preview = {
                        "target_skew": float(target_v),
                        "pred_skew": float(pred_v),
                        "applied_skew": float(applied_v),
                        "remaining_skew": float(remain_v),
                        "post_residual_skew": float(post_v),
                        "confidence": float(conf_v),
                    }
                except Exception:
                    last_deskew_preview = None
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
            running_score_rank_gap = running_score_rank_gap + score_rank_gap.detach().to(torch.float32)
            running_score_spurious_gap = running_score_spurious_gap + score_spurious_gap.detach().to(torch.float32)
            running_signal_gap = running_signal_gap + signal_gap.detach().to(torch.float32)
            running_aux_bundle_loss = running_aux_bundle_loss + aux_bundle_loss.detach().to(torch.float32)
            running_prefilter_loss = running_prefilter_loss + prefilter_loss.detach().to(torch.float32)
            running_post_residual_loss = running_post_residual_loss + post_residual_loss.detach().to(torch.float32)
            running_skew_mae = running_skew_mae + skew_mae.detach().to(torch.float32)
            running_deskew_target_abs = running_deskew_target_abs + deskew_target_abs.detach().to(torch.float32)
            running_deskew_pred_abs = running_deskew_pred_abs + deskew_pred_abs.detach().to(torch.float32)
            running_deskew_applied_abs = running_deskew_applied_abs + deskew_applied_abs.detach().to(torch.float32)
            running_deskew_remaining_abs = running_deskew_remaining_abs + deskew_remaining_abs.detach().to(torch.float32)
            running_deskew_post_abs = running_deskew_post_abs + deskew_post_abs.detach().to(torch.float32)
            running_deskew_conf = running_deskew_conf + deskew_conf.detach().to(torch.float32)
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
                avg_rank_gap = float((running_score_rank_gap / float(steps_done)).item())
                avg_spurious_gap = float((running_score_spurious_gap / float(steps_done)).item())
                avg_signal_gap = float((running_signal_gap / float(steps_done)).item())
                avg_aux_bundle = float((running_aux_bundle_loss / float(steps_done)).item())
                avg_prefilter = float((running_prefilter_loss / float(steps_done)).item())
                avg_post_residual = float((running_post_residual_loss / float(steps_done)).item())
                avg_skew_mae = float((running_skew_mae / float(steps_done)).item())
                avg_deskew_target_abs = float((running_deskew_target_abs / float(steps_done)).item())
                avg_deskew_pred_abs = float((running_deskew_pred_abs / float(steps_done)).item())
                avg_deskew_applied_abs = float((running_deskew_applied_abs / float(steps_done)).item())
                avg_deskew_remaining_abs = float((running_deskew_remaining_abs / float(steps_done)).item())
                avg_deskew_post_abs = float((running_deskew_post_abs / float(steps_done)).item())
                avg_deskew_conf = float((running_deskew_conf / float(steps_done)).item())
                avg_deg = float(running_degrade / float(steps_done))
                if (
                    (step_preview_callback is not None)
                    and (last_xclean_preview is not None)
                    and (last_xb_preview is not None)
                    and (last_xh_preview is not None)
                ):
                    try:
                        step_preview_callback(
                            {
                                "epoch": int(epoch),
                                "step": int(steps_done),
                                "steps_per_epoch": int(num_steps),
                                "x_clean": last_xclean_preview.detach().to(torch.float32),
                                "x_in": last_xb_preview.detach().to(torch.float32),
                                "x_out": last_xh_preview.detach().to(torch.float32),
                                "target_condition": (
                                    last_target_condition_preview.detach().to(torch.float32)
                                    if last_target_condition_preview is not None
                                    else None
                                ),
                                "score_after": float(avg_score_after),
                                "score_gap": float(avg_score_gap),
                                "loss": float(avg_loss),
                                "denoise_l1": float(avg_denoise),
                                "high_bits_l1": float(avg_hi_bits),
                                "low_bits_l1": float(avg_lo_bits),
                                "entropy_excess": float(avg_ent),
                                "score_target": float(avg_score_tgt),
                                "degrade_strength": float(avg_deg),
                                "deskew_target_abs": float(avg_deskew_target_abs),
                                "deskew_pred_abs": float(avg_deskew_pred_abs),
                                "deskew_applied_abs": float(avg_deskew_applied_abs),
                                "deskew_remaining_abs": float(avg_deskew_remaining_abs),
                                "deskew_post_abs": float(avg_deskew_post_abs),
                                "deskew_confidence_mean": float(avg_deskew_conf),
                                "deskew_preview": dict(last_deskew_preview) if isinstance(last_deskew_preview, dict) else None,
                            }
                        )
                    except Exception:
                        pass
                print(
                    f"[transformer-train] epoch={epoch}/{epochs} step={steps_done}/{num_steps} "
                    f"loss={avg_loss:.4f} denoise_l1={avg_denoise:.4f} "
                    f"hi_bits={avg_hi_bits:.4f} lo_bits={avg_lo_bits:.4f} entropy_excess={avg_ent:.4f} "
                    f"score_target={avg_score_tgt:.4f} score_after={avg_score_after:.4f} "
                    f"score_gap={avg_score_gap:.4f} rank_gap={avg_rank_gap:.4f} spurious_gap={avg_spurious_gap:.4f} signal_gap={avg_signal_gap:.4f} "
                    f"aux={avg_aux_bundle:.4f} deskew_pre={avg_prefilter:.4f} deskew_post={avg_post_residual:.4f} deskew_mae={avg_skew_mae:.4f} "
                    f"deskew_abs(tgt/pred/app/rem/post)=({avg_deskew_target_abs:.4f}/{avg_deskew_pred_abs:.4f}/{avg_deskew_applied_abs:.4f}/{avg_deskew_remaining_abs:.4f}/{avg_deskew_post_abs:.4f}) "
                    f"deskew_conf={avg_deskew_conf:.3f} "
                    f"degrade={avg_deg:.3f} "
                    f"samp_per_sec={(steps_done * int(batch_size)) / elapsed:.1f} "
                    f"sample_ms={(1000.0 * t_sample) / steps_done:.2f} "
                    f"forward_ms={(1000.0 * t_forward) / steps_done:.2f} "
                    f"backward_ms={(1000.0 * t_backward) / steps_done:.2f}",
                    flush=True,
                )

        if stop_now:
            break
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
            "score_rank_gap": float((running_score_rank_gap * s).item()),
            "score_spurious_gap": float((running_score_spurious_gap * s).item()),
            "signal_gap": float((running_signal_gap * s).item()),
            "aux_bundle_loss": float((running_aux_bundle_loss * s).item()),
            "deskew_prefilter_loss": float((running_prefilter_loss * s).item()),
            "deskew_post_residual_loss": float((running_post_residual_loss * s).item()),
            "deskew_skew_mae": float((running_skew_mae * s).item()),
            "deskew_target_abs": float((running_deskew_target_abs * s).item()),
            "deskew_pred_abs": float((running_deskew_pred_abs * s).item()),
            "deskew_applied_abs": float((running_deskew_applied_abs * s).item()),
            "deskew_remaining_abs": float((running_deskew_remaining_abs * s).item()),
            "deskew_post_abs": float((running_deskew_post_abs * s).item()),
            "deskew_confidence": float((running_deskew_conf * s).item()),
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
            "train_score_rank_gap": row_train["score_rank_gap"],
            "train_score_spurious_gap": row_train["score_spurious_gap"],
            "train_signal_gap": row_train["signal_gap"],
            "train_aux_bundle_loss": row_train["aux_bundle_loss"],
            "train_deskew_prefilter_loss": row_train["deskew_prefilter_loss"],
            "train_deskew_post_residual_loss": row_train["deskew_post_residual_loss"],
            "train_deskew_skew_mae": row_train["deskew_skew_mae"],
            "train_deskew_target_abs": row_train["deskew_target_abs"],
            "train_deskew_pred_abs": row_train["deskew_pred_abs"],
            "train_deskew_applied_abs": row_train["deskew_applied_abs"],
            "train_deskew_remaining_abs": row_train["deskew_remaining_abs"],
            "train_deskew_post_abs": row_train["deskew_post_abs"],
            "train_deskew_confidence": row_train["deskew_confidence"],
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
            f"score_gap={row_train['score_gap']:.4f} rank_gap={row_train['score_rank_gap']:.4f} "
            f"spurious_gap={row_train['score_spurious_gap']:.4f} signal_gap={row_train['signal_gap']:.4f} "
            f"aux={row_train['aux_bundle_loss']:.4f} deskew_pre={row_train['deskew_prefilter_loss']:.4f} "
            f"deskew_post={row_train['deskew_post_residual_loss']:.4f} deskew_mae={row_train['deskew_skew_mae']:.4f} "
            f"deskew_abs(tgt/pred/app/rem/post)=({row_train['deskew_target_abs']:.4f}/{row_train['deskew_pred_abs']:.4f}/"
            f"{row_train['deskew_applied_abs']:.4f}/{row_train['deskew_remaining_abs']:.4f}/{row_train['deskew_post_abs']:.4f}) "
            f"deskew_conf={row_train['deskew_confidence']:.3f} "
            f"degrade={row_train['degrade_strength']:.3f} "
            f"val_gain={gain:.4f} "
            f"epoch_sec={epoch_sec:.2f} eval_sec={eval_sec:.2f} "
            f"sample_pct={(100.0 * t_sample / max(1e-6, epoch_sec)):.1f} "
            f"forward_pct={(100.0 * t_forward / max(1e-6, epoch_sec)):.1f} "
            f"backward_pct={(100.0 * t_backward / max(1e-6, epoch_sec)):.1f}",
            flush=True,
        )

        if gain > best_gain:
            best_gain = gain
            best_state = copy.deepcopy(transformer.state_dict())

    if best_state is not None:
        transformer.load_state_dict(best_state)
    for p, req in zip(classifier.parameters(), classifier_requires_grad):
        p.requires_grad_(bool(req))
    classifier.train(classifier_was_training)
    return transformer, history
