import argparse
import json
import math
import os
import time
import wave
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wav_ml_core import (
    COLOR_MODES,
    RenderConfig,
    WaveRecord,
    decode_pcm,
    discover_wavs,
    image_u8_to_tensor,
    read_wav_record,
    render_record,
    render_mono_wave_to_tensor,
)
from wav_ml_models import (
    SinusoidalLRController,
    SinusoidalLROptions,
    TinyConvClassifier,
    WavePatchTransformer,
    evaluate_classifier,
    evaluate_feature_score_before_after,
    evaluate_transformer_accuracy,
    multilabel_feature_score,
    configure_torch_runtime,
    maybe_compile_module,
    set_seed,
    train_classifier,
    train_transformer_feature_metric,
)


def _log(msg: str):
    print(msg, flush=True)


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    key = str(amp_dtype).strip().lower()
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}")


def _autocast_context(device: torch.device, enabled: bool, amp_dtype_t: torch.dtype):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=amp_dtype_t)
    return nullcontext()


def _dataloader_perf_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: int):
    if int(num_workers) <= 0:
        return {}
    out = {"persistent_workers": bool(persistent_workers)}
    if int(prefetch_factor) > 0:
        out["prefetch_factor"] = int(prefetch_factor)
    return out


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise RuntimeError("Checkpoint does not contain a usable state dict.")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("_orig_mod."):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


def _apply_classifier_init(model: TinyConvClassifier, ckpt_path: str, scope: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier init checkpoint not found: {p}")

    blob = torch.load(str(p), map_location="cpu")
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0

    for k, v in src_state.items():
        if scope == "features" and not k.startswith("features."):
            skipped += 1
            continue
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        else:
            skipped += 1

    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
        "scope": scope,
    }


def _apply_model_init(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": ""}
    p = Path(ckpt_path)
    if not p.exists():
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "path": str(p)}

    blob = torch.load(str(p), map_location="cpu")
    src_state = _strip_module_prefix(_extract_state_dict(blob))
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "path": str(p.resolve()),
    }


def _apply_state_dict(model: nn.Module, src_state: Dict[str, torch.Tensor], source_name: str):
    if not isinstance(src_state, dict):
        return {"used": False, "loaded_keys": 0, "skipped_keys": 0, "source": source_name}
    src_state = _strip_module_prefix(src_state)
    dst_state = model.state_dict()
    to_load = {}
    skipped = 0
    for k, v in src_state.items():
        if k in dst_state and tuple(dst_state[k].shape) == tuple(v.shape):
            to_load[k] = v
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(to_load, strict=False)
    return {
        "used": True,
        "loaded_keys": len(to_load),
        "skipped_keys": skipped,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
        "source": source_name,
    }


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_pipeline_checkpoint(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_training_segment_snapshot(
    enabled: bool,
    out_dir: Path,
    objective_mode: str,
    run_tag: str,
    segment: str,
    best_cfg: Optional[RenderConfig],
    classifier: Optional[nn.Module] = None,
    transformer: Optional[nn.Module] = None,
    wave_classifier: Optional[nn.Module] = None,
    classifier_history: Optional[Sequence[Dict]] = None,
    transformer_history: Optional[Sequence[Dict]] = None,
    wave_classifier_history: Optional[Sequence[Dict]] = None,
    orchestration_history: Optional[Sequence[Dict]] = None,
    refresh_history: Optional[Sequence[Dict]] = None,
    gate_history: Optional[Sequence[Dict]] = None,
    gate_status: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    if not bool(enabled):
        return

    payload = {
        "run_tag": run_tag,
        "objective_mode": str(objective_mode),
        "segment": str(segment),
        "timestamp": time.time(),
    }
    if best_cfg is not None:
        try:
            payload["best_cfg"] = best_cfg.to_dict()
        except Exception:
            pass
    if classifier is not None:
        payload["classifier_state"] = classifier.state_dict()
    if transformer is not None:
        payload["transformer_state"] = transformer.state_dict()
    if wave_classifier is not None:
        payload["wave_classifier_state"] = wave_classifier.state_dict()
    if classifier_history is not None:
        payload["classifier_history"] = list(classifier_history)
    if transformer_history is not None:
        payload["transformer_history"] = list(transformer_history)
    if wave_classifier_history is not None:
        payload["wave_classifier_history"] = list(wave_classifier_history)
    if orchestration_history is not None:
        payload["orchestration_history"] = list(orchestration_history)
    if refresh_history is not None:
        payload["refresh_history"] = list(refresh_history)
    if gate_history is not None:
        payload["gate_history"] = list(gate_history)
    if isinstance(gate_status, dict):
        payload["gate_status"] = dict(gate_status)
    if isinstance(extra, dict):
        payload.update(extra)

    _save_pipeline_checkpoint(out_dir / "pipeline_checkpoint.pt", payload)
    if classifier is not None:
        torch.save({"state_dict": classifier.state_dict()}, out_dir / "classifier.pt")
    if transformer is not None:
        torch.save({"state_dict": transformer.state_dict()}, out_dir / "transformer.pt")
    if wave_classifier is not None:
        torch.save({"state_dict": wave_classifier.state_dict()}, out_dir / "wave_classifier.pt")


def _set_feature_freeze(model: TinyConvClassifier, freeze: bool):
    for name, p in model.named_parameters():
        if name.startswith("features."):
            p.requires_grad_(not freeze)


def _make_classifier_from_ckpt_meta(model_name: str, num_classes: int):
    if model_name == "tiny":
        return TinyConvClassifier(num_classes=num_classes)
    if model_name == "resnet18":
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise RuntimeError(f"Unsupported classifier model in checkpoint: {model_name}")


def _load_berkeley_classifier_for_metric(ckpt_path: str, device: torch.device):
    if not ckpt_path:
        raise RuntimeError("Berkeley metric mode requires --classifier-init-ckpt.")

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {p}")

    blob = torch.load(str(p), map_location="cpu")
    model_name = str(blob.get("model_name", "tiny"))
    class_names = blob.get("class_names", [])
    num_classes = int(blob.get("num_classes", len(class_names) if len(class_names) > 0 else 20))
    model = _make_classifier_from_ckpt_meta(model_name=model_name, num_classes=num_classes)
    state = _strip_module_prefix(_extract_state_dict(blob))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    info = {
        "path": str(p.resolve()),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "missing_after_load": len(missing),
        "unexpected_after_load": len(unexpected),
    }
    return model, info


@torch.no_grad()
def _score_config_with_classifier(
    records: Sequence[WaveRecord],
    indices: Sequence[int],
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
    classifier: nn.Module,
    device: torch.device,
    batch_size: int,
    score_topk: int,
    score_threshold: float,
    score_w_topk: float,
    score_w_cov: float,
    score_w_mean: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
):
    if len(indices) == 0:
        return {
            "score": float("-inf"),
            "topk_mean": 0.0,
            "hard_coverage": 0.0,
            "mean_prob": 0.0,
            "num_images": 0,
        }

    logits_all = []
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    for i in range(0, len(indices), max(1, int(batch_size))):
        batch_idx = indices[i : i + max(1, int(batch_size))]
        xb = []
        for idx in batch_idx:
            img_u8, _ = render_record(records[idx], cfg)
            xb.append(image_u8_to_tensor(img_u8, image_hw=image_hw))
        x = torch.stack(xb, dim=0).to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
            logits = classifier(x)
        logits_all.append(logits.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    probs = torch.sigmoid(logits_cat)
    ent = -(probs * torch.log2(torch.clamp(probs, 1e-8, 1.0)) + (1.0 - probs) * torch.log2(torch.clamp(1.0 - probs, 1e-8, 1.0)))
    s = multilabel_feature_score(
        logits_cat,
        topk=score_topk,
        threshold=score_threshold,
        w_topk=score_w_topk,
        w_cov=score_w_cov,
        w_mean=score_w_mean,
    )
    return {
        "score": float(s["score"].item()),
        "topk_mean": float(s["topk_mean"].item()),
        "hard_coverage": float(s["hard_coverage"].item()),
        "mean_prob": float(s["mean_prob"].item()),
        "mean_entropy": float(ent.mean().item()),
        "num_images": int(logits_cat.shape[0]),
    }


def _build_berkeley_refresh_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_train: int,
    seed: int,
    device: torch.device,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

    train_ds, _ = prepare_sbd_multilabel(
        data_root=data_root,
        image_size=image_size,
        auto_install_scipy=auto_install_scipy,
    )
    if max_train > 0 and len(train_ds) > int(max_train):
        idx = np.random.default_rng(seed).choice(len(train_ds), size=int(max_train), replace=False)
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())

    loader = DataLoader(
        train_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    return loader, len(train_ds)


def _build_berkeley_refresh_cache(
    loader: DataLoader,
    device: torch.device,
    cache_batches: int,
    cache_device: str,
    channels_last: bool,
):
    n_batches = max(0, int(cache_batches))
    mode = str(cache_device).strip().lower()
    if n_batches <= 0 or mode == "none":
        return None

    if mode == "cuda":
        target = torch.device("cuda:0" if device.type == "cuda" else "cpu")
    elif mode == "cpu":
        target = torch.device("cpu")
    else:
        target = torch.device("cuda:0" if device.type == "cuda" else "cpu")

    xs = []
    ys = []
    for i, (xb, yb) in enumerate(loader, start=1):
        xs.append(xb)
        ys.append(yb)
        if i >= n_batches:
            break

    if len(xs) == 0:
        return None

    x = torch.cat(xs, dim=0).contiguous()
    y = torch.cat(ys, dim=0).contiguous()
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    if target.type == "cuda":
        x = x.to(target, non_blocking=True)
        y = y.to(target, non_blocking=True)
    elif device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()

    return {
        "x": x,
        "y": y,
        "num_samples": int(y.shape[0]),
        "num_batches": len(xs),
        "device": target.type,
    }


def _auto_berkeley_refresh_batch_size(
    cache_x: torch.Tensor,
    cache_y: torch.Tensor,
    device: torch.device,
    vram_fraction: float,
    activation_multiplier: float,
    max_cap: int,
):
    if int(cache_x.shape[0]) <= 0:
        return 1
    max_cap = int(max_cap)
    hard_cap = int(cache_x.shape[0]) if max_cap <= 0 else min(int(cache_x.shape[0]), max_cap)
    if device.type != "cuda":
        return max(1, hard_cap)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = 0

    sample_bytes = int(cache_x[0].numel()) * int(cache_x.element_size())
    if cache_y is not None:
        sample_bytes += int(cache_y[0].numel()) * int(cache_y.element_size())
    sample_bytes = max(1, sample_bytes)
    mult = max(1.0, float(activation_multiplier))
    target = int(max(0.0, min(0.99, float(vram_fraction))) * float(max(0, free_bytes)))
    if target <= 0:
        return max(1, min(64, hard_cap))
    est = int(target // int(sample_bytes * mult))
    return max(1, min(est, hard_cap))


def _build_berkeley_gate_val_loader(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    batch_size: int,
    num_workers: int,
    max_val: int,
    seed: int,
    device: torch.device,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
):
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

    _, val_ds = prepare_sbd_multilabel(
        data_root=data_root,
        image_size=image_size,
        auto_install_scipy=auto_install_scipy,
    )
    if max_val > 0 and len(val_ds) > int(max_val):
        idx = np.random.default_rng(seed).choice(len(val_ds), size=int(max_val), replace=False)
        val_ds = torch.utils.data.Subset(val_ds, idx.tolist())

    loader = DataLoader(
        val_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=max(0, int(num_workers)),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        ),
    )
    return loader, len(val_ds)


@torch.no_grad()
def _evaluate_berkeley_classifier_gate(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
):
    classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    total_loss = 0.0
    n = 0
    logits_all = []
    targets_all = []
    steps = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
            logits = classifier(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
        total_loss += float(loss.item()) * int(xb.shape[0])
        n += int(xb.shape[0])
        logits_all.append(logits.detach().cpu())
        targets_all.append(yb.detach().cpu())
        steps += 1
        if max_steps > 0 and steps >= int(max_steps):
            break

    if n <= 0:
        return {
            "loss": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "bit_acc": 0.0,
            "mean_confidence": 0.0,
            "num_samples": 0,
        }

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    probs = torch.sigmoid(logits_cat)
    preds = (probs >= 0.5).float()
    t = targets_cat.float()

    tp = (preds * t).sum(dim=0)
    fp = (preds * (1.0 - t)).sum(dim=0)
    fn = ((1.0 - preds) * t).sum(dim=0)
    macro_f1 = torch.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).item()
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_f1 = ((2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-8)).item()
    bit_acc = preds.eq(t).float().mean().item()
    mean_conf = torch.maximum(probs, 1.0 - probs).mean().item()
    return {
        "loss": total_loss / max(1, n),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "bit_acc": float(bit_acc),
        "mean_confidence": float(mean_conf),
        "num_samples": int(n),
    }


def _run_berkeley_refresh_epochs(
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    max_steps: int,
    lr_sine_cycles: float,
    lr_sine_frequency: float,
    lr_sine_tail_fraction: float,
    lr_sine_min_scale: float,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    grad_accum_steps: int = 1,
    log_every: int = 0,
    max_seconds: float = 0.0,
    cache_x: Optional[torch.Tensor] = None,
    cache_y: Optional[torch.Tensor] = None,
    cache_batch_size: int = 0,
):
    if epochs <= 0:
        return {"ran": False, "loss": 0.0}

    # A prior transformer stage may have frozen this classifier for feature scoring.
    # Ensure refresh has trainable params before building autograd graph.
    if not any(bool(p.requires_grad) for p in classifier.parameters()):
        for p in classifier.parameters():
            p.requires_grad_(True)

    classifier.train()
    if channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    use_scaler = bool(amp_enabled and device.type == "cuda" and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)
    grad_accum_steps = max(1, int(grad_accum_steps))
    use_cache = (
        (cache_x is not None)
        and (cache_y is not None)
        and int(getattr(cache_x, "shape", [0])[0]) > 0
        and int(getattr(cache_y, "shape", [0])[0]) == int(getattr(cache_x, "shape", [0])[0])
        and int(cache_batch_size) > 0
    )
    cache_batch_size = max(1, int(cache_batch_size)) if use_cache else 0
    refresh_source = "cache" if use_cache else "loader"
    opt = torch.optim.AdamW(classifier.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if use_cache:
        cache_n = int(cache_x.shape[0])
        if max_steps > 0:
            steps_per_epoch = max(1, int(max_steps))
        else:
            steps_per_epoch = max(1, int(math.ceil(float(cache_n) / float(cache_batch_size))))
    else:
        steps_per_epoch = len(loader)
        if max_steps > 0:
            steps_per_epoch = min(steps_per_epoch, int(max_steps))
        steps_per_epoch = max(1, int(steps_per_epoch))
    updates_per_epoch = int(math.ceil(float(steps_per_epoch) / float(grad_accum_steps)))
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=max(1, int(epochs) * updates_per_epoch),
        options=SinusoidalLROptions(
            cycles=float(lr_sine_cycles),
            frequency=float(lr_sine_frequency),
            tail_fraction=float(lr_sine_tail_fraction),
            min_scale=float(lr_sine_min_scale),
        ),
    )
    total_loss = 0.0
    n = 0
    t_start = time.time()
    total_target_steps = max(1, int(epochs) * steps_per_epoch)
    global_step = 0
    for _ in range(int(epochs)):
        step = 0
        opt.zero_grad(set_to_none=True)
        if use_cache:
            cache_n = int(cache_x.shape[0])
        else:
            loader_iter = iter(loader)

        while step < steps_per_epoch:
            if use_cache:
                idx = torch.randint(
                    low=0,
                    high=cache_n,
                    size=(cache_batch_size,),
                    device=cache_x.device,
                )
                xb = cache_x.index_select(0, idx)
                yb = cache_y.index_select(0, idx)
            else:
                try:
                    xb, yb = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    xb, yb = next(loader_iter)
            if xb.device != device:
                xb = xb.to(device, non_blocking=True)
            if yb.device != device:
                yb = yb.to(device, non_blocking=True)
            if channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                logits = classifier(xb)
            if logits.shape[1] != yb.shape[1]:
                raise RuntimeError(
                    f"Berkeley refresh target mismatch: logits={tuple(logits.shape)} vs targets={tuple(yb.shape)}"
                )
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()
            do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == steps_per_epoch)
            if do_step:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                lr_ctl.step()
                opt.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
            step += 1
            global_step += 1
            if int(log_every) > 0 and (global_step % int(log_every) == 0):
                elapsed = max(1e-6, time.time() - t_start)
                ips = float(n) / elapsed
                print(
                    f"[berkeley-refresh] step={global_step}/{total_target_steps} "
                    f"loss={total_loss / max(1, n):.4f} samples={n} samp_per_sec={ips:.1f}",
                    flush=True,
                )
            if float(max_seconds) > 0.0 and (time.time() - t_start) >= float(max_seconds):
                classifier.eval()
                return {
                    "ran": True,
                    "loss": total_loss / max(1, n),
                    "samples": n,
                    "truncated_by_time": True,
                    "elapsed_sec": float(time.time() - t_start),
                    "source": refresh_source,
                }
            if max_steps > 0 and step >= int(max_steps):
                break
    classifier.eval()
    return {
        "ran": True,
        "loss": total_loss / max(1, n),
        "samples": n,
        "truncated_by_time": False,
        "elapsed_sec": float(time.time() - t_start),
        "source": refresh_source,
    }


def _sample_stream_chunks_with_labels(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    batch_size: int,
    chunk_samples: int,
    rng: np.random.Generator,
):
    xb = np.zeros((batch_size, chunk_samples), dtype=np.float32)
    yb = np.zeros((batch_size,), dtype=np.int64)
    picked = []
    for i in range(batch_size):
        j = int(rng.integers(0, len(streams)))
        s = streams[j]
        yb[i] = int(labels[j])
        picked.append(metas[j])
        if s.size == 0:
            continue
        if s.size >= chunk_samples:
            start = int(rng.integers(0, s.size - chunk_samples + 1))
            xb[i, :] = s[start : start + chunk_samples]
        else:
            xb[i, : s.size] = s
    return xb, yb, picked


def _save_mono_wav(path: Path, mono_f32: np.ndarray, framerate: int):
    y = np.clip(mono_f32, -1.0, 1.0)
    i16 = np.round(y * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(framerate))
        wf.writeframes(i16.tobytes())


def _fit_wave_length(x: np.ndarray, target_samples: int, rng: np.random.Generator):
    target = max(1, int(target_samples))
    if x.size == 0:
        return np.zeros((target,), dtype=np.float32)
    if x.size >= target:
        if x.size == target:
            return x.astype(np.float32, copy=False)
        start = int(rng.integers(0, x.size - target + 1))
        return x[start : start + target].astype(np.float32, copy=False)
    reps = int(np.ceil(float(target) / float(x.size)))
    y = np.tile(x, reps)[:target]
    return y.astype(np.float32, copy=False)


def _synthesize_structured_wave(target_samples: int, framerate: int, rng: np.random.Generator) -> np.ndarray:
    n = max(256, int(target_samples))
    sr = max(1000, int(framerate))
    t = (np.arange(n, dtype=np.float32) / float(sr)).astype(np.float32, copy=False)
    y = np.zeros((n,), dtype=np.float32)

    n_comp = int(rng.integers(2, 6))
    dur = float(n) / float(sr)
    for _ in range(n_comp):
        f0 = float(rng.uniform(40.0, 1800.0))
        f1 = float(np.clip(f0 * float(rng.uniform(0.65, 1.6)), 20.0, 2400.0))
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        if bool(rng.random() < 0.5):
            # Linear chirp in phase form for cheap generation.
            k = (f1 - f0) / max(1e-6, dur)
            phase = (2.0 * np.pi * ((f0 * t) + (0.5 * k * t * t))) + phase0
        else:
            phase = (2.0 * np.pi * f0 * t) + phase0
        amp = float(rng.uniform(0.3, 1.0))
        y += (amp * np.sin(phase)).astype(np.float32, copy=False)

    # Envelope + tremolo keeps shapes nonstationary and less noise-like.
    attack = int(min(n, max(0, int(float(rng.uniform(0.01, 0.12)) * n))))
    release = int(min(n, max(0, int(float(rng.uniform(0.08, 0.35)) * n))))
    env = np.ones((n,), dtype=np.float32)
    if attack > 1:
        env[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False, dtype=np.float32)
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True, dtype=np.float32)
    mod_f = float(rng.uniform(0.4, 8.0))
    mod_depth = float(rng.uniform(0.1, 0.65))
    mod = (1.0 - mod_depth) + (mod_depth * (0.5 * (1.0 + np.sin((2.0 * np.pi * mod_f * t) + float(rng.uniform(0.0, 2.0 * np.pi))))))
    y = y * env * mod.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32, copy=False)


def _load_reinject_wave_paths(library_dir: Path, limit: int = 0):
    idx_path = library_dir / "index.jsonl"
    if not idx_path.exists():
        return []

    out: List[str] = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            fp = str(row.get("file", "")).strip()
            if not fp:
                continue
            p = Path(fp)
            candidates = []
            if p.is_absolute():
                candidates.append(p)
            else:
                candidates.append(Path.cwd() / p)
                candidates.append(library_dir / p)
            found = None
            for c in candidates:
                if c.exists():
                    found = c.resolve()
                    break
            if found is None:
                continue
            out.append(str(found))
            if limit > 0 and len(out) >= int(limit):
                break
    return out


def _bootstrap_latent_wav_pool(
    out_dir: Path,
    seed: int,
    count: int,
    framerate: int,
    seconds: float,
    noise_std: float,
    reinject_dir: str,
    reinject_ratio: float,
    reinject_copy_gain: float,
    reinject_noise_gain: float,
    structured_ratio: float,
    structured_gain: float,
    structured_noise_gain: float,
):
    rng = np.random.default_rng(int(seed) + 424242)
    sample_count = max(256, int(float(framerate) * max(0.05, float(seconds))))
    pool_dir = out_dir / "latent_wave_pool"
    noise_dir = pool_dir / "noise"
    mix_dir = pool_dir / "mix"
    noise_dir.mkdir(parents=True, exist_ok=True)
    mix_dir.mkdir(parents=True, exist_ok=True)

    if reinject_dir:
        lib_dir = Path(reinject_dir)
    else:
        lib_dir = out_dir / "accepted_wave_library"
    reinject_paths = _load_reinject_wave_paths(lib_dir, limit=100000)

    rows = []
    mixed_count = 0
    structured_count = 0
    for i in range(max(1, int(count))):
        noise = (rng.standard_normal(sample_count).astype(np.float32) * float(noise_std))
        y = noise.copy()
        src = ""
        used_mix = False

        if len(reinject_paths) > 0 and float(rng.random()) < float(reinject_ratio):
            src = reinject_paths[int(rng.integers(0, len(reinject_paths)))]
            try:
                rec = read_wav_record(src)
                mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=0)
                ref = _fit_wave_length(mono, target_samples=sample_count, rng=rng)
                peak = float(np.max(np.abs(ref))) if ref.size > 0 else 0.0
                if peak > 1e-6:
                    ref = ref / peak
                y = (float(reinject_noise_gain) * noise) + (float(reinject_copy_gain) * ref)
                used_mix = True
                mixed_count += 1
            except Exception:
                src = ""
                y = noise
        elif float(rng.random()) < float(structured_ratio):
            proto = _synthesize_structured_wave(
                target_samples=sample_count,
                framerate=framerate,
                rng=rng,
            )
            y = (float(structured_noise_gain) * noise) + (float(structured_gain) * proto)
            src = "__structured_seed__"
            used_mix = True
            structured_count += 1

        y = np.clip(y, -1.0, 1.0)
        target_dir = mix_dir if used_mix else noise_dir
        out_path = target_dir / f"latent_{i:05d}.wav"
        _save_mono_wav(out_path, y, framerate=framerate)
        rows.append({"file": str(out_path), "mixed": bool(used_mix), "source": src})

    manifest = pool_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "count": int(count),
                "framerate": int(framerate),
                "seconds": float(seconds),
                "sample_count": int(sample_count),
                "noise_std": float(noise_std),
                "reinject_dir": str(lib_dir),
                "reinject_ratio": float(reinject_ratio),
                "reinject_copy_gain": float(reinject_copy_gain),
                "reinject_noise_gain": float(reinject_noise_gain),
                "reinject_candidates": len(reinject_paths),
                "mixed_count": int(mixed_count),
                "structured_ratio": float(structured_ratio),
                "structured_gain": float(structured_gain),
                "structured_noise_gain": float(structured_noise_gain),
                "structured_count": int(structured_count),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "pool_dir": str(pool_dir),
        "manifest": str(manifest),
        "count": int(count),
        "reinject_dir": str(lib_dir),
        "reinject_candidates": len(reinject_paths),
        "mixed_count": int(mixed_count),
        "structured_count": int(structured_count),
    }


def _build_wave_classifier_dataset_from_transformer(
    streams: Sequence[np.ndarray],
    labels: Sequence[int],
    metas: Sequence[Dict],
    cfg: RenderConfig,
    transformer: nn.Module,
    berkeley_classifier: nn.Module,
    sample_bits: int,
    image_hw: Tuple[int, int],
    chunk_samples: int,
    total_samples: int,
    batch_size: int,
    device: torch.device,
    rng_seed: int,
    accept_score_threshold: float,
    accept_l1_threshold: float,
    library_dir: Path,
    library_limit: int,
    cycle_id: int,
    round_id: int,
    accepted_only: bool = False,
    run_tag: str = "",
    library_serial_start: int = 0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    pin_memory: bool = False,
):
    rng = np.random.default_rng(rng_seed)
    x_all = []
    y_all = []
    accepted_rows = []
    transformer.eval()
    berkeley_classifier.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16

    target_samples = max(1, int(total_samples))
    accepted_only = bool(accepted_only)
    # When accepted_only is enabled, keep drawing until we reach the requested
    # accepted-set size (or a bounded attempt budget).
    max_draw_samples = target_samples if (not accepted_only) else max(target_samples, target_samples * 8)

    drawn = 0
    kept = 0
    while drawn < max_draw_samples and ((kept < target_samples) if accepted_only else (drawn < target_samples)):
        if accepted_only:
            b = min(int(batch_size), max_draw_samples - drawn)
        else:
            b = min(int(batch_size), target_samples - drawn)
        if b <= 0:
            break
        xb_np, yb_np, picked = _sample_stream_chunks_with_labels(
            streams=streams,
            labels=labels,
            metas=metas,
            batch_size=b,
            chunk_samples=chunk_samples,
            rng=rng,
        )
        xb = torch.from_numpy(xb_np)
        if pin_memory and device.type == "cuda":
            xb = xb.pin_memory()
        xb = xb.to(device, non_blocking=True)
        with torch.no_grad():
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                xh = transformer(xb)
                imgs = render_mono_wave_to_tensor(xh, cfg=cfg, image_hw=image_hw, sample_bits=sample_bits)
                if channels_last:
                    imgs = imgs.contiguous(memory_format=torch.channels_last)
                logits = berkeley_classifier(imgs)
            probs = torch.sigmoid(logits)
            k = max(1, min(3, int(probs.shape[1])))
            per_score = torch.topk(probs, k=k, dim=1).values.mean(dim=1)
            per_l1 = torch.mean(torch.abs(xh - xb), dim=1)

        accept_mask = (per_score >= float(accept_score_threshold)) & (per_l1 <= float(accept_l1_threshold))
        if accepted_only:
            if bool(torch.any(accept_mask)):
                keep_idx = torch.nonzero(accept_mask, as_tuple=False).squeeze(1)
                keep_np = keep_idx.detach().cpu().numpy()
                x_all.append(imgs.index_select(0, keep_idx).detach().cpu())
                y_all.append(torch.from_numpy(yb_np[keep_np]))
                kept += int(keep_idx.numel())
        else:
            x_all.append(imgs.detach().cpu())
            y_all.append(torch.from_numpy(yb_np))
            kept += int(b)
        drawn += int(b)

        if library_limit > 0 and len(accepted_rows) < int(library_limit):
            library_dir.mkdir(parents=True, exist_ok=True)
            index_path = library_dir / "index.jsonl"
            with index_path.open("a", encoding="utf-8") as f:
                for i in range(b):
                    sc = float(per_score[i].item())
                    l1 = float(per_l1[i].item())
                    if sc >= float(accept_score_threshold) and l1 <= float(accept_l1_threshold):
                        serial = int(library_serial_start) + len(accepted_rows)
                        if run_tag:
                            fn = f"{run_tag}_cycle{cycle_id:03d}_round{round_id:03d}_{serial:08d}.wav"
                        else:
                            fn = f"cycle{cycle_id:03d}_round{round_id:03d}_{serial:08d}.wav"
                        out_path = library_dir / fn
                        _save_mono_wav(out_path, xh[i].detach().cpu().numpy(), framerate=int(picked[i]["framerate"]))
                        row = {
                            "file": str(out_path),
                            "label": int(yb_np[i]),
                            "score": sc,
                            "l1": l1,
                            "source": picked[i]["path"],
                            "cycle": int(cycle_id),
                            "round": int(round_id),
                        }
                        f.write(json.dumps(row) + "\n")
                        accepted_rows.append(row)
                    if len(accepted_rows) >= int(library_limit):
                        break

    if len(x_all) == 0:
        x = torch.empty((0, 3, int(image_hw[0]), int(image_hw[1])), dtype=torch.float32)
        y = torch.empty((0,), dtype=torch.long)
    else:
        x = torch.cat(x_all, dim=0)
        y = torch.cat(y_all, dim=0).long()
        if accepted_only and int(x.shape[0]) > target_samples:
            x = x[:target_samples]
            y = y[:target_samples]
    if accepted_only and int(x.shape[0]) < target_samples:
        print(
            f"[wave-cls-dataset] accepted_only target={target_samples} got={int(x.shape[0])} "
            f"(drawn={drawn}, max_draw={max_draw_samples})",
            flush=True,
        )
    return x, y, accepted_rows


def _resolve_decode_mode(sampwidth: int, bitmode: str) -> str:
    if bitmode != "auto":
        return bitmode
    if sampwidth == 1:
        return "8"
    if sampwidth == 2:
        return "16"
    if sampwidth == 3:
        return "24"
    if sampwidth == 4:
        return "32"
    return "16"


def _decode_record_to_mono(record: WaveRecord, cfg: RenderConfig, max_points: int = 0):
    if cfg.use_channels == "auto":
        user_nch = record.nch if record.nch in (1, 2) else 2
    else:
        user_nch = int(cfg.use_channels)

    mode = _resolve_decode_mode(record.sampwidth, cfg.bitmode)
    samples_f32, samples_i32, sample_bits, _ = decode_pcm(record.frames, user_nch=user_nch, mode=mode)
    if samples_f32.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), (sample_bits if sample_bits is not None else 16)

    if samples_f32.shape[1] == 1:
        mono = samples_f32[:, 0]
    else:
        if cfg.mono_pick == "L":
            mono = samples_f32[:, 0]
        elif cfg.mono_pick == "R":
            mono = samples_f32[:, 1]
        elif cfg.mono_pick == "LR stride":
            out = np.empty((samples_f32.shape[0] * 2,), dtype=np.float32)
            out[0::2] = samples_f32[:, 0]
            out[1::2] = samples_f32[:, 1]
            mono = out
        else:
            mono = 0.5 * (samples_f32[:, 0] + samples_f32[:, 1])

    if max_points > 0 and mono.size > max_points:
        mono = mono[:max_points]
    bits = sample_bits if sample_bits is not None else (32 if mode == "float32" else 16)
    return mono.astype(np.float32, copy=False), int(bits)


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    if x.size < 64:
        return 0.0
    n = min(4096, x.size)
    n = int(2 ** np.floor(np.log2(max(64, n))))
    y = x[:n].astype(np.float64, copy=False)
    y = y * np.hanning(n)
    spec = np.abs(np.fft.rfft(y))
    denom = float(np.sum(spec))
    if denom <= 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    return float(np.sum(freqs * spec) / denom)


def _build_labels(records: Sequence[WaveRecord], data_root: str, label_mode: str, pseudo_classes: int):
    root = Path(data_root).resolve()
    if label_mode == "folder":
        names: List[str] = []
        for rec in records:
            p = Path(rec.path).resolve()
            try:
                rel = p.relative_to(root)
                if len(rel.parts) >= 2:
                    names.append(rel.parts[0])
                else:
                    names.append(p.parent.name)
            except Exception:
                names.append(p.parent.name)
        uniq = sorted(set(names))
        if len(uniq) >= 2:
            lut = {n: i for i, n in enumerate(uniq)}
            labels = np.array([lut[n] for n in names], dtype=np.int64)
            return labels, uniq, "folder"

        _log("Folder labels are single-class; switching to pseudo labels from spectral centroid bins.")

    pseudo_classes = max(2, int(pseudo_classes))
    base_cfg = RenderConfig()
    centroids = []
    for rec in records:
        mono, _ = _decode_record_to_mono(rec, base_cfg, max_points=65536)
        centroids.append(_spectral_centroid(mono, sr=rec.framerate))
    vals = np.array(centroids, dtype=np.float64)
    q = np.linspace(0.0, 1.0, pseudo_classes + 1)
    edges = np.quantile(vals, q)
    if np.allclose(edges, edges[0]):
        rank = np.argsort(np.argsort(vals))
        labels = (rank * pseudo_classes) // max(1, len(vals))
    else:
        labels = np.searchsorted(edges[1:-1], vals, side="right")
    labels = labels.astype(np.int64)
    class_names = [f"pseudo_bin_{i}" for i in range(int(labels.max()) + 1)]
    return labels, class_names, "spectral_quantile"


def _stratified_split(labels: np.ndarray, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    labels = np.asarray(labels, dtype=np.int64)
    for cls in sorted(set(labels.tolist())):
        idx = np.where(labels == cls)[0]
        idx = np.array(idx, copy=True)
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_frac))
        cut = min(max(cut, 1), len(idx) - 1) if len(idx) > 1 else 1
        train_idx.extend(idx[:cut].tolist())
        val_idx.extend(idx[cut:].tolist())

    if len(val_idx) == 0:
        rng.shuffle(train_idx)
        val_idx = train_idx[-max(1, len(train_idx) // 5) :]
        train_idx = train_idx[: -len(val_idx)]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _sample_indices(indices: Sequence[int], limit: int, rng: np.random.Generator):
    idx = list(indices)
    if limit <= 0 or len(idx) <= limit:
        return idx
    return rng.choice(np.array(idx), size=int(limit), replace=False).tolist()


def _render_indices_to_tensors(
    records: Sequence[WaveRecord],
    labels: np.ndarray,
    indices: Sequence[int],
    cfg: RenderConfig,
    image_hw: Tuple[int, int],
):
    x_list = []
    y_list = []
    bits = []
    for idx in indices:
        img_u8, meta = render_record(records[idx], cfg)
        x_list.append(image_u8_to_tensor(img_u8, image_hw=image_hw))
        y_list.append(int(labels[idx]))
        if meta.get("sample_bits") is not None:
            bits.append(int(meta["sample_bits"]))
    x = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.long)
    return x, y, bits


def _random_configs(num_trials: int, rng: np.random.Generator, max_points: int):
    color_choices = [m[0] for m in COLOR_MODES]
    widths = [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    downs = [1, 2, 3, 4, 6, 8]
    mono_choices = ["Mix", "L", "R", "LR stride"]

    cfgs = [
        RenderConfig(
            width=1024,
            downsample=1,
            color_mode=COLOR_MODES[0][0],
            mono_pick="Mix",
            bitmask_enable=False,
            max_points=max_points,
        )
    ]
    for _ in range(max(0, int(num_trials) - 1)):
        use_mask = bool(rng.random() < 0.5)
        lo = int(rng.integers(0, 12))
        hi = int(rng.integers(lo, min(16, lo + 9)))
        cfgs.append(
            RenderConfig(
                width=int(rng.choice(widths)),
                downsample=int(rng.choice(downs)),
                color_mode=str(rng.choice(color_choices)),
                mono_pick=str(rng.choice(mono_choices)),
                empty_fill=0,
                bitmask_enable=use_mask,
                bitmask_low=lo,
                bitmask_high=hi,
                max_points=max_points,
            )
        )
    return cfgs


def _try_scipy_refine(best_cfg: RenderConfig, max_points: int, enabled: bool):
    if not enabled:
        return [best_cfg]
    try:
        from scipy import optimize  # noqa: F401
    except Exception:
        _log("scipy is not installed; skipping scipy config refinement.")
        return [best_cfg]

    _log("scipy detected; running lightweight local config jitter around the current best.")
    rng = np.random.default_rng(777)
    width_jitter = [max(128, best_cfg.width + int(k) * 64) for k in range(-2, 3)]
    ds_jitter = [max(1, best_cfg.downsample + k) for k in (-1, 0, 1)]
    mask_lows = [max(0, best_cfg.bitmask_low + k) for k in (-2, -1, 0, 1, 2)]
    mask_highs = [max(0, best_cfg.bitmask_high + k) for k in (-2, -1, 0, 1, 2)]
    cfgs = []
    for _ in range(6):
        cfgs.append(
            RenderConfig(
                bitmode=best_cfg.bitmode,
                use_channels=best_cfg.use_channels,
                mono_pick=best_cfg.mono_pick,
                width=int(rng.choice(width_jitter)),
                downsample=int(rng.choice(ds_jitter)),
                color_mode=best_cfg.color_mode,
                empty_fill=best_cfg.empty_fill,
                bitmask_enable=best_cfg.bitmask_enable,
                bitmask_low=int(rng.choice(mask_lows)),
                bitmask_high=int(rng.choice(mask_highs)),
                max_points=max_points,
            )
        )
    return [best_cfg] + cfgs


def _prepare_streams(
    records: Sequence[WaveRecord],
    labels: np.ndarray,
    indices: Sequence[int],
    cfg: RenderConfig,
    max_points: int,
):
    streams = []
    ys = []
    bits = []
    meta = []
    for idx in indices:
        mono, bit_depth = _decode_record_to_mono(records[idx], cfg, max_points=max_points)
        if mono.size == 0:
            continue
        streams.append(mono.astype(np.float32, copy=False))
        ys.append(int(labels[idx]))
        bits.append(int(bit_depth))
        meta.append({"path": records[idx].path, "framerate": int(records[idx].framerate)})
    return streams, ys, bits, meta


def _write_mono_wav(path: str, mono_f32: np.ndarray, framerate: int):
    y = np.clip(mono_f32, -1.0, 1.0)
    i16 = np.round(y * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(framerate))
        wf.writeframes(i16.tobytes())


def _append_accepted_rows_to_transformer_streams(
    rows: Sequence[Dict],
    seen_paths: set,
    train_streams: List[np.ndarray],
    train_stream_labels: List[int],
    train_meta: List[Dict],
    max_points: int,
) -> int:
    added = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        p = str(row.get("file", "")).strip()
        if not p or p in seen_paths:
            continue
        try:
            rec = read_wav_record(p)
            mono, _ = _decode_record_to_mono(rec, cfg=RenderConfig(), max_points=max_points)
            if mono.size <= 0:
                continue
            train_streams.append(mono.astype(np.float32, copy=False))
            train_stream_labels.append(int(row.get("label", 0)))
            train_meta.append({"path": str(p), "framerate": int(rec.framerate)})
            seen_paths.add(p)
            added += 1
        except Exception:
            continue
    return int(added)


def _latent_pool_stream_kind(path: str) -> str:
    p = str(path).replace("\\", "/").lower()
    if "/latent_wave_pool/noise/" in p:
        return "noise"
    if "/latent_wave_pool/mix/" in p:
        return "mix"
    return "other"


def _encode_target_classes_bitmask(target_classes: Sequence[int]) -> int:
    mask = 0
    for cls in target_classes:
        c = int(cls)
        if c < 0:
            continue
        mask |= (1 << c)
    return int(mask)


def _decode_target_classes_from_bitmask(mask: int, num_classes: int) -> List[int]:
    out: List[int] = []
    m = int(mask)
    for c in range(max(0, int(num_classes))):
        if (m & (1 << c)) != 0:
            out.append(int(c))
    return out


def _render_source_shape_for_stream(stream_len: int, cfg: RenderConfig):
    downsample = max(1, int(cfg.downsample))
    width = max(1, int(cfg.width))
    base_len = max(1, (max(1, int(stream_len)) + downsample - 1) // downsample)
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


def _resize_chw_nearest(img_chw: np.ndarray, height: int, width: int) -> np.ndarray:
    t = torch.from_numpy(img_chw.astype(np.float32, copy=False)).unsqueeze(0)
    out = F.interpolate(t, size=(max(1, int(height)), max(1, int(width))), mode="nearest")
    return out.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _compose_payload_canvas(
    payload_images: Sequence[np.ndarray],
    payload_masks: Sequence[int],
    rng: np.random.Generator,
    target_h: int,
    target_w: int,
    min_payloads: int,
    max_payloads: int,
):
    if len(payload_images) <= 0:
        return None, 0, 0
    lo = max(1, int(min_payloads))
    hi = max(lo, int(max_payloads))
    k = int(rng.integers(lo, hi + 1))
    if k <= 0:
        k = 1

    idx = rng.choice(np.arange(len(payload_images)), size=int(k), replace=(k > len(payload_images))).astype(np.int64)
    canvas = np.zeros((3, max(1, int(target_h)), max(1, int(target_w))), dtype=np.float32)
    rows = np.linspace(0, int(target_h), num=int(k) + 1, dtype=np.int64)
    target_mask = 0
    used = 0
    for i, pidx in enumerate(idx.tolist()):
        r0 = int(rows[i])
        r1 = int(rows[i + 1])
        if r1 <= r0:
            continue
        patch = _resize_chw_nearest(payload_images[int(pidx)], height=(r1 - r0), width=int(target_w))
        canvas[:, r0:r1, :] = patch
        target_mask |= int(payload_masks[int(pidx)])
        used += 1
    return canvas, int(target_mask), int(used)


def _payload_canvas_to_base_sequence(canvas_chw: np.ndarray, shape_info: Dict, empty_fill: float):
    mapping = shape_info["mapping"]
    special = bool(shape_info["special"])
    base_len = int(shape_info["base_len"])
    max_len = int(shape_info["max_len"])
    total = int(shape_info["total"])

    if not special:
        streams = []
        if mapping.get("R") is not None:
            streams.append(canvas_chw[0].reshape(-1))
        if mapping.get("G") is not None:
            streams.append(canvas_chw[1].reshape(-1))
        if mapping.get("B") is not None:
            streams.append(canvas_chw[2].reshape(-1))
        if len(streams) == 0:
            flat = np.full((int(total),), float(empty_fill), dtype=np.float32)
        elif len(streams) == 1:
            flat = streams[0].astype(np.float32, copy=False)
        else:
            flat = np.mean(np.stack(streams, axis=0).astype(np.float32, copy=False), axis=0)
        if flat.size < max_len:
            pad = np.full((max_len - flat.size,), float(empty_fill), dtype=np.float32)
            flat = np.concatenate([flat, pad], axis=0)
        return flat[:base_len].astype(np.float32, copy=False)

    r = canvas_chw[0].reshape(-1).astype(np.float32, copy=False)
    g = canvas_chw[1].reshape(-1).astype(np.float32, copy=False)
    b = canvas_chw[2].reshape(-1).astype(np.float32, copy=False)
    if r.size < max_len:
        r = np.concatenate([r, np.full((max_len - r.size,), float(empty_fill), dtype=np.float32)], axis=0)
    if g.size < max_len:
        g = np.concatenate([g, np.full((max_len - g.size,), float(empty_fill), dtype=np.float32)], axis=0)
    if b.size < max_len:
        b = np.concatenate([b, np.full((max_len - b.size,), float(empty_fill), dtype=np.float32)], axis=0)
    idx = np.arange(base_len, dtype=np.int64)
    src_idx = idx // 3
    mod = idx % 3
    out = np.where(mod == 0, r[src_idx], np.where(mod == 1, g[src_idx], b[src_idx]))
    return out.astype(np.float32, copy=False)


def _embed_base_sequence_into_stream(
    stream_f32: np.ndarray,
    base_seq_01: np.ndarray,
    cfg: RenderConfig,
    sample_bits: int,
):
    x = np.asarray(stream_f32, dtype=np.float32).copy()
    shape = _render_source_shape_for_stream(stream_len=int(x.size), cfg=cfg)
    base_len = int(shape["base_len"])
    downsample = int(shape["downsample"])
    n_use = min(int(base_len), int(base_seq_01.size))
    if n_use <= 0:
        return x

    idx = (np.arange(n_use, dtype=np.int64) * downsample).astype(np.int64)
    idx = idx[idx < int(x.size)]
    n_use = int(idx.size)
    if n_use <= 0:
        return x

    base_vals = np.clip(base_seq_01[:n_use], 0.0, 1.0).astype(np.float32, copy=False)
    if bool(cfg.bitmask_enable):
        lo, hi, depth = normalize_bit_window(int(cfg.bitmask_low), int(cfg.bitmask_high), int(sample_bits))
        full_mask = (1 << int(sample_bits)) - 1
        window_mask = (1 << int(depth)) - 1
        q_min = -(1 << (int(sample_bits) - 1))
        q_max = (1 << (int(sample_bits) - 1)) - 1
        q_scale = float(1 << (int(sample_bits) - 1))

        current = np.clip(x[idx], -1.0, 1.0).astype(np.float32, copy=False)
        q_signed = np.round(current * q_scale).astype(np.int64)
        q_signed = np.clip(q_signed, q_min, q_max).astype(np.int64)

        desired = np.round(base_vals * float(window_mask)).astype(np.int64)
        desired = np.clip(desired, 0, window_mask).astype(np.int64)

        unsigned = np.bitwise_and(q_signed, full_mask)
        clear_mask = np.int64(~(window_mask << int(lo)))
        unsigned = np.bitwise_and(unsigned, clear_mask)
        unsigned = np.bitwise_or(unsigned, np.left_shift(desired, int(lo)))
        sign_cut = (1 << (int(sample_bits) - 1))
        signed = np.where(unsigned >= sign_cut, unsigned - (1 << int(sample_bits)), unsigned)
        x[idx] = np.clip(signed.astype(np.float32) / q_scale, -1.0, 1.0)
        return x.astype(np.float32, copy=False)

    x[idx] = (base_vals * 2.0) - 1.0
    x[idx] = np.clip(x[idx], -1.0, 1.0)
    return x.astype(np.float32, copy=False)


def _build_berkeley_payload_bank(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    max_samples: int,
    seed: int,
):
    try:
        from berkeley_sbd_pretrain import prepare_sbd_multilabel
    except ModuleNotFoundError:
        from toys_to_survive_development.berkeley_sbd_pretrain import prepare_sbd_multilabel

    _, val_ds = prepare_sbd_multilabel(
        data_root=data_root,
        image_size=int(image_size),
        auto_install_scipy=bool(auto_install_scipy),
    )
    n = int(len(val_ds))
    if n <= 0:
        return [], [], {"available": 0, "used": 0}

    take = n if int(max_samples) <= 0 else min(n, int(max_samples))
    rng = np.random.default_rng(seed)
    if take >= n:
        picks = np.arange(n, dtype=np.int64)
    else:
        picks = rng.choice(np.arange(n, dtype=np.int64), size=take, replace=False).astype(np.int64)

    images: List[np.ndarray] = []
    masks: List[int] = []
    for idx in picks.tolist():
        try:
            x, y = val_ds[int(idx)]
            x01 = torch.clamp((x.float() * 0.5) + 0.5, 0.0, 1.0).cpu().numpy().astype(np.float32, copy=False)
            cls = torch.nonzero(y > 0.5, as_tuple=False).squeeze(1).cpu().numpy().astype(np.int64, copy=False)
            if cls.size <= 0:
                continue
            mask = _encode_target_classes_bitmask(cls.tolist())
            if mask <= 0:
                continue
            images.append(x01)
            masks.append(int(mask))
        except Exception:
            continue

    return images, masks, {"available": n, "used": len(images)}


def _imprint_latent_stream_targets(
    streams: Sequence[np.ndarray],
    metas: Sequence[Dict],
    payload_images: Sequence[np.ndarray],
    payload_masks: Sequence[int],
    num_classes: int,
    chunk_samples: int,
    classifier: nn.Module,
    cfg: RenderConfig,
    sample_bits: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    seed: int,
    steps: int,
    lr: float,
    l2_weight: float,
    min_target_classes: int,
    max_target_classes: int,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
):
    if len(streams) != len(metas):
        raise RuntimeError(
            f"Latent imprint expects stream/meta parity, got streams={len(streams)} metas={len(metas)}"
        )

    rng = np.random.default_rng(seed)
    out_streams: List[np.ndarray] = []
    out_targets: List[int] = []
    mix_total = 0
    noise_total = 0
    imprinted = 0
    probs: List[float] = []
    target_class_count: List[int] = []
    empty_fill = float(int(cfg.empty_fill) & 0xFF) / 255.0

    for s, m in zip(streams, metas):
        kind = _latent_pool_stream_kind(m.get("path", ""))
        if kind == "noise":
            noise_total += 1
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(-1)
            continue
        if kind != "mix":
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(-1)
            continue

        mix_total += 1
        if int(num_classes) <= 0 or len(payload_images) <= 0:
            out_streams.append(np.asarray(s, dtype=np.float32))
            out_targets.append(-1)
            continue

        shape = _render_source_shape_for_stream(stream_len=int(max(1, s.size)), cfg=cfg)
        canvas, target_mask, n_target = _compose_payload_canvas(
            payload_images=payload_images,
            payload_masks=payload_masks,
            rng=rng,
            target_h=int(shape["height"]),
            target_w=int(shape["width"]),
            min_payloads=int(min_target_classes),
            max_payloads=int(max_target_classes),
        )
        if canvas is None or int(target_mask) <= 0:
            raise RuntimeError("Latent mix-target imprint failed to build a valid payload canvas.")
        base_seq = _payload_canvas_to_base_sequence(
            canvas_chw=canvas,
            shape_info=shape,
            empty_fill=float(empty_fill),
        )
        out_stream = _embed_base_sequence_into_stream(
            stream_f32=np.asarray(s, dtype=np.float32),
            base_seq_01=base_seq,
            cfg=cfg,
            sample_bits=int(sample_bits),
        )
        out_streams.append(np.clip(out_stream, -1.0, 1.0).astype(np.float32, copy=False))
        out_targets.append(int(target_mask))
        imprinted += 1
        with torch.no_grad():
            xb = torch.from_numpy(out_streams[-1][None, :]).to(device)
            amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                img = render_mono_wave_to_tensor(xb, cfg=cfg, image_hw=image_hw, sample_bits=int(sample_bits))
                if channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                logits = classifier(img)
            tgt = _decode_target_classes_from_bitmask(int(target_mask), int(num_classes))
            if len(tgt) > 0:
                probs.append(float(torch.sigmoid(logits[:, tgt]).mean().item()))
        target_class_count.append(int(n_target))

    info = {
        "streams": int(len(streams)),
        "mix_streams": int(mix_total),
        "noise_streams": int(noise_total),
        "imprinted_streams": int(imprinted),
        "target_supervised_streams": int(sum(1 for t in out_targets if int(t) > 0)),
        "mean_target_classes_per_stream": (float(np.mean(target_class_count)) if len(target_class_count) > 0 else 0.0),
        "mean_target_prob_after_imprint": (float(np.mean(probs)) if len(probs) > 0 else 0.0),
    }
    return out_streams, out_targets, info


def parse_args():
    p = argparse.ArgumentParser(description="Config-search + classifier + waveform transformer training pipeline.")
    p.add_argument(
        "--wav-root",
        default="",
        help="Directory to recursively scan for .wav files. If omitted, latent fallback pool is generated.",
    )
    p.add_argument("--output-dir", default="toys_to_survive_development/wav_pipeline_runs/latest")
    p.add_argument("--max-files", type=int, default=300)
    p.add_argument("--latent-fallback-count", type=int, default=192)
    p.add_argument("--latent-fallback-seconds", type=float, default=2.0)
    p.add_argument("--latent-fallback-rate", type=int, default=16000)
    p.add_argument("--latent-noise-std", type=float, default=0.20)
    p.add_argument(
        "--latent-reinject-dir",
        default="",
        help="Optional accepted-wave library dir to reinject from (expects index.jsonl). Defaults to <output-dir>/accepted_wave_library.",
    )
    p.add_argument("--latent-reinject-ratio", type=float, default=0.50)
    p.add_argument("--latent-reinject-copy-gain", type=float, default=0.70)
    p.add_argument("--latent-reinject-noise-gain", type=float, default=0.35)
    p.add_argument("--latent-structured-ratio", type=float, default=0.85)
    p.add_argument("--latent-structured-gain", type=float, default=0.80)
    p.add_argument("--latent-structured-noise-gain", type=float, default=0.20)
    p.add_argument(
        "--latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_true",
        help="When latent fallback is active in berkeley mode, write deterministic Berkeley image payloads into mix-stream target bits.",
    )
    p.add_argument(
        "--no-latent-berkeley-imprint-mix-targets",
        dest="latent_berkeley_imprint_mix_targets",
        action="store_false",
    )
    p.add_argument("--latent-berkeley-imprint-steps", type=int, default=12)
    p.add_argument("--latent-berkeley-imprint-lr", type=float, default=0.08)
    p.add_argument("--latent-berkeley-imprint-l2", type=float, default=0.02)
    p.add_argument("--latent-berkeley-imprint-min-target-classes", type=int, default=2)
    p.add_argument("--latent-berkeley-imprint-max-target-classes", type=int, default=4)
    p.add_argument(
        "--resume-from",
        default="",
        help="Optional prior run directory. If set, load checkpoints/config/history from there.",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="Auto-resume from current output-dir artifacts when present.",
    )
    p.add_argument(
        "--force-config-search",
        action="store_true",
        help="When resuming, ignore prior best config and run config search again.",
    )
    p.add_argument(
        "--checkpoint-every-round",
        type=int,
        default=1,
        help="Save pipeline checkpoint every N orchestration rounds (0 disables periodic snapshots).",
    )
    p.add_argument(
        "--checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_true",
        help="Save pipeline/model checkpoints immediately after each completed training segment.",
    )
    p.add_argument(
        "--no-checkpoint-after-training-segment",
        dest="checkpoint_after_training_segment",
        action="store_false",
    )
    p.add_argument(
        "--objective-mode",
        choices=["berkeley_multilabel", "wave_labels"],
        default="berkeley_multilabel",
        help="Search/optimize objective source. 'berkeley_multilabel' is the intended feature-presence metric mode.",
    )
    p.add_argument("--label-mode", choices=["folder", "spectral"], default="folder")
    p.add_argument("--pseudo-classes", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision on CUDA.")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--channels-last", action="store_true", help="Use NHWC memory format for image tensors/models.")
    p.add_argument("--compile-models", action="store_true", help="Use torch.compile for classifier/transformer models.")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--grad-accum-steps", type=int, default=1, help="Micro-batch accumulation factor.")
    p.add_argument("--pin-memory-wave-batches", action="store_true", help="Pin sampled wave batches before H2D copy.")
    p.add_argument("--cache-wave-cls-dataset-on-device", action="store_true", help="Cache rendered wave-classifier datasets on GPU.")
    p.add_argument("--cache-wave-streams-on-device", action="store_true", help="Cache waveform stream pools on GPU for vectorized batch sampling.")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--matmul-precision", choices=["high", "medium", "highest"], default="high")

    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--render-max-points", type=int, default=262144)

    p.add_argument("--config-trials", type=int, default=10)
    p.add_argument("--config-epochs", type=int, default=2)
    p.add_argument("--search-train-limit", type=int, default=96)
    p.add_argument("--search-val-limit", type=int, default=48)
    p.add_argument("--try-scipy-refine", action="store_true")
    p.add_argument("--score-topk", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=0.35)
    p.add_argument("--score-w-topk", type=float, default=0.60)
    p.add_argument("--score-w-cov", type=float, default=0.30)
    p.add_argument("--score-w-mean", type=float, default=0.10)

    p.add_argument("--classifier-epochs", type=int, default=6)
    p.add_argument("--classifier-batch-size", type=int, default=32)
    p.add_argument("--classifier-lr", type=float, default=2e-3)
    p.add_argument("--lr-sine-cycles", type=float, default=1.0)
    p.add_argument("--lr-sine-frequency", type=float, default=0.0, help="Cycles per optimizer step; overrides --lr-sine-cycles when > 0.")
    p.add_argument("--lr-sine-tail-fraction", type=float, default=0.15)
    p.add_argument("--lr-sine-min-scale", type=float, default=0.0)
    p.add_argument(
        "--classifier-init-ckpt",
        default="",
        help="Optional pretrained classifier checkpoint (for warm-starting wave classifier).",
    )
    p.add_argument(
        "--classifier-init-scope",
        choices=["features", "all"],
        default="features",
        help="When loading init checkpoint, load only feature extractor or all matching keys.",
    )
    p.add_argument(
        "--classifier-freeze-features",
        action="store_true",
        help="Freeze classifier feature extractor during wave-domain classifier training.",
    )
    p.add_argument("--berkeley-data-root", default="toys_to_survive_development/data/berkeley_sbd")
    p.add_argument("--berkeley-image-size", type=int, default=0, help="0 means reuse --image-size")
    p.add_argument("--berkeley-auto-install-scipy", action="store_true")
    p.add_argument("--berkeley-refresh-every", type=int, default=0, help="Run Berkeley refresh every N config trials; 0=off.")
    p.add_argument(
        "--berkeley-refresh-round-every",
        type=int,
        default=0,
        help="In orchestration mode, run Berkeley refresh every N rounds; 0=off.",
    )
    p.add_argument("--berkeley-refresh-epochs", type=int, default=1)
    p.add_argument("--berkeley-refresh-lr", type=float, default=2e-4)
    p.add_argument("--berkeley-refresh-weight-decay", type=float, default=1e-4)
    p.add_argument("--berkeley-refresh-batch-size", type=int, default=32, help="Refresh train batch size. Set 0 to auto-size from GPU free memory when cache is enabled.")
    p.add_argument("--berkeley-refresh-loader-batch-size", type=int, default=128, help="Batch size used only while building Berkeley refresh cache/loader input queue.")
    p.add_argument("--berkeley-refresh-workers", type=int, default=2)
    p.add_argument("--berkeley-refresh-log-every", type=int, default=0, help="0 disables periodic Berkeley-refresh progress logs.")
    p.add_argument("--berkeley-refresh-max-seconds", type=float, default=0.0, help="0 disables wall-time cap per refresh call.")
    p.add_argument(
        "--berkeley-refresh-cache-batches",
        type=int,
        default=0,
        help="If >0, pre-cache this many Berkeley refresh batches for repeated GPU-fed refresh.",
    )
    p.add_argument(
        "--berkeley-refresh-cache-device",
        choices=["none", "auto", "cpu", "cuda"],
        default="none",
        help="Storage device for cached Berkeley refresh pool.",
    )
    p.add_argument("--berkeley-refresh-vram-fraction", type=float, default=0.90, help="Target fraction of free VRAM to fill for auto refresh batch.")
    p.add_argument("--berkeley-refresh-activation-mult", type=float, default=10.0, help="Activation overhead multiplier used for auto refresh batch sizing.")
    p.add_argument("--berkeley-refresh-max-batch-cap", type=int, default=0, help="Hard cap for auto refresh batch size (0 = no cap).")
    p.add_argument("--loader-persistent-workers", dest="loader_persistent_workers", action="store_true")
    p.add_argument("--no-loader-persistent-workers", dest="loader_persistent_workers", action="store_false")
    p.add_argument("--loader-prefetch-factor", type=int, default=4)
    p.add_argument("--berkeley-refresh-max-train", type=int, default=0, help="0 means full Berkeley train set.")
    p.add_argument("--berkeley-refresh-max-steps", type=int, default=0, help="0 means full epoch over refresh loader.")
    p.add_argument("--final-train-limit", type=int, default=0, help="0 means all.")
    p.add_argument("--final-val-limit", type=int, default=0, help="0 means all.")

    p.add_argument("--transformer-epochs", type=int, default=6)
    p.add_argument("--transformer-steps", type=int, default=80)
    p.add_argument("--transformer-batch-size", type=int, default=16)
    p.add_argument("--transformer-log-every", type=int, default=0, help="If >0, print transformer stage timing/progress every N train steps.")
    p.add_argument(
        "--transformer-visualize-status",
        dest="transformer_visualize_status",
        action="store_true",
        help="Open pygame OpenGL viewer and blit transformer input/output images at each status report.",
    )
    p.add_argument("--no-transformer-visualize-status", dest="transformer_visualize_status", action="store_false")
    p.add_argument("--transformer-viz-scale", type=int, default=3, help="Display scale multiplier for status viewer.")
    p.add_argument("--transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_true")
    p.add_argument("--no-transformer-cache-eval-batches", dest="transformer_cache_eval_batches", action="store_false")
    p.add_argument("--transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_true")
    p.add_argument("--no-transformer-degrade-inputs", dest="transformer_degrade_inputs", action="store_false")
    p.add_argument("--transformer-degrade-min-strength", type=float, default=0.25)
    p.add_argument("--transformer-degrade-max-strength", type=float, default=0.95)
    p.add_argument("--transformer-degrade-noise-std-min", type=float, default=0.01)
    p.add_argument("--transformer-degrade-noise-std-max", type=float, default=0.14)
    p.add_argument("--transformer-degrade-dropout-max", type=float, default=0.25)
    p.add_argument("--transformer-degrade-quant-bits-min", type=int, default=3)
    p.add_argument("--transformer-degrade-quant-bits-max", type=int, default=10)
    p.add_argument("--transformer-loss-entropy-weight", type=float, default=0.60)
    p.add_argument("--transformer-loss-high-bit-weight", type=float, default=0.50)
    p.add_argument("--transformer-loss-low-bit-weight", type=float, default=0.08)
    p.add_argument("--transformer-loss-wave-l1-weight", type=float, default=1.00)
    p.add_argument(
        "--transformer-loss-score-target-weight",
        type=float,
        default=1.50,
        help="Weight for keeping transformed-wave classifier feature score at/above clean-wave target.",
    )
    p.add_argument(
        "--transformer-loss-score-target-margin",
        type=float,
        default=0.02,
        help="Required improvement margin for target score (after vs before); loss penalizes shortfall.",
    )
    p.add_argument(
        "--transformer-accepted-preload-max",
        type=int,
        default=0,
        help="If >0, preload up to this many accepted-library waves into transformer train streams before rounds start.",
    )
    p.add_argument(
        "--transformer-accepted-append-max-per-round",
        type=int,
        default=0,
        help="If >0, append at most this many newly accepted waves per round into transformer train streams (0=all).",
    )
    p.add_argument(
        "--orchestration-mode",
        choices=["none", "sequential", "alternating"],
        default="sequential",
        help="In berkeley_multilabel mode: add wave-classifier co-training schedule.",
    )
    p.add_argument("--orchestration-cycles", type=int, default=3)
    p.add_argument("--orchestration-rounds", type=int, default=2, help="Used when mode=alternating.")
    p.add_argument("--transformer-epochs-per-round", type=int, default=1)
    p.add_argument("--transformer-steps-per-round", type=int, default=40)
    p.add_argument("--wave-cls-epochs-per-round", type=int, default=1)
    p.add_argument("--wave-cls-train-samples", type=int, default=768)
    p.add_argument("--wave-cls-val-samples", type=int, default=256)
    p.add_argument("--wave-cls-batch-size", type=int, default=32)
    p.add_argument("--wave-cls-lr", type=float, default=1e-3)
    p.add_argument(
        "--wave-cls-accepted-only",
        action="store_true",
        help="Train wave classifier on only accepted transformer outputs (score/l1 gated).",
    )
    p.add_argument(
        "--gate-berkeley-min-score",
        type=float,
        default=0.0,
        help="Legacy wave-score floor gate (kept for compatibility).",
    )
    p.add_argument(
        "--gate-berkeley-min-confidence",
        type=float,
        default=0.0,
        help="If >0, Berkeley classifier mean-confidence gate (evaluated on Berkeley val set).",
    )
    p.add_argument(
        "--gate-berkeley-min-macro-f1",
        type=float,
        default=0.0,
        help="If >0, Berkeley classifier macro-F1 gate (evaluated on Berkeley val set).",
    )
    p.add_argument(
        "--gate-berkeley-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive Berkeley gate passes before downstream is allowed.",
    )
    p.add_argument("--gate-berkeley-val-max", type=int, default=512, help="Max Berkeley-val samples used for classifier-confidence gate.")
    p.add_argument("--gate-berkeley-eval-max-steps", type=int, default=24, help="Max Berkeley-val batches per gate eval; 0=full loader.")
    p.add_argument(
        "--gate-wave-min-entropy",
        type=float,
        default=0.12,
        help="Wave-material gate: minimum mean binary entropy on wave-rendered logits.",
    )
    p.add_argument(
        "--gate-wave-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive wave-entropy gate passes before transformer stage.",
    )
    p.add_argument(
        "--gate-transformer-min-score-after",
        type=float,
        default=0.0,
        help="If >0, transformer post-score gate threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-min-gain",
        type=float,
        default=-1e9,
        help="Transformer gain threshold before wave-classifier stage.",
    )
    p.add_argument(
        "--gate-transformer-maintain-rounds",
        type=int,
        default=1,
        help="Required consecutive transformer gate passes before wave-classifier stage.",
    )
    p.add_argument("--accept-score-threshold", type=float, default=0.40)
    p.add_argument("--accept-l1-threshold", type=float, default=0.20)
    p.add_argument("--library-max-per-round", type=int, default=24)
    p.add_argument("--chunk-samples", type=int, default=32768)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--transformer-lr", type=float, default=2e-4)
    p.add_argument("--max-delta", type=float, default=0.2)
    p.add_argument("--stream-max-points", type=int, default=262144)
    p.add_argument("--save-preview", type=int, default=2)
    p.set_defaults(
        cudnn_benchmark=True,
        allow_tf32=True,
        loader_persistent_workers=True,
        transformer_cache_eval_batches=True,
        transformer_degrade_inputs=True,
        transformer_visualize_status=False,
        latent_berkeley_imprint_mix_targets=True,
        checkpoint_after_training_segment=False,
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device if ("cuda" not in args.device or torch.cuda.is_available()) else "cpu")
    if ("cuda" in str(args.device).lower()) and (device.type != "cuda"):
        _log(f"Requested CUDA device '{args.device}' but CUDA is unavailable; falling back to CPU.")
    configure_torch_runtime(
        device=device,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        allow_tf32=bool(args.allow_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_accum_steps = max(1, int(args.grad_accum_steps))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {out_dir}")
    _log(f"Device: {device}")
    _log(
        "Perf: "
        f"amp={amp_enabled}({args.amp_dtype}), channels_last={bool(args.channels_last)}, "
        f"compile={bool(args.compile_models)}({args.compile_mode}), grad_accum={grad_accum_steps}, "
        f"wave_stream_cache={bool(args.cache_wave_streams_on_device)}, "
        f"tf32={bool(args.allow_tf32)}, cudnn_benchmark={bool(args.cudnn_benchmark)}"
    )
    _log(
        "Feature metric: "
        f"score_threshold={float(args.score_threshold):.4f}, "
        f"w_topk={float(args.score_w_topk):.3f}, w_cov={float(args.score_w_cov):.3f}, w_mean={float(args.score_w_mean):.3f}"
    )
    _log(
        "Transformer objective: classifier-score-target + entropy + high-bit + low-bit + waveform-L1 "
        f"(w_score_target={float(args.transformer_loss_score_target_weight):.3f}, "
        f"score_margin={float(args.transformer_loss_score_target_margin):.3f}, "
        f"w_entropy={float(args.transformer_loss_entropy_weight):.3f}, "
        f"w_hi={float(args.transformer_loss_high_bit_weight):.3f}, "
        f"w_wave={float(args.transformer_loss_wave_l1_weight):.3f}, "
        f"w_lo={float(args.transformer_loss_low_bit_weight):.3f})"
    )
    _log(
        "Degrade curriculum: "
        f"enabled={1 if bool(args.transformer_degrade_inputs) else 0}, "
        f"strength=[{float(args.transformer_degrade_min_strength):.3f},{float(args.transformer_degrade_max_strength):.3f}], "
        f"noise_std=[{float(args.transformer_degrade_noise_std_min):.3f},{float(args.transformer_degrade_noise_std_max):.3f}], "
        f"dropout_max={float(args.transformer_degrade_dropout_max):.3f}, "
        f"quant_bits=[{int(args.transformer_degrade_quant_bits_min)},{int(args.transformer_degrade_quant_bits_max)}]"
    )
    _log(
        "Latent Berkeley imprint: "
        f"enabled={1 if bool(args.latent_berkeley_imprint_mix_targets) else 0}, "
        f"steps={int(args.latent_berkeley_imprint_steps)}, "
        f"lr={float(args.latent_berkeley_imprint_lr):.4f}, "
        f"l2={float(args.latent_berkeley_imprint_l2):.4f}, "
        f"targets_per_mix=[{int(args.latent_berkeley_imprint_min_target_classes)},{int(args.latent_berkeley_imprint_max_target_classes)}]"
    )
    _log(
        "Transformer status viz: "
        f"enabled={1 if bool(args.transformer_visualize_status) else 0}, "
        f"scale={max(1, int(args.transformer_viz_scale))}"
    )
    _log(
        "Checkpoint policy: "
        f"every_round={int(args.checkpoint_every_round)}, "
        f"after_training_segment={1 if bool(args.checkpoint_after_training_segment) else 0}"
    )

    t0 = time.time()
    latent_fallback_info = None
    wav_data_root = str(args.wav_root).strip()
    wav_paths: List[str] = []

    if wav_data_root:
        wav_paths = discover_wavs(wav_data_root)
        if len(wav_paths) == 0:
            _log(f"No wav files found under: {wav_data_root}; switching to latent fallback.")

    if len(wav_paths) == 0:
        latent_fallback_info = _bootstrap_latent_wav_pool(
            out_dir=out_dir,
            seed=args.seed,
            count=args.latent_fallback_count,
            framerate=args.latent_fallback_rate,
            seconds=args.latent_fallback_seconds,
            noise_std=args.latent_noise_std,
            reinject_dir=args.latent_reinject_dir,
            reinject_ratio=max(0.0, min(1.0, float(args.latent_reinject_ratio))),
            reinject_copy_gain=float(args.latent_reinject_copy_gain),
            reinject_noise_gain=float(args.latent_reinject_noise_gain),
            structured_ratio=max(0.0, min(1.0, float(args.latent_structured_ratio))),
            structured_gain=float(args.latent_structured_gain),
            structured_noise_gain=float(args.latent_structured_noise_gain),
        )
        wav_data_root = str(latent_fallback_info["pool_dir"])
        wav_paths = discover_wavs(wav_data_root)
        _log(
            "Latent fallback active: "
            f"root={wav_data_root}, wavs={len(wav_paths)}, mixed={latent_fallback_info['mixed_count']}, "
            f"structured={latent_fallback_info.get('structured_count', 0)}, "
            f"reinject_candidates={latent_fallback_info['reinject_candidates']}"
        )

    if len(wav_paths) == 0:
        raise RuntimeError("No wav files available after latent fallback bootstrap.")
    if args.max_files > 0:
        wav_paths = wav_paths[: args.max_files]
    _log(f"Loading {len(wav_paths)} wav files from: {wav_data_root}")

    records: List[WaveRecord] = []
    for p in wav_paths:
        try:
            records.append(read_wav_record(p))
        except Exception as e:
            _log(f"[skip] {p} ({e})")
    if len(records) < 8:
        raise RuntimeError("Need at least 8 readable WAV files.")

    resume_enabled = bool(args.auto_resume or str(args.resume_from).strip())
    resume_dir = Path(str(args.resume_from).strip()) if str(args.resume_from).strip() else out_dir
    resume_summary_path = resume_dir / "summary.json"
    resume_cfg_path = resume_dir / "best_render_config.json"
    resume_search_path = resume_dir / "search_results.json"
    resume_classifier_path = resume_dir / "classifier.pt"
    resume_transformer_path = resume_dir / "transformer.pt"
    resume_wave_classifier_path = resume_dir / "wave_classifier.pt"
    resume_pipeline_ckpt_path = resume_dir / "pipeline_checkpoint.pt"

    resume_summary = _load_json(resume_summary_path) if resume_enabled else None
    resume_cfg = _load_json(resume_cfg_path) if resume_enabled else None
    resume_pipeline_ckpt = None
    if resume_enabled and resume_pipeline_ckpt_path.exists():
        try:
            resume_pipeline_ckpt = torch.load(str(resume_pipeline_ckpt_path), map_location="cpu")
        except Exception:
            resume_pipeline_ckpt = None

    if resume_enabled:
        _log(
            "Resume mode: "
            f"dir={resume_dir}, summary={resume_summary_path.exists()}, "
            f"cfg={resume_cfg_path.exists()}, ckpt={resume_pipeline_ckpt_path.exists()}"
        )

    run_tag = time.strftime("%Y%m%d_%H%M%S")

    if args.objective_mode == "berkeley_multilabel":
        _log("Objective mode: berkeley_multilabel")
        label_mode = "folder" if args.label_mode == "folder" else "spectral"
        labels_for_split, split_class_names, label_source = _build_labels(
            records=records,
            data_root=wav_data_root,
            label_mode=label_mode,
            pseudo_classes=args.pseudo_classes,
        )
        split_num_classes = len(set(labels_for_split.tolist()))
        if split_num_classes < 2:
            raise RuntimeError("Need at least 2 classes for data splitting.")
        if len(split_class_names) != split_num_classes:
            split_class_names = [f"wave_cls_{i}" for i in range(split_num_classes)]
        train_idx, val_idx = _stratified_split(labels_for_split, train_frac=args.train_frac, seed=args.seed)
        _log(f"Split: train={len(train_idx)} val={len(val_idx)} (source={label_source})")

        search_train_idx = _sample_indices(train_idx, args.search_train_limit, rng)
        search_val_idx = _sample_indices(val_idx, args.search_val_limit, rng)
        final_train_idx = _sample_indices(train_idx, args.final_train_limit, rng)
        final_val_idx = _sample_indices(val_idx, args.final_val_limit, rng)
        search_indices = sorted(set(search_train_idx + search_val_idx))

        image_hw = (int(args.image_size), int(args.image_size))
        candidates = _random_configs(num_trials=args.config_trials, rng=rng, max_points=args.render_max_points)

        classifier, classifier_init_info = _load_berkeley_classifier_for_metric(
            args.classifier_init_ckpt,
            device=device,
        )
        num_classes = int(classifier_init_info["num_classes"])
        class_names = list(classifier_init_info.get("class_names", []))
        if len(class_names) != num_classes:
            class_names = [f"berkeley_cls_{i}" for i in range(num_classes)]
        classifier_resume_info = {"used": False}
        if resume_enabled and resume_classifier_path.exists():
            classifier_resume_info = _apply_model_init(classifier, str(resume_classifier_path))
            _log(
                f"Resumed Berkeley classifier from {resume_classifier_path} "
                f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
            )
        if args.channels_last:
            classifier = classifier.to(memory_format=torch.channels_last)
        classifier = maybe_compile_module(
            classifier,
            enabled=bool(args.compile_models),
            mode=str(args.compile_mode),
        )

        refresh_loader = None
        refresh_cache = None
        refresh_run_batch_size = max(1, int(args.berkeley_refresh_batch_size))
        refresh_rows: List[Dict] = []
        if int(args.berkeley_refresh_epochs) > 0 and (
            int(args.berkeley_refresh_every) > 0 or int(args.berkeley_refresh_round_every) > 0
        ):
            refresh_size = int(args.berkeley_image_size) if int(args.berkeley_image_size) > 0 else int(args.image_size)
            refresh_loader_batch = (
                max(1, int(args.berkeley_refresh_loader_batch_size))
                if int(args.berkeley_refresh_batch_size) <= 0
                else max(1, int(args.berkeley_refresh_batch_size))
            )
            refresh_loader, refresh_count = _build_berkeley_refresh_loader(
                data_root=args.berkeley_data_root,
                image_size=refresh_size,
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                batch_size=refresh_loader_batch,
                num_workers=args.berkeley_refresh_workers,
                max_train=args.berkeley_refresh_max_train,
                seed=args.seed + 7300,
                device=device,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            _log(
                "Berkeley refresh loader ready: "
                f"search_every={args.berkeley_refresh_every}, round_every={args.berkeley_refresh_round_every}, "
                f"epochs={args.berkeley_refresh_epochs}, train={refresh_count}"
            )
            if int(args.berkeley_refresh_cache_batches) > 0:
                refresh_cache = _build_berkeley_refresh_cache(
                    loader=refresh_loader,
                    device=device,
                    cache_batches=int(args.berkeley_refresh_cache_batches),
                    cache_device=str(args.berkeley_refresh_cache_device),
                    channels_last=bool(args.channels_last),
                )
                if isinstance(refresh_cache, dict):
                    _log(
                        "Berkeley refresh cache ready: "
                        f"samples={refresh_cache.get('num_samples', 0)}, "
                        f"batches={refresh_cache.get('num_batches', 0)}, "
                        f"device={refresh_cache.get('device', 'cpu')}"
                    )
            if int(args.berkeley_refresh_batch_size) <= 0:
                if isinstance(refresh_cache, dict):
                    refresh_run_batch_size = _auto_berkeley_refresh_batch_size(
                        cache_x=refresh_cache.get("x"),
                        cache_y=refresh_cache.get("y"),
                        device=device,
                        vram_fraction=float(args.berkeley_refresh_vram_fraction),
                        activation_multiplier=float(args.berkeley_refresh_activation_mult),
                        max_cap=int(args.berkeley_refresh_max_batch_cap),
                    )
                else:
                    refresh_run_batch_size = int(refresh_loader_batch)
                _log(f"Auto Berkeley refresh batch size: {refresh_run_batch_size}")
            else:
                refresh_run_batch_size = max(1, int(args.berkeley_refresh_batch_size))

        berkeley_gate_loader = None
        if float(args.gate_berkeley_min_confidence) > 0.0 or float(args.gate_berkeley_min_macro_f1) > 0.0:
            gate_size = int(args.berkeley_image_size) if int(args.berkeley_image_size) > 0 else int(args.image_size)
            berkeley_gate_loader, gate_count = _build_berkeley_gate_val_loader(
                data_root=args.berkeley_data_root,
                image_size=gate_size,
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                batch_size=max(1, int(args.berkeley_refresh_loader_batch_size)),
                num_workers=max(0, int(args.berkeley_refresh_workers)),
                max_val=int(args.gate_berkeley_val_max),
                seed=args.seed + 7399,
                device=device,
                persistent_workers=bool(args.loader_persistent_workers),
                prefetch_factor=int(args.loader_prefetch_factor),
            )
            _log(
                "Berkeley confidence gate enabled: "
                f"min_conf={args.gate_berkeley_min_confidence}, min_macro_f1={args.gate_berkeley_min_macro_f1}, "
                f"val_samples={gate_count}"
            )

        best_cfg = None
        best_score = -1.0
        search_rows = []
        resume_search_rows = _load_json(resume_search_path) if (resume_enabled and resume_search_path.exists()) else None
        did_config_search = False

        if resume_enabled and (not args.force_config_search):
            if isinstance(resume_cfg, dict):
                try:
                    best_cfg = RenderConfig.from_dict(resume_cfg)
                    _log(f"Resumed best render config from {resume_cfg_path}")
                except Exception:
                    best_cfg = None
            if best_cfg is None and isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("best_cfg"), dict):
                try:
                    best_cfg = RenderConfig.from_dict(resume_pipeline_ckpt.get("best_cfg"))
                    _log("Resumed best render config from pipeline checkpoint")
                except Exception:
                    best_cfg = None
            if isinstance(resume_summary, dict):
                best_score = float(resume_summary.get("best_search_score", best_score))
                prior_refresh = resume_summary.get("berkeley_refresh_history", [])
                if isinstance(prior_refresh, list):
                    refresh_rows.extend(prior_refresh)
            if isinstance(resume_search_rows, list):
                search_rows = list(resume_search_rows)

        if best_cfg is None:
            did_config_search = True
            _log(f"Config search started on {len(search_indices)} rendered items...")
            for i, cfg in enumerate(candidates, start=1):
                try:
                    metric = _score_config_with_classifier(
                        records=records,
                        indices=search_indices,
                        cfg=cfg,
                        image_hw=image_hw,
                        classifier=classifier,
                        device=device,
                        batch_size=args.classifier_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                    )
                    score = float(metric["score"])
                    search_rows.append({"trial": i, "score": score, "cfg": cfg.to_dict(), **metric})
                    _log(
                        f"  [{i}/{len(candidates)}] score={score:.4f} topk={metric['topk_mean']:.4f} "
                        f"cov={metric['hard_coverage']:.4f}"
                    )
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg

                    if (
                        refresh_loader is not None
                        and int(args.berkeley_refresh_every) > 0
                        and (i % int(args.berkeley_refresh_every) == 0)
                    ):
                        ref = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=refresh_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(args.berkeley_refresh_max_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=(refresh_cache.get("x") if isinstance(refresh_cache, dict) else None),
                            cache_y=(refresh_cache.get("y") if isinstance(refresh_cache, dict) else None),
                            cache_batch_size=int(refresh_run_batch_size),
                        )
                        refresh_rows.append({"stage": "config_search", "trial": int(i), **ref})
                        _log(
                            "    refresh: "
                            f"loss={ref['loss']:.4f} samples={ref.get('samples', 0)} "
                            f"source={ref.get('source', 'loader')} elapsed={ref.get('elapsed_sec', 0.0):.1f}s"
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"config_refresh_trial_{int(i)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            refresh_history=refresh_rows,
                            extra={"best_search_score": float(best_score)},
                        )
                except Exception as e:
                    _log(f"  [{i}/{len(candidates)}] failed: {e}")

            if best_cfg is None:
                raise RuntimeError("Config search failed to produce a usable config.")
        else:
            _log("Skipping config search due to resume config (use --force-config-search to override).")

        if args.try_scipy_refine and did_config_search:
            refine_candidates = _try_scipy_refine(best_cfg, max_points=args.render_max_points, enabled=True)
            if len(refine_candidates) > 1:
                _log("Running optional refinement candidates...")
                for j, cfg in enumerate(refine_candidates[1:], start=1):
                    try:
                        metric = _score_config_with_classifier(
                            records=records,
                            indices=search_indices,
                            cfg=cfg,
                            image_hw=image_hw,
                            classifier=classifier,
                            device=device,
                            batch_size=args.classifier_batch_size,
                            score_topk=args.score_topk,
                            score_threshold=args.score_threshold,
                            score_w_topk=args.score_w_topk,
                            score_w_cov=args.score_w_cov,
                            score_w_mean=args.score_w_mean,
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                        )
                        score = float(metric["score"])
                        search_rows.append({"trial": len(search_rows) + 1, "score": score, "cfg": cfg.to_dict(), **metric})
                        _log(f"  [refine {j}] score={score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_cfg = cfg
                    except Exception as e:
                        _log(f"  [refine {j}] failed: {e}")

        _log(f"Best cfg score={best_score:.4f}: {best_cfg.to_dict()}")
        cls_val = _score_config_with_classifier(
            records=records,
            indices=final_val_idx,
            cfg=best_cfg,
            image_hw=image_hw,
            classifier=classifier,
            device=device,
            batch_size=args.classifier_batch_size,
            score_topk=args.score_topk,
            score_threshold=args.score_threshold,
            score_w_topk=args.score_w_topk,
            score_w_cov=args.score_w_cov,
            score_w_mean=args.score_w_mean,
            amp_enabled=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
        )
        _log(
            f"Best-config val metric: score={cls_val['score']:.4f} "
            f"topk={cls_val['topk_mean']:.4f} coverage={cls_val['hard_coverage']:.4f}"
        )

        gate_config = {
            "berkeley_min_score": float(args.gate_berkeley_min_score),
            "berkeley_min_confidence": float(args.gate_berkeley_min_confidence),
            "berkeley_min_macro_f1": float(args.gate_berkeley_min_macro_f1),
            "berkeley_maintain_rounds": max(1, int(args.gate_berkeley_maintain_rounds)),
            "wave_min_entropy": float(args.gate_wave_min_entropy),
            "wave_maintain_rounds": max(1, int(args.gate_wave_maintain_rounds)),
            "transformer_min_score_after": float(args.gate_transformer_min_score_after),
            "transformer_min_gain": float(args.gate_transformer_min_gain),
            "transformer_maintain_rounds": max(1, int(args.gate_transformer_maintain_rounds)),
        }

        train_streams, train_stream_labels, bits_train, train_meta = _prepare_streams(
            records=records,
            labels=labels_for_split,
            indices=final_train_idx,
            cfg=best_cfg,
            max_points=args.stream_max_points,
        )
        val_streams, val_stream_labels, bits_val, val_meta = _prepare_streams(
            records=records,
            labels=labels_for_split,
            indices=final_val_idx,
            cfg=best_cfg,
            max_points=args.stream_max_points,
        )
        if len(train_streams) < 4 or len(val_streams) < 2:
            raise RuntimeError("Not enough decoded streams for transformer stage.")

        bits_all = bits_train + bits_val
        sample_bits = int(Counter(bits_all).most_common(1)[0][0]) if len(bits_all) > 0 else 16
        chunk_samples = int(args.chunk_samples)
        if chunk_samples % int(args.patch_size) != 0:
            chunk_samples = (chunk_samples // int(args.patch_size)) * int(args.patch_size)
            if chunk_samples <= 0:
                raise RuntimeError("chunk-samples must be >= patch-size")

        if latent_fallback_info is not None and bool(args.latent_berkeley_imprint_mix_targets):
            if not bool(best_cfg.bitmask_enable):
                raise RuntimeError(
                    "Latent Berkeley mix-target imprint requires bitmask-enabled render config "
                    "(best cfg has bitmask_enable=0)."
                )
            lo, hi, depth = normalize_bit_window(
                int(best_cfg.bitmask_low),
                int(best_cfg.bitmask_high),
                int(sample_bits),
            )
            _log(f"Latent Berkeley imprint bit window: [{lo}..{hi}] ({depth} bits)")

        payload_images: List[np.ndarray] = []
        payload_masks: List[int] = []
        payload_bank_info = {"available": 0, "used": 0}
        if (
            latent_fallback_info is not None
            and bool(args.latent_berkeley_imprint_mix_targets)
            and int(num_classes) > 0
        ):
            payload_size = int(args.berkeley_image_size) if int(args.berkeley_image_size) > 0 else int(args.image_size)
            payload_max_samples = max(256, min(4096, int(args.latent_fallback_count) * 8))
            payload_images, payload_masks, payload_bank_info = _build_berkeley_payload_bank(
                data_root=args.berkeley_data_root,
                image_size=int(payload_size),
                auto_install_scipy=bool(args.berkeley_auto_install_scipy),
                max_samples=int(payload_max_samples),
                seed=args.seed + 7067,
            )
            _log(
                "Latent Berkeley payload bank: "
                f"available={payload_bank_info.get('available', 0)} "
                f"used={payload_bank_info.get('used', 0)} image_size={payload_size}"
            )
            if len(payload_images) <= 0:
                raise RuntimeError(
                    "Latent Berkeley imprint is enabled but no Berkeley payload images were prepared. "
                    "Check --berkeley-data-root and dataset availability."
                )

        train_stream_target_labels: Optional[List[int]] = None
        val_stream_target_labels: Optional[List[int]] = None
        latent_targeting_info = {"enabled": False, "payload_bank": payload_bank_info}
        if (
            latent_fallback_info is not None
            and bool(args.latent_berkeley_imprint_mix_targets)
            and int(num_classes) > 0
            and len(payload_images) > 0
        ):
            train_streams, train_stream_target_labels, train_target_info = _imprint_latent_stream_targets(
                streams=train_streams,
                metas=train_meta,
                payload_images=payload_images,
                payload_masks=payload_masks,
                num_classes=int(num_classes),
                chunk_samples=int(chunk_samples),
                classifier=classifier,
                cfg=best_cfg,
                sample_bits=int(sample_bits),
                image_hw=image_hw,
                device=device,
                seed=args.seed + 111,
                steps=int(args.latent_berkeley_imprint_steps),
                lr=float(args.latent_berkeley_imprint_lr),
                l2_weight=float(args.latent_berkeley_imprint_l2),
                min_target_classes=int(args.latent_berkeley_imprint_min_target_classes),
                max_target_classes=int(args.latent_berkeley_imprint_max_target_classes),
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
            )
            val_streams, val_stream_target_labels, val_target_info = _imprint_latent_stream_targets(
                streams=val_streams,
                metas=val_meta,
                payload_images=payload_images,
                payload_masks=payload_masks,
                num_classes=int(num_classes),
                chunk_samples=int(chunk_samples),
                classifier=classifier,
                cfg=best_cfg,
                sample_bits=int(sample_bits),
                image_hw=image_hw,
                device=device,
                seed=args.seed + 222,
                steps=int(args.latent_berkeley_imprint_steps),
                lr=float(args.latent_berkeley_imprint_lr),
                l2_weight=float(args.latent_berkeley_imprint_l2),
                min_target_classes=int(args.latent_berkeley_imprint_min_target_classes),
                max_target_classes=int(args.latent_berkeley_imprint_max_target_classes),
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
            )
            latent_targeting_info = {
                "enabled": True,
                "payload_bank": payload_bank_info,
                "train": train_target_info,
                "val": val_target_info,
            }
            _log(
                "Latent Berkeley target imprint: "
                f"train_imprinted={train_target_info['imprinted_streams']}/{train_target_info['mix_streams']} "
                f"(targeted={train_target_info['target_supervised_streams']}, "
                f"mean_targets={train_target_info['mean_target_classes_per_stream']:.2f}, "
                f"mean_prob={train_target_info['mean_target_prob_after_imprint']:.4f}), "
                f"val_imprinted={val_target_info['imprinted_streams']}/{val_target_info['mix_streams']} "
                f"(targeted={val_target_info['target_supervised_streams']}, "
                f"mean_targets={val_target_info['mean_target_classes_per_stream']:.2f}, "
                f"mean_prob={val_target_info['mean_target_prob_after_imprint']:.4f})"
            )

        transformer = WavePatchTransformer(
            chunk_samples=chunk_samples,
            patch_size=int(args.patch_size),
            d_model=96,
            nhead=4,
            num_layers=3,
            ff_mult=4,
            max_delta=float(args.max_delta),
            dropout=0.1,
        ).to(device)

        baseline = evaluate_feature_score_before_after(
            transformer=transformer,
            classifier=classifier,
            streams=val_streams,
            cfg=best_cfg,
            sample_bits=sample_bits,
            image_hw=image_hw,
            chunk_samples=chunk_samples,
            device=device,
            max_batches=16,
            batch_size=args.transformer_batch_size,
            score_topk=args.score_topk,
            score_threshold=args.score_threshold,
            score_w_topk=args.score_w_topk,
            score_w_cov=args.score_w_cov,
            score_w_mean=args.score_w_mean,
            amp=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
            pin_memory=bool(args.pin_memory_wave_batches),
            stream_cache_on_device=bool(args.cache_wave_streams_on_device),
        )
        _log(
            f"Transformer baseline: before={baseline['score_before']:.4f} "
            f"after={baseline['score_after']:.4f} gain={baseline['score_gain']:.4f}"
        )

        mode = str(args.orchestration_mode)
        transformer_hist: List[Dict[str, float]] = []
        wave_classifier_hist: List[Dict[str, float]] = []
        orchestration_rows: List[Dict] = []
        accepted_total: List[Dict] = []
        gate_history: List[Dict] = []
        wave_classifier: Optional[nn.Module] = None
        wave_classifier_init_info = {"used": False}
        wave_classifier_resume_info = {"used": False}
        transformer_resume_info = {"used": False}
        wave_classifier_last_eval = None
        library_dir = out_dir / "accepted_wave_library"
        library_index_path = library_dir / "index.jsonl"
        pipeline_checkpoint_path = out_dir / "pipeline_checkpoint.pt"

        if resume_enabled:
            if isinstance(resume_summary, dict):
                if isinstance(resume_summary.get("transformer_history"), list):
                    transformer_hist.extend(list(resume_summary.get("transformer_history", [])))
                if isinstance(resume_summary.get("wave_classifier_history"), list):
                    wave_classifier_hist.extend(list(resume_summary.get("wave_classifier_history", [])))
                elif isinstance(resume_summary.get("classifier_history"), list):
                    wave_classifier_hist.extend(list(resume_summary.get("classifier_history", [])))
                if isinstance(resume_summary.get("orchestration_history"), list):
                    orchestration_rows.extend(list(resume_summary.get("orchestration_history", [])))
                if resume_summary.get("wave_classifier_last_eval") is not None:
                    wave_classifier_last_eval = resume_summary.get("wave_classifier_last_eval")
                if isinstance(resume_summary.get("berkeley_refresh_history"), list) and len(refresh_rows) == 0:
                    refresh_rows.extend(list(resume_summary.get("berkeley_refresh_history", [])))
                if isinstance(resume_summary.get("gate_history"), list):
                    gate_history.extend(list(resume_summary.get("gate_history", [])))
            if isinstance(resume_pipeline_ckpt, dict):
                if isinstance(resume_pipeline_ckpt.get("transformer_history"), list):
                    transformer_hist = list(resume_pipeline_ckpt.get("transformer_history", transformer_hist))
                if isinstance(resume_pipeline_ckpt.get("wave_classifier_history"), list):
                    wave_classifier_hist = list(resume_pipeline_ckpt.get("wave_classifier_history", wave_classifier_hist))
                if isinstance(resume_pipeline_ckpt.get("orchestration_history"), list):
                    orchestration_rows = list(resume_pipeline_ckpt.get("orchestration_history", orchestration_rows))
                if isinstance(resume_pipeline_ckpt.get("refresh_history"), list):
                    refresh_rows = list(resume_pipeline_ckpt.get("refresh_history", refresh_rows))
                if isinstance(resume_pipeline_ckpt.get("gate_history"), list):
                    gate_history = list(resume_pipeline_ckpt.get("gate_history", gate_history))
                if resume_pipeline_ckpt.get("wave_classifier_last_eval") is not None:
                    wave_classifier_last_eval = resume_pipeline_ckpt.get("wave_classifier_last_eval")

        if resume_enabled and resume_transformer_path.exists():
            transformer_resume_info = _apply_model_init(transformer, str(resume_transformer_path))
            _log(
                f"Resumed transformer from {resume_transformer_path} "
                f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
            )
        elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_state"), dict):
            transformer_resume_info = _apply_state_dict(
                transformer,
                resume_pipeline_ckpt.get("transformer_state"),
                source_name="pipeline_checkpoint",
            )
            _log(
                "Resumed transformer from pipeline checkpoint "
                f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
            )
        transformer = maybe_compile_module(
            transformer,
            enabled=bool(args.compile_models),
            mode=str(args.compile_mode),
        )

        cycle_offset = 0
        for row in orchestration_rows:
            if isinstance(row, dict):
                try:
                    cycle_offset = max(cycle_offset, int(float(row.get("cycle", 0))))
                except Exception:
                    pass

        library_serial = 0
        if library_index_path.exists():
            try:
                with library_index_path.open("r", encoding="utf-8") as f:
                    library_serial = sum(1 for line in f if line.strip())
            except Exception:
                library_serial = 0

        berkeley_gate_streak = 0
        wave_gate_streak = 0
        transformer_gate_streak = 0
        if isinstance(resume_summary, dict):
            gs = resume_summary.get("gate_status", {})
            if isinstance(gs, dict):
                berkeley_gate_streak = int(gs.get("berkeley_streak", berkeley_gate_streak))
                wave_gate_streak = int(gs.get("wave_streak", wave_gate_streak))
                transformer_gate_streak = int(gs.get("transformer_streak", transformer_gate_streak))
        if isinstance(resume_pipeline_ckpt, dict):
            gs = resume_pipeline_ckpt.get("gate_status", {})
            if isinstance(gs, dict):
                berkeley_gate_streak = int(gs.get("berkeley_streak", berkeley_gate_streak))
                wave_gate_streak = int(gs.get("wave_streak", wave_gate_streak))
                transformer_gate_streak = int(gs.get("transformer_streak", transformer_gate_streak))

        initial_berkeley_eval = None
        if berkeley_gate_loader is not None:
            initial_berkeley_eval = _evaluate_berkeley_classifier_gate(
                classifier=classifier,
                loader=berkeley_gate_loader,
                device=device,
                max_steps=int(args.gate_berkeley_eval_max_steps),
                amp_enabled=amp_enabled,
                amp_dtype=args.amp_dtype,
                channels_last=bool(args.channels_last),
            )
        initial_berkeley_gate_pass = True
        if initial_berkeley_eval is not None:
            initial_berkeley_gate_pass = (
                (float(initial_berkeley_eval["mean_confidence"]) >= gate_config["berkeley_min_confidence"])
                and (float(initial_berkeley_eval["macro_f1"]) >= gate_config["berkeley_min_macro_f1"])
            )
        if initial_berkeley_gate_pass:
            berkeley_gate_streak = max(berkeley_gate_streak, 1)
        else:
            berkeley_gate_streak = 0
        initial_wave_gate_pass = (
            (float(cls_val.get("mean_entropy", 0.0)) >= gate_config["wave_min_entropy"])
            and (float(cls_val["score"]) >= gate_config["berkeley_min_score"])
        )
        if initial_wave_gate_pass:
            wave_gate_streak = max(wave_gate_streak, 1)
        else:
            wave_gate_streak = 0

        if mode == "none":
            berkeley_gate_ready = (
                berkeley_gate_streak >= gate_config["berkeley_maintain_rounds"]
            )
            wave_gate_ready = wave_gate_streak >= gate_config["wave_maintain_rounds"]
            if berkeley_gate_ready:
                if wave_gate_ready:
                    transformer, run_hist = train_transformer_feature_metric(
                        transformer=transformer,
                        classifier=classifier,
                        train_streams=train_streams,
                        train_target_labels=train_stream_target_labels,
                        val_streams=val_streams,
                        cfg=best_cfg,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        device=device,
                        epochs=args.transformer_epochs,
                        steps_per_epoch=args.transformer_steps,
                        batch_size=args.transformer_batch_size,
                        chunk_samples=chunk_samples,
                        lr=args.transformer_lr,
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        degrade_inputs=bool(args.transformer_degrade_inputs),
                        degrade_min_strength=float(args.transformer_degrade_min_strength),
                        degrade_max_strength=float(args.transformer_degrade_max_strength),
                        degrade_noise_std_min=float(args.transformer_degrade_noise_std_min),
                        degrade_noise_std_max=float(args.transformer_degrade_noise_std_max),
                        degrade_dropout_max=float(args.transformer_degrade_dropout_max),
                        degrade_quant_bits_min=int(args.transformer_degrade_quant_bits_min),
                        degrade_quant_bits_max=int(args.transformer_degrade_quant_bits_max),
                        entropy_penalty_weight=float(args.transformer_loss_entropy_weight),
                        high_bit_penalty_weight=float(args.transformer_loss_high_bit_weight),
                        low_bit_penalty_weight=float(args.transformer_loss_low_bit_weight),
                        wave_l1_weight=float(args.transformer_loss_wave_l1_weight),
                        score_target_weight=float(args.transformer_loss_score_target_weight),
                        score_target_margin=float(args.transformer_loss_score_target_margin),
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        pin_memory=bool(args.pin_memory_wave_batches),
                        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
                        log_every_steps=max(0, int(args.transformer_log_every)),
                        cache_eval_batches=bool(args.transformer_cache_eval_batches),
                        visualize_status=bool(args.transformer_visualize_status),
                        visualize_scale=max(1, int(args.transformer_viz_scale)),
                        seed=args.seed + 9000,
                    )
                    transformer_hist.extend(run_hist)
                    _save_training_segment_snapshot(
                        enabled=bool(args.checkpoint_after_training_segment),
                        out_dir=out_dir,
                        objective_mode=args.objective_mode,
                        run_tag=run_tag,
                        segment="mode_none_transformer_train",
                        best_cfg=best_cfg,
                        classifier=classifier,
                        transformer=transformer,
                        transformer_history=transformer_hist,
                        wave_classifier_history=wave_classifier_hist,
                        orchestration_history=orchestration_rows,
                        refresh_history=refresh_rows,
                        gate_history=gate_history,
                        gate_status={
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        extra={
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                        },
                    )
                else:
                    _log(
                        "Transformer training skipped by wave-entropy gate "
                        f"(streak={wave_gate_streak}/{gate_config['wave_maintain_rounds']}, "
                        f"min_entropy={gate_config['wave_min_entropy']:.4f})."
                    )
            else:
                _log(
                    "Transformer training skipped by Berkeley-confidence gate "
                    f"(streak={berkeley_gate_streak}/{gate_config['berkeley_maintain_rounds']}, "
                    f"min_conf={gate_config['berkeley_min_confidence']:.4f}, "
                    f"min_macro_f1={gate_config['berkeley_min_macro_f1']:.4f})."
                )
            _save_pipeline_checkpoint(
                pipeline_checkpoint_path,
                {
                    "run_tag": run_tag,
                    "objective_mode": args.objective_mode,
                    "orchestration_mode": mode,
                    "last_cycle": cycle_offset,
                    "last_round": 0,
                    "best_cfg": best_cfg.to_dict(),
                    "classifier_state": classifier.state_dict(),
                    "transformer_state": transformer.state_dict(),
                    "wave_classifier_state": None,
                    "classifier_init_info": classifier_init_info,
                    "classifier_resume_info": classifier_resume_info,
                    "transformer_resume_info": transformer_resume_info,
                    "wave_classifier_init_info": wave_classifier_init_info,
                    "wave_classifier_resume_info": wave_classifier_resume_info,
                    "transformer_history": transformer_hist,
                    "wave_classifier_history": wave_classifier_hist,
                    "orchestration_history": orchestration_rows,
                    "refresh_history": refresh_rows,
                    "gate_history": gate_history,
                    "gate_status": {
                        "berkeley_streak": int(berkeley_gate_streak),
                        "wave_streak": int(wave_gate_streak),
                        "transformer_streak": int(transformer_gate_streak),
                    },
                    "wave_classifier_last_eval": wave_classifier_last_eval,
                    "accepted_library_count": library_serial,
                    "sample_bits": int(sample_bits),
                    "chunk_samples": int(chunk_samples),
                    "patch_size": int(args.patch_size),
                    "timestamp": time.time(),
                },
            )
        else:
            wave_classifier = TinyConvClassifier(num_classes=split_num_classes)
            wave_classifier_init_info = _apply_classifier_init(
                model=wave_classifier,
                ckpt_path=args.classifier_init_ckpt,
                scope=args.classifier_init_scope,
            )
            if resume_enabled and resume_wave_classifier_path.exists():
                wave_classifier_resume_info = _apply_model_init(wave_classifier, str(resume_wave_classifier_path))
                _log(
                    f"Resumed wave classifier from {resume_wave_classifier_path} "
                    f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                    f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                )
            elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("wave_classifier_state"), dict):
                wave_classifier_resume_info = _apply_state_dict(
                    wave_classifier,
                    resume_pipeline_ckpt.get("wave_classifier_state"),
                    source_name="pipeline_checkpoint",
                )
                _log(
                    "Resumed wave classifier from pipeline checkpoint "
                    f"(loaded={wave_classifier_resume_info.get('loaded_keys', 0)}, "
                    f"skipped={wave_classifier_resume_info.get('skipped_keys', 0)})"
                )
            if args.classifier_freeze_features:
                _set_feature_freeze(wave_classifier, freeze=True)
            if args.channels_last:
                wave_classifier = wave_classifier.to(memory_format=torch.channels_last)
            wave_classifier = maybe_compile_module(
                wave_classifier,
                enabled=bool(args.compile_models),
                mode=str(args.compile_mode),
            )

            cycles = max(1, int(args.orchestration_cycles))
            rounds_per_cycle = 1 if mode == "sequential" else max(1, int(args.orchestration_rounds))
            _log(
                f"Orchestration: mode={mode}, cycles={cycles}, rounds_per_cycle={rounds_per_cycle}, "
                f"transformer_steps_per_round={args.transformer_steps_per_round}, "
                f"wave_cls_samples={args.wave_cls_train_samples}/{args.wave_cls_val_samples}, "
                f"accepted_to_transformer_preload={int(args.transformer_accepted_preload_max)}, "
                f"accepted_to_transformer_append={int(args.transformer_accepted_append_max_per_round)}, "
                f"accepted_only={bool(args.wave_cls_accepted_only)}, "
                f"resume_cycle_offset={cycle_offset}, library_serial_start={library_serial}"
            )

            accepted_stream_seen: set = set()
            preload_cap = max(0, int(args.transformer_accepted_preload_max))
            if preload_cap > 0 and library_index_path.exists():
                preload_rows = []
                try:
                    with library_index_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                preload_rows.append(json.loads(line))
                            except Exception:
                                continue
                except Exception:
                    preload_rows = []
                if len(preload_rows) > preload_cap:
                    preload_rows = preload_rows[-preload_cap:]
                preload_added = _append_accepted_rows_to_transformer_streams(
                    rows=preload_rows,
                    seen_paths=accepted_stream_seen,
                    train_streams=train_streams,
                    train_stream_labels=train_stream_labels,
                    train_meta=train_meta,
                    max_points=int(args.stream_max_points),
                )
                if preload_added > 0:
                    if train_stream_target_labels is not None:
                        train_stream_target_labels.extend([-1] * int(preload_added))
                    _log(
                        "Preloaded accepted-library streams for transformer: "
                        f"added={preload_added} total_train_streams={len(train_streams)}"
                    )

            global_round = len(orchestration_rows)
            for cycle_local in range(1, cycles + 1):
                cycle_id = cycle_offset + cycle_local
                for round_id in range(1, rounds_per_cycle + 1):
                    global_round += 1
                    _log(f"[cycle {cycle_id}/{cycle_offset + cycles} round {round_id}/{rounds_per_cycle}]")

                    if (
                        refresh_loader is not None
                        and int(args.berkeley_refresh_round_every) > 0
                        and (global_round % int(args.berkeley_refresh_round_every) == 0)
                    ):
                        ref = _run_berkeley_refresh_epochs(
                            classifier=classifier,
                            loader=refresh_loader,
                            device=device,
                            epochs=int(args.berkeley_refresh_epochs),
                            lr=float(args.berkeley_refresh_lr),
                            weight_decay=float(args.berkeley_refresh_weight_decay),
                            max_steps=int(args.berkeley_refresh_max_steps),
                            lr_sine_cycles=float(args.lr_sine_cycles),
                            lr_sine_frequency=float(args.lr_sine_frequency),
                            lr_sine_tail_fraction=float(args.lr_sine_tail_fraction),
                            lr_sine_min_scale=float(args.lr_sine_min_scale),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            log_every=int(args.berkeley_refresh_log_every),
                            max_seconds=float(args.berkeley_refresh_max_seconds),
                            cache_x=(refresh_cache.get("x") if isinstance(refresh_cache, dict) else None),
                            cache_y=(refresh_cache.get("y") if isinstance(refresh_cache, dict) else None),
                            cache_batch_size=int(refresh_run_batch_size),
                        )
                        refresh_rows.append({"stage": "orchestration", "cycle": cycle_id, "round": round_id, **ref})
                        _log(
                            "  Berkeley refresh: "
                            f"loss={ref['loss']:.4f} samples={ref.get('samples', 0)} "
                            f"source={ref.get('source', 'loader')} elapsed={ref.get('elapsed_sec', 0.0):.1f}s"
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"orchestration_refresh_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            wave_classifier=wave_classifier,
                            transformer_history=transformer_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                            },
                        )

                    berkeley_round_metric = _score_config_with_classifier(
                        records=records,
                        indices=final_val_idx,
                        cfg=best_cfg,
                        image_hw=image_hw,
                        classifier=classifier,
                        device=device,
                        batch_size=args.classifier_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                    )
                    berkeley_round_eval = None
                    if berkeley_gate_loader is not None:
                        berkeley_round_eval = _evaluate_berkeley_classifier_gate(
                            classifier=classifier,
                            loader=berkeley_gate_loader,
                            device=device,
                            max_steps=int(args.gate_berkeley_eval_max_steps),
                            amp_enabled=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                        )

                    berkeley_pass = True
                    if berkeley_round_eval is not None:
                        berkeley_pass = (
                            (float(berkeley_round_eval["mean_confidence"]) >= gate_config["berkeley_min_confidence"])
                            and (float(berkeley_round_eval["macro_f1"]) >= gate_config["berkeley_min_macro_f1"])
                        )
                    if berkeley_pass:
                        berkeley_gate_streak += 1
                    else:
                        berkeley_gate_streak = 0
                    berkeley_ready = berkeley_gate_streak >= gate_config["berkeley_maintain_rounds"]

                    wave_pass = (
                        (float(berkeley_round_metric.get("mean_entropy", 0.0)) >= gate_config["wave_min_entropy"])
                        and (float(berkeley_round_metric["score"]) >= gate_config["berkeley_min_score"])
                    )
                    if wave_pass:
                        wave_gate_streak += 1
                    else:
                        wave_gate_streak = 0
                    wave_ready = wave_gate_streak >= gate_config["wave_maintain_rounds"]

                    round_row: Dict = {
                        "cycle": cycle_id,
                        "round": round_id,
                        "berkeley_metric": berkeley_round_metric,
                        "berkeley_classifier_eval": berkeley_round_eval,
                        "berkeley_gate_pass": bool(berkeley_pass),
                        "berkeley_streak": int(berkeley_gate_streak),
                        "berkeley_ready": bool(berkeley_ready),
                        "wave_entropy_gate_pass": bool(wave_pass),
                        "wave_entropy_streak": int(wave_gate_streak),
                        "wave_entropy_ready": bool(wave_ready),
                    }

                    if not berkeley_ready or not wave_ready:
                        round_row["downstream_skipped"] = "berkeley_gate" if not berkeley_ready else "wave_entropy_gate"
                        conf_txt = ""
                        if berkeley_round_eval is not None:
                            conf_txt = (
                                f"conf={berkeley_round_eval['mean_confidence']:.4f}, "
                                f"macro_f1={berkeley_round_eval['macro_f1']:.4f}, "
                            )
                        _log(
                            "  downstream skipped by pre-transformer gates "
                            f"({conf_txt}"
                            f"wave_entropy={berkeley_round_metric.get('mean_entropy', 0.0):.4f}, "
                            f"berk_streak={berkeley_gate_streak}/{gate_config['berkeley_maintain_rounds']}, "
                            f"wave_streak={wave_gate_streak}/{gate_config['wave_maintain_rounds']})"
                        )
                        gate_history.append(
                            {
                                "cycle": cycle_id,
                                "round": round_id,
                                "berkeley_pass": bool(berkeley_pass),
                                "berkeley_streak": int(berkeley_gate_streak),
                                "berkeley_ready": bool(berkeley_ready),
                                "wave_pass": bool(wave_pass),
                                "wave_streak": int(wave_gate_streak),
                                "wave_ready": bool(wave_ready),
                                "transformer_pass": None,
                                "transformer_streak": int(transformer_gate_streak),
                                "transformer_ready": None,
                            }
                        )
                        orchestration_rows.append(round_row)
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_pipeline_checkpoint(
                                pipeline_checkpoint_path,
                                {
                                    "run_tag": run_tag,
                                    "objective_mode": args.objective_mode,
                                    "orchestration_mode": mode,
                                    "last_cycle": int(cycle_id),
                                    "last_round": int(round_id),
                                    "best_cfg": best_cfg.to_dict(),
                                    "classifier_state": classifier.state_dict(),
                                    "transformer_state": transformer.state_dict(),
                                    "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                    "classifier_init_info": classifier_init_info,
                                    "classifier_resume_info": classifier_resume_info,
                                    "transformer_resume_info": transformer_resume_info,
                                    "wave_classifier_init_info": wave_classifier_init_info,
                                    "wave_classifier_resume_info": wave_classifier_resume_info,
                                    "transformer_history": transformer_hist,
                                    "wave_classifier_history": wave_classifier_hist,
                                    "orchestration_history": orchestration_rows,
                                    "refresh_history": refresh_rows,
                                    "gate_history": gate_history,
                                    "gate_status": {
                                        "berkeley_streak": int(berkeley_gate_streak),
                                        "wave_streak": int(wave_gate_streak),
                                        "transformer_streak": int(transformer_gate_streak),
                                    },
                                    "wave_classifier_last_eval": wave_classifier_last_eval,
                                    "accepted_library_count": int(library_serial),
                                    "sample_bits": int(sample_bits),
                                    "chunk_samples": int(chunk_samples),
                                    "patch_size": int(args.patch_size),
                                    "timestamp": time.time(),
                                },
                            )
                        continue

                    transformer, round_hist = train_transformer_feature_metric(
                        transformer=transformer,
                        classifier=classifier,
                        train_streams=train_streams,
                        train_target_labels=train_stream_target_labels,
                        val_streams=val_streams,
                        cfg=best_cfg,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        device=device,
                        epochs=max(1, int(args.transformer_epochs_per_round)),
                        steps_per_epoch=max(1, int(args.transformer_steps_per_round)),
                        batch_size=args.transformer_batch_size,
                        chunk_samples=chunk_samples,
                        lr=args.transformer_lr,
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        degrade_inputs=bool(args.transformer_degrade_inputs),
                        degrade_min_strength=float(args.transformer_degrade_min_strength),
                        degrade_max_strength=float(args.transformer_degrade_max_strength),
                        degrade_noise_std_min=float(args.transformer_degrade_noise_std_min),
                        degrade_noise_std_max=float(args.transformer_degrade_noise_std_max),
                        degrade_dropout_max=float(args.transformer_degrade_dropout_max),
                        degrade_quant_bits_min=int(args.transformer_degrade_quant_bits_min),
                        degrade_quant_bits_max=int(args.transformer_degrade_quant_bits_max),
                        entropy_penalty_weight=float(args.transformer_loss_entropy_weight),
                        high_bit_penalty_weight=float(args.transformer_loss_high_bit_weight),
                        low_bit_penalty_weight=float(args.transformer_loss_low_bit_weight),
                        wave_l1_weight=float(args.transformer_loss_wave_l1_weight),
                        score_target_weight=float(args.transformer_loss_score_target_weight),
                        score_target_margin=float(args.transformer_loss_score_target_margin),
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        pin_memory=bool(args.pin_memory_wave_batches),
                        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
                        log_every_steps=max(0, int(args.transformer_log_every)),
                        cache_eval_batches=bool(args.transformer_cache_eval_batches),
                        visualize_status=bool(args.transformer_visualize_status),
                        visualize_scale=max(1, int(args.transformer_viz_scale)),
                        seed=args.seed + 9000 + global_round,
                    )
                    for row in round_hist:
                        rr = dict(row)
                        rr["cycle"] = float(cycle_id)
                        rr["round"] = float(round_id)
                        transformer_hist.append(rr)
                    _save_training_segment_snapshot(
                        enabled=bool(args.checkpoint_after_training_segment),
                        out_dir=out_dir,
                        objective_mode=args.objective_mode,
                        run_tag=run_tag,
                        segment=f"round_transformer_cycle_{int(cycle_id)}_round_{int(round_id)}",
                        best_cfg=best_cfg,
                        classifier=classifier,
                        transformer=transformer,
                        wave_classifier=wave_classifier,
                        transformer_history=transformer_hist,
                        wave_classifier_history=wave_classifier_hist,
                        orchestration_history=orchestration_rows,
                        refresh_history=refresh_rows,
                        gate_history=gate_history,
                        gate_status={
                            "berkeley_streak": int(berkeley_gate_streak),
                            "wave_streak": int(wave_gate_streak),
                            "transformer_streak": int(transformer_gate_streak),
                        },
                        extra={
                            "sample_bits": int(sample_bits),
                            "chunk_samples": int(chunk_samples),
                            "patch_size": int(args.patch_size),
                            "accepted_library_count": int(library_serial),
                        },
                    )

                    feature_eval = evaluate_feature_score_before_after(
                        transformer=transformer,
                        classifier=classifier,
                        streams=val_streams,
                        cfg=best_cfg,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        device=device,
                        max_batches=20,
                        batch_size=args.transformer_batch_size,
                        score_topk=args.score_topk,
                        score_threshold=args.score_threshold,
                        score_w_topk=args.score_w_topk,
                        score_w_cov=args.score_w_cov,
                        score_w_mean=args.score_w_mean,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
                    )
                    _log(
                        f"  transformer score: before={feature_eval['score_before']:.4f} "
                        f"after={feature_eval['score_after']:.4f} gain={feature_eval['score_gain']:.4f}"
                    )

                    transformer_pass = (
                        (float(feature_eval["score_after"]) >= gate_config["transformer_min_score_after"])
                        and (float(feature_eval["score_gain"]) >= gate_config["transformer_min_gain"])
                    )
                    if transformer_pass:
                        transformer_gate_streak += 1
                    else:
                        transformer_gate_streak = 0
                    transformer_ready = (
                        transformer_gate_streak >= gate_config["transformer_maintain_rounds"]
                    )
                    round_row["transformer_eval"] = feature_eval
                    round_row["transformer_gate_pass"] = bool(transformer_pass)
                    round_row["transformer_streak"] = int(transformer_gate_streak)
                    round_row["transformer_ready"] = bool(transformer_ready)

                    gate_history.append(
                        {
                            "cycle": cycle_id,
                            "round": round_id,
                            "berkeley_pass": bool(berkeley_pass),
                            "berkeley_streak": int(berkeley_gate_streak),
                            "berkeley_ready": bool(berkeley_ready),
                            "wave_pass": bool(wave_pass),
                            "wave_streak": int(wave_gate_streak),
                            "wave_ready": bool(wave_ready),
                            "transformer_pass": bool(transformer_pass),
                            "transformer_streak": int(transformer_gate_streak),
                            "transformer_ready": bool(transformer_ready),
                        }
                    )

                    if not transformer_ready:
                        round_row["downstream_skipped"] = "transformer_gate"
                        _log(
                            "  wave-classifier stage skipped by transformer gate "
                            f"(after={feature_eval['score_after']:.4f} min_after={gate_config['transformer_min_score_after']:.4f}, "
                            f"gain={feature_eval['score_gain']:.4f} min_gain={gate_config['transformer_min_gain']:.4f}, "
                            f"streak={transformer_gate_streak}/{gate_config['transformer_maintain_rounds']})"
                        )
                        orchestration_rows.append(round_row)
                        if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                            _save_pipeline_checkpoint(
                                pipeline_checkpoint_path,
                                {
                                    "run_tag": run_tag,
                                    "objective_mode": args.objective_mode,
                                    "orchestration_mode": mode,
                                    "last_cycle": int(cycle_id),
                                    "last_round": int(round_id),
                                    "best_cfg": best_cfg.to_dict(),
                                    "classifier_state": classifier.state_dict(),
                                    "transformer_state": transformer.state_dict(),
                                    "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                    "classifier_init_info": classifier_init_info,
                                    "classifier_resume_info": classifier_resume_info,
                                    "transformer_resume_info": transformer_resume_info,
                                    "wave_classifier_init_info": wave_classifier_init_info,
                                    "wave_classifier_resume_info": wave_classifier_resume_info,
                                    "transformer_history": transformer_hist,
                                    "wave_classifier_history": wave_classifier_hist,
                                    "orchestration_history": orchestration_rows,
                                    "refresh_history": refresh_rows,
                                    "gate_history": gate_history,
                                    "gate_status": {
                                        "berkeley_streak": int(berkeley_gate_streak),
                                        "wave_streak": int(wave_gate_streak),
                                        "transformer_streak": int(transformer_gate_streak),
                                    },
                                    "wave_classifier_last_eval": wave_classifier_last_eval,
                                    "accepted_library_count": int(library_serial),
                                    "sample_bits": int(sample_bits),
                                    "chunk_samples": int(chunk_samples),
                                    "patch_size": int(args.patch_size),
                                    "timestamp": time.time(),
                                },
                            )
                        continue

                    x_wave_train, y_wave_train, accepted_rows = _build_wave_classifier_dataset_from_transformer(
                        streams=train_streams,
                        labels=train_stream_labels,
                        metas=train_meta,
                        cfg=best_cfg,
                        transformer=transformer,
                        berkeley_classifier=classifier,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        total_samples=max(1, int(args.wave_cls_train_samples)),
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        device=device,
                        rng_seed=args.seed + 12000 + (global_round * 11),
                        accept_score_threshold=float(args.accept_score_threshold),
                        accept_l1_threshold=float(args.accept_l1_threshold),
                        library_dir=library_dir,
                        library_limit=max(0, int(args.library_max_per_round)),
                        cycle_id=cycle_id,
                        round_id=round_id,
                        accepted_only=bool(args.wave_cls_accepted_only),
                        run_tag=run_tag,
                        library_serial_start=library_serial,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                    )
                    x_wave_val, y_wave_val, _ = _build_wave_classifier_dataset_from_transformer(
                        streams=val_streams,
                        labels=val_stream_labels,
                        metas=val_meta,
                        cfg=best_cfg,
                        transformer=transformer,
                        berkeley_classifier=classifier,
                        sample_bits=sample_bits,
                        image_hw=image_hw,
                        chunk_samples=chunk_samples,
                        total_samples=max(1, int(args.wave_cls_val_samples)),
                        batch_size=max(1, int(args.wave_cls_batch_size)),
                        device=device,
                        rng_seed=args.seed + 12100 + (global_round * 11),
                        accept_score_threshold=float(args.accept_score_threshold),
                        accept_l1_threshold=float(args.accept_l1_threshold),
                        library_dir=library_dir,
                        library_limit=0,
                        cycle_id=cycle_id,
                        round_id=round_id,
                        accepted_only=bool(args.wave_cls_accepted_only),
                        run_tag=run_tag,
                        library_serial_start=library_serial,
                        amp_enabled=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        pin_memory=bool(args.pin_memory_wave_batches),
                    )
                    accepted_total.extend(accepted_rows)
                    library_serial += len(accepted_rows)
                    append_cap = max(0, int(args.transformer_accepted_append_max_per_round))
                    to_append = accepted_rows if append_cap <= 0 else accepted_rows[:append_cap]
                    appended_streams = _append_accepted_rows_to_transformer_streams(
                        rows=to_append,
                        seen_paths=accepted_stream_seen,
                        train_streams=train_streams,
                        train_stream_labels=train_stream_labels,
                        train_meta=train_meta,
                        max_points=int(args.stream_max_points),
                    )
                    if appended_streams > 0 and train_stream_target_labels is not None:
                        train_stream_target_labels.extend([-1] * int(appended_streams))
                    round_row["transformer_train_stream_add"] = int(appended_streams)
                    round_row["transformer_train_stream_total"] = int(len(train_streams))
                    if appended_streams > 0:
                        _log(
                            "  transformer stream pool extended from accepted library: "
                            f"+{appended_streams} (total={len(train_streams)})"
                        )

                    round_row["wave_train_samples"] = int(x_wave_train.shape[0])
                    round_row["wave_val_samples"] = int(x_wave_val.shape[0])
                    round_row["accepted_library"] = len(accepted_rows)
                    can_train_wave = (
                        int(x_wave_train.shape[0]) >= 2
                        and int(x_wave_val.shape[0]) >= 2
                        and int(torch.unique(y_wave_train).numel()) >= 2
                    )
                    if can_train_wave:
                        wave_classifier, round_wave_hist = train_classifier(
                            model=wave_classifier,
                            x_train=x_wave_train,
                            y_train=y_wave_train,
                            x_val=x_wave_val,
                            y_val=y_wave_val,
                            device=device,
                            epochs=max(1, int(args.wave_cls_epochs_per_round)),
                            batch_size=max(1, int(args.wave_cls_batch_size)),
                            lr=float(args.wave_cls_lr),
                            lr_sine_cycles=args.lr_sine_cycles,
                            lr_sine_frequency=args.lr_sine_frequency,
                            lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                            lr_sine_min_scale=args.lr_sine_min_scale,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                            grad_accum_steps=grad_accum_steps,
                            cache_dataset_on_device=bool(args.cache_wave_cls_dataset_on_device),
                            seed=args.seed + 13000 + global_round,
                        )
                        for row in round_wave_hist:
                            rr = dict(row)
                            rr["cycle"] = float(cycle_id)
                            rr["round"] = float(round_id)
                            wave_classifier_hist.append(rr)
                        wave_classifier_last_eval = evaluate_classifier(
                            model=wave_classifier,
                            x=x_wave_val,
                            y=y_wave_val,
                            batch_size=max(1, int(args.wave_cls_batch_size)),
                            device=device,
                            amp=amp_enabled,
                            amp_dtype=args.amp_dtype,
                            channels_last=bool(args.channels_last),
                        )
                        round_row["wave_classifier_val"] = wave_classifier_last_eval
                        _log(
                            f"  wave classifier val: acc={wave_classifier_last_eval['acc']:.4f} "
                            f"loss={wave_classifier_last_eval['loss']:.4f} accepted={len(accepted_rows)}"
                        )
                        _save_training_segment_snapshot(
                            enabled=bool(args.checkpoint_after_training_segment),
                            out_dir=out_dir,
                            objective_mode=args.objective_mode,
                            run_tag=run_tag,
                            segment=f"round_wave_classifier_cycle_{int(cycle_id)}_round_{int(round_id)}",
                            best_cfg=best_cfg,
                            classifier=classifier,
                            transformer=transformer,
                            wave_classifier=wave_classifier,
                            transformer_history=transformer_hist,
                            wave_classifier_history=wave_classifier_hist,
                            orchestration_history=orchestration_rows,
                            refresh_history=refresh_rows,
                            gate_history=gate_history,
                            gate_status={
                                "berkeley_streak": int(berkeley_gate_streak),
                                "wave_streak": int(wave_gate_streak),
                                "transformer_streak": int(transformer_gate_streak),
                            },
                            extra={
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "accepted_library_count": int(library_serial),
                            },
                        )
                    else:
                        round_row["wave_classifier_val"] = None
                        round_row["wave_classifier_skipped"] = True
                        round_row["wave_unique_train_labels"] = int(torch.unique(y_wave_train).numel())
                        _log(
                            f"  wave classifier skipped (samples={x_wave_train.shape[0]}/{x_wave_val.shape[0]}, "
                            f"unique_train_labels={round_row['wave_unique_train_labels']})"
                        )
                    orchestration_rows.append(round_row)

                    if int(args.checkpoint_every_round) > 0 and (global_round % int(args.checkpoint_every_round) == 0):
                        _save_pipeline_checkpoint(
                            pipeline_checkpoint_path,
                            {
                                "run_tag": run_tag,
                                "objective_mode": args.objective_mode,
                                "orchestration_mode": mode,
                                "last_cycle": int(cycle_id),
                                "last_round": int(round_id),
                                "best_cfg": best_cfg.to_dict(),
                                "classifier_state": classifier.state_dict(),
                                "transformer_state": transformer.state_dict(),
                                "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                                "classifier_init_info": classifier_init_info,
                                "classifier_resume_info": classifier_resume_info,
                                "transformer_resume_info": transformer_resume_info,
                                "wave_classifier_init_info": wave_classifier_init_info,
                                "wave_classifier_resume_info": wave_classifier_resume_info,
                                "transformer_history": transformer_hist,
                                "wave_classifier_history": wave_classifier_hist,
                                "orchestration_history": orchestration_rows,
                                "refresh_history": refresh_rows,
                                "gate_history": gate_history,
                                "gate_status": {
                                    "berkeley_streak": int(berkeley_gate_streak),
                                    "wave_streak": int(wave_gate_streak),
                                    "transformer_streak": int(transformer_gate_streak),
                                },
                                "wave_classifier_last_eval": wave_classifier_last_eval,
                                "accepted_library_count": int(library_serial),
                                "sample_bits": int(sample_bits),
                                "chunk_samples": int(chunk_samples),
                                "patch_size": int(args.patch_size),
                                "timestamp": time.time(),
                            },
                        )

        final_eval = evaluate_feature_score_before_after(
            transformer=transformer,
            classifier=classifier,
            streams=val_streams,
            cfg=best_cfg,
            sample_bits=sample_bits,
            image_hw=image_hw,
            chunk_samples=chunk_samples,
            device=device,
            max_batches=20,
            batch_size=args.transformer_batch_size,
            score_topk=args.score_topk,
            score_threshold=args.score_threshold,
            score_w_topk=args.score_w_topk,
            score_w_cov=args.score_w_cov,
            score_w_mean=args.score_w_mean,
            amp=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
            pin_memory=bool(args.pin_memory_wave_batches),
            stream_cache_on_device=bool(args.cache_wave_streams_on_device),
        )
        _log(
            f"Transformer final: before={final_eval['score_before']:.4f} "
            f"after={final_eval['score_after']:.4f} gain={final_eval['score_gain']:.4f}"
        )

        _save_pipeline_checkpoint(
            out_dir / "pipeline_checkpoint.pt",
            {
                "run_tag": run_tag,
                "objective_mode": args.objective_mode,
                "orchestration_mode": mode,
                "last_cycle": int(cycle_offset + max(1, int(args.orchestration_cycles)) if mode != "none" else cycle_offset),
                "last_round": int(max(1, int(args.orchestration_rounds)) if mode == "alternating" else 0),
                "best_cfg": best_cfg.to_dict(),
                "classifier_state": classifier.state_dict(),
                "transformer_state": transformer.state_dict(),
                "wave_classifier_state": (wave_classifier.state_dict() if wave_classifier is not None else None),
                "classifier_init_info": classifier_init_info,
                "classifier_resume_info": classifier_resume_info,
                "transformer_resume_info": transformer_resume_info,
                "wave_classifier_init_info": wave_classifier_init_info,
                "wave_classifier_resume_info": wave_classifier_resume_info,
                "transformer_history": transformer_hist,
                "wave_classifier_history": wave_classifier_hist,
                "orchestration_history": orchestration_rows,
                "refresh_history": refresh_rows,
                "gate_history": gate_history,
                "gate_status": {
                    "berkeley_streak": int(berkeley_gate_streak),
                    "wave_streak": int(wave_gate_streak),
                    "transformer_streak": int(transformer_gate_streak),
                },
                "wave_classifier_last_eval": wave_classifier_last_eval,
                "accepted_library_count": int(library_serial),
                "latent_berkeley_targeting": latent_targeting_info,
                "transformer_eval": final_eval,
                "sample_bits": int(sample_bits),
                "chunk_samples": int(chunk_samples),
                "patch_size": int(args.patch_size),
                "timestamp": time.time(),
            },
        )

        classifier_path = out_dir / "classifier.pt"
        transformer_path = out_dir / "transformer.pt"
        wave_classifier_path = out_dir / "wave_classifier.pt"
        cfg_path = out_dir / "best_render_config.json"
        summary_path = out_dir / "summary.json"
        search_path = out_dir / "search_results.json"
        orchestration_path = out_dir / "orchestration_history.json"

        torch.save(
            {
                "state_dict": classifier.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "model_name": classifier_init_info.get("model_name", "tiny"),
            },
            classifier_path,
        )
        torch.save(
            {"state_dict": transformer.state_dict(), "chunk_samples": chunk_samples, "patch_size": int(args.patch_size)},
            transformer_path,
        )
        if wave_classifier is not None:
            torch.save(
                {
                    "state_dict": wave_classifier.state_dict(),
                    "num_classes": split_num_classes,
                    "class_names": split_class_names,
                    "model_name": "tiny",
                    "init_info": wave_classifier_init_info,
                },
                wave_classifier_path,
            )

        cfg_path.write_text(json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8")
        search_path.write_text(json.dumps(search_rows, indent=2), encoding="utf-8")
        if len(orchestration_rows) > 0:
            orchestration_path.write_text(json.dumps(orchestration_rows, indent=2), encoding="utf-8")

        if args.save_preview > 0:
            preview_dir = out_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            transformer.eval()
            with torch.no_grad():
                for i in range(min(int(args.save_preview), len(val_streams))):
                    s = val_streams[i]
                    if s.size >= chunk_samples:
                        x_np = s[:chunk_samples]
                    else:
                        x_np = np.pad(s, (0, chunk_samples - s.size), constant_values=0.0)
                    xb = torch.from_numpy(x_np[None, :]).to(device)
                    xh = transformer(xb).detach().cpu().numpy()[0]
                    _write_mono_wav(str(preview_dir / f"sample_{i:02d}_original.wav"), x_np, framerate=val_meta[i]["framerate"])
                    _write_mono_wav(str(preview_dir / f"sample_{i:02d}_enhanced.wav"), xh, framerate=val_meta[i]["framerate"])

        summary = {
            "args": vars(args),
            "run_tag": run_tag,
            "resume_enabled": bool(resume_enabled),
            "resume_dir": str(resume_dir) if resume_enabled else "",
            "objective_mode": args.objective_mode,
            "orchestration_mode": mode,
            "wav_data_root": wav_data_root,
            "latent_fallback": latent_fallback_info,
            "latent_berkeley_targeting": latent_targeting_info,
            "num_records": len(records),
            "num_classes": num_classes,
            "class_names": class_names,
            "wave_label_num_classes": split_num_classes,
            "wave_label_class_names": split_class_names,
            "label_source_for_split": label_source,
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "best_cfg": best_cfg.to_dict(),
            "best_search_score": best_score,
            "classifier_init": classifier_init_info,
            "classifier_resume": classifier_resume_info,
            "classifier_init_trials": [{"mode": "berkeley_multilabel"}],
            "classifier_val": cls_val,
            "berkeley_refresh_history": refresh_rows,
            "gate_config": gate_config,
            "gate_history": gate_history,
            "gate_status": {
                "berkeley_streak": int(berkeley_gate_streak),
                "wave_streak": int(wave_gate_streak),
                "transformer_streak": int(transformer_gate_streak),
            },
            "transformer_baseline": baseline,
            "transformer_eval": final_eval,
            "transformer_history": transformer_hist,
            "transformer_resume": transformer_resume_info,
            "orchestration_history": orchestration_rows,
            "classifier_history": wave_classifier_hist,
            "wave_classifier_init": wave_classifier_init_info,
            "wave_classifier_resume": wave_classifier_resume_info,
            "wave_classifier_history": wave_classifier_hist,
            "wave_classifier_last_eval": wave_classifier_last_eval,
            "accepted_library_count": int(library_serial),
            "accepted_library_index": str(library_index_path) if library_index_path.exists() else "",
            "elapsed_sec": time.time() - t0,
            "artifacts": {
                "classifier": str(classifier_path),
                "transformer": str(transformer_path),
                "wave_classifier": str(wave_classifier_path) if wave_classifier is not None else "",
                "config": str(cfg_path),
                "search_results": str(search_path),
                "orchestration_history": str(orchestration_path) if orchestration_path.exists() else "",
                "accepted_library_index": str(library_index_path) if library_index_path.exists() else "",
                "latent_manifest": str(latent_fallback_info["manifest"]) if latent_fallback_info else "",
                "pipeline_checkpoint": str(out_dir / "pipeline_checkpoint.pt"),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _log(f"Finished. Summary: {summary_path}")
        return

    label_mode = "folder" if args.label_mode == "folder" else "spectral"
    labels, class_names, label_source = _build_labels(
        records=records,
        data_root=wav_data_root,
        label_mode=label_mode,
        pseudo_classes=args.pseudo_classes,
    )
    num_classes = len(set(labels.tolist()))
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes for classifier training.")
    _log(f"Labels: {label_source}, classes={num_classes}")

    train_idx, val_idx = _stratified_split(labels, train_frac=args.train_frac, seed=args.seed)
    _log(f"Split: train={len(train_idx)} val={len(val_idx)}")

    search_train_idx = _sample_indices(train_idx, args.search_train_limit, rng)
    search_val_idx = _sample_indices(val_idx, args.search_val_limit, rng)
    _log(f"Search subset: train={len(search_train_idx)} val={len(search_val_idx)}")

    image_hw = (int(args.image_size), int(args.image_size))

    candidates = _random_configs(num_trials=args.config_trials, rng=rng, max_points=args.render_max_points)
    best_cfg = None
    best_score = -1.0
    search_rows = []
    init_info_search = []
    resume_search_rows = _load_json(resume_search_path) if (resume_enabled and resume_search_path.exists()) else None
    did_config_search = False

    if resume_enabled and (not args.force_config_search):
        if isinstance(resume_cfg, dict):
            try:
                best_cfg = RenderConfig.from_dict(resume_cfg)
                _log(f"Resumed best render config from {resume_cfg_path}")
            except Exception:
                best_cfg = None
        if best_cfg is None and isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("best_cfg"), dict):
            try:
                best_cfg = RenderConfig.from_dict(resume_pipeline_ckpt.get("best_cfg"))
                _log("Resumed best render config from pipeline checkpoint")
            except Exception:
                best_cfg = None
        if isinstance(resume_summary, dict):
            best_score = float(resume_summary.get("best_search_score", best_score))
            if isinstance(resume_summary.get("classifier_init_trials"), list):
                init_info_search = list(resume_summary.get("classifier_init_trials", init_info_search))
        if isinstance(resume_search_rows, list):
            search_rows = list(resume_search_rows)

    if best_cfg is None:
        did_config_search = True
        _log("Config search started...")
        for i, cfg in enumerate(candidates, start=1):
            try:
                xtr, ytr, _ = _render_indices_to_tensors(records, labels, search_train_idx, cfg, image_hw=image_hw)
                xva, yva, _ = _render_indices_to_tensors(records, labels, search_val_idx, cfg, image_hw=image_hw)
                model = TinyConvClassifier(num_classes=num_classes)
                init_info = _apply_classifier_init(model, args.classifier_init_ckpt, args.classifier_init_scope)
                if args.classifier_freeze_features:
                    _set_feature_freeze(model, freeze=True)
                model, _ = train_classifier(
                    model=model,
                    x_train=xtr,
                    y_train=ytr,
                    x_val=xva,
                    y_val=yva,
                    device=device,
                    epochs=args.config_epochs,
                    batch_size=args.classifier_batch_size,
                    lr=args.classifier_lr,
                    lr_sine_cycles=args.lr_sine_cycles,
                    lr_sine_frequency=args.lr_sine_frequency,
                    lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                    lr_sine_min_scale=args.lr_sine_min_scale,
                    amp=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                    grad_accum_steps=grad_accum_steps,
                    seed=args.seed + i,
                )
                val_stats = evaluate_classifier(
                    model,
                    xva,
                    yva,
                    batch_size=args.classifier_batch_size,
                    device=device,
                    amp=amp_enabled,
                    amp_dtype=args.amp_dtype,
                    channels_last=bool(args.channels_last),
                )
                score = float(val_stats["acc"])
                row = {"trial": i, "score": score, "cfg": cfg.to_dict(), "val_loss": float(val_stats["loss"])}
                search_rows.append(row)
                init_info_search.append({"trial": i, **init_info})
                _log(f"  [{i}/{len(candidates)}] val_acc={score:.4f} cfg={cfg.to_dict()}")
                if score > best_score:
                    best_score = score
                    best_cfg = cfg
            except Exception as e:
                _log(f"  [{i}/{len(candidates)}] failed: {e}")
    else:
        _log("Skipping config search due to resume config (use --force-config-search to override).")

    if best_cfg is None:
        raise RuntimeError("Config search failed to produce a usable config.")

    if args.try_scipy_refine and did_config_search:
        refine_candidates = _try_scipy_refine(best_cfg, max_points=args.render_max_points, enabled=True)
        if len(refine_candidates) > 1:
            _log("Running optional refinement candidates...")
            for j, cfg in enumerate(refine_candidates[1:], start=1):
                try:
                    xtr, ytr, _ = _render_indices_to_tensors(records, labels, search_train_idx, cfg, image_hw=image_hw)
                    xva, yva, _ = _render_indices_to_tensors(records, labels, search_val_idx, cfg, image_hw=image_hw)
                    model = TinyConvClassifier(num_classes=num_classes)
                    init_info = _apply_classifier_init(model, args.classifier_init_ckpt, args.classifier_init_scope)
                    if args.classifier_freeze_features:
                        _set_feature_freeze(model, freeze=True)
                    model, _ = train_classifier(
                        model=model,
                        x_train=xtr,
                        y_train=ytr,
                        x_val=xva,
                        y_val=yva,
                        device=device,
                        epochs=max(1, args.config_epochs),
                        batch_size=args.classifier_batch_size,
                        lr=args.classifier_lr,
                        lr_sine_cycles=args.lr_sine_cycles,
                        lr_sine_frequency=args.lr_sine_frequency,
                        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
                        lr_sine_min_scale=args.lr_sine_min_scale,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                        grad_accum_steps=grad_accum_steps,
                        seed=args.seed + 999 + j,
                    )
                    val_stats = evaluate_classifier(
                        model,
                        xva,
                        yva,
                        batch_size=args.classifier_batch_size,
                        device=device,
                        amp=amp_enabled,
                        amp_dtype=args.amp_dtype,
                        channels_last=bool(args.channels_last),
                    )
                    score = float(val_stats["acc"])
                    search_rows.append({"trial": len(search_rows) + 1, "score": score, "cfg": cfg.to_dict()})
                    init_info_search.append({"trial": len(search_rows), **init_info})
                    _log(f"  [refine {j}] val_acc={score:.4f} cfg={cfg.to_dict()}")
                    if score > best_score:
                        best_score = score
                        best_cfg = cfg
                except Exception as e:
                    _log(f"  [refine {j}] failed: {e}")

    _log(f"Best cfg val_acc={best_score:.4f}: {best_cfg.to_dict()}")

    final_train_idx = _sample_indices(train_idx, args.final_train_limit, rng)
    final_val_idx = _sample_indices(val_idx, args.final_val_limit, rng)
    _log(f"Final classifier subset: train={len(final_train_idx)} val={len(final_val_idx)}")

    x_train, y_train, bits_train = _render_indices_to_tensors(records, labels, final_train_idx, best_cfg, image_hw=image_hw)
    x_val, y_val, bits_val = _render_indices_to_tensors(records, labels, final_val_idx, best_cfg, image_hw=image_hw)
    bits_all = bits_train + bits_val
    sample_bits = int(Counter(bits_all).most_common(1)[0][0]) if len(bits_all) > 0 else 16
    _log(f"Sample bits for transformer renderer: {sample_bits}")

    classifier = TinyConvClassifier(num_classes=num_classes)
    classifier_init_info = _apply_classifier_init(classifier, args.classifier_init_ckpt, args.classifier_init_scope)
    classifier_resume_info = {"used": False}
    if resume_enabled and resume_classifier_path.exists():
        classifier_resume_info = _apply_model_init(classifier, str(resume_classifier_path))
        _log(
            f"Resumed classifier from {resume_classifier_path} "
            f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
        )
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("classifier_state"), dict):
        classifier_resume_info = _apply_state_dict(
            classifier,
            resume_pipeline_ckpt.get("classifier_state"),
            source_name="pipeline_checkpoint",
        )
        _log(
            "Resumed classifier from pipeline checkpoint "
            f"(loaded={classifier_resume_info.get('loaded_keys', 0)}, skipped={classifier_resume_info.get('skipped_keys', 0)})"
        )
    if classifier_init_info["used"]:
        _log(
            f"Classifier init loaded {classifier_init_info['loaded_keys']} keys "
            f"(scope={classifier_init_info['scope']}, skipped={classifier_init_info['skipped_keys']})."
        )
    if args.classifier_freeze_features:
        _set_feature_freeze(classifier, freeze=True)
        _log("Classifier feature extractor frozen.")
    if args.channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
    classifier = maybe_compile_module(
        classifier,
        enabled=bool(args.compile_models),
        mode=str(args.compile_mode),
    )
    classifier_hist: List[Dict[str, float]] = []
    if isinstance(resume_summary, dict) and isinstance(resume_summary.get("classifier_history"), list):
        classifier_hist.extend(list(resume_summary.get("classifier_history", [])))
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("classifier_history"), list):
        classifier_hist.extend(list(resume_pipeline_ckpt.get("classifier_history", [])))

    classifier, run_classifier_hist = train_classifier(
        model=classifier,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        device=device,
        epochs=args.classifier_epochs,
        batch_size=args.classifier_batch_size,
        lr=args.classifier_lr,
        lr_sine_cycles=args.lr_sine_cycles,
        lr_sine_frequency=args.lr_sine_frequency,
        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
        lr_sine_min_scale=args.lr_sine_min_scale,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        grad_accum_steps=grad_accum_steps,
        seed=args.seed + 5000,
    )
    classifier_hist.extend(run_classifier_hist)
    _save_training_segment_snapshot(
        enabled=bool(args.checkpoint_after_training_segment),
        out_dir=out_dir,
        objective_mode=args.objective_mode,
        run_tag=run_tag,
        segment="final_classifier_train",
        best_cfg=best_cfg,
        classifier=classifier,
        classifier_history=classifier_hist,
        extra={"best_search_score": float(best_score)},
    )
    cls_val = evaluate_classifier(
        classifier,
        x_val,
        y_val,
        batch_size=args.classifier_batch_size,
        device=device,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
    )
    _log(f"Classifier val: acc={cls_val['acc']:.4f} loss={cls_val['loss']:.4f}")

    train_streams, train_stream_labels, _, _ = _prepare_streams(
        records=records,
        labels=labels,
        indices=final_train_idx,
        cfg=best_cfg,
        max_points=args.stream_max_points,
    )
    val_streams, val_stream_labels, _, val_meta = _prepare_streams(
        records=records,
        labels=labels,
        indices=final_val_idx,
        cfg=best_cfg,
        max_points=args.stream_max_points,
    )
    if len(train_streams) < 4 or len(val_streams) < 2:
        raise RuntimeError("Not enough decoded streams for transformer stage.")

    chunk_samples = int(args.chunk_samples)
    if chunk_samples % int(args.patch_size) != 0:
        chunk_samples = (chunk_samples // int(args.patch_size)) * int(args.patch_size)
        if chunk_samples <= 0:
            raise RuntimeError("chunk-samples must be >= patch-size")
        _log(f"Adjusted chunk-samples to {chunk_samples} to satisfy patch divisibility.")

    transformer = WavePatchTransformer(
        chunk_samples=chunk_samples,
        patch_size=int(args.patch_size),
        d_model=96,
        nhead=4,
        num_layers=3,
        ff_mult=4,
        max_delta=float(args.max_delta),
        dropout=0.1,
    )
    transformer = transformer.to(device)
    transformer_resume_info = {"used": False}
    if resume_enabled and resume_transformer_path.exists():
        transformer_resume_info = _apply_model_init(transformer, str(resume_transformer_path))
        _log(
            f"Resumed transformer from {resume_transformer_path} "
            f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
        )
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_state"), dict):
        transformer_resume_info = _apply_state_dict(
            transformer,
            resume_pipeline_ckpt.get("transformer_state"),
            source_name="pipeline_checkpoint",
        )
        _log(
            "Resumed transformer from pipeline checkpoint "
            f"(loaded={transformer_resume_info.get('loaded_keys', 0)}, skipped={transformer_resume_info.get('skipped_keys', 0)})"
        )
    transformer = maybe_compile_module(
        transformer,
        enabled=bool(args.compile_models),
        mode=str(args.compile_mode),
    )

    baseline = evaluate_transformer_accuracy(
        transformer=transformer,
        classifier=classifier,
        streams=val_streams,
        labels=val_stream_labels,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        chunk_samples=chunk_samples,
        device=device,
        max_batches=16,
        batch_size=args.transformer_batch_size,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
    )
    _log(f"Transformer baseline val: before={baseline['acc_before']:.4f} after={baseline['acc_after']:.4f}")

    transformer_hist: List[Dict[str, float]] = []
    if isinstance(resume_summary, dict) and isinstance(resume_summary.get("transformer_history"), list):
        transformer_hist.extend(list(resume_summary.get("transformer_history", [])))
    elif isinstance(resume_pipeline_ckpt, dict) and isinstance(resume_pipeline_ckpt.get("transformer_history"), list):
        transformer_hist.extend(list(resume_pipeline_ckpt.get("transformer_history", [])))

    transformer, run_transformer_hist = train_transformer_feature_metric(
        transformer=transformer,
        classifier=classifier,
        train_streams=train_streams,
        train_target_labels=None,
        val_streams=val_streams,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        device=device,
        epochs=args.transformer_epochs,
        steps_per_epoch=args.transformer_steps,
        batch_size=args.transformer_batch_size,
        chunk_samples=chunk_samples,
        lr=args.transformer_lr,
        lr_sine_cycles=args.lr_sine_cycles,
        lr_sine_frequency=args.lr_sine_frequency,
        lr_sine_tail_fraction=args.lr_sine_tail_fraction,
        lr_sine_min_scale=args.lr_sine_min_scale,
        score_topk=args.score_topk,
        score_threshold=args.score_threshold,
        score_w_topk=args.score_w_topk,
        score_w_cov=args.score_w_cov,
        score_w_mean=args.score_w_mean,
        degrade_inputs=bool(args.transformer_degrade_inputs),
        degrade_min_strength=float(args.transformer_degrade_min_strength),
        degrade_max_strength=float(args.transformer_degrade_max_strength),
        degrade_noise_std_min=float(args.transformer_degrade_noise_std_min),
        degrade_noise_std_max=float(args.transformer_degrade_noise_std_max),
        degrade_dropout_max=float(args.transformer_degrade_dropout_max),
        degrade_quant_bits_min=int(args.transformer_degrade_quant_bits_min),
        degrade_quant_bits_max=int(args.transformer_degrade_quant_bits_max),
        entropy_penalty_weight=float(args.transformer_loss_entropy_weight),
        high_bit_penalty_weight=float(args.transformer_loss_high_bit_weight),
        low_bit_penalty_weight=float(args.transformer_loss_low_bit_weight),
        wave_l1_weight=float(args.transformer_loss_wave_l1_weight),
        score_target_weight=float(args.transformer_loss_score_target_weight),
        score_target_margin=float(args.transformer_loss_score_target_margin),
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        grad_accum_steps=grad_accum_steps,
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
        log_every_steps=max(0, int(args.transformer_log_every)),
        cache_eval_batches=bool(args.transformer_cache_eval_batches),
        visualize_status=bool(args.transformer_visualize_status),
        visualize_scale=max(1, int(args.transformer_viz_scale)),
        seed=args.seed + 9000,
    )
    transformer_hist.extend(run_transformer_hist)
    _save_training_segment_snapshot(
        enabled=bool(args.checkpoint_after_training_segment),
        out_dir=out_dir,
        objective_mode=args.objective_mode,
        run_tag=run_tag,
        segment="final_transformer_train",
        best_cfg=best_cfg,
        classifier=classifier,
        transformer=transformer,
        classifier_history=classifier_hist,
        transformer_history=transformer_hist,
        extra={
            "sample_bits": int(sample_bits),
            "chunk_samples": int(chunk_samples),
            "patch_size": int(args.patch_size),
            "best_search_score": float(best_score),
        },
    )

    final_eval = evaluate_transformer_accuracy(
        transformer=transformer,
        classifier=classifier,
        streams=val_streams,
        labels=val_stream_labels,
        cfg=best_cfg,
        sample_bits=sample_bits,
        image_hw=image_hw,
        chunk_samples=chunk_samples,
        device=device,
        max_batches=20,
        batch_size=args.transformer_batch_size,
        amp=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        pin_memory=bool(args.pin_memory_wave_batches),
        stream_cache_on_device=bool(args.cache_wave_streams_on_device),
    )
    _log(f"Transformer final val: before={final_eval['acc_before']:.4f} after={final_eval['acc_after']:.4f}")

    _save_pipeline_checkpoint(
        out_dir / "pipeline_checkpoint.pt",
        {
            "run_tag": run_tag,
            "objective_mode": args.objective_mode,
            "best_cfg": best_cfg.to_dict(),
            "classifier_state": classifier.state_dict(),
            "transformer_state": transformer.state_dict(),
            "classifier_init_info": classifier_init_info,
            "classifier_resume_info": classifier_resume_info,
            "transformer_resume_info": transformer_resume_info,
            "classifier_history": classifier_hist,
            "transformer_history": transformer_hist,
            "classifier_val": cls_val,
            "transformer_eval": final_eval,
            "sample_bits": int(sample_bits),
            "chunk_samples": int(chunk_samples),
            "patch_size": int(args.patch_size),
            "timestamp": time.time(),
        },
    )

    classifier_path = out_dir / "classifier.pt"
    transformer_path = out_dir / "transformer.pt"
    cfg_path = out_dir / "best_render_config.json"
    summary_path = out_dir / "summary.json"
    search_path = out_dir / "search_results.json"

    torch.save({"state_dict": classifier.state_dict(), "num_classes": num_classes, "class_names": class_names}, classifier_path)
    torch.save({"state_dict": transformer.state_dict(), "chunk_samples": chunk_samples, "patch_size": int(args.patch_size)}, transformer_path)
    cfg_path.write_text(json.dumps(best_cfg.to_dict(), indent=2), encoding="utf-8")
    search_path.write_text(json.dumps(search_rows, indent=2), encoding="utf-8")

    if args.save_preview > 0:
        preview_dir = out_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        transformer.eval()
        with torch.no_grad():
            for i in range(min(int(args.save_preview), len(val_streams))):
                s = val_streams[i]
                if s.size >= chunk_samples:
                    x_np = s[:chunk_samples]
                else:
                    x_np = np.pad(s, (0, chunk_samples - s.size), constant_values=0.0)
                xb = torch.from_numpy(x_np[None, :]).to(device)
                xh = transformer(xb).detach().cpu().numpy()[0]
                _write_mono_wav(str(preview_dir / f"sample_{i:02d}_original.wav"), x_np, framerate=val_meta[i]["framerate"])
                _write_mono_wav(str(preview_dir / f"sample_{i:02d}_enhanced.wav"), xh, framerate=val_meta[i]["framerate"])

    summary = {
        "args": vars(args),
        "run_tag": run_tag,
        "resume_enabled": bool(resume_enabled),
        "resume_dir": str(resume_dir) if resume_enabled else "",
        "wav_data_root": wav_data_root,
        "latent_fallback": latent_fallback_info,
        "num_records": len(records),
        "num_classes": num_classes,
        "class_names": class_names,
        "label_source": label_source,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "best_cfg": best_cfg.to_dict(),
        "best_search_score": best_score,
        "classifier_init": classifier_init_info,
        "classifier_resume": classifier_resume_info,
        "classifier_init_trials": init_info_search,
        "classifier_val": cls_val,
        "transformer_eval": final_eval,
        "classifier_history": classifier_hist,
        "transformer_history": transformer_hist,
        "transformer_resume": transformer_resume_info,
        "elapsed_sec": time.time() - t0,
        "artifacts": {
            "classifier": str(classifier_path),
            "transformer": str(transformer_path),
            "config": str(cfg_path),
            "search_results": str(search_path),
            "latent_manifest": str(latent_fallback_info["manifest"]) if latent_fallback_info else "",
            "pipeline_checkpoint": str(out_dir / "pipeline_checkpoint.pt"),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Finished. Summary: {summary_path}")


if __name__ == "__main__":
    main()
