import argparse
import json
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import SBDataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

try:
    from wav_ml_models import (
        SinusoidalLRController,
        SinusoidalLROptions,
        TinyConvClassifier,
        configure_torch_runtime,
        maybe_compile_module,
        set_seed,
    )
except ModuleNotFoundError:
    # Allows running as module: python -m toys_to_survive_development.berkeley_sbd_pretrain
    from toys_to_survive_development.wav_ml_models import (
        SinusoidalLRController,
        SinusoidalLROptions,
        TinyConvClassifier,
        configure_torch_runtime,
        maybe_compile_module,
        set_seed,
    )


VOC20_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv monitor",
]


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


def _clean_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[7:]] = v
        elif k.startswith("_orig_mod."):
            out[k[10:]] = v
        else:
            out[k] = v
    return out


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


def ensure_scipy(auto_install: bool):
    try:
        import scipy  # noqa: F401
        return
    except Exception:
        if not auto_install:
            raise RuntimeError(
                "scipy is required by torchvision SBDataset. "
                "Install manually (`pip install scipy`) or rerun with --auto-install-scipy."
            )
    _log("scipy not found; installing via pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    import scipy  # noqa: F401


class SymmetryMix:
    def __init__(self, p: float = 0.45, alpha_min: float = 0.2, alpha_max: float = 0.6):
        self.p = float(p)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        op = int(torch.randint(0, 4, (1,)).item())
        if op == 0:
            y = torch.flip(x, dims=[2])  # hflip
        elif op == 1:
            y = torch.flip(x, dims=[1])  # vflip
        elif op == 2:
            y = torch.rot90(x, k=1, dims=[1, 2])
        else:
            y = torch.rot90(x, k=2, dims=[1, 2])

        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.rand(1).item()
        return torch.clamp((1.0 - alpha) * x + (alpha * y), 0.0, 1.0)


class NoiseInject:
    def __init__(
        self,
        p: float = 0.5,
        sigma_min: float = 0.005,
        sigma_max: float = 0.06,
        sp_prob: float = 0.003,
    ):
        self.p = float(p)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sp_prob = float(sp_prob)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.rand(1).item()
        y = x + (torch.randn_like(x) * sigma)

        if self.sp_prob > 0:
            r = torch.rand_like(y[:1, :, :])  # one mask for all channels
            y = torch.where(r < (self.sp_prob * 0.5), torch.zeros_like(y), y)
            y = torch.where(r > (1.0 - self.sp_prob * 0.5), torch.ones_like(y), y)

        return torch.clamp(y, 0.0, 1.0)


class SBDMultiLabelDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: np.ndarray, transform):
        self.image_paths = image_paths
        self.labels = labels.astype(np.float32, copy=False)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        y = torch.from_numpy(self.labels[idx])
        return x, y


def _binary_mask_from_segmentation(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg, dtype=np.int32)
    if int(arr.ndim) != 2:
        raise RuntimeError(f"Invalid Berkeley segmentation mask shape: {tuple(arr.shape)}")
    return (arr > 0).astype(np.float32, copy=False)


def _transform_image_mask_pair(
    img: Image.Image,
    mask: Image.Image,
    image_size: int,
    train: bool,
    symmetry_mix: Optional[SymmetryMix] = None,
    noise_inject: Optional[NoiseInject] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    size = max(8, int(image_size))
    if train:
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            img,
            scale=(0.65, 1.0),
            ratio=(0.8, 1.25),
        )
        img = TF.resized_crop(
            img,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[int(size), int(size)],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            top=top,
            left=left,
            height=height,
            width=width,
            size=[int(size), int(size)],
            interpolation=InterpolationMode.NEAREST,
        )
        if torch.rand(1).item() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if torch.rand(1).item() < 0.15:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
    else:
        img = TF.resize(img, [int(size), int(size)], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [int(size), int(size)], interpolation=InterpolationMode.NEAREST)

    x = TF.to_tensor(img)
    if train and symmetry_mix is not None:
        x = symmetry_mix(x)
    if train and noise_inject is not None:
        x = noise_inject(x)
    x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    mask_t = TF.to_tensor(mask)
    mask_t = (mask_t >= 0.5).to(dtype=torch.float32)
    return x, mask_t


class SBDMultiLabelMaskDataset(Dataset):
    def __init__(self, sbd_ds: SBDataset, labels: np.ndarray, image_size: int, train: bool):
        self.sbd_ds = sbd_ds
        self.labels = labels.astype(np.float32, copy=False)
        self.image_size = max(8, int(image_size))
        self.train = bool(train)
        self.symmetry_mix = SymmetryMix(p=0.45, alpha_min=0.2, alpha_max=0.6) if self.train else None
        self.noise_inject = NoiseInject(p=0.55, sigma_min=0.005, sigma_max=0.06, sp_prob=0.003) if self.train else None

    def __len__(self):
        return len(self.sbd_ds.images)

    def __getitem__(self, idx: int):
        img = Image.open(self.sbd_ds.images[idx]).convert("RGB")
        seg = np.array(self.sbd_ds._get_segmentation_target(self.sbd_ds.masks[idx]), dtype=np.int32)
        mask = Image.fromarray((_binary_mask_from_segmentation(seg) * 255.0).astype(np.uint8, copy=False), mode="L")
        x, mask_t = _transform_image_mask_pair(
            img=img,
            mask=mask,
            image_size=int(self.image_size),
            train=bool(self.train),
            symmetry_mix=self.symmetry_mix,
            noise_inject=self.noise_inject,
        )
        y = torch.from_numpy(self.labels[idx])
        return x, y, mask_t


def _labels_from_segmentation_masks(ds: SBDataset, cache_path: Path):
    target_dim = max(1, int(len(VOC20_CLASSES)))

    def _adapt_shape(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if int(x.ndim) != 2:
            raise RuntimeError(f"Invalid cached label shape: {tuple(x.shape)}")
        n = int(x.shape[0])
        if int(x.shape[1]) == int(target_dim):
            return x.astype(np.float32, copy=False)
        if int(x.shape[1]) > int(target_dim):
            return x[:, : int(target_dim)].astype(np.float32, copy=False)
        pad = np.zeros((int(n), int(target_dim - int(x.shape[1]))), dtype=np.float32)
        return np.concatenate([x.astype(np.float32, copy=False), pad], axis=1).astype(np.float32, copy=False)

    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=False)
        labels = _adapt_shape(blob["labels"])
        try:
            np.savez_compressed(cache_path, labels=labels.astype(np.float32, copy=False))
        except Exception:
            pass
        return labels

    _log(f"Building label cache: {cache_path}")
    n = len(ds)
    labels = np.zeros((n, int(target_dim)), dtype=np.float32)
    for i, mpath in enumerate(ds.masks):
        seg = np.array(ds._get_segmentation_target(mpath), dtype=np.int32)
        ids = np.unique(seg)
        ids = ids[(ids >= 1) & (ids <= min(20, int(target_dim)))]
        if ids.size > 0:
            labels[i, ids - 1] = 1.0
        if (i + 1) % 500 == 0 or (i + 1) == n:
            _log(f"  parsed {i + 1}/{n} masks")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, labels=labels)
    return labels


def _sbd_looks_ready(root: Path) -> bool:
    needed = ["img", "cls", "train.txt", "val.txt"]
    return all((root / n).exists() for n in needed)


def _load_sbd_split(root: Path, image_set: str, download: bool):
    try:
        return SBDataset(root=str(root), image_set=image_set, mode="segmentation", download=download)
    except Exception as e:
        # torchvision can error here if download/extract was already done and folders exist.
        if download and "already exists" in str(e):
            _log(f"SBD split '{image_set}': existing extracted files detected, retrying with download=False.")
            return SBDataset(root=str(root), image_set=image_set, mode="segmentation", download=False)
        raise


def prepare_sbd_multilabel(
    data_root: str,
    image_size: int,
    auto_install_scipy: bool,
    include_masks: bool = True,
):
    ensure_scipy(auto_install=auto_install_scipy)
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    # Download/extract at most once; subsequent split loads must be local-only.
    should_download = not _sbd_looks_ready(root)
    train_sbd = _load_sbd_split(root, image_set="train", download=should_download)
    val_sbd = _load_sbd_split(root, image_set="val", download=False)

    train_labels = _labels_from_segmentation_masks(train_sbd, root / "cache" / "sbd_train_multilabel.npz")
    val_labels = _labels_from_segmentation_masks(val_sbd, root / "cache" / "sbd_val_multilabel.npz")

    tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.8, 1.25),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.15),
            transforms.ToTensor(),
            SymmetryMix(p=0.45, alpha_min=0.2, alpha_max=0.6),
            NoiseInject(p=0.55, sigma_min=0.005, sigma_max=0.06, sp_prob=0.003),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if bool(include_masks):
        train_ds = SBDMultiLabelMaskDataset(train_sbd, train_labels, image_size=image_size, train=True)
        val_ds = SBDMultiLabelMaskDataset(val_sbd, val_labels, image_size=image_size, train=False)
    else:
        train_ds = SBDMultiLabelDataset(train_sbd.images, train_labels, transform=tf_train)
        val_ds = SBDMultiLabelDataset(val_sbd.images, val_labels, transform=tf_val)
    return train_ds, val_ds


def multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    t = targets.float()

    tp = (preds * t).sum(dim=0)
    fp = (preds * (1.0 - t)).sum(dim=0)
    fn = ((1.0 - preds) * t).sum(dim=0)

    macro_f1 = torch.mean((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).item()

    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    micro_f1 = ((2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-8)).item()

    acc = (preds.eq(t).float().mean()).item()
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1), "bit_acc": float(acc)}


def mask_metrics(mask_logits: torch.Tensor, mask_targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(mask_logits)
    preds = (probs >= float(threshold)).float()
    t = (mask_targets >= 0.5).float()

    inter = (preds * t).sum(dim=(1, 2, 3))
    union = ((preds + t) > 0.0).float().sum(dim=(1, 2, 3))
    pred_mass = preds.sum(dim=(1, 2, 3))
    target_mass = t.sum(dim=(1, 2, 3))

    mean_iou = torch.mean(inter / (union + 1e-8)).item()
    mean_dice = torch.mean((2.0 * inter) / (pred_mass + target_mass + 1e-8)).item()
    pixel_acc = preds.eq(t).float().mean().item()
    return {
        "mask_iou": float(mean_iou),
        "mask_dice": float(mean_dice),
        "mask_pixel_acc": float(pixel_acc),
    }


def _forward_classifier_outputs(model: nn.Module, xb: torch.Tensor) -> Dict[str, torch.Tensor]:
    if hasattr(model, "forward_with_aux") and callable(getattr(model, "forward_with_aux")):
        out = model.forward_with_aux(xb)
        if (
            isinstance(out, dict)
            and isinstance(out.get("logits"), torch.Tensor)
            and isinstance(out.get("mask_logits"), torch.Tensor)
        ):
            return out
    raise RuntimeError("Berkeley pretrain requires classifier outputs with both logits and mask_logits.")


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
    raise RuntimeError("Berkeley pretrain requires (image, target, mask) batch structure.")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    channels_last: bool = False,
    mask_loss_weight: float = 1.0,
    mask_threshold: float = 0.5,
):
    model.eval()
    amp_dtype_t = _resolve_amp_dtype(amp_dtype) if amp_enabled else torch.float16
    total_loss = 0.0
    n = 0
    logits_all = []
    targets_all = []
    mask_logits_all = []
    mask_targets_all = []
    mask_weight = max(0.0, float(mask_loss_weight))
    for batch in loader:
        xb, yb, mb = _unpack_batch(batch)
        xb = xb.to(device, non_blocking=True)
        if channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)
        if mb is not None:
            mb = mb.to(device, non_blocking=True)
        with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
            out = _forward_classifier_outputs(model, xb)
            logits = out["logits"]
            loss = F.mse_loss(torch.sigmoid(logits), yb)
            mask_logits = out.get("mask_logits")
            if mb is None or not isinstance(mask_logits, torch.Tensor):
                raise RuntimeError("Berkeley evaluation requires both mask targets and mask logits.")
            loss = loss + (mask_weight * F.binary_cross_entropy_with_logits(mask_logits, mb))
        total_loss += float(loss.item()) * int(xb.shape[0])
        n += int(xb.shape[0])
        logits_all.append(logits.detach().cpu())
        targets_all.append(yb.detach().cpu())
        mask_logits_all.append(mask_logits.detach().cpu())
        mask_targets_all.append(mb.detach().cpu())

    logits_cat = torch.cat(logits_all, dim=0)
    targets_cat = torch.cat(targets_all, dim=0)
    m = multilabel_metrics(logits_cat, targets_cat, threshold=0.5)
    m["loss"] = total_loss / max(1, n)
    if len(mask_logits_all) > 0:
        mask_logits_cat = torch.cat(mask_logits_all, dim=0)
        mask_targets_cat = torch.cat(mask_targets_all, dim=0)
        m.update(mask_metrics(mask_logits_cat, mask_targets_cat, threshold=float(mask_threshold)))
    return m


def make_model(
    model_name: str,
    num_classes: int,
    enable_mask_head: bool = False,
    mask_decoder_channels: int = 64,
):
    if model_name == "tiny":
        return TinyConvClassifier(
            num_classes=num_classes,
            mask_decoder_channels=(int(mask_decoder_channels) if bool(enable_mask_head) else 0),
        )
    if model_name == "resnet18":
        if bool(enable_mask_head):
            raise ValueError("Mask head is currently supported only for the 'tiny' Berkeley classifier.")
        from torchvision.models import resnet18

        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {model_name}")


def _load_checkpoint_for_resume(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {p}")
    return torch.load(str(p), map_location="cpu")


def parse_args():
    p = argparse.ArgumentParser(
        description="Auto-download Berkeley SBD and pretrain a multi-label classifier with symmetry+noise augmentations."
    )
    p.add_argument("--data-root", default="toys_to_survive_development/data/berkeley_sbd")
    p.add_argument("--output-dir", default="toys_to_survive_development/wav_pipeline_runs/berkeley_pretrain")
    p.add_argument("--auto-install-scipy", action="store_true")
    p.add_argument("--model", choices=["tiny", "resnet18"], default="tiny")
    p.add_argument("--image-size", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-sine-cycles", type=float, default=1.0)
    p.add_argument("--lr-sine-frequency", type=float, default=0.0, help="Cycles per optimizer step; overrides --lr-sine-cycles when > 0.")
    p.add_argument("--lr-sine-tail-fraction", type=float, default=0.15)
    p.add_argument("--lr-sine-min-scale", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train", type=int, default=0, help="0 means full dataset")
    p.add_argument("--max-val", type=int, default=0, help="0 means full dataset")
    p.add_argument("--resume-from", default="", help="Path to a checkpoint to continue training from.")
    p.add_argument("--reset-optimizer", action="store_true", help="When resuming, ignore optimizer/LR-controller states.")
    p.add_argument(
        "--resume-lr",
        type=float,
        default=None,
        help="Override LR after resume (or fresh start). Useful when continuing from an old checkpoint.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--mask-head", dest="mask_head", action="store_true", help="Enable auxiliary foreground mask output for the tiny classifier.")
    p.add_argument("--no-mask-head", dest="mask_head", action="store_false")
    p.add_argument("--mask-loss-weight", type=float, default=1.0)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--mask-decoder-channels", type=int, default=64)
    p.add_argument("--compile-model", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.add_argument("--no-tf32", dest="allow_tf32", action="store_false")
    p.add_argument("--matmul-precision", choices=["high", "medium", "highest"], default="high")
    p.add_argument("--loader-persistent-workers", dest="loader_persistent_workers", action="store_true")
    p.add_argument("--no-loader-persistent-workers", dest="loader_persistent_workers", action="store_false")
    p.add_argument("--loader-prefetch-factor", type=int, default=4)
    p.set_defaults(cudnn_benchmark=True, allow_tf32=True, loader_persistent_workers=True, mask_head=True)
    return p.parse_args()


def main():
    args = parse_args()
    if not bool(args.mask_head):
        _log("Overriding --no-mask-head: Berkeley pretrain now requires masks for every label.")
        args.mask_head = True
    set_seed(args.seed)
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
    amp_dtype_t = _resolve_amp_dtype(args.amp_dtype) if amp_enabled else torch.float16
    grad_accum_steps = max(1, int(args.grad_accum_steps))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Output dir: {out_dir}")
    _log(f"Device: {device}")
    _log(
        "Perf: "
        f"amp={amp_enabled}({args.amp_dtype}), channels_last={bool(args.channels_last)}, "
        f"compile={bool(args.compile_model)}({args.compile_mode}), grad_accum={grad_accum_steps}, "
        f"tf32={bool(args.allow_tf32)}, cudnn_benchmark={bool(args.cudnn_benchmark)}"
    )

    t0 = time.time()
    train_ds, val_ds = prepare_sbd_multilabel(
        data_root=args.data_root,
        image_size=args.image_size,
        auto_install_scipy=args.auto_install_scipy,
        include_masks=True,
    )

    if args.max_train > 0 and len(train_ds) > args.max_train:
        idx = np.random.default_rng(args.seed).choice(len(train_ds), size=args.max_train, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
    if args.max_val > 0 and len(val_ds) > args.max_val:
        idx = np.random.default_rng(args.seed + 1).choice(len(val_ds), size=args.max_val, replace=False)
        val_ds = torch.utils.data.Subset(val_ds, idx.tolist())

    train_workers = max(0, int(args.num_workers))
    val_workers = max(0, int(args.num_workers) // 2)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=train_workers,
            persistent_workers=bool(args.loader_persistent_workers),
            prefetch_factor=int(args.loader_prefetch_factor),
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        **_dataloader_perf_kwargs(
            num_workers=val_workers,
            persistent_workers=bool(args.loader_persistent_workers),
            prefetch_factor=int(args.loader_prefetch_factor),
        ),
    )

    model = make_model(
        args.model,
        num_classes=len(VOC20_CLASSES),
        enable_mask_head=True,
        mask_decoder_channels=int(args.mask_decoder_channels),
    ).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_batches = max(1, len(train_loader))
    updates_per_epoch = max(1, int(np.ceil(float(train_batches) / float(grad_accum_steps))))
    total_steps = max(1, int(args.epochs) * updates_per_epoch)
    lr_ctl = SinusoidalLRController(
        optimizer=opt,
        total_steps=total_steps,
        options=SinusoidalLROptions(
            cycles=float(args.lr_sine_cycles),
            frequency=float(args.lr_sine_frequency),
            tail_fraction=float(args.lr_sine_tail_fraction),
            min_scale=float(args.lr_sine_min_scale),
        ),
    )
    use_scaler = bool(amp_enabled and amp_dtype_t == torch.float16)
    scaler = _make_grad_scaler(enabled=use_scaler)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -1.0
    history: List[Dict[str, float]] = []
    start_epoch = 0
    resumed_from = ""

    if args.resume_from:
        ckpt = _load_checkpoint_for_resume(args.resume_from)
        ckpt_model = ckpt.get("model_name", args.model)
        ckpt_classes = int(ckpt.get("num_classes", len(VOC20_CLASSES)))
        ckpt_mask_head = bool(ckpt.get("mask_head_enabled", False))
        if ckpt_model != args.model:
            raise RuntimeError(
                f"Resume model mismatch: checkpoint has '{ckpt_model}', current --model is '{args.model}'."
            )
        if ckpt_classes != len(VOC20_CLASSES):
            raise RuntimeError(
                f"Resume class-count mismatch: checkpoint has {ckpt_classes}, expected {len(VOC20_CLASSES)}."
            )
        if bool(ckpt_mask_head) != bool(args.mask_head):
            raise RuntimeError(
                "Resume mask-head mismatch: "
                f"checkpoint mask_head_enabled={bool(ckpt_mask_head)} current={bool(args.mask_head)}."
            )

        model.load_state_dict(_clean_state_dict_prefixes(ckpt["state_dict"]), strict=True)
        resumed_from = str(Path(args.resume_from).resolve())
        _log(f"Resumed model weights from: {resumed_from}")

        best_score = float(ckpt.get("best_val_macro_f1", best_score))
        history = list(ckpt.get("history", []))
        start_epoch = int(ckpt.get("last_epoch", len(history)))
        best_state_blob = ckpt.get("best_state_dict", None)
        best_state = _clean_state_dict_prefixes(best_state_blob) if isinstance(best_state_blob, dict) else None

        resumed_opt = False
        resumed_lr_controller = False
        if (not args.reset_optimizer) and ("optimizer_state_dict" in ckpt):
            try:
                opt.load_state_dict(ckpt["optimizer_state_dict"])
                if "lr_controller_state_dict" in ckpt:
                    lr_ctl.load_state_dict(ckpt["lr_controller_state_dict"])
                    resumed_lr_controller = True
                else:
                    # Back-compat for checkpoints saved before LR-controller state existed.
                    lr_ctl.set_step(max(0, start_epoch) * updates_per_epoch)
                resumed_opt = True
                _log("Resumed optimizer/LR-controller state.")
            except Exception as e:
                _log(f"Could not load optimizer/LR-controller state ({e}); using fresh optimizer/LR-controller.")
        elif args.reset_optimizer:
            _log("Resetting optimizer/LR-controller as requested.")
        else:
            _log("Checkpoint has no optimizer state; using fresh optimizer/LR-controller.")

        if args.resume_lr is not None:
            new_lr = float(args.resume_lr)
            for g in opt.param_groups:
                g["lr"] = new_lr
            lr_ctl.set_base_lrs(new_lr)
            _log(f"Overriding LR to {new_lr:g}")
        elif resumed_opt:
            resumed_lr = float(opt.param_groups[0]["lr"])
            if resumed_lr <= 0.0:
                new_lr = float(args.lr)
                for g in opt.param_groups:
                    g["lr"] = new_lr
                lr_ctl.set_base_lrs(new_lr)
                _log(f"Resumed LR was <= 0; resetting LR to --lr ({new_lr:g}).")
            else:
                phase_note = "controller_state" if resumed_lr_controller else "epoch-derived_phase"
                _log(f"Continuing with resumed LR={resumed_lr:.6g} ({phase_note})")
    elif args.resume_lr is not None:
        new_lr = float(args.resume_lr)
        for g in opt.param_groups:
            g["lr"] = new_lr
        lr_ctl.set_base_lrs(new_lr)
        _log(f"Using overridden fresh-start LR={new_lr:g}")

    model = maybe_compile_module(
        model,
        enabled=bool(args.compile_model),
        mode=str(args.compile_mode),
    )

    _log(f"Training on train={len(train_ds)} val={len(val_ds)} images")
    _log(f"Epoch plan: starting_from={start_epoch}, run_epochs={args.epochs}, final_epoch={start_epoch + args.epochs}")
    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        run_loss = 0.0
        seen = 0
        opt.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader, start=1):
            xb, yb, mb = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            if args.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            with _autocast_context(device=device, enabled=amp_enabled, amp_dtype_t=amp_dtype_t):
                out = _forward_classifier_outputs(model, xb)
                logits = out["logits"]
                loss = F.mse_loss(torch.sigmoid(logits), yb)
                mask_logits = out.get("mask_logits")
                if not isinstance(mask_logits, torch.Tensor):
                    raise RuntimeError("Berkeley training requires mask logits from the classifier.")
                loss = loss + (
                    max(0.0, float(args.mask_loss_weight))
                    * F.binary_cross_entropy_with_logits(mask_logits, mb)
                )
            loss_to_backprop = loss / float(grad_accum_steps)
            if use_scaler:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()
            do_step = ((step_idx % grad_accum_steps) == 0) or (step_idx == train_batches)
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
            run_loss += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])
        train_loss = run_loss / max(1, seen)
        val_stats = evaluate(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=args.amp_dtype,
            channels_last=bool(args.channels_last),
            mask_loss_weight=float(args.mask_loss_weight),
            mask_threshold=float(args.mask_threshold),
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": float(val_stats["loss"]),
            "val_macro_f1": float(val_stats["macro_f1"]),
            "val_micro_f1": float(val_stats["micro_f1"]),
            "val_bit_acc": float(val_stats["bit_acc"]),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        if "mask_iou" in val_stats:
            row["val_mask_iou"] = float(val_stats["mask_iou"])
            row["val_mask_dice"] = float(val_stats["mask_dice"])
        history.append(row)
        _log(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_micro_f1={row['val_micro_f1']:.4f}"
            + (f" val_mask_iou={float(row['val_mask_iou']):.4f}" if "val_mask_iou" in row else "")
        )

        if row["val_macro_f1"] > best_score:
            best_score = row["val_macro_f1"]
            best_state = {
                k: v.detach().cpu()
                for k, v in _clean_state_dict_prefixes(model.state_dict()).items()
            }

    if best_state is not None:
        try:
            model.load_state_dict(best_state, strict=True)
        except RuntimeError:
            prefixed = {f"_orig_mod.{k}": v for k, v in best_state.items()}
            model.load_state_dict(prefixed, strict=True)

    final_val = evaluate(
        model,
        val_loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=args.amp_dtype,
        channels_last=bool(args.channels_last),
        mask_loss_weight=float(args.mask_loss_weight),
        mask_threshold=float(args.mask_threshold),
    )
    final_epoch = start_epoch + int(args.epochs)
    model_state = _clean_state_dict_prefixes(model.state_dict())

    ckpt = {
        "state_dict": model_state,
        "best_state_dict": best_state,
        "optimizer_state_dict": opt.state_dict(),
        "lr_controller_state_dict": lr_ctl.state_dict(),
        "model_name": args.model,
        "num_classes": len(VOC20_CLASSES),
        "class_names": VOC20_CLASSES,
        "mask_head_enabled": bool(args.mask_head),
        "mask_decoder_channels": int(args.mask_decoder_channels),
        "args": vars(args),
        "history": history,
        "last_epoch": final_epoch,
        "best_val_macro_f1": best_score,
        "final_val": final_val,
    }
    ckpt_path = out_dir / "berkeley_sbd_multilabel_classifier.pt"
    torch.save(ckpt, ckpt_path)

    summary = {
        "args": vars(args),
        "resumed_from": resumed_from,
        "start_epoch": start_epoch,
        "final_epoch": final_epoch,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "class_names": VOC20_CLASSES,
        "best_val_macro_f1": best_score,
        "final_val": final_val,
        "history": history,
        "checkpoint": str(ckpt_path),
        "elapsed_sec": time.time() - t0,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(f"Done. Checkpoint: {ckpt_path}")
    _log(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
