from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from pipeline.weight_map import (
    WEIGHT_IMAGE_MODES,
    annotate_weight_map,
    render_weight_image,
    snapshot_parameter_state_from_state_dict,
)


def _safe_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text or ""))
    token = token.strip("._")
    return token or "weights"


def _to_float_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        if value.numel() <= 0:
            return None
        if not torch.is_floating_point(value) and not value.dtype.is_complex:
            if value.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.bool,
            }:
                return None
        return value.detach().to(device="cpu", dtype=torch.float32).clone()

    if isinstance(value, np.ndarray):
        if value.size <= 0:
            return None
        if value.dtype.kind not in {"f", "i", "u", "b"}:
            return None
        return torch.from_numpy(np.asarray(value)).to(dtype=torch.float32).clone()

    if isinstance(value, (float, int, bool, np.floating, np.integer)):
        return torch.tensor([value], dtype=torch.float32)

    return None


def _to_tensor_state_dict(mapping: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for key, value in mapping.items():
        tensor = _to_float_tensor(value)
        if tensor is None:
            continue
        state[str(key)] = tensor
    return state


def _extract_pt_candidates(blob: Any) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    seen: set[str] = set()

    def add(label: str, value: Any) -> None:
        if str(label) in seen:
            return
        if torch.is_tensor(value):
            state = {"tensor": _to_float_tensor(value)}
            state = {k: v for k, v in state.items() if v is not None}
        elif isinstance(value, dict):
            state = _to_tensor_state_dict(value)
        else:
            state = {}
        if not state:
            return
        seen.add(str(label))
        candidates.append((str(label), state))

    if torch.is_tensor(blob):
        add("root", blob)
        return candidates

    if not isinstance(blob, dict):
        return candidates

    add("root", blob)

    state_dict = blob.get("state_dict")
    if isinstance(state_dict, dict):
        add("state_dict", state_dict)

    for key, value in blob.items():
        if isinstance(value, dict):
            add(str(key), value)

    return candidates


def _extract_numpy_candidates(path: Path) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    blob = np.load(path, allow_pickle=True)
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = []

    if isinstance(blob, np.lib.npyio.NpzFile):
        state = _to_tensor_state_dict({str(k): blob[k] for k in blob.files})
        blob.close()
        if state:
            candidates.append(("root", state))
        return candidates

    if isinstance(blob, np.ndarray) and blob.dtype == object and blob.shape == ():
        try:
            item = blob.item()
        except Exception:
            item = None
        if isinstance(item, dict):
            state = _to_tensor_state_dict(item)
            if state:
                candidates.append(("root", state))
            return candidates

    tensor = _to_float_tensor(blob)
    if tensor is not None:
        candidates.append(("root", {"array": tensor}))
    return candidates


def _load_candidates(path: Path) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return _extract_pt_candidates(blob)
    if suffix in {".npy", ".np", ".npz"}:
        return _extract_numpy_candidates(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _select_candidates(
    candidates: Sequence[Tuple[str, Dict[str, torch.Tensor]]],
    selected_key: str,
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    if not selected_key:
        return list(candidates)
    picked = [(label, state) for label, state in candidates if str(label) == str(selected_key)]
    if not picked:
        labels = ", ".join(label for label, _ in candidates) or "<none>"
        raise KeyError(f"Requested key {selected_key!r} not found. Available: {labels}")
    return picked


def _output_path(input_path: Path, label: str, output_dir: Path | None, mode: str) -> Path:
    stem = input_path.stem
    if str(label) and str(label) != "root":
        stem = f"{stem}.{_safe_token(label)}"
    stem = f"{stem}.{_safe_token(mode)}"
    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{stem}.png"


def _sidecar_payload(
    *,
    source_path: Path,
    label: str,
    mode: str,
    rgb: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source": str(source_path),
        "label": str(label),
        "mode": str(mode),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
    }
    if str(mode) == "parameter_groups":
        payload["node_count"] = int(meta.get("node_count", 0))
        payload["grid_side"] = int(meta.get("grid_side", 0))
        payload["nodes"] = [
            {
                "node": str(row.get("node", "")),
                "count": int(row.get("count", 0)),
                "keys": list(row.get("keys", [])),
            }
            for row in list(meta.get("stats", []))
        ]
        return payload

    payload["layer_count"] = int(meta.get("layer_count", 0))
    payload["raw_width"] = int(meta.get("raw_width", 0))
    payload["raw_height"] = int(meta.get("raw_height", 0))
    payload["transposed"] = bool(meta.get("transposed", False))
    payload["layers"] = [
        {
            "name": str(layer.get("name", "")),
            "label": str(layer.get("label", "")),
            "unit_count": int(layer.get("unit_count", 0)),
            "keys": list(layer.get("keys", [])),
        }
        for layer in list(meta.get("layers", []))
    ]
    return payload


def _render_one(
    *,
    source_path: Path,
    label: str,
    state: Dict[str, torch.Tensor],
    image_size: int,
    output_dir: Path | None,
    mode: str,
) -> Path:
    parameter_keys, parameter_state = snapshot_parameter_state_from_state_dict(state)
    if not parameter_state:
        raise ValueError(f"No float tensors found in {source_path} candidate {label!r}")

    rgb, meta = render_weight_image(
        parameter_state,
        parameter_keys=parameter_keys,
        reference_state=None,
        target_width=image_size,
        target_height=image_size,
    )
    subtitle = ""
    if str(mode) == "parameter_groups":
        subtitle = f"{source_path.name} nodes={int(meta.get('node_count', 0))}"
    rgb = annotate_weight_map(
        rgb,
        title=f"{(str(label) if str(label) != 'root' else source_path.stem)} [{mode}]",
        subtitle=subtitle,
    )

    out_path = _output_path(source_path, label, output_dir, mode=mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG")
    sidecar = _sidecar_payload(
        source_path=source_path,
        label=label,
        mode=mode,
        rgb=rgb,
        meta=meta,
    )
    out_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render weight images using the shared pipeline.weight_map renderer.",
    )
    parser.add_argument("inputs", nargs="+", help="One or more .pt, .npy/.np, or .npz files.")
    parser.add_argument(
        "--key",
        default="",
        help="Optional candidate key inside the file, such as state_dict or classifier_state.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Output image size in pixels. Default: 256.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for output PNG files. Defaults next to each input.",
    )
    parser.add_argument(
        "--mode",
        default="parameter_groups",
        choices=WEIGHT_IMAGE_MODES,
        help="Render mode. Default: parameter_groups.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else None
    image_size = max(8, int(args.size))
    failures = 0

    for raw_path in args.inputs:
        path = Path(raw_path).resolve()
        if not path.exists():
            print(f"[weight-image] missing file: {path}", file=sys.stderr)
            failures += 1
            continue

        try:
            candidates = _load_candidates(path)
            if not candidates:
                raise ValueError("no usable float-tensor state dicts found")
            chosen = _select_candidates(candidates, str(args.key or "").strip())
            for label, state in chosen:
                out_path = _render_one(
                    source_path=path,
                    label=label,
                    state=state,
                    image_size=image_size,
                    output_dir=output_dir,
                    mode=str(args.mode),
                )
                print(f"[weight-image] {path.name} [{label}] ({args.mode}) -> {out_path}")
        except Exception as exc:
            print(f"[weight-image] failed for {path}: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
