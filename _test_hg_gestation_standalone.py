"""Real integration test: TinyConvClassifier + HypergraphNet on real gestation dataset.

Requirements:
  - Must use the actual gestation dataset (SemanticWheelDataset on disk)
  - Must use a real DataLoader (with semantic_mask_stack_collate)
  - Must demonstrate full HypergraphNet integration in a controlled setting

The context is set up by running the real InitVocabNode then BuildSymbolPoolNode,
exactly as the orchestrator does — no ad-hoc vocabulary or synthetic images.
The symbol pool is sourced from data/semantic_symbol_pool (real MNIST/EMNIST on disk).
provide_gestation() then builds the real SemanticWheelDataset and DataLoader.

Run:  python _test_hg_gestation_standalone.py
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pipeline.context import PipelineContext
from pipeline.nodes.data_nodes import (
    BerkeleyDataConfig,
    BerkeleyPayloadConfig,
    DataNode,
    GestationDataConfig,
    PregestationDataConfig,
)
from pipeline.nodes.vocab_node import (
    BuildSymbolPoolNode,
    InitVocabNode,
    VocabConfig,
)
from pipeline.word_cooccurrence_hypergraph import (
    HypergraphNet,
    WordCooccurrenceHypergraph,
    _term_key,
)
from wav_ml_models import TinyConvClassifier


IMAGE_SIZE       = 128
BATCH_SIZE       = 32
SAMPLES_PER_TERM = 32
EPOCHS           = 1024
LR               = 1e-2
TOP_K            = 12   # top-k scores to display in the score report
REPORT_EVERY     = 32    # print score report every N epochs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(output_dir: Path) -> PipelineContext:
    """Replicate orchestrator context init — runs InitVocabNode + BuildSymbolPoolNode."""
    ctx = PipelineContext()
    ctx.args = argparse.Namespace(
        seed=42,
        # InitVocabNode reads these
        classifier_init_ckpt="",
        semantic_vocab_extra_texts="",
        extra_semantic_terms="",
        stage_c_lora_max_terms=50,
    )
    ctx.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx.output_dir = output_dir
    ctx.total_rounds_completed = 0

    vocab_cfg = VocabConfig(
        total_slots=50,
        symbol_pool_mode="auto",
        symbol_pool_samples_per_term=SAMPLES_PER_TERM,
        image_size=IMAGE_SIZE,
        seed=42,
        symbol_pool_root=str(Path(__file__).parent / "data" / "semantic_symbol_pool"),
        symbol_include_pictograms=True,
    )

    # Run the real vocab init node — sets ctx.class_names, ctx.supervised_class_names,
    # ctx.active_extra_terms, ctx.semantic_term_to_idx
    InitVocabNode(vocab_cfg).execute(ctx)
    print(f"[vocab-init] class_names={len(ctx.class_names)} "
          f"(supervised={len(ctx.supervised_class_names)}, "
          f"extra={len(ctx.active_extra_terms)})")

    # Run the real symbol pool builder — sets ctx.symbol_pool from MNIST/EMNIST/KMNIST
    BuildSymbolPoolNode(vocab_cfg).execute(ctx)
    assert ctx.symbol_pool, (
        f"Symbol pool empty — real dataset files not found at {vocab_cfg.symbol_pool_root}. "
        "Run the pipeline once first to download MNIST/EMNIST."
    )
    n_terms  = len(ctx.symbol_pool)
    n_images = sum(len(v) for v in ctx.symbol_pool.values())
    print(f"[symbol-pool] {n_terms} terms, {n_images} images")

    return ctx


def _build_targets(
    terms_rows: list[list[str]],
    term_to_idx: dict[str, int],
    n_full: int,
    device: torch.device,
) -> torch.Tensor:
    vecs = []
    for row in terms_rows:
        y = np.zeros(n_full, dtype=np.float32)
        for t in row:
            idx = term_to_idx.get(_term_key(t), -1)
            if 0 <= idx < n_full:
                y[idx] = 1.0
        vecs.append(y)
    return torch.from_numpy(np.stack(vecs, 0)).to(device)


def _supervised_loss(logits: torch.Tensor, targets: torch.Tensor, n_sup: int) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits[:, :n_sup].float(), targets[:, :n_sup].float(), reduction="mean"
    )


def _print_score_report(
    epoch: int,
    classifier: TinyConvClassifier,
    loader: torch.utils.data.DataLoader,
    full_vocab: list[str],
    term_to_idx: dict[str, int],
    extra_nodes: list[str],
    active_hg_keys: list[str],
    n_full: int,
    device: torch.device,
    top_k: int,
) -> None:
    """
    Aggregate over the entire loader, then print:
      - Every term that is a target anywhere in the dataset with its mean score
      - Top-K predicted terms across the full combined score vector
      - Full extra-vocab section so HG improvement is clearly visible
    """
    classifier.eval()

    sum_probs  = torch.zeros(n_full)
    sum_target = torch.zeros(n_full)
    n_samples  = 0

    with torch.no_grad():
        for batch in loader:
            xb        = batch["x"].to(device)
            terms_rows = [list(r) for r in (batch.get("terms_rows") or [])]
            yb        = _build_targets(terms_rows, term_to_idx, n_full, device)

            out      = classifier.forward_with_aux(xb, active_hg_keys=active_hg_keys)
            sup_p    = torch.sigmoid(out["logits"]).cpu()
            hg_raw   = out.get("hypergraph_logits")
            hg_p     = torch.sigmoid(hg_raw).cpu() if hg_raw is not None else None

            full_p = torch.cat([sup_p, hg_p], dim=1) if hg_p is not None else sup_p
            # pad to n_full if heads don't cover it fully
            if full_p.shape[1] < n_full:
                pad = torch.zeros(full_p.shape[0], n_full - full_p.shape[1])
                full_p = torch.cat([full_p, pad], dim=1)

            B = int(xb.shape[0])
            sum_probs  += full_p.sum(dim=0)
            sum_target += yb.cpu().sum(dim=0)
            n_samples  += B

    mean_probs  = sum_probs  / max(1, n_samples)
    mean_target = sum_target / max(1, n_samples)

    # n_class / n_hg derived from the last batch's outputs
    n_class = int(out["logits"].shape[1])   # type: ignore[possibly-undefined]
    n_hg    = int(hg_p.shape[1]) if hg_p is not None else 0  # type: ignore[possibly-undefined]

    n_class = sum_probs.shape[1]
    n_hg    = hg_p.shape[1] if hg_p is not None else 0

    print(f"\n{'='*72}")
    print(f"  SCORE REPORT — after epoch {epoch}  "
          f"({n_samples} samples, full loader averaged)")
    print(f"{'='*72}")

    # ---- targets: every term that is a target in this batch ----
    target_indices = (mean_target > 0).nonzero(as_tuple=True)[0].tolist()
    print(f"\n  TARGETS ({len(target_indices)} active across batch):")
    print(f"  {'term':<30}  {'target':>6}  {'score':>6}  head")
    print(f"  {'-'*54}")
    for idx in sorted(target_indices):
        term  = full_vocab[idx] if idx < len(full_vocab) else f"idx{idx}"
        tgt   = float(mean_target[idx])
        score = float(mean_probs[idx]) if idx < len(mean_probs) else 0.0
        head  = "HG " if idx >= n_class else "sup"
        marker = ">>>" if idx >= n_class else "   "
        print(f"  {marker} {term:<28}  {tgt:>6.3f}  {score:>6.3f}  {head}")

    # ---- top-k over entire score vector ----
    if len(mean_probs) > 0:
        topk_vals, topk_idxs = torch.topk(mean_probs, k=min(top_k, len(mean_probs)))
        print(f"\n  TOP-{min(top_k, len(mean_probs))} SCORES (all heads combined):")
        print(f"  {'term':<30}  {'score':>6}  {'target':>6}  head")
        print(f"  {'-'*54}")
        for val, idx in zip(topk_vals.tolist(), topk_idxs.tolist()):
            term   = full_vocab[idx] if idx < len(full_vocab) else f"idx{idx}"
            tgt    = float(mean_target[idx]) if idx < len(mean_target) else 0.0
            head   = "HG " if idx >= n_class else "sup"
            marker = ">>>" if idx >= n_class else "   "
            hit    = " ✓" if tgt > 0 else ""
            print(f"  {marker} {term:<28}  {val:>6.3f}  {tgt:>6.3f}  {head}{hit}")

    # ---- extra-vocab section: all HG nodes with scores ----
    if n_hg > 0 and extra_nodes:
        print(f"\n  EXTRA VOCAB (HG head) — all {n_hg} nodes:")
        print(f"  {'term':<30}  {'score':>6}  {'target':>6}")
        print(f"  {'-'*44}")
        hg_start = n_class
        hg_scores = [(extra_nodes[i], float(mean_probs[hg_start + i]) if (hg_start + i) < len(mean_probs) else 0.0,
                      float(mean_target[hg_start + i]) if (hg_start + i) < len(mean_target) else 0.0)
                     for i in range(n_hg)]
        for term, score, tgt in sorted(hg_scores, key=lambda x: -x[1]):
            hit = " ✓" if tgt > 0 else ""
            print(f"  {'>>>'} {term:<28}  {score:>6.3f}  {tgt:>6.3f}{hit}")

    classifier.train()


def _hg_loss(hg_logits: torch.Tensor | None, targets: torch.Tensor, n_sup: int) -> torch.Tensor:
    if hg_logits is None:
        return torch.tensor(0.0)
    n_extra = int(targets.shape[1]) - n_sup
    if n_extra <= 0:
        return torch.tensor(0.0, device=targets.device)
    w = min(int(hg_logits.shape[1]), n_extra)
    return F.binary_cross_entropy_with_logits(
        hg_logits[:, :w].float(),
        targets[:, n_sup : n_sup + w].float(),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("HypergraphNet — real gestation pipeline integration test")
    print("=" * 72)

    tmpdir = Path(tempfile.mkdtemp(prefix="hg_gest_real_"))
    print(f"Working dir: {tmpdir}\n")

    try:
        # ------------------------------------------------------------------ #
        # 1. Real context — InitVocabNode + BuildSymbolPoolNode, same as orch  #
        # ------------------------------------------------------------------ #
        ctx = _make_ctx(tmpdir)

        # ------------------------------------------------------------------ #
        # 2. Real DataNode.provide_gestation() — builds SemanticWheelDataset  #
        # ------------------------------------------------------------------ #
        data_node = DataNode(
            preg_cfg=PregestationDataConfig(),
            gest_cfg=GestationDataConfig(
                image_size=IMAGE_SIZE,
                batch_size=BATCH_SIZE,
                samples_per_term=SAMPLES_PER_TERM,
                deformations_per_clean=1,
                cache_mb=640,
            ),
            payload_cfg=BerkeleyPayloadConfig(),
            bdata_cfg=BerkeleyDataConfig(),
        )

        print("\nCalling DataNode.provide_gestation() ...")
        data_node.provide_gestation(ctx)

        loader  = ctx.gestation_loader
        dataset = ctx.gestation_dataset
        assert loader is not None,  "provide_gestation did not set ctx.gestation_loader"
        assert dataset is not None, "provide_gestation did not set ctx.gestation_dataset"

        print(f"\n[ok] SemanticWheelDataset: {len(dataset)} rows")
        print(f"     local_vocab ({len(dataset.local_vocab)} terms): "
              f"{dataset.local_vocab[:12]}{'...' if len(dataset.local_vocab) > 12 else ''}")

        # ------------------------------------------------------------------ #
        # 3. Hypergraph from real dataset term rows                           #
        #    Nodes = terms in rows but NOT in ctx.class_names                 #
        # ------------------------------------------------------------------ #
        class_set = {_term_key(t) for t in ctx.class_names}
        real_term_rows = [dataset.read_terms_entry(i) for i in range(len(dataset))]
        extra_only_rows = [
            [t for t in row if _term_key(t) not in class_set]
            for row in real_term_rows
        ]

        hg = WordCooccurrenceHypergraph(extra_only_rows)
        print(f"\n[ok] Hypergraph (extra terms outside ctx.class_names):")
        print(f"     rows={len(hg)}, unique_edges={len(hg.unique_edges)}, "
              f"nodes={len(hg.nodes)}: {hg.nodes[:8]}{'...' if len(hg.nodes) > 8 else ''}")

        # ------------------------------------------------------------------ #
        # 4. TinyConvClassifier + HypergraphNet                               #
        # ------------------------------------------------------------------ #
        classifier = TinyConvClassifier(
            num_classes=len(ctx.class_names),
            base_ch=32,
            max_ch=128,
            context_blocks=2,
        ).to(ctx.device)

        feat_dim     = classifier.head[2].in_features
        extra_nodes  = hg.nodes
        n_extra      = max(1, len(extra_nodes))
        max_edge_len = max(4, max((len(e) for e in hg.unique_edges), default=4))

        hg_net = HypergraphNet(
            input_width=feat_dim,
            hidden_width=128,
            output_width=n_extra,
            max_hyperedge_length=max_edge_len,
        )
        hg_net.observe_hypergraph(hg)
        hg_net = hg_net.to(ctx.device)
        classifier.attach_hypergraph_net(hg_net)

        print(f"\n[ok] Classifier num_classes={len(ctx.class_names)}, feat_dim={feat_dim}")
        print(f"     HypergraphNet neurons={len(hg_net.neuron_library)}, "
              f"edges={len(hg_net._edge_data)}, output_width={n_extra}")

        # Full target map: class_names slots first, then HG extra slots
        full_vocab    = list(ctx.class_names) + extra_nodes
        n_full        = len(full_vocab)
        n_class_names = len(ctx.class_names)   # 151 — the split between sup head and HG head
        term_to_idx   = {_term_key(t): i for i, t in enumerate(full_vocab)}
        active_hg_keys = list(extra_nodes)

        # ------------------------------------------------------------------ #
        # 5. Training loop over the real gestation DataLoader                 #
        # ------------------------------------------------------------------ #
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=LR)
        classifier.train()
        step_losses: list[float] = []

        print(f"\nTraining: {EPOCHS} epochs, batch_size={BATCH_SIZE}")
        print("-" * 72)

        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            n_batches  = 0

            for batch in loader:
                xb = batch["x"].to(ctx.device)
                mb = batch.get("mask")
                if mb is not None:
                    mb = mb.to(ctx.device)
                    if mb.ndim == 3:
                        mb = mb.unsqueeze(1)

                terms_rows = [list(r) for r in (batch.get("terms_rows") or [])]
                yb = _build_targets(terms_rows, term_to_idx, n_full, ctx.device)

                out         = classifier.forward_with_aux(xb, active_hg_keys=active_hg_keys)
                logits      = out["logits"]
                hg_logits   = out.get("hypergraph_logits")
                mask_logits = out.get("mask_logits")

                loss = _supervised_loss(logits, yb, n_class_names)
                if mask_logits is not None and mb is not None:
                    loss = loss + F.binary_cross_entropy_with_logits(
                        mask_logits.float(), mb.float(), reduction="mean"
                    )
                loss = loss + 0.5 * _hg_loss(hg_logits, yb, n_class_names)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()

                v = float(loss.item())
                assert torch.isfinite(torch.tensor(v)), \
                    f"non-finite loss at step {len(step_losses)}: {v}"
                step_losses.append(v)
                epoch_loss += v
                n_batches  += 1

            avg = epoch_loss / max(1, n_batches)
            print(f"  epoch {epoch+1}/{EPOCHS}  avg_loss={avg:.4f}  batches={n_batches}")

            if (epoch + 1) % REPORT_EVERY == 0 or epoch == EPOCHS - 1:
                _print_score_report(
                    epoch=epoch + 1,
                    classifier=classifier,
                    loader=loader,
                    full_vocab=full_vocab,
                    term_to_idx=term_to_idx,
                    extra_nodes=extra_nodes,
                    active_hg_keys=active_hg_keys,
                    n_full=n_full,
                    device=ctx.device,
                    top_k=TOP_K,
                )

        # ------------------------------------------------------------------ #
        # 6. Assertions                                                        #
        # ------------------------------------------------------------------ #
        print()
        assert len(step_losses) > 0, "No training steps ran — loader was empty"
        print(f"[ok] {len(step_losses)} steps, all losses finite")

        assert step_losses[-1] < step_losses[0] * 20, \
            f"Loss exploded: {step_losses[0]:.4f} → {step_losses[-1]:.4f}"
        print(f"[ok] Loss stable: {step_losses[0]:.4f} → {step_losses[-1]:.4f}")

        classifier.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            xb_s = sample_batch["x"].to(ctx.device)
            out_s = classifier.forward_with_aux(xb_s, active_hg_keys=active_hg_keys)

        assert "logits" in out_s,            "missing 'logits'"
        assert "hypergraph_logits" in out_s, "missing 'hypergraph_logits'"
        assert out_s["logits"].shape[1] == len(ctx.class_names), \
            f"logits dim: {out_s['logits'].shape[1]} vs {len(ctx.class_names)}"
        assert out_s["hypergraph_logits"].shape[1] == n_extra, \
            f"hg_logits dim: {out_s['hypergraph_logits'].shape[1]} vs {n_extra}"
        print("[ok] both logit heads correct shape")

        print("\n[PASS] Full real-gestation HypergraphNet integration test passed.")

    finally:
        shutil.rmtree(str(tmpdir), ignore_errors=True)
        print(f"Cleaned up {tmpdir}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
