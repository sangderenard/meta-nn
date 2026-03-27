# HypergraphNet Integration Plan

## Goal

Replace the per-slot LoRA adapter mechanism (Stage C) with a permanently resident
`HypergraphNet` that is attached to `TinyConvClassifier` at build time, observed
against the live dataset term hypergraph before the first training stage, and trained
continuously alongside (or independently of) the classifier backbone across all stages.

---

## Context you must hold

### What exists today

| Symbol | File | Role |
|---|---|---|
| `TinyConvClassifier` | `wav_ml_models.py` | Backbone; already has `attach_hypergraph_net`, `set_backbone_training`, `set_hypergraph_net_training`, `forward_with_aux(x, active_hg_keys)` |
| `HypergraphNet` | `pipeline/word_cooccurrence_hypergraph.py` | Vocab-extension network; `observe_hypergraph(hg)` populates neuron library from a `WordCooccurrenceHypergraph` |
| `WordCooccurrenceHypergraph` | same file | Pure data structure; nodes = terms, hyperedges = dataset rows (frozensets of term keys) |
| `ClassifierConfig` | `pipeline/nodes/classifier_node.py` | All classifier hyperparameters; already has `stageC_max_terms: int = 50` — this integer IS `max_hyperedge_length` |
| `BuildClassifierNode` | same file | Builds classifier, optimizer, puts both on `ctx`; runs once |
| `_run_classifier_refresh_epochs` | same file | Shared inner training loop used by all four classifier stages; calls `_forward_classifier_outputs_require_mask` and consumes `out["logits"]` |
| `_forward_classifier_outputs_require_mask` | `pipeline/nodes/data_nodes.py` | Thin wrapper that calls `classifier.forward_with_aux(xb)` and raises on missing mask |
| `batch_meta["terms_rows"]` | inside `_run_classifier_refresh_epochs` | Per-sample term lists for the current batch — already present, already used to build `yb`; this is the `active_hg_keys` source |
| `_dataset_terms_rows` | `pipeline/nodes/data_nodes.py` | Scans an entire dataset and returns all term rows; already used in `LoRARoundNode` |
| `LoRARoundNode` | `pipeline/nodes/classifier_node.py` | Stage C: sweeps planned vocab-LoRA slots; this is what the HypergraphNet replaces |
| `ctx.active_extra_terms` | `pipeline/context.py` | List of dynamic vocab terms currently active; source for `active_hg_keys` when `batch_meta` is unavailable |
| `ctx.semantic_term_to_idx` | same | Full term → class-index map |
| Orchestrator | `pipeline/orchestrator.py` | Builds the pipeline graph; parses `ClassifierConfig` from args; wires all nodes and edges |

### What the forward pass currently does

`_run_classifier_refresh_epochs` inner step:
```
out = _forward_classifier_outputs_require_mask(classifier, xb_part)
logits = out["logits"]          # (B, num_classes)
# hypergraph_logits key exists in out only if hypergraph_net is attached — currently unused
loss = _classifier_supervision_loss(classifier, logits, yb_part, ...)
mask_loss = _semantic_mask_bce_loss(out["mask_logits"], mb_part)
total_loss = loss * SCALE + mask_loss
```

The `active_hg_keys` argument to `forward_with_aux` is never passed — so
`HypergraphNet._compose_dynamic_head_weight` always sees `active_keys=None`
and uses all edges. It also never receives gradient because `hypergraph_logits`
is never included in the loss.

### Pooled feature dimension

`TinyConvClassifier` computes `c3 = min(max_ch, round(base_ch * 4))`.
With defaults `base_ch=64, max_ch=384` this is `min(384, 256) = 256`.
This is **not exposed** as a property. The plan adds one.

---

## Changes — ordered by dependency

### 1. Expose `pooled_feature_dim` on `TinyConvClassifier` (`wav_ml_models.py`)

Add a `@property` that returns `c3` (the channel count after the conv backbone,
before `AdaptiveAvgPool2d`). `c3` is currently computed inline in `__init__` as
a local variable. Save it as `self._pooled_feature_dim = c3` immediately after
it is computed, then expose it:

```python
@property
def pooled_feature_dim(self) -> int:
    return self._pooled_feature_dim
```

This lets `BuildClassifierNode` read the right `input_width` for `HypergraphNet`
without duplicating the `min(max_ch, round(base_ch * 4))` formula.

---

### 2. Add HypergraphNet fields to `ClassifierConfig` (`pipeline/nodes/classifier_node.py`)

Add these fields to the `@dataclass ClassifierConfig`:

```python
# ---- HypergraphNet (vocab-extension, replaces Stage C LoRA) -------------
hypernet_enabled: bool = True
hypernet_hidden_width: int = 0      # 0 = use pooled_feature_dim (auto)
hypernet_lr: float = 0.0            # 0.0 = share classifier optimizer
hypernet_train_with_backbone: bool = True   # train hypernet during backbone stages
hypernet_backbone_frozen_in_stageC: bool = False  # freeze backbone during stageC equivalent
```

`max_hyperedge_length` is not a new field — it is `stageC_max_terms` (already 50).
The two concepts are the same thing: the maximum number of dynamic vocabulary terms
the network must distinguish at once.

---

### 3. Add `ctx.hypergraph_net` and `ctx.hypergraph_optimizer` to `PipelineContext` (`pipeline/context.py`)

```python
hypergraph_net: Optional[Any] = None          # HypergraphNet instance
hypergraph_optimizer: Optional[Any] = None    # separate optimizer, or None to share
```

Typed as `Any` to avoid importing from the pipeline sub-package (mirrors the
`nn.Module` annotation strategy in `wav_ml_models.py`).

---

### 4. Build and attach `HypergraphNet` in `BuildClassifierNode.execute()` (`pipeline/nodes/classifier_node.py`)

Immediately after `ctx.classifier = model`, when `cfg.hypernet_enabled` is true:

```python
if cfg.hypernet_enabled:
    from pipeline.word_cooccurrence_hypergraph import HypergraphNet
    _hidden = int(cfg.hypernet_hidden_width) if int(cfg.hypernet_hidden_width) > 0 \
              else int(model.pooled_feature_dim)
    hg_net = HypergraphNet(
        input_width=int(model.pooled_feature_dim),
        hidden_width=_hidden,
        output_width=int(n_classes),
        max_hyperedge_length=int(cfg.stageC_max_terms),
    ).to(ctx.device)
    model.attach_hypergraph_net(hg_net)
    ctx.hypergraph_net = hg_net

    if float(cfg.hypernet_lr) > 0.0:
        ctx.hypergraph_optimizer = torch.optim.AdamW(
            hg_net.parameters(),
            lr=float(cfg.hypernet_lr),
            weight_decay=float(cfg.weight_decay),
        )
    # else: hg_net parameters are already in model.parameters() via add_module,
    #       so the classifier optimizer covers them.
```

**Important**: because `attach_hypergraph_net` calls `self.hypergraph_net = net`
(routed through `__setattr__` into `_modules`), the net's parameters are already
included in `model.parameters()`. If `hypernet_lr == 0.0`, no second optimizer is
needed — they share the classifier's AdamW. If a separate optimizer is used, the
calling code must exclude `hypergraph_net` parameters from the classifier optimizer
(or accept double-counting, which is harmless for correctness but wastes memory).
Recommendation: keep `hypernet_lr = 0.0` by default (shared optimizer) and only
split when explicit separate LR is needed.

If a checkpoint is being resumed, `model.load_state_dict(..., strict=False)` already
handles the new `hypergraph_net.*` keys gracefully (unknown keys are ignored with
`strict=False`). The hypernet starts fresh on resume if no checkpoint keys match.

---

### 5. Add `ObserveHypergraphNode` — a new pipeline node

Create this class in `pipeline/nodes/classifier_node.py` (or a new thin file
`pipeline/nodes/hypergraph_node.py`).

**Purpose**: scan the gestation dataset's term rows, build a
`WordCooccurrenceHypergraph`, call `ctx.hypergraph_net.observe_hypergraph(hg)`.
This populates the neuron library and edge data so the dynamic head stops using
the stub weight.

```python
class ObserveHypergraphNode(PipelineNode):
    node_id = "observe_hypergraph"
    description = "Scan dataset terms and prime HypergraphNet neuron library"

    def should_run(self, ctx):
        return (
            ctx.hypergraph_net is not None
            and getattr(ctx, "gestation_dataset", None) is not None
            and not getattr(ctx, "_hypergraph_observed", False)
        )

    def execute(self, ctx):
        from pipeline.word_cooccurrence_hypergraph import WordCooccurrenceHypergraph
        from pipeline.nodes.data_nodes import _dataset_terms_rows
        term_rows = _dataset_terms_rows(ctx.gestation_dataset, progress_control=ctx)
        hg = WordCooccurrenceHypergraph(term_rows)
        ctx.hypergraph_net.observe_hypergraph(hg)
        ctx._hypergraph_observed = True
        _log(f"[hypergraph] observed {len(hg)} rows, "
             f"{len(hg.unique_edges)} unique edges, "
             f"{len(hg.nodes)} nodes registered in neuron library")
```

**When to run**: after gestation data is built, before gestation training.
Edge placement in the graph: `"provide_gestation" → "observe_hypergraph" → "stage_1_gestation"`.

**Re-observation**: set `ctx._hypergraph_observed = False` in `should_run`
whenever `ctx.gestation_dataset` changes identity (i.e., on rebuild). The simplest
implementation: track the `id()` of the dataset object and re-observe when it changes.

---

### 6. Thread `active_hg_keys` through the training loop

#### 6a. `_forward_classifier_outputs_require_mask` (`pipeline/nodes/data_nodes.py`)

Add `active_hg_keys: Optional[List[str]] = None` parameter and pass it to
`classifier.forward_with_aux(xb, active_hg_keys)`. All existing call sites pass
no argument → default None → backward-compatible.

#### 6b. `_run_classifier_refresh_epochs` (`pipeline/nodes/classifier_node.py`)

Add `active_hg_keys: Optional[List[str]] = None` parameter.

Inside the inner step, after `_terms` is resolved from `batch_meta["terms_rows"]`,
derive batch-level active keys:

```python
_batch_hg_keys: Optional[List[str]] = None
if active_hg_keys is not None:
    _batch_hg_keys = active_hg_keys
elif _terms:
    # Union of all term sets in this batch — tells the hypernet what's
    # actively co-occurring right now so it can filter its edge set.
    _seen: set = set()
    _batch_hg_keys = []
    for row in _terms:
        for t in row:
            k = str(t).strip().lower()
            if k and k not in _seen:
                _seen.add(k)
                _batch_hg_keys.append(str(t))
```

Then pass `_batch_hg_keys` to `_forward_classifier_outputs_require_mask`.

#### 6c. All four stage call sites

`PregestationTrainNode`, `GestationTrainNode`, `BerkeleyRefreshTrainNode`, and
`LoRARoundNode` all call `_run_classifier_refresh_epochs`.  No changes needed to
the call sites — the default `active_hg_keys=None` means the hypernet uses all
edges, which is correct for stages that do not have a narrowed vocab focus.

---

### 7. Include `hypergraph_logits` in the training loss

In `_run_classifier_refresh_epochs`, immediately after `logits = out["logits"]`:

```python
hg_logits = out.get("hypergraph_logits")   # None if hypergraph_net not attached
```

Then in the loss computation block, after `mask_loss` is computed:

```python
if hg_logits is not None:
    hg_loss, _, _, _ = _classifier_supervision_loss(
        classifier=classifier,
        logits=hg_logits,
        y_multihot=yb_part,
        supervised_dim=int(supervised_dim),
        semantic_cosine_weight=float(semantic_cosine_weight),
        semantic_soft_target_max=float(semantic_soft_target_max),
    )
    loss = loss + hg_loss
```

This is safe: `_classifier_supervision_loss` only uses `logits` and `y_multihot`
dimensionally — `hg_logits` is `(B, num_classes)`, the same shape as `logits`.
The hypernet's BCE+cosine loss is added at full weight alongside the backbone loss.
No new hyperparameter is needed at this stage; if tuning is required later, a
`hypernet_loss_weight` field can be added to `ClassifierConfig`.

The `del` line at the end of the step already handles `out` and `logits`.
Add `hg_logits` to the same `del` statement to avoid holding references.

---

### 8. Training mode control per stage

Each stage's `execute()` method currently calls `ensure_vocab_lora_active` (which
loads a LoRA adapter) and `deactivate_vocab_lora_slot` (which unloads it). The
HypergraphNet does not swap — it is always loaded. The LoRA calls can be left
in place for now (they are no-ops when `lora_enabled=False`) or removed stage by
stage. The new training mode control is additive:

**Before each stage's `_run_classifier_refresh_epochs` call**, add:

```python
if ctx.hypergraph_net is not None:
    ctx.classifier.set_hypergraph_net_training(cfg.hypernet_train_with_backbone)
```

**For Stage C replacement** (see section 9 below), use:

```python
ctx.classifier.set_backbone_training(not cfg.hypernet_backbone_frozen_in_stageC)
ctx.classifier.set_hypergraph_net_training(True)
```

After the stage completes, restore both to True (or whatever the next stage needs).
Because `set_backbone_training` iterates `named_children()` and skips
`"hypergraph_net"`, and `set_hypergraph_net_training` touches only
`self.hypergraph_net`, these calls are fully independent.

---

### 9. Replace `LoRARoundNode` (Stage C)

`LoRARoundNode` sweeps vocab-LoRA slots one at a time, loading and unloading
adapters for each group of terms. With `HypergraphNet` resident, this sweep is
replaced by a single training pass on Berkeley data where the backbone may be
frozen and the hypernet trains against the full vocabulary.

**Option A (minimal disruption)**: gate `LoRARoundNode.should_run` to return
`False` when `cfg.hypernet_enabled and not cfg.lora_enabled`. Add a new
`HypergraphStageC_Node` that runs the Berkeley loader through
`_run_classifier_refresh_epochs` with `set_backbone_training(not frozen)` and
`set_hypergraph_net_training(True)`.

**Option B (immediate)**: in `LoRARoundNode.execute()`, add an early return
when `ctx.hypergraph_net is not None and not cfg.lora_enabled`. This is the
simplest change — the node exits cleanly and the hypernet trains during the
upstream Berkeley stage instead.

Recommendation: implement Option B first for speed, then promote to Option A
once the hypernet is confirmed to be learning.

---

### 10. Orchestrator wiring (`pipeline/orchestrator.py`)

#### 10a. Import `ObserveHypergraphNode`

```python
from pipeline.nodes.classifier_node import (   # or hypergraph_node
    ...
    ObserveHypergraphNode,
)
```

#### 10b. Register and wire the node

In `_build_pipeline_graph` (or equivalent), after the gestation data node is
added:

```python
g.add_node(ObserveHypergraphNode())

# observe fires after gestation data is ready, before gestation training
g.add_edge("provide_gestation", "observe_hypergraph", label="data_ready")
g.add_edge("observe_hypergraph", "stage_1_gestation", label="hypernet_primed")
```

If the gestation provider is named differently, find the edge currently going
directly to `"stage_1_gestation"` and insert `"observe_hypergraph"` into that path.

#### 10c. Config parsing

`ClassifierConfig` fields added in step 2 are already picked up by the orchestrator's
existing `ClassifierConfig(...)` construction from args — no new argument parsing
is required as long as the fields have defaults. If the operator wants to override
them, add to the arg namespace:

```python
hypernet_enabled=bool(_g("hypernet_enabled", default=True)),
hypernet_hidden_width=int(_g("hypernet_hidden_width", default=0)),
hypernet_lr=float(_g("hypernet_lr", default=0.0)),
hypernet_train_with_backbone=bool(_g("hypernet_train_with_backbone", default=True)),
hypernet_backbone_frozen_in_stageC=bool(_g("hypernet_backbone_frozen_in_stageC", default=False)),
```

---

### 11. Checkpoint compatibility

`model.load_state_dict(..., strict=False)` is already in `BuildClassifierNode`.
New `hypergraph_net.*` keys in the state dict are loaded when present and silently
skipped when absent. No migration is needed.

However: when saving a checkpoint, the existing code does
`ctx.classifier.state_dict()` — because `hypergraph_net` is a registered submodule
via `add_module`, its parameters are already included in that dict automatically.
No save-path changes are needed.

---

## Implementation order

1. `wav_ml_models.py` — expose `pooled_feature_dim` property
2. `classifier_node.py` — add `ClassifierConfig` fields
3. `context.py` — add `hypergraph_net`, `hypergraph_optimizer` fields
4. `BuildClassifierNode.execute()` — attach `HypergraphNet`
5. `data_nodes.py` — thread `active_hg_keys` through `_forward_classifier_outputs_require_mask`
6. `_run_classifier_refresh_epochs` — add `active_hg_keys` param, derive batch keys, add `hg_logits` to loss
7. `ObserveHypergraphNode` — new class
8. `orchestrator.py` — wire the new node
9. Stage execute() methods — add `set_hypergraph_net_training` calls
10. `LoRARoundNode` — add early-exit when hypernet replaces it

Each step is independently testable. After step 6, running
`_test_hypergraph_net.py` with a patched `_run_classifier_refresh_epochs`
call already validates the gradient path. After step 8, the live pipeline
will use the hypernet on every training step without any operator action.

---

## Invariants to maintain

- `_run_classifier_refresh_epochs` must remain backward-compatible — all new
  parameters default to `None` / `False` / `0`.
- `TinyConvClassifier.forward(x)` with no second argument must still return
  plain logits — the `active_hg_keys=None` default ensures this.
- `hypergraph_logits` in the loss must be guarded by `if hg_logits is not None`
  so pipelines without the hypernet are unaffected.
- `ObserveHypergraphNode.should_run` must be idempotent — re-entry without
  dataset change must return False immediately.
- The hypernet's parameters must NOT appear twice in any optimizer (check when
  `hypernet_lr > 0` that the classifier optimizer is built from a filtered
  `model.parameters()` call).
