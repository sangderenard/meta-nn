# DataNode Architecture Guidance
## Pipeline Design Principles for the nodus Training System

---

## 1. Graph Execution Model

### Edge-Driven Data Provision

The pipeline graph executor fires `PipelineEdge.on_traverse(ctx)` for every **active** incoming
edge immediately before a target node's `execute()` runs. This is the sole mechanism by which
data is delivered to training nodes. No training node calls into the DataNode directly.

The edge list in `orchestrator.py` is both the documentation and the execution specification.
Reading `build_pipeline_graph()` tells you exactly what data flows where, under what condition,
and which DataNode method provisions it.

```
data_node → stage_0_pregestation   on_traverse=provide_pregestation   (unconditional)
data_node → stage_1_gestation      on_traverse=provide_gestation       (after gate_pregestation)
data_node → stage_2_berkeley       on_traverse=provide_berkeley_data   (after early_gates)
data_node → gate_berkeley          on_traverse=provide_gate_data       (after early_gates)
data_node → stage_g_generator      on_traverse=provide_payload         (after all_gates)
data_node → build_flashcard_rows   on_traverse=provide_payload         (unconditional)
```

### DataNode.execute() — Active Housekeeping

`DataNode.execute()` is **not** a no-op. It is the per-round housekeeping sweep over all
registered possessions:

- Check each possession for expiry against its policy
- Evict stale possessions (free GPU tensors, drop dataset refs)
- Log storage sizes per tier (GPU / RAM / disk)

It does **not** build data — building is JIT via `on_traverse`. The housekeeping sweep runs
after building (possessions are present or expired; expired ones get rebuilt on the next
`on_traverse` call for the first consumer that needs them).

`DataNode.should_run()` gates this sweep. If no vocabulary exists there is nothing to expire,
evict, or measure.

---

## 2. DataNode as Storage Authority

The DataNode is the single authority over all storage interactions across GPU memory, RAM,
and disk. It tracks every data object it manages as a **possession** with explicit metadata:

```python
@dataclass
class DataPossession:
    name: str                           # "pregestation", "gestation", "berkeley", "payload"
    tier: str                           # "disk" | "ram" | "gpu"
    ctx_attrs: List[str]                # attribute names on PipelineContext this possession owns
    build_fn: Callable[[ctx], None]     # how to (re)build; called by on_traverse
    expiry_fn: Callable[[ctx], bool]    # returns True when this possession is stale
    size_fn: Callable[[], int]          # bytes currently held, 0 if not built
```

The sweep in `execute()` iterates the registry. The `on_traverse` callbacks delegate to
`build_fn` through the public `provide_*` methods. Every `provide_*` method is idempotent:
it checks the possession's expiry state before building.

---

## 3. Possession Specifications

### 3a. Pregestation Possession

**Source:** Fully synthetic — geometric shapes, directional indicators, noise textures.
No real imagery.

**Expiry:** Vocab hash change. On expiry, the DataNode evicts the previous mode-pass results
down to the configured cap; it holds the cap and governs how much prior data survives.

**N-pass structure:** Pregestation has multiple mode configurations
(`direction_color`, `symbol`, `noise_texture`, and any future additions). Each mode is a
separate pass. The DataNode manages which passes are live, their age, and their cumulative
row count against the cap. This branching is intentional: the pregestation system shifts
mode parameters to diversify learning across rounds.

**Built-in validation reserve:** Each pass produces a reserved validation split.
No separate gate_val_loader for pregestation data exists or will be created.

### 3b. Gestation Possession

**Source:** Symbol image pool — per-term reference images from MNIST/EMNIST-style sources,
synthetically rendered characters, or internal bootstrap images. Built by `BuildSymbolPoolNode`
and placed in `ctx.symbol_pool`. Gestation is **not** a subset of Berkeley data.

**Expiry:** Vocab hash change. Same N-pass / cap model as pregestation.

**Source mixing:** The DataNode may conglomerate gestation from multiple sources (symbol pool,
selected pregestation rows, real-image rows) via a wrapper that manages per-source caps and
applies the overall possession cap across the combined result.

**Built-in validation reserve:** Same policy as pregestation. No separate loader.

### 3c. Berkeley Possession

**Source:** Static SBD image files on disk. Expresses the full vocabulary via semantic
segmentation masks.

**Expiry / Churn:** Rows are periodically replaced on a churn schedule (configurable N rounds).
Churn and any row pruning **invalidate the entire composite** — partial updates are not allowed
because the composite is the full-vocabulary state. When the composite is invalidated, the
DataNode rebuilds it from scratch.

**Pre-vectorized deformations (hot loop elimination):**
At possession build time, the DataNode generates N committed deformed variants per source image
using `DiskSemanticRowsDataset._apply_degrade()` from `semantic_dataset_loaders.py`. This
function applies a stochastic combination of: blur (avg_pool2d), stride-skew (odd-row shift),
pixel dropout, quantization, and Gaussian noise to the image. Critically, degrade also
**produces its own mask**: each degradation operation accumulates a "touch" map recording
where and how strongly the image was modified. This touch map is then blended with the
original mask via `_blend_attention_maps()` to produce a new composite mask for the
deformed variant. The mask is not merely transformed geometrically — **each deformation
supplies its own mask contribution**.

For each deformed variant, composite mask regeneration is mandatory:

1. Apply `_apply_degrade(image_chw, mask_hw, idx)` → `(deformed_image, degrade_blended_mask)`
2. The deformed variant's mask goes into the label stack alongside all other masks:
   run `build_term_mask_stack_from_image(deformed_image, label_vec, ...)` or
   `build_label_mask_stack(degrade_blended_mask, label_vec, ...)` from
   `semantic_dataset_loaders.py` → per-label mask stack `[K, H, W]`
3. Call `_composite_mask_stack(stack)` (canonical function in `semantic_dataset_loaders.py:403`)
   → specially normalized gestalt composite `[H, W]`
4. Store: `(deformed_image, label_vec, composite_mask, per_label_stack)` as a committed row

Each deformed variant is a full row in the dataset. It carries its own label (same as source),
its own per-label mask stack (incorporating the deformation touch map), and its own composite
regenerated directly from `_composite_mask_stack()`. **The composite from the original
undeformed image is never reused for any deformed variant.** When rows are pruned for
reduced representation, the composite must be regenerated in full because the composite
is the full vocabulary state.

The hot-loop degrade flag (`degrade=True`) is retired for Berkeley. The loader becomes a
simple indexed reader over pre-built rows.

**Cache:** Retired as a separate object. Pre-built rows stored by the DataNode serve this role,
managed using the same model as pregestation caching.

**Built-in validation reserve:** Every dataset managed by DataNode contains a reserve pool
of validation rows that is maintained as an **orphan-free split**: every class label present
in the validation reserve must also have at least one training row. If a churn/prune cycle
removes the last training row for a class, that class's validation rows are either promoted
to training or removed. This constraint is re-evaluated after every composite rebuild.
No separate `berkeley_gate_val_loader` exists or will be created.

### 3d. Payload Possession

**Source:** A filtered view of the Berkeley possession. Every payload row must carry the
`"berkeley sbd dataset"` term tag and a valid supervised label vector. Rows are screened
at possession build time.

**What it holds:**
- Semantic segmentation masks (for spatial conditioning of the GAN)
- Conditioning vectors built from label embeddings (for class conditioning)
- Pixel images — present because the discriminator requires real images for adversarial training,
  not for pixel-level supervision of the generator

**Expiry:** Built once per run. Rebuilds only if the Berkeley possession is invalidated
(invalidation propagates).

**Payload validation:** Retired as a separate loader. The Berkeley possession's validation
reserve covers this. No `payload_validation_loader` or `payload_validation_dataset` exists
in the new design.

---

## 4. GAN Training Principle — Classifier as IR

The classifier's analysis of generated content is the graph's intermediate representation.
Every gradient path into the generator must pass through the classifier's semantic assessment
of what the generated image contains.

**Acceptable generator losses:**
- Adversarial loss: `F.softplus(-discriminator(fake))` — standard GAN, no image comparison
- Classifier feature score loss: `(1 - target_prob)` where `target_prob` comes from running
  the classifier on the generated image and comparing to the condition vector

**Losses that must remain weight=0.0 (inert):**
- `mask_loss`: `F.binary_cross_entropy_with_logits(fake_mask_logits, real_mask_g)` —
  compares generator mask output to real Berkeley segmentation mask (reference-based)
- `outside_loss`: `|fake_image - real_image| * outside_weight` —
  direct pixel comparison between generated image and real Berkeley image (reference-based)

These are preserved in code at weight zero because they are the wrong gradient for this
system's purpose — not because they are optional enhancements. The generator's job is not
to reproduce images. It is to produce images that contain the requested semantic content
in the requested spatial regions, as judged by the classifier.

The `wave_recon_weight` loss operates in the audio/bit-plane domain derived from the
generated image's encoded bit sequence. It does not compare generated pixels to real image
pixels and is not a reference image loss.

**No agent or automated process may add image comparison losses to the generator
gradient path.** The configurability and inert-default state is the premium state.

---

## 5. What This Means for New Contributions

**Adding a new data source to a possession:**
Define the source's cap, expiry policy, and how its rows integrate into the combined cap.
Register it in the DataNode's possession registry. Do not add it to a training node's
`execute()` or `should_run()`.

**Adding a new training node that needs data:**
Add an edge from `data_node` to the new node with `on_traverse=_data_node.provide_X` and
`condition=<appropriate gate predicate>`. The data flows from the edge, not from the node.

**Adding a new deformation type:**
It must go through `DiskSemanticRowsDataset._apply_degrade()` in `semantic_dataset_loaders.py`
or follow the same protocol: apply the deformation to the image, accumulate a touch map
recording where and how strongly the image was modified, blend the touch map into the mask
via `_blend_attention_maps()`, then recompute the per-label mask stack and call
`_composite_mask_stack()` to regenerate the gestalt composite. Every deformation supplies
its own mask. Do not reuse the original undeformed composite.

**Adding a new GAN loss:**
It must be classifier-mediated. The classifier's sigmoid output on the generated image,
compared against the conditioning vector, is the permitted signal path. Any loss that
compares generated image content to real image content at the pixel level must default
to weight=0.0 and must not be increased during normal operation.
