# Remaining Bugs & Work Items

## Investigation Required

### System instability
System service exception — hunch: stale GPU allocation accessed lazily after race condition.

### GPU underutilization in faux model adapter processes
Adapter processes not saturating GPU. May need profiling to identify bottleneck.

### Microbatch cap too sensitive
Tuning needed — current cap triggers too aggressively.

### Eval threading vs GPU management
If evals aren't on their own thread they need the GPU management system; if they are on their own thread they're not doing their job. Priority is expressibility through the IR+mermaid+runtime.

---

## Vocab / Symbol Pool — Investigate Only

### 38 symbol pool terms not in vocab
`_flatten_symbol_pool` warns: digit 0-9, letter a-z, emnist/mnist dataset not in vocab.

**Investigation findings** (see `/memories/repo/meta-nn-notes.md`):
- `_default_class_names()` in `pipeline/utils.py` has 101 entries (100 unique). Digits, letters, and dataset terms were **never** in the fixed vocab or bootstrap primitives.
- Datasets are explicitly blacklisted from churn via `_REMOVED_SEMANTIC_SEED_TERMS` in `pipeline/nodes/vocab_node.py`.
- Symbol pool generates images for these terms but they have no path into any vocabulary list.
- **Decision needed**: either add these to the fixed vocab / churn bootstraps, or suppress the warning and accept the pool as image-only.

### Duplicate "edge" in fixed vocab
`_default_class_names()` has "edge" at two positions (~index 67 in features block, ~index 91 in color terms block). Second occurrence is a dead slot. Contributes to "edges not showing up as target or found" — downstream term-to-index mapping picks one index, the other is unreachable.
- **Decision needed**: remove the duplicate (shifts indices, breaks existing checkpoints) or leave as-is.

---

## GUI Enhancements

### History needs scaling / normalizing / range controls
History graph values need user-adjustable scaling and range.

### Checkpoints selectable in GUI
~~Add next/prev buttons near the scrub wheel for checkpoint navigation.~~ **DONE** — PREV/NEXT buttons in `_render_btn_panel`, jump scrub offset to nearest checkpoint marker. Click detection in `_poll_events`.

### Graph display rework
Current graph rendering on GUI needs redesign. Per-channel visibility toggles added — click legend entries to show/hide individual loss channels. Hidden channels are dimmed in the legend and excluded from Y-axis scaling.

### Diagnostic weight image panel
Figure out the right panel layout to show a diagnostic image of weights as they change over time. Consider whether shipping to the GUI GPU makes sense (different devices?). Showing last checkpoint weights is a reasonable fallback.

---

## IR / Runtime

### IR+mermaid+runtime drift check
Partially addressed: `plan_protocol.py validate()` now returns warnings for condition consistency (mixed gated/ungated edges from same source) and condition_expr mismatches. Full structural audit of IR-to-runtime alignment still needed.

---

## Resolved (this session)

- ~~saving/restoring blank initial snapshot~~ → `wav_ml_viewer.py`
- ~~targets truncated / right-aligned~~ → `wav_ml_viewer.py`
- ~~eager loading before gates pass~~ → `pipeline/orchestrator.py` (flashcard edge gated)
- ~~overloading (more rows than planned)~~ → same gate fix
- ~~move step/batch loss into controls panel~~ → `pipeline/preview.py` + `wav_ml_viewer.py`
- ~~mask stats accuracy (target_only / detected_only)~~ → `pipeline/preview.py` (binarized masks)
- ~~don't redraw GUI if nothing new~~ → `wav_ml_viewer.py` (dirty-flag comparison)
