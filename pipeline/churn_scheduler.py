from __future__ import annotations

from typing import Any, Dict, List, Sequence

from pipeline.context import PipelineContext
from pipeline.word_cooccurrence_hypergraph import WordCooccurrenceHypergraph


def build_churn_row_schedule(
    *,
    ctx: PipelineContext,
    plan: Dict[str, Any],
    term_rows: Sequence[Sequence[str]],
    source: str,
    stage_label: str,
    max_terms_per_slot: int,
) -> Dict[str, Any]:
    _ = (ctx, plan, source, stage_label, max_terms_per_slot)

    hg = WordCooccurrenceHypergraph(term_rows)

    # TODO: implement slot partitioning and row ordering using hg
    _ = hg

    ordered_row_indices: List[int] = list(range(len(term_rows)))

    return {
        "slots": [],
        "slot_groups": [],
        "ordered_row_indices": ordered_row_indices,
        "scheduler_stub": True,
        "scheduler_status": "pending",
        "scheduler_reason": (
            "Churn scheduling is not yet implemented. Rows are returned in "
            "natural order; slot partitioning and swap-point emission are TODO."
        ),
    }
