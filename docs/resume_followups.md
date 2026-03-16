# Resume Follow-Ups

These items are intentionally deferred from the current resume fix.

## Dataset Schedule State

The runtime still rebuilds loader/sampler schedule state from scratch on process restart.
That is acceptable for the current fix because the immediate issue was restore ordering, not dataset continuity.

Eventual work:
- Persist sequential sampler cursor and any stage-local schedule counters.
- Restore those values through `SaveRestoreNode` so the next batch choice can continue exactly where the previous process stopped.

## Gate State

The current checkpoint path still treats gate state mostly as boolean pass/fail state.
That is acceptable for the current fix because restart correctness was dominated by restore timing.

Eventual work:
- Persist `consecutive_passes`, `required_consecutive`, `last_metric`, and gate history.
- Restore that richer gate state through `SaveRestoreNode` so restart preserves gate-progress semantics exactly.
