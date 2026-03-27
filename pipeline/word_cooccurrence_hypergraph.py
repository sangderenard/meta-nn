"""
Word co-occurrence hypergraph.

    Nodes      = vocabulary terms  (normalised lowercase canonical keys)
    Hyperedges = dataset rows      (each row is a frozenset of term keys)

The hyperedges are stored as-is; no pairwise projection is computed.
"""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _term_key(term: str) -> str:
    return re.sub(r"\s+", " ", str(term)).strip().lower()


class WordCooccurrenceHypergraph:
    """
    Hypergraph built from a sequence of term rows.

    Parameters
    ----------
    term_rows:
        Sequence of rows; each row is a sequence of term strings.

    Attributes (read-only via properties)
    --------------------------------------
    hyperedges     — one ``frozenset[str]`` of canonical term keys per row,
                     in input order.
    unique_edges   — ``dict[frozenset, int]`` mapping each distinct hyperedge
                     to the number of rows that produce it.
    nodes          — sorted list of all canonical term keys in the graph.
    node_freq      — ``dict[str, int]`` mapping each term key to the number
                     of rows that contain it.
    """

    def __init__(self, term_rows: Sequence[Sequence[str]]) -> None:
        hyperedges: List[FrozenSet[str]] = []
        unique_edges: Dict[FrozenSet[str], int] = {}
        node_freq: Dict[str, int] = {}

        for row in term_rows:
            edge: FrozenSet[str] = frozenset(
                _term_key(str(t)) for t in row if _term_key(str(t))
            )
            hyperedges.append(edge)
            unique_edges[edge] = unique_edges.get(edge, 0) + 1
            for k in edge:
                node_freq[k] = node_freq.get(k, 0) + 1

        self._hyperedges: List[FrozenSet[str]] = hyperedges
        self._unique_edges: Dict[FrozenSet[str], int] = unique_edges
        self._node_freq: Dict[str, int] = node_freq

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hyperedges(self) -> List[FrozenSet[str]]:
        """One frozenset of term keys per input row, in input order."""
        return self._hyperedges

    @property
    def unique_edges(self) -> Dict[FrozenSet[str], int]:
        """Distinct hyperedges mapped to their row-occurrence count."""
        return self._unique_edges

    @property
    def nodes(self) -> List[str]:
        """Sorted list of all term keys present in at least one row."""
        return sorted(self._node_freq)

    @property
    def node_freq(self) -> Dict[str, int]:
        """Term key → number of rows that contain it."""
        return dict(self._node_freq)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def row_edge(self, row_idx: int) -> FrozenSet[str]:
        """Return the hyperedge (frozenset of term keys) for *row_idx*."""
        return self._hyperedges[int(row_idx)]

    def edges_containing(self, term: str) -> List[FrozenSet[str]]:
        """Return all unique hyperedges that contain *term*."""
        k = _term_key(term)
        return [e for e in self._unique_edges if k in e]

    def rows_containing(self, term: str) -> List[int]:
        """Return indices of all rows whose hyperedge contains *term*."""
        k = _term_key(term)
        return [i for i, e in enumerate(self._hyperedges) if k in e]

    def project(self, keep_keys: Set[str]) -> "WordCooccurrenceHypergraph":
        """
        Return a new hypergraph restricted to *keep_keys*.

        Each hyperedge is intersected with *keep_keys*; empty results are
        kept (as empty frozensets) so row indices remain aligned.
        """
        projected_rows: List[List[str]] = [
            [k for k in edge if k in keep_keys]
            for edge in self._hyperedges
        ]
        return WordCooccurrenceHypergraph(projected_rows)

    def __len__(self) -> int:
        return len(self._hyperedges)

    def __repr__(self) -> str:
        return (
            f"WordCooccurrenceHypergraph("
            f"rows={len(self._hyperedges)}, "
            f"unique_edges={len(self._unique_edges)}, "
            f"nodes={len(self._node_freq)})"
        )


# ---------------------------------------------------------------------------
# HypergraphNet
# ---------------------------------------------------------------------------

class HypergraphNet(nn.Module):
    """
    input_proj   : input_width  → hidden_width
    neck_proj    : hidden_width → max_hyperedge_length
    pre_head_norm: LayerNorm(max_hyperedge_length)
    dynamic_head : max_hyperedge_length → max_hyperedge_length   (weight tensor via F.linear)
    post_head_norm: LayerNorm(max_hyperedge_length)
    output_hidden: max_hyperedge_length → hidden_width
    logit_proj   : hidden_width → output_width
    """

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        max_hyperedge_length: int,
    ) -> None:
        super().__init__()
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.output_width = int(output_width)
        self.max_hyperedge_length = int(max_hyperedge_length)

        mhl = self.max_hyperedge_length

        self.input_proj   = nn.Linear(self.input_width, self.hidden_width)
        self.neck_proj    = nn.Linear(self.hidden_width, mhl)
        self.pre_head_norm  = nn.LayerNorm(mhl)
        self.post_head_norm = nn.LayerNorm(mhl)
        self.output_hidden = nn.Linear(mhl, self.hidden_width)
        self.logit_proj   = nn.Linear(self.hidden_width, self.output_width)

        self.act = nn.GELU()

        # Neuron library: one learnable vector per registered term key.
        # Keys are sanitised for ParameterDict (no dots).
        # Expands via register_neuron(); participates fully in .to(device),
        # state_dict(), and autograd.
        self.neuron_library = nn.ParameterDict()

        # Fallback weight used when no hypergraph has been observed yet.
        # Shape: (mhl, mhl). Eye init keeps the head as near-identity at t=0.
        self._stub_head_weight = nn.Parameter(torch.eye(mhl))

        # Pre-scanned edge data: list of (frozenset of sanitised library keys, freq).
        # Populated by observe_hypergraph(); empty until then.
        self._edge_data: List[Tuple[FrozenSet[str], int]] = []

    # ------------------------------------------------------------------
    # Neuron library
    # ------------------------------------------------------------------

    @staticmethod
    def _library_key(term_key: str) -> str:
        """Sanitise a term key for use as a ParameterDict key."""
        return re.sub(r"[^a-zA-Z0-9_]", "_", str(term_key))

    def register_neuron(self, term_key: str) -> None:
        """Add a learnable neuron vector for *term_key* if not already present."""
        k = self._library_key(term_key)
        if k not in self.neuron_library:
            p = nn.Parameter(torch.empty(self.max_hyperedge_length))
            nn.init.normal_(p, std=1.0 / math.sqrt(self.max_hyperedge_length))
            self.neuron_library[k] = p

    def observe_hypergraph(self, hg: "WordCooccurrenceHypergraph") -> None:
        """
        Pre-scan *hg*: register a neuron for every term in the graph and
        cache the edge frequency data for use during weight composition.
        After this call the stub weight is no longer used.
        """
        self._edge_data = [
            (frozenset(self._library_key(k) for k in edge), freq)
            for edge, freq in hg.unique_edges.items()
        ]
        for term_key in hg.nodes:
            self.register_neuron(term_key)

    # ------------------------------------------------------------------
    # Dynamic head composition
    # ------------------------------------------------------------------

    def _compose_dynamic_head_weight(
        self, active_keys: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Assemble the (mhl, mhl) dynamic head weight via hypergraph message passing.

        For each unique hyperedge (edge_keys, freq):
          - Gather neuron vectors V for terms present in the library  → (n, mhl)
          - Each term v_i receives context = mean of all other term vectors in the edge
          - Contribution to W = freq * V.T @ C   where C[i] = (V_sum - V[i]) / (n-1)
          - Singletons contribute freq * outer(v, v) — self-directed signal

        W is accumulated frequency-weighted and normalised by total frequency.
        Falls back to _stub_head_weight if no edge data has been observed yet.

        active_keys, when provided, restricts composition to edges that contain
        at least one active term — focusing the head on the current batch's demand.
        """
        if not self._edge_data:
            return self._stub_head_weight

        ref = self._stub_head_weight
        device, dtype = ref.device, ref.dtype
        mhl = self.max_hyperedge_length

        active: Optional[Set[str]] = (
            {self._library_key(k) for k in active_keys} if active_keys else None
        )

        W_accum = torch.zeros(mhl, mhl, device=device, dtype=dtype)
        total_weight = 0.0

        for edge_keys, freq in self._edge_data:
            if active is not None and not (edge_keys & active):
                continue

            vecs = [
                self.neuron_library[k]
                for k in edge_keys
                if k in self.neuron_library
            ]
            if not vecs:
                continue

            n = len(vecs)
            V = torch.stack(vecs)           # (n, mhl)

            if n == 1:
                # Singleton: no neighbour context, self outer product
                W_accum = W_accum + freq * torch.outer(V[0], V[0])
            else:
                V_sum = V.sum(0)            # (mhl,)
                # C[i] = mean of all other term vectors in this edge
                C = (V_sum.unsqueeze(0) - V) / (n - 1)   # (n, mhl)
                # sum_i( v_i ⊗ context_i ) — vectorised, no inner Python loop
                W_accum = W_accum + freq * (V.T @ C)      # (mhl, mhl)

            total_weight += freq

        if total_weight == 0.0:
            return self._stub_head_weight

        return W_accum / total_weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def logits(
        self, x: torch.Tensor, active_keys: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Batch-safe forward pass.  *x* may be (B, input_width) or
        (B, T, input_width); all ops broadcast over leading dims.
        """
        x = self.act(self.input_proj(x))           # → (..., hidden_width)
        x = self.act(self.neck_proj(x))            # → (..., mhl)
        x = self.pre_head_norm(x)                  # stabilise head input
        w = self._compose_dynamic_head_weight(active_keys)  # (mhl, mhl)
        x = F.linear(x, w)                         # → (..., mhl)  batch-safe
        x = self.act(self.post_head_norm(x))       # stabilise + nonlinearity
        x = self.act(self.output_hidden(x))        # → (..., hidden_width)
        return self.logit_proj(x)                  # → (..., output_width)

    def forward(
        self, x: torch.Tensor, active_keys: Optional[List[str]] = None
    ) -> torch.Tensor:
        return self.logits(x, active_keys)
