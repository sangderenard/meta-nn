from __future__ import annotations

import contextlib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.context import PipelineContext
from pipeline.nodes.data_nodes import (
    BerkeleyDataConfig,
    BerkeleyPayloadConfig,
    DataNode,
    GestationDataConfig,
    PregestationDataConfig,
    ensure_runtime_loader_contract,
)


class _DummyLoader:
    def __init__(self, dataset: object | None = None) -> None:
        self.dataset = dataset if dataset is not None else object()

    def __len__(self) -> int:
        return 1


class DataNodePossessionGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.node = DataNode(
            preg_cfg=PregestationDataConfig(),
            gest_cfg=GestationDataConfig(),
            payload_cfg=BerkeleyPayloadConfig(),
            bdata_cfg=BerkeleyDataConfig(),
        )
        self.ctx = PipelineContext(device=torch.device("cpu"))
        self.ctx.total_rounds_completed = 7
        self.ctx.output_dir = "."

    def _mark_ready(self, possession_name: str) -> None:
        poss = self.node.possessions[possession_name]
        poss.mark_built()
        poss.force_next_rebuild = False
        for attr in self.node._required_ctx_attrs_for_possession(possession_name):
            setattr(self.ctx, attr, object())

    def _mark_missing(self, possession_name: str) -> None:
        poss = self.node.possessions[possession_name]
        poss.mark_expired()
        poss.force_next_rebuild = False
        for attr in poss.ctx_attrs:
            setattr(self.ctx, attr, None)

    def test_provider_should_reuse_ready_possessions_for_all_loading_scenarios(self) -> None:
        for name in self.node.possessions:
            with self.subTest(possession=name):
                self._mark_ready(name)
                self.assertFalse(self.node._provider_should_rebuild(self.ctx, name))

    def test_provider_should_rebuild_missing_possessions_even_if_last_build_round_matches(self) -> None:
        round_fields = {
            "pregestation": "_preg_last_build_round",
            "gestation": "_gest_last_build_round",
            "berkeley": "_bdata_last_build_round",
            "payload": "_payload_last_build_round",
            "payload_validation": "_payload_validation_last_build_round",
        }
        for name, round_field in round_fields.items():
            with self.subTest(possession=name):
                setattr(self.node, round_field, int(self.ctx.total_rounds_completed))
                self._mark_missing(name)
                self.assertTrue(self.node._provider_should_rebuild(self.ctx, name))

    def test_force_rebuild_overrides_ready_possessions_for_all_loading_scenarios(self) -> None:
        for name, poss in self.node.possessions.items():
            with self.subTest(possession=name):
                self._mark_ready(name)
                poss.force_next_rebuild = True
                self.assertTrue(self.node._provider_should_rebuild(self.ctx, name))

    def test_runtime_loader_contract_rebuilds_pregestation_after_same_round_expiry(self) -> None:
        self.ctx.class_names = ["signal"]
        self.ctx.semantic_term_to_idx = {"signal": 0}
        self.ctx.data = self.node
        self.node._preg_last_build_round = int(self.ctx.total_rounds_completed)
        self._mark_missing("pregestation")

        dataset = object()
        loader = _DummyLoader(dataset)
        eval_loader = _DummyLoader(dataset)
        raw_cache = {
            "all_images": [np.zeros((3, 4, 4), dtype=np.float16)],
            "all_masks": [np.zeros((4, 4), dtype=np.float16)],
            "all_label_stacks": [np.ones((1, 4, 4), dtype=np.float16)],
            "all_label_indices": [np.asarray([0], dtype=np.int64)],
            "all_term_rows": [["signal"]],
            "local_vocab": ["signal"],
        }

        with patch("pipeline.nodes.data_nodes._load_raw_stage_cache", return_value=raw_cache), \
             patch("pipeline.nodes.data_nodes._log", return_value=None), \
             patch("pipeline.nodes.data_nodes.semantic_processing_device", return_value=contextlib.nullcontext(None)), \
             patch("pipeline.nodes.data_nodes._build_semantic_stage_cache_dataset", return_value=(dataset, [0], {"entries_per_clean": 1})), \
             patch("pipeline.nodes.data_nodes._orphan_free_split", return_value=([0], [0])), \
             patch("pipeline.nodes.data_nodes.build_stage_loaders", return_value=(loader, eval_loader)), \
             patch("pipeline.nodes.data_nodes._dataset_terms_rows", return_value=[["signal"]]), \
             patch("pipeline.nodes.data_nodes._register_churn_terms", return_value={"slot_count": 0}):
            loader_info = ensure_runtime_loader_contract(self.ctx, consumer_id="stage_0_pregestation")

        self.assertIs(loader_info["loader"], loader)
        self.assertIs(self.ctx.pregestation_loader, loader)
        self.assertIs(self.ctx.pregestation_eval_loader, eval_loader)
        self.assertTrue(self.node.possessions["pregestation"].built)


if __name__ == "__main__":
    unittest.main()
