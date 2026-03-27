from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from pipeline.context import PipelineContext
from pipeline.nodes.classifier_node import BerkeleyRefreshTrainNode, ClassifierConfig, GestationTrainNode
from pipeline.nodes.gate_nodes import BerkeleyGateConfig, BerkeleyGateNode, GestationEvalNode


class _DummyLoader:
    def __init__(self) -> None:
        self.dataset = object()

    def __len__(self) -> int:
        return 1


class _NoopDataNode:
    def provide_berkeley_data(self, ctx: PipelineContext) -> None:
        return None


class _GestationDataNode:
    def provide_gestation(self, ctx: PipelineContext) -> None:
        ctx.gestation_loader = _DummyLoader()

    def provide_gestation_eval(self, ctx: PipelineContext) -> None:
        ctx.gestation_eval_loader = _DummyLoader()


class Stage2RuntimeGuardTests(unittest.TestCase):
    def test_stage2_raises_when_loader_provisioning_returns_none(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.data = _NoopDataNode()

        node = BerkeleyRefreshTrainNode(ClassifierConfig())

        with self.assertRaisesRegex(RuntimeError, "requires berkeley_refresh_loader"):
            node.execute(ctx)

    def test_stage2_registers_runtime_terms_before_training(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.classifier_optimizer = torch.optim.SGD(ctx.classifier.parameters(), lr=0.01)
        ctx.berkeley_refresh_loader = _DummyLoader()

        node = BerkeleyRefreshTrainNode(ClassifierConfig())
        call_order: list[str] = []

        def _record_register(*args, **kwargs):
            call_order.append("register")
            return {
                "required_extra_term_count": 1,
                "current_vocab_fit": False,
                "slot_count": 1,
            }

        def _record_run(*args, **kwargs):
            call_order.append("run")
            return {"ran": True, "loss": 0.25}

        with patch("pipeline.nodes.data_nodes._dataset_terms_rows", return_value=[["boat"]]), \
             patch("pipeline.nodes.data_nodes._register_churn_terms", side_effect=_record_register), \
             patch("pipeline.nodes.classifier_node._run_classifier_refresh_epochs", side_effect=_record_run):
            node.execute(ctx)

        self.assertEqual(call_order, ["register", "run"])

    def test_gate2_registers_payload_terms_before_eval(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.payload_validation_loader = _DummyLoader()

        node = BerkeleyGateNode(BerkeleyGateConfig(required_consecutive=1), classifier_cfg=ClassifierConfig())
        call_order: list[str] = []

        def _record_register(*args, **kwargs):
            call_order.append("register")
            return {
                "required_extra_term_count": 1,
                "current_vocab_fit": False,
                "slot_count": 1,
            }

        def _record_ensure(*args, **kwargs):
            call_order.append("ensure")
            return {"needed": False}

        def _record_eval(*args, **kwargs):
            call_order.append("eval")
            return {
                "mean_confidence": 0.95,
                "macro_f1": 0.90,
                "loss": 0.05,
            }

        with patch("pipeline.nodes.data_nodes._dataset_terms_rows", return_value=[["boat"]]), \
             patch("pipeline.nodes.data_nodes._register_churn_terms", side_effect=_record_register), \
             patch("pipeline.nodes.classifier_node.ensure_vocab_lora_active", side_effect=_record_ensure), \
             patch("pipeline.nodes.gate_nodes._gate_eval_model", return_value=(ctx.classifier, torch.device("cpu"), "classifier")), \
             patch("pipeline.nodes.gate_nodes._evaluate_berkeley_classifier_gate", side_effect=_record_eval):
            node.execute(ctx)

        self.assertEqual(call_order, ["register", "ensure", "eval"])

    def test_stage1_should_run_without_loader_when_classifier_ready(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.gate_pregestation.passed = True

        node = GestationTrainNode(ClassifierConfig())

        self.assertTrue(node.should_run(ctx))

    def test_stage1_acquires_loader_via_data_node(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.classifier_optimizer = torch.optim.SGD(ctx.classifier.parameters(), lr=0.01)
        ctx.gate_pregestation.passed = True
        ctx.data = _GestationDataNode()

        node = GestationTrainNode(ClassifierConfig())
        call_order: list[str] = []

        def _record_register(*args, **kwargs):
            call_order.append("register")
            return {
                "required_extra_term_count": 1,
                "current_vocab_fit": False,
                "slot_count": 1,
            }

        def _record_ensure(*args, **kwargs):
            call_order.append("ensure")
            return {"needed": False}

        def _record_run(*args, **kwargs):
            call_order.append("run")
            return {"ran": True, "loss": 0.20}

        with patch("pipeline.nodes.data_nodes._dataset_terms_rows", return_value=[["boat"]]), \
             patch("pipeline.nodes.data_nodes._register_churn_terms", side_effect=_record_register), \
             patch("pipeline.nodes.classifier_node.ensure_vocab_lora_active", side_effect=_record_ensure), \
             patch("pipeline.nodes.classifier_node._run_classifier_refresh_epochs", side_effect=_record_run), \
             patch("pipeline.nodes.classifier_node.deactivate_vocab_lora_slot", return_value={"deactivated": False}):
            node.execute(ctx)

        self.assertEqual(call_order, ["register", "ensure", "run"])
        self.assertIsNotNone(ctx.gestation_loader)

    def test_gate1_acquires_loader_via_data_node(self) -> None:
        ctx = PipelineContext(device=torch.device("cpu"))
        ctx.classifier = torch.nn.Linear(1, 1)
        ctx.gate_pregestation.passed = True
        ctx.data = _GestationDataNode()

        node = GestationEvalNode(ClassifierConfig())
        call_order: list[str] = []

        def _record_register(*args, **kwargs):
            call_order.append("register")
            return {
                "required_extra_term_count": 1,
                "current_vocab_fit": False,
                "slot_count": 1,
            }

        def _record_ensure(*args, **kwargs):
            call_order.append("ensure")
            return {"needed": False}

        def _record_eval(*args, **kwargs):
            call_order.append("eval")
            return {"loss": 0.10}

        with patch("pipeline.nodes.data_nodes._dataset_terms_rows", return_value=[["boat"]]), \
             patch("pipeline.nodes.data_nodes._register_churn_terms", side_effect=_record_register), \
             patch("pipeline.nodes.classifier_node.ensure_vocab_lora_active", side_effect=_record_ensure), \
             patch("pipeline.nodes.gate_nodes._evaluate_loss_gate", side_effect=_record_eval), \
             patch("pipeline.nodes.vocab_node.deactivate_vocab_lora_slot", return_value={"deactivated": False}):
            node.execute(ctx)

        self.assertEqual(call_order, ["register", "ensure", "eval"])
        self.assertIsNotNone(ctx.gestation_eval_loader)


if __name__ == "__main__":
    unittest.main()
