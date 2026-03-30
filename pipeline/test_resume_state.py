import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pipeline.nodes.save_restore_node import SaveRestoreNode
from pipeline.orchestrator import _load_resume_state, _restore_rng_from_resume_dir


def test_restore_rng_from_resume_dir_restores_python_numpy_and_torch(tmp_path: Path):
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    bank_blob = [
        {
            "round_id": 7,
            "cycle": 2,
            "python_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_cpu_state": torch.random.get_rng_state(),
        }
    ]
    torch.save(bank_blob, tmp_path / "seed_bank.pt")

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    _restore_rng_from_resume_dir(tmp_path)

    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1).item()) == expected_torch


def test_load_resume_state_prefers_newer_speculative_step_checkpoint(tmp_path: Path):
    torch.save(
        {
            "round_id": 7,
            "cycle": 2,
            "timestamp": 100.0,
            "metrics_history": [{"stage": "old"}],
            "active_network_state": {"weight": torch.tensor([1.0], dtype=torch.float32)},
            "active_network_optimizer_state": {"state": {}, "param_groups": [{"lr": 0.10}]},
        },
        tmp_path / "pipeline_checkpoint.pt",
    )
    torch.save(
        {
            "checkpoint_kind": "speculative_optimizer_step",
            "round_id": 8,
            "cycle": 3,
            "timestamp": 200.0,
            "active_network_state": {"weight": torch.tensor([2.0], dtype=torch.float32)},
            "state_dict": {"weight": torch.tensor([2.0], dtype=torch.float32)},
            "active_network_optimizer_state": {"state": {}, "param_groups": [{"lr": 0.50}]},
        },
        tmp_path / "active_network.pt",
    )

    args = SimpleNamespace(auto_resume=True, resume_from="")
    resume = _load_resume_state(args, tmp_path)
    ckpt = resume["pipeline_ckpt"]

    assert isinstance(ckpt, dict)
    assert float(ckpt["active_network_state"]["weight"].item()) == 2.0
    assert float(ckpt["active_network_optimizer_state"]["param_groups"][0]["lr"]) == 0.50
    assert int(ckpt["round_id"]) == 8
    assert int(ckpt["cycle"]) == 3
    assert ckpt["metrics_history"] == [{"stage": "old"}]
    assert bool(ckpt["_resume_overlay_applied"]) is True
    assert bool(ckpt["_resume_replay_safe"]) is False
    assert bool(resume["pipeline_ckpt_meta"]["overlay_applied"]) is True


def test_save_restore_load_checkpoint_for_restore_merges_newer_speculative_step_checkpoint(tmp_path: Path):
    torch.save(
        {
            "round_id": 4,
            "cycle": 1,
            "timestamp": 10.0,
            "active_network_state": {"weight": torch.tensor([1.0], dtype=torch.float32)},
        },
        tmp_path / "pipeline_checkpoint.pt",
    )
    torch.save(
        {
            "checkpoint_kind": "speculative_optimizer_step",
            "round_id": 5,
            "cycle": 2,
            "timestamp": 20.0,
            "active_network_state": {"weight": torch.tensor([3.0], dtype=torch.float32)},
            "state_dict": {"weight": torch.tensor([3.0], dtype=torch.float32)},
            "active_network_optimizer_state": {"state": {}, "param_groups": [{"lr": 0.75}]},
        },
        tmp_path / "active_network.pt",
    )

    sr = SaveRestoreNode()
    ckpt_path, ckpt = sr._load_checkpoint_for_restore(tmp_path, target_round=5, target_cycle=2)

    assert ckpt_path == (tmp_path / "pipeline_checkpoint.pt")
    assert isinstance(ckpt, dict)
    assert float(ckpt["active_network_state"]["weight"].item()) == 3.0
    assert float(ckpt["active_network_optimizer_state"]["param_groups"][0]["lr"]) == 0.75
    assert int(ckpt["round_id"]) == 5
    assert int(ckpt["cycle"]) == 2
    assert bool(ckpt["_resume_overlay_applied"]) is True
    assert bool(ckpt["_resume_replay_safe"]) is False
