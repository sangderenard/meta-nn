import random
from pathlib import Path

import numpy as np
import torch

from pipeline.orchestrator import _restore_rng_from_resume_dir


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
