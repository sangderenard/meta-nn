from __future__ import annotations

import unittest

import torch

from pipeline.nodes.data_nodes import _dataset_terms_rows


class _FiveTupleDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.rows = [
            ["letter w"],
            ["letter l"],
            ["digit 6"],
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return (
            torch.zeros(3, 8, 8),
            torch.zeros(1, 8, 8),
            torch.zeros(1, 8, 8),
            torch.zeros(1, dtype=torch.long),
            list(self.rows[int(index)]),
        )


class _ThreeTupleDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.rows = [
            ["signal"],
            ["noise"],
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return (
            torch.zeros(3, 8, 8),
            torch.zeros(1, 8, 8),
            list(self.rows[int(index)]),
        )


class DatasetTermsRowsTests(unittest.TestCase):
    def test_subset_of_five_tuple_dataset_preserves_terms(self) -> None:
        dataset = torch.utils.data.Subset(_FiveTupleDataset(), [2, 0, 1])

        rows = _dataset_terms_rows(dataset)

        self.assertEqual(rows, [["digit 6"], ["letter w"], ["letter l"]])

    def test_subset_of_three_tuple_dataset_preserves_terms(self) -> None:
        dataset = torch.utils.data.Subset(_ThreeTupleDataset(), [1, 0])

        rows = _dataset_terms_rows(dataset)

        self.assertEqual(rows, [["noise"], ["signal"]])


if __name__ == "__main__":
    unittest.main()
