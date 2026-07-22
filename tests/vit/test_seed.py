"""Unit tests for vit.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vit.utils.seed import make_generator, seed_worker, set_global_seed


class _IndexDataset(Dataset):
    """Tiny dataset that returns its own index, for order-determinism checks."""

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> int:
        return idx


class TestSetGlobalSeed:
    def test_rejects_negative_seed(self) -> None:
        with pytest.raises(ValueError):
            set_global_seed(-1)

    def test_torch_reproducibility(self) -> None:
        set_global_seed(123)
        a = torch.rand(5)
        set_global_seed(123)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_numpy_reproducibility(self) -> None:
        set_global_seed(123)
        a = np.random.rand(5)
        set_global_seed(123)
        b = np.random.rand(5)
        assert np.array_equal(a, b)

    def test_python_random_reproducibility(self) -> None:
        set_global_seed(123)
        a = [random.random() for _ in range(5)]
        set_global_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_different_seeds_differ(self) -> None:
        set_global_seed(1)
        a = torch.rand(5)
        set_global_seed(2)
        b = torch.rand(5)
        assert not torch.equal(a, b)


class TestMakeGenerator:
    def test_seeded_generator_is_deterministic(self) -> None:
        gen_a = make_generator(42)
        gen_b = make_generator(42)
        a = torch.randperm(10, generator=gen_a)
        b = torch.randperm(10, generator=gen_b)
        assert torch.equal(a, b)

    def test_unseeded_generator_returns_generator(self) -> None:
        gen = make_generator(None)
        assert isinstance(gen, torch.Generator)


class TestDataLoaderDeterminism:
    def test_shuffled_batch_order_reproducible_across_runs(self) -> None:
        dataset = _IndexDataset(size=20)

        def _collect_order() -> list:
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                worker_init_fn=seed_worker,
                generator=make_generator(7),
            )
            return [batch.tolist() for batch in loader]

        order_a = _collect_order()
        order_b = _collect_order()
        assert order_a == order_b