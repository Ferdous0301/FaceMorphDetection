"""Reproducibility utilities.

This module centralizes every source of randomness that must be controlled
for deterministic ViT training runs: Python's ``random`` module, ``numpy``,
and ``torch`` (CPU and CUDA). It also provides a deterministic
``worker_init_fn`` for ``torch.utils.data.DataLoader`` so that multi-process
data loading does not silently reintroduce non-determinism, and a seeded
``torch.Generator`` factory for deterministic shuffling.

Design notes:
    * :func:`set_global_seed` is intentionally the *only* place in the whole
      ``vit`` package allowed to call the underlying ``random.seed`` /
      ``np.random.seed`` / ``torch.manual_seed`` functions directly. Every
      other module should obtain randomness through a ``torch.Generator``
      passed in explicitly (see :func:`make_generator`), which keeps
      reproducibility auditable from a single call site.
    * Enabling ``torch.use_deterministic_algorithms(True)`` can make some CUDA
      operations slower or raise ``RuntimeError`` for operations without a
      deterministic implementation. This is accepted as a deliberate
      correctness-over-speed trade-off for a thesis project where exact
      reproducibility of reported results matters more than raw throughput.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_global_seed", "seed_worker", "make_generator"]


def set_global_seed(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed every relevant random number generator for reproducible runs.

    Seeds, in order: the ``PYTHONHASHSEED`` environment variable (affects
    hash-based ordering, e.g. of sets/dicts, in child processes spawned
    afterwards), Python's ``random`` module, ``numpy``, and ``torch`` (both
    CPU and, if available, all CUDA devices). Optionally also requests
    deterministic algorithm implementations from PyTorch.

    Args:
        seed: The seed value to apply everywhere. Must be a non-negative
            integer.
        deterministic_algorithms: If True, calls
            ``torch.use_deterministic_algorithms(True)`` and sets
            ``torch.backends.cudnn.deterministic = True`` /
            ``torch.backends.cudnn.benchmark = False``. Disable only for
            quick experimentation where bit-exact reproducibility is not
            required, as it may noticeably reduce throughput on GPU.

    Raises:
        ValueError: If ``seed`` is negative.

    Example:
        >>> set_global_seed(42)
        >>> torch.rand(1)  # doctest: +SKIP
        tensor([0.8823])
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_algorithms:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # `warn_only=True` avoids hard failures on the (rare) op without a
        # deterministic CUDA kernel, while still enforcing determinism
        # everywhere it is available.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Deterministically seed a ``DataLoader`` worker process.

    Intended to be passed as ``worker_init_fn=seed_worker`` to
    ``torch.utils.data.DataLoader``. Combined with a seeded
    ``torch.Generator`` passed via the ``generator=`` argument (see
    :func:`make_generator`), this guarantees that batch order and any
    worker-local randomness (e.g. in ``Dataset.__getitem__``) are
    reproducible across runs and across ``num_workers`` settings.

    Args:
        worker_id: The DataLoader-assigned worker id (unused directly, but
            required by the ``worker_init_fn`` signature; the actual seed is
            derived from PyTorch's per-worker initial seed, which already
            incorporates the base seed set via :func:`set_global_seed` and
            the worker id).

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(
        ...     dataset, worker_init_fn=seed_worker, generator=make_generator(42)
        ... )  # doctest: +SKIP
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: Optional[int]) -> torch.Generator:
    """Create a CPU :class:`torch.Generator` seeded deterministically.

    Intended for use as the ``generator=`` argument of a ``DataLoader`` so
    that shuffling order is reproducible independently of global RNG state.

    Args:
        seed: Seed for the generator. If ``None``, the generator is left
            unseeded (non-deterministic), which is only appropriate for
            ad-hoc/manual usage outside of the reproducible training
            pipeline.

    Returns:
        A ``torch.Generator`` instance, seeded if ``seed`` is not ``None``.
    """
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator