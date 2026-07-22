"""
Checkpoint management for the ViT morph-attack-detection module.

Responsibilities:
    - Save `last.pt` on every call to `save()`.
    - Save `best.pt` whenever `is_best=True` is passed in.
    - Save rolling `epoch_{N:04d}.pt` snapshots, keeping only the most
      recent `keep_last_k` of them (older ones are deleted).
    - Load any of the above back into a `CheckpointState`, including
      RNG state for exact resumability.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vit.configs.schema import ExperimentConfig

_BEST_FILENAME = "best.pt"
_LATEST_FILENAME = "last.pt"
_EPOCH_FILENAME_TEMPLATE = "epoch_{epoch:04d}.pt"


@dataclass
class CheckpointState:
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    scaler_state_dict: Optional[Dict[str, Any]]
    best_metric_value: float
    config: ExperimentConfig
    rng_state: Dict[str, Any]


class CheckpointManager:
    """
    Manages checkpoint persistence for a single training run.

    Args:
        checkpoint_dir: directory in which checkpoints are written.
            Created if it does not already exist.
        monitor_metric: name of the metric used to determine "best"
            (informational; the actual comparison is done by the caller,
            e.g. EarlyStopping — this class just persists whatever
            `is_best` flag it's given).
        mode: "min" or "max". Stored for reference / validation only.
        keep_last_k: number of rolling per-epoch checkpoints to retain.
            Must be >= 1. Older epoch checkpoints beyond this count are
            deleted automatically on each save.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor_metric: str,
        mode: str,
        keep_last_k: int = 3,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if keep_last_k < 1:
            raise ValueError(f"keep_last_k must be >= 1, got {keep_last_k}")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.keep_last_k = keep_last_k

    def save(self, state: CheckpointState, is_best: bool) -> Path:
        """
        Persist `state` to disk.

        Always writes/overwrites `last.pt` and an `epoch_{N}.pt` snapshot,
        pruning old epoch snapshots beyond `keep_last_k`. Additionally
        writes `best.pt` when `is_best` is True.

        Args:
            state: the CheckpointState to persist.
            is_best: whether this state represents the best model so far.

        Returns:
            Path to the `last.pt` file that was written.
        """
        epoch_path = self.checkpoint_dir / _EPOCH_FILENAME_TEMPLATE.format(
            epoch=state.epoch
        )
        latest_path = self.checkpoint_dir / _LATEST_FILENAME

        torch.save(state, epoch_path)
        shutil.copyfile(epoch_path, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / _BEST_FILENAME
            shutil.copyfile(epoch_path, best_path)

        self._rotate_epoch_checkpoints()
        return latest_path

    def _rotate_epoch_checkpoints(self) -> None:
        epoch_checkpoints = sorted(
            self.checkpoint_dir.glob("epoch_*.pt"),
            key=lambda p: p.name,
        )
        excess = len(epoch_checkpoints) - self.keep_last_k
        if excess > 0:
            for stale_path in epoch_checkpoints[:excess]:
                stale_path.unlink(missing_ok=True)

    def load(self, path: Path, map_location: torch.device) -> CheckpointState:
        """
        Load a CheckpointState from an explicit path.

        Args:
            path: path to a checkpoint file previously written by `save`.
            map_location: device to map tensors onto (e.g. for loading a
                GPU-trained checkpoint on a CPU-only machine).

        Returns:
            The deserialized CheckpointState.

        Raises:
            FileNotFoundError: if `path` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(state, CheckpointState):
            raise TypeError(
                f"Checkpoint at {path} did not contain a CheckpointState "
                f"(got {type(state)})"
            )
        return state

    def load_best(self, map_location: torch.device = torch.device("cpu")) -> CheckpointState:
        """Load the best checkpoint (`best.pt`)."""
        return self.load(self.checkpoint_dir / _BEST_FILENAME, map_location)

    def load_latest(self, map_location: torch.device = torch.device("cpu")) -> CheckpointState:
        """Load the most recent checkpoint (`last.pt`)."""
        return self.load(self.checkpoint_dir / _LATEST_FILENAME, map_location)

    def list_epoch_checkpoints(self) -> List[Path]:
        """Return sorted paths of all retained rolling epoch checkpoints."""
        return sorted(self.checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.name)