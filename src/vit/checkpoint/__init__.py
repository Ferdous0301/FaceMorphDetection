"""Checkpoint persistence for the ViT stage.

Public API:

    from vit.checkpoint import CheckpointManager, CheckpointState
"""

from __future__ import annotations

from vit.checkpoint.checkpoint_manager import CheckpointManager, CheckpointState

__all__ = ["CheckpointManager", "CheckpointState"]