"""Typed configuration objects for the ViT training/evaluation/inference stage.

All configuration objects in this module are implemented as **frozen**
dataclasses. Immutability is a deliberate design choice:

* It prevents accidental mutation of hyperparameters mid-run (e.g. inside a
  training loop), which would silently break reproducibility.
* It makes configuration objects safe to hash, log, and store verbatim inside
  checkpoints.
* It forces any legitimate "modification" (e.g. overriding the learning rate
  from the CLI) to go through explicit, auditable reconstruction via
  :func:`dataclasses.replace`.

The only public entry points most callers need are :class:`ExperimentConfig`,
:func:`load_experiment_config`, and :func:`save_experiment_config`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TypeVar

import yaml

__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "ConfigValidationError",
    "load_experiment_config",
    "save_experiment_config",
]

_T = TypeVar("_T")

#: Backbones supported by ``vit.models.backbone_factory``. Declared here (and
#: not imported from the models package) to avoid a config -> model import
#: cycle; the models package re-validates against its own registry too.
SUPPORTED_BACKBONES: Tuple[str, ...] = (
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
)

SUPPORTED_OPTIMIZERS: Tuple[str, ...] = ("adamw", "adam", "sgd")
SUPPORTED_SCHEDULERS: Tuple[str, ...] = ("cosine", "step", "plateau", "none")
SUPPORTED_LOSSES: Tuple[str, ...] = ("cross_entropy", "focal_loss")


class ConfigValidationError(ValueError):
    """Raised when a configuration object fails semantic validation.

    Distinguished from a plain :class:`ValueError` so callers (e.g. the CLI)
    can catch configuration problems specifically and report them with a
    dedicated, user-friendly exit path instead of a generic traceback.
    """


def _require(condition: bool, message: str) -> None:
    """Raise :class:`ConfigValidationError` with ``message`` if ``condition`` is False."""
    if not condition:
        raise ConfigValidationError(message)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset loading and data pipeline behaviour.

    Attributes:
        train_csv: Path to the training manifest CSV produced by the
            Dataset Split stage.
        val_csv: Path to the validation manifest CSV.
        test_csv: Path to the test manifest CSV.
        image_root: Root directory that ``path_column`` entries in the CSVs
            are relative to. If CSV paths are already absolute, this may be
            set to ``Path(".")``.
        image_size: Square side length (pixels) images are resized to before
            being fed to the ViT backbone.
        batch_size: Mini-batch size used for all three splits.
        num_workers: Number of ``DataLoader`` worker processes.
        pin_memory: Whether to pin host memory for faster host-to-device
            transfer. Ignored (safely) when running on CPU.
        label_column: Name of the CSV column holding the integer class label.
        path_column: Name of the CSV column holding the (relative) image path.
        drop_last_train: Whether to drop the last incomplete training batch.
            Improves batch-norm/attention statistics stability; irrelevant
            for correctness but kept explicit for reproducibility.
    """

    train_csv: Path
    val_csv: Path
    test_csv: Path
    image_root: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    label_column: str = "label"
    path_column: str = "image_path"
    drop_last_train: bool = False

    def __post_init__(self) -> None:
        _require(self.image_size > 0, f"image_size must be positive, got {self.image_size}")
        _require(self.batch_size > 0, f"batch_size must be positive, got {self.batch_size}")
        _require(self.num_workers >= 0, f"num_workers must be >= 0, got {self.num_workers}")
        _require(bool(self.label_column), "label_column must be a non-empty string")
        _require(bool(self.path_column), "path_column must be a non-empty string")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the ViT backbone and classification head.

    Attributes:
        backbone: Name of the ViT variant, must be a key registered in
            ``vit.models.backbone_factory.SUPPORTED_BACKBONES``.
        pretrained: Whether to initialize the backbone from ImageNet-pretrained
            weights (via ``torchvision``).
        num_classes: Number of output classes. ``2`` for binary
            bona-fide/morph classification.
        freeze_backbone: If True, backbone parameters are frozen at
            construction time and only the classification head is trained.
        dropout: Dropout probability applied inside the classification head.
    """

    backbone: str = "vit_b_16"
    pretrained: bool = True
    num_classes: int = 2
    freeze_backbone: bool = False
    dropout: float = 0.1

    def __post_init__(self) -> None:
        _require(
            self.backbone in SUPPORTED_BACKBONES,
            f"backbone '{self.backbone}' is not supported. "
            f"Supported values: {SUPPORTED_BACKBONES}",
        )
        _require(self.num_classes >= 2, f"num_classes must be >= 2, got {self.num_classes}")
        _require(0.0 <= self.dropout < 1.0, f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
        name: One of ``SUPPORTED_OPTIMIZERS``.
        lr: Base learning rate.
        weight_decay: Weight decay (L2 regularization) coefficient.
        betas: Adam/AdamW beta coefficients. Ignored for SGD.
        momentum: SGD momentum. Ignored for Adam/AdamW.
    """

    name: str = "adamw"
    lr: float = 3e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9

    def __post_init__(self) -> None:
        _require(
            self.name in SUPPORTED_OPTIMIZERS,
            f"optimizer '{self.name}' is not supported. Supported values: {SUPPORTED_OPTIMIZERS}",
        )
        _require(self.lr > 0.0, f"lr must be positive, got {self.lr}")
        _require(self.weight_decay >= 0.0, f"weight_decay must be >= 0, got {self.weight_decay}")
        _require(len(self.betas) == 2, "betas must be a 2-tuple")


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for the learning-rate scheduler.

    Attributes:
        name: One of ``SUPPORTED_SCHEDULERS``. ``"none"`` disables scheduling.
        warmup_epochs: Number of linear warmup epochs (used by ``"cosine"``).
        total_epochs: Total number of epochs the schedule is defined over.
            Must match ``TrainingConfig.epochs`` (validated at the
            ``ExperimentConfig`` level).
        min_lr: Minimum learning rate reached at the end of the schedule.
        step_size: Epoch interval between decays (used by ``"step"``).
        gamma: Multiplicative decay factor (used by ``"step"``).
        plateau_patience: Epochs with no improvement before decaying
            (used by ``"plateau"``).
        plateau_factor: Multiplicative decay factor (used by ``"plateau"``).
    """

    name: str = "cosine"
    warmup_epochs: int = 2
    total_epochs: int = 30
    min_lr: float = 1e-7
    step_size: int = 10
    gamma: float = 0.1
    plateau_patience: int = 3
    plateau_factor: float = 0.5

    def __post_init__(self) -> None:
        _require(
            self.name in SUPPORTED_SCHEDULERS,
            f"scheduler '{self.name}' is not supported. Supported values: {SUPPORTED_SCHEDULERS}",
        )
        _require(self.warmup_epochs >= 0, f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        _require(self.total_epochs > 0, f"total_epochs must be positive, got {self.total_epochs}")
        _require(
            self.warmup_epochs < self.total_epochs or self.total_epochs == 0,
            "warmup_epochs must be smaller than total_epochs",
        )
        _require(self.min_lr >= 0.0, f"min_lr must be >= 0, got {self.min_lr}")


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training loop, reproducibility, and I/O locations.

    Attributes:
        epochs: Total number of training epochs.
        loss_name: One of ``SUPPORTED_LOSSES``.
        use_class_weights: Whether to weight the loss inversely proportional
            to class frequency (useful for imbalanced bona-fide/morph sets).
        mixed_precision: Whether to use automatic mixed precision (AMP).
            Safely ignored (disabled) when running on CPU.
        grad_clip_norm: Max-norm for gradient clipping. ``None`` disables
            clipping.
        early_stopping_patience: Epochs with no improvement before stopping.
            ``0`` disables early stopping.
        early_stopping_metric: Name of the metric monitored for early
            stopping and "best" checkpoint selection (e.g. ``"val_eer"``,
            ``"val_accuracy"``).
        early_stopping_mode: ``"min"`` if lower metric values are better
            (e.g. loss, EER) or ``"max"`` if higher is better (e.g. accuracy,
            AUC).
        seed: Global random seed for full reproducibility.
        checkpoint_dir: Directory checkpoints are written to.
        log_dir: Directory CSV/TensorBoard logs are written to.
        results_dir: Directory evaluation results/plots are written to.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
        log_every_n_steps: Frequency (in optimizer steps) of within-epoch
            console/TensorBoard scalar logging.
    """

    epochs: int = 30
    loss_name: str = "cross_entropy"
    use_class_weights: bool = True
    mixed_precision: bool = True
    grad_clip_norm: Optional[float] = 1.0
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_eer"
    early_stopping_mode: str = "min"
    seed: int = 42
    checkpoint_dir: Path = Path("checkpoints/vit")
    log_dir: Path = Path("logs/vit")
    results_dir: Path = Path("results/vit")
    device: str = "auto"
    log_every_n_steps: int = 50

    def __post_init__(self) -> None:
        _require(self.epochs > 0, f"epochs must be positive, got {self.epochs}")
        _require(
            self.loss_name in SUPPORTED_LOSSES,
            f"loss_name '{self.loss_name}' is not supported. Supported values: {SUPPORTED_LOSSES}",
        )
        _require(
            self.grad_clip_norm is None or self.grad_clip_norm > 0.0,
            f"grad_clip_norm must be None or positive, got {self.grad_clip_norm}",
        )
        _require(
            self.early_stopping_patience >= 0,
            f"early_stopping_patience must be >= 0, got {self.early_stopping_patience}",
        )
        _require(
            self.early_stopping_mode in ("min", "max"),
            f"early_stopping_mode must be 'min' or 'max', got '{self.early_stopping_mode}'",
        )
        _require(self.log_every_n_steps > 0, "log_every_n_steps must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration bundling all sub-configs for one experiment.

    Attributes:
        data: See :class:`DataConfig`.
        model: See :class:`ModelConfig`.
        optimizer: See :class:`OptimizerConfig`.
        scheduler: See :class:`SchedulerConfig`.
        training: See :class:`TrainingConfig`.
        experiment_name: Human-readable identifier used to name log/checkpoint
            sub-directories and TensorBoard runs.
    """

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    experiment_name: str = "vit_morph_detection"

    def __post_init__(self) -> None:
        _require(bool(self.experiment_name.strip()), "experiment_name must be non-empty")
        _require(
            self.scheduler.total_epochs == self.training.epochs,
            "scheduler.total_epochs must equal training.epochs "
            f"({self.scheduler.total_epochs} != {self.training.epochs})",
        )

    def with_overrides(self, **overrides: Any) -> "ExperimentConfig":
        """Return a new :class:`ExperimentConfig` with top-level fields overridden.

        Since instances are frozen, this is the supported way to derive a
        modified configuration (e.g. from CLI flags) without mutating the
        original object. Nested sub-config overrides should instead be
        applied to the sub-config directly via :func:`dataclasses.replace`
        before constructing the new ``ExperimentConfig``.

        Args:
            **overrides: Field name/value pairs corresponding to top-level
                attributes of :class:`ExperimentConfig`.

        Returns:
            A new, independently-validated :class:`ExperimentConfig`.
        """
        return replace(self, **overrides)


def _dataclass_from_dict(cls: Type[_T], data: Dict[str, Any]) -> _T:
    """Recursively construct a (possibly nested) dataclass from a plain dict.

    ``Path``-typed fields receive automatic ``str`` -> ``Path`` conversion.
    ``Tuple``-typed fields receive automatic ``list`` -> ``tuple`` conversion.
    Unknown keys in ``data`` raise :class:`ConfigValidationError` to catch
    typos in YAML configs early rather than silently ignoring them.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")

    field_defs = {f.name: f for f in fields(cls)}
    unknown_keys = set(data.keys()) - set(field_defs.keys())
    _require(
        not unknown_keys,
        f"Unknown configuration key(s) for {cls.__name__}: {sorted(unknown_keys)}",
    )

    kwargs: Dict[str, Any] = {}
    for name, f in field_defs.items():
        if name not in data:
            continue  # fall back to the dataclass default
        value = data[name]
        f_type = f.type

        if isinstance(f_type, str):
            # Handle stringified annotations (e.g. `from __future__ import annotations`).
            f_type_name = f_type
        else:
            f_type_name = getattr(f_type, "__name__", str(f_type))

        if "Path" in f_type_name and value is not None:
            value = Path(value)
        elif "Tuple" in f_type_name and isinstance(value, list):
            value = tuple(value)

        kwargs[name] = value

    return cls(**kwargs)  # type: ignore[return-value]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig` from a YAML file.

    Args:
        path: Path to a YAML file with top-level keys ``data``, ``model``,
            ``optimizer``, ``scheduler``, ``training``, and optionally
            ``experiment_name``.

    Returns:
        A fully validated, immutable :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigValidationError: If the file is missing required sections,
            contains unknown keys, or fails semantic validation (e.g. an
            unsupported backbone name, a non-positive learning rate).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    required_sections = ("data", "model", "optimizer", "scheduler", "training")
    missing = [s for s in required_sections if s not in raw]
    _require(not missing, f"Missing required config section(s): {missing}")

    data_cfg = _dataclass_from_dict(DataConfig, raw["data"])
    model_cfg = _dataclass_from_dict(ModelConfig, raw["model"])
    optimizer_cfg = _dataclass_from_dict(OptimizerConfig, raw["optimizer"])
    scheduler_cfg = _dataclass_from_dict(SchedulerConfig, raw["scheduler"])
    training_cfg = _dataclass_from_dict(TrainingConfig, raw["training"])

    experiment_name = raw.get("experiment_name", "vit_morph_detection")

    return ExperimentConfig(
        data=data_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=training_cfg,
        experiment_name=experiment_name,
    )


def _to_serializable(obj: Any) -> Any:
    """Recursively convert dataclass/Path/tuple values into YAML-serializable types."""
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


def save_experiment_config(config: ExperimentConfig, path: Path) -> None:
    """Serialize an :class:`ExperimentConfig` to a YAML file.

    Used to snapshot the exact configuration alongside checkpoints/logs for
    an experiment, guaranteeing later reproducibility and auditability.

    Args:
        config: The configuration to serialize.
        path: Destination YAML file path. Parent directories are created if
            they do not already exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # asdict() on a nested frozen dataclass gives us a plain nested dict;
    # deepcopy guards against accidental aliasing of mutable defaults.
    serializable = copy.deepcopy(_to_serializable(config))

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(serializable, fh, sort_keys=False, default_flow_style=False)