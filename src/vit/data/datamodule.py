"""DataModule facade unifying dataset and dataloader construction.

``ViTDataModule`` is the single place that decides how the train/val/test
``MorphDataset`` instances and their corresponding ``DataLoader``s are built,
so that transform selection, worker seeding, and batching behaviour cannot
drift between splits or between different call sites (Trainer, Evaluator,
CLI scripts) that need a loader for the same split.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from torch import Tensor
from torch.utils.data import DataLoader

from src.vit.configs.schema import DataConfig
from src.vit.data.dataset import MorphDataset
from src.vit.data.transforms import build_eval_transforms, build_train_transforms
from src.vit.utils.seed import make_generator, seed_worker

__all__ = ["ViTDataModule"]


class ViTDataModule:
    """Builds and serves train/val/test ``DataLoader``s from a :class:`DataConfig`.

    Usage:
        >>> from vit.config.schema import DataConfig
        >>> config = DataConfig(
        ...     train_csv="train.csv", val_csv="val.csv", test_csv="test.csv",
        ...     image_root=".",
        ... )  # doctest: +SKIP
        >>> dm = ViTDataModule(config, seed=42)  # doctest: +SKIP
        >>> dm.setup()  # doctest: +SKIP
        >>> train_loader = dm.train_dataloader()  # doctest: +SKIP

    Attributes:
        config: The :class:`~vit.config.schema.DataConfig` driving dataset
            and dataloader construction.
        seed: Base random seed, used to derive a deterministic
            ``torch.Generator`` for training-set shuffling and to seed
            DataLoader worker processes.
    """

    def __init__(
        self,
        config: DataConfig,
        seed: int,
        train_transform: Optional[Callable] = None,
        eval_transform: Optional[Callable] = None,
        validate_files: bool = True,
    ) -> None:
        """Construct the DataModule. Does not touch disk until :meth:`setup` is called.

        Args:
            config: Data configuration (paths, batch size, worker count, etc.).
            seed: Base seed for deterministic shuffling/worker seeding.
            train_transform: Transform applied to the training split. If
                ``None``, built automatically via
                :func:`vit.data.transforms.build_train_transforms` using
                ``config.image_size``.
            eval_transform: Transform applied to the val/test splits. If
                ``None``, built automatically via
                :func:`vit.data.transforms.build_eval_transforms` using
                ``config.image_size``.
            validate_files: Forwarded to :class:`~vit.data.dataset.MorphDataset`
                for every split; see its docstring for details.
        """
        self.config = config
        self.seed = seed
        self._validate_files = validate_files
        self._train_transform = train_transform or build_train_transforms(config.image_size)
        self._eval_transform = eval_transform or build_eval_transforms(config.image_size)

        self._train_dataset: Optional[MorphDataset] = None
        self._val_dataset: Optional[MorphDataset] = None
        self._test_dataset: Optional[MorphDataset] = None

    def setup(self) -> None:
        """Construct the underlying :class:`MorphDataset` for each split.

        Must be called once before any of :meth:`train_dataloader`,
        :meth:`val_dataloader`, :meth:`test_dataloader`, or
        :meth:`class_counts`/:meth:`class_weights` are used. Idempotent:
        calling it multiple times simply rebuilds the datasets.

        Raises:
            FileNotFoundError: If any manifest CSV or (when
                ``validate_files=True``) any referenced image file is missing.
            ValueError: If a manifest is missing required columns.
        """
        self._train_dataset = MorphDataset(
            csv_path=self.config.train_csv,
            image_root=self.config.image_root,
            transform=self._train_transform,
            path_column=self.config.path_column,
            label_column=self.config.label_column,
            validate_files=self._validate_files,
        )
        self._val_dataset = MorphDataset(
            csv_path=self.config.val_csv,
            image_root=self.config.image_root,
            transform=self._eval_transform,
            path_column=self.config.path_column,
            label_column=self.config.label_column,
            validate_files=self._validate_files,
        )
        self._test_dataset = MorphDataset(
            csv_path=self.config.test_csv,
            image_root=self.config.image_root,
            transform=self._eval_transform,
            path_column=self.config.path_column,
            label_column=self.config.label_column,
            validate_files=self._validate_files,
        )

    def _require_setup(self) -> None:
        if self._train_dataset is None or self._val_dataset is None or self._test_dataset is None:
            raise RuntimeError(
                "ViTDataModule.setup() must be called before accessing datasets or dataloaders."
            )

    def train_dataloader(self) -> DataLoader:
        """Build the training ``DataLoader``.

        Shuffles each epoch using a deterministic ``torch.Generator`` seeded
        from ``self.seed``, and seeds worker processes via
        :func:`vit.utils.seed.seed_worker`, so batch order is reproducible
        across runs and across ``num_workers`` settings.

        Returns:
            A ``DataLoader`` over the training split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last_train,
            worker_init_fn=seed_worker,
            generator=make_generator(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        """Build the validation ``DataLoader`` (no shuffling, no dropped batches).

        Returns:
            A ``DataLoader`` over the validation split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

    def test_dataloader(self) -> DataLoader:
        """Build the test ``DataLoader`` (no shuffling, no dropped batches).

        Returns:
            A ``DataLoader`` over the test split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            worker_init_fn=seed_worker,
        )

    def class_counts(self) -> Dict[int, int]:
        """Return per-class sample counts for the training split.

        Used to decide/construct class-weighted loss functions.

        Returns:
            Mapping from class label to sample count in the training split.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        assert self._train_dataset is not None
        return self._train_dataset.class_counts

    def class_weights(self) -> Tensor:
        """Return inverse-frequency class weights computed from the training split.

        Returns:
            A 1-D float tensor indexed by class label; see
            :meth:`vit.data.dataset.MorphDataset.class_weights`.

        Raises:
            RuntimeError: If :meth:`setup` has not been called yet.
        """
        self._require_setup()
        assert self._train_dataset is not None
        return self._train_dataset.class_weights()