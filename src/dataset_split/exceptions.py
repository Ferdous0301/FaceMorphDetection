"""Custom exception hierarchy for the Dataset Split stage.

This module defines all exceptions raised by the dataset split component
of the face morphing attack detection pipeline. All exceptions derive
from :class:`DatasetSplitError`, allowing callers to catch any
dataset-split-related failure with a single except clause while still
permitting fine-grained handling of specific failure modes.

No implementation logic lives in this module; it contains exception
class definitions only.
"""

from __future__ import annotations


class DatasetSplitError(Exception):
    """Base exception for all errors raised by the Dataset Split stage.

    All other exceptions in this module inherit from this class. Catching
    ``DatasetSplitError`` will catch any error originating from the
    dataset split pipeline stage.
    """


class InvalidSplitRatioError(DatasetSplitError):
    """Raised when the requested train/validation/test split ratios are invalid.

    Examples of invalid ratios include values that are negative, exceed
    1.0, or do not sum to 1.0 within an acceptable tolerance.
    """


class IdentityLeakageError(DatasetSplitError):
    """Raised when the same subject identity appears across multiple splits.

    Identity leakage between splits (e.g. train and test) undermines the
    validity of morphing attack detection evaluation and must be treated
    as a fatal error.
    """


class ManifestValidationError(DatasetSplitError):
    """Raised when a dataset manifest fails schema or content validation.

    This includes manifests that are malformed, missing required columns,
    or contain entries that are inconsistent with the expected dataset
    structure.
    """


class DuplicateImageError(DatasetSplitError):
    """Raised when duplicate images are detected within or across splits.

    Duplicate images can bias evaluation metrics and must be identified
    and rejected prior to finalizing a dataset split.
    """


class MissingMetadataError(DatasetSplitError):
    """Raised when required metadata for a sample or dataset is absent.

    This includes missing identity labels, missing morph/bona-fide labels,
    or any other metadata field required to perform a valid split.
    """


class InvalidConfigurationError(DatasetSplitError):
    """Raised when the Dataset Split stage is configured incorrectly.

    This includes invalid or conflicting configuration parameters that
    prevent the stage from executing correctly, such as unsupported
    strategies, malformed paths, or contradictory options.
    """