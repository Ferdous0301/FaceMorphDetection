"""Unit tests for the Dataset Split stage exception hierarchy.

These tests verify:
    * Correct inheritance relationships between exception classes.
    * That every exception can be raised and caught as expected.
    * That exception messages and arguments propagate correctly.
    * That all exceptions are catchable via the common base class.
"""

from __future__ import annotations

import pytest

from src.dataset_split.exceptions import (
    DatasetSplitError,
    DuplicateImageError,
    IdentityLeakageError,
    InvalidConfigurationError,
    InvalidSplitRatioError,
    ManifestValidationError,
    MissingMetadataError,
)

ALL_SUBCLASSES: tuple[type[DatasetSplitError], ...] = (
    InvalidSplitRatioError,
    IdentityLeakageError,
    ManifestValidationError,
    DuplicateImageError,
    MissingMetadataError,
    InvalidConfigurationError,
)


class TestInheritanceHierarchy:
    """Tests verifying the exception class inheritance structure."""

    def test_base_exception_inherits_from_exception(self) -> None:
        """DatasetSplitError must inherit directly from Exception."""
        assert issubclass(DatasetSplitError, Exception)

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclasses_inherit_from_dataset_split_error(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Every specific exception must inherit from DatasetSplitError."""
        assert issubclass(exc_cls, DatasetSplitError)

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclasses_are_exceptions(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Every specific exception must ultimately be an Exception."""
        assert issubclass(exc_cls, Exception)

    def test_no_unintended_cross_inheritance(self) -> None:
        """Sibling exceptions must not inherit from one another."""
        for exc_cls in ALL_SUBCLASSES:
            others = [c for c in ALL_SUBCLASSES if c is not exc_cls]
            for other in others:
                assert not issubclass(exc_cls, other)


class TestExceptionsCanBeRaised:
    """Tests verifying that each exception can be raised and caught."""

    def test_dataset_split_error_can_be_raised(self) -> None:
        """DatasetSplitError can be raised and caught directly."""
        with pytest.raises(DatasetSplitError):
            raise DatasetSplitError("base error")

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_can_be_raised_and_caught_directly(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Each subclass can be raised and caught by its own type."""
        with pytest.raises(exc_cls):
            raise exc_cls("something went wrong")

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_can_be_caught_as_base_class(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Each subclass can be caught via the common base exception."""
        with pytest.raises(DatasetSplitError):
            raise exc_cls("something went wrong")

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_can_be_caught_as_exception(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Each subclass can be caught via the built-in Exception type."""
        with pytest.raises(Exception):
            raise exc_cls("something went wrong")


class TestExceptionMessagePropagation:
    """Tests verifying that exception messages propagate correctly."""

    def test_base_exception_message_propagates(self) -> None:
        """The message passed to DatasetSplitError is preserved."""
        message = "dataset split failed"
        with pytest.raises(DatasetSplitError) as exc_info:
            raise DatasetSplitError(message)
        assert str(exc_info.value) == message

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_message_propagates(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """The message passed to each subclass is preserved."""
        message = f"{exc_cls.__name__} occurred"
        with pytest.raises(exc_cls) as exc_info:
            raise exc_cls(message)
        assert str(exc_info.value) == message

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_supports_multiple_args(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Exceptions support multiple positional arguments like Exception."""
        with pytest.raises(exc_cls) as exc_info:
            raise exc_cls("bad value", 42)
        assert exc_info.value.args == ("bad value", 42)

    def test_no_message_still_raisable(self) -> None:
        """Exceptions can be raised without any message argument."""
        with pytest.raises(DatasetSplitError) as exc_info:
            raise DatasetSplitError()
        assert str(exc_info.value) == ""


class TestExceptionDocstrings:
    """Tests verifying that all exceptions are properly documented."""

    def test_base_exception_has_docstring(self) -> None:
        """DatasetSplitError must have a non-empty docstring."""
        assert DatasetSplitError.__doc__ is not None
        assert DatasetSplitError.__doc__.strip() != ""

    @pytest.mark.parametrize("exc_cls", ALL_SUBCLASSES)
    def test_subclass_has_docstring(
        self, exc_cls: type[DatasetSplitError]
    ) -> None:
        """Every subclass must have a non-empty docstring."""
        assert exc_cls.__doc__ is not None
        assert exc_cls.__doc__.strip() != ""