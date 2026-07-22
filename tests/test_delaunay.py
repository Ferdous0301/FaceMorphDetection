"""
tests/test_delaunay.py
=======================

Tests for ``src/morphing/delaunay.py``.

No MediaPipe dependency; all inputs are synthetic NumPy arrays.

Coverage
--------
* _border_points         – shape, dtype, clamping, all-in-bounds
* _add_border_points     – augmented shape, original rows preserved
* _build_coord_lookup    – keys are rounded, all points present
* triangulate            – shape mismatch raises, output shapes, index
                           dtype, index bounds, non-empty result,
                           alpha=0 mean equals A, alpha=1 mean equals B
* warp_triangle          – canvas modified inside triangle, degenerate
                           triangles silently skipped, canvas shape preserved
* warp_and_blend         – shape mismatch raises, out-of-range alpha raises,
                           output shape and dtype, output not all-zeros,
                           alpha=0 dominated by A, alpha=1 dominated by B,
                           identical images stay identical
"""

from __future__ import annotations

import numpy as np
import pytest

from src.morphing.delaunay import (
    _add_border_points,
    _border_points,
    _build_coord_lookup,
    triangulate,
    warp_and_blend,
    warp_triangle,
)
from src.morphing.mediapipe_landmarks import NUM_LANDMARKS


# ---------------------------------------------------------------------------
# Module-level constants / shared fixtures
# ---------------------------------------------------------------------------

IMAGE_SHAPE: tuple[int, int] = (128, 128)   # (height, width)
N: int = NUM_LANDMARKS                        # 468


def _landmarks(seed: int = 0) -> np.ndarray:
    """Return ``(N, 2)`` float32 landmarks uniformly inside IMAGE_SHAPE.

    A 10-pixel margin from each edge is applied so that landmarks never
    coincide with the border points added by the triangulation code.
    """
    rng = np.random.default_rng(seed)
    h, w = IMAGE_SHAPE
    return rng.uniform(
        low=[10.0, 10.0],
        high=[float(w - 10), float(h - 10)],
        size=(N, 2),
    ).astype(np.float32)


def _solid(color: tuple[int, int, int], shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    """Return a solid-colour BGR uint8 image."""
    img = np.empty((*shape, 3), dtype=np.uint8)
    img[:] = color
    return img


# ---------------------------------------------------------------------------
# _border_points
# ---------------------------------------------------------------------------

class TestBorderPoints:
    def test_shape_and_dtype(self) -> None:
        pts = _border_points(IMAGE_SHAPE)
        assert pts.shape == (8, 2)
        assert pts.dtype == np.float32

    def test_top_left_corner(self) -> None:
        pts = _border_points(IMAGE_SHAPE)
        assert pts[0, 0] == 0.0 and pts[0, 1] == 0.0

    def test_bottom_right_corner(self) -> None:
        h, w = IMAGE_SHAPE
        pts = _border_points(IMAGE_SHAPE)
        assert pts[7, 0] == float(w - 1)
        assert pts[7, 1] == float(h - 1)

    def test_all_points_within_image(self) -> None:
        h, w = IMAGE_SHAPE
        pts = _border_points(IMAGE_SHAPE)
        assert pts[:, 0].min() >= 0.0
        assert pts[:, 0].max() <= float(w - 1)
        assert pts[:, 1].min() >= 0.0
        assert pts[:, 1].max() <= float(h - 1)

    @pytest.mark.parametrize("shape", [(64, 64), (100, 200), (256, 128)])
    def test_various_image_shapes(self, shape: tuple[int, int]) -> None:
        h, w = shape
        pts = _border_points(shape)
        assert pts[:, 0].max() <= float(w - 1)
        assert pts[:, 1].max() <= float(h - 1)


# ---------------------------------------------------------------------------
# _add_border_points
# ---------------------------------------------------------------------------

class TestAddBorderPoints:
    def test_augmented_row_count(self) -> None:
        pts = _landmarks()
        aug = _add_border_points(pts, IMAGE_SHAPE)
        assert aug.shape == (N + 8, 2)

    def test_original_rows_preserved(self) -> None:
        pts = _landmarks()
        aug = _add_border_points(pts, IMAGE_SHAPE)
        np.testing.assert_array_equal(aug[:N], pts)

    def test_border_rows_appended_at_end(self) -> None:
        pts = _landmarks()
        aug = _add_border_points(pts, IMAGE_SHAPE)
        border = _border_points(IMAGE_SHAPE)
        np.testing.assert_array_equal(aug[N:], border)


# ---------------------------------------------------------------------------
# _build_coord_lookup
# ---------------------------------------------------------------------------

class TestBuildCoordLookup:
    def test_all_points_in_lookup(self) -> None:
        pts = _landmarks(seed=7).astype(np.float32)
        lookup = _build_coord_lookup(pts)
        assert len(lookup) == len(pts)

    def test_lookup_returns_correct_index(self) -> None:
        pts = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        lookup = _build_coord_lookup(pts)
        assert lookup[(10.0, 20.0)] == 0
        assert lookup[(30.0, 40.0)] == 1


# ---------------------------------------------------------------------------
# triangulate
# ---------------------------------------------------------------------------

class TestTriangulate:
    def test_shape_mismatch_raises(self) -> None:
        pts_a = _landmarks(0)
        pts_b = _landmarks(1)[:100]  # wrong length
        with pytest.raises(ValueError, match="same shape"):
            triangulate(pts_a, pts_b, 0.5, IMAGE_SHAPE)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_augmented_output_shapes(self, alpha: float) -> None:
        pts_a, pts_b = _landmarks(0), _landmarks(1)
        aug_a, aug_b, aug_m, tri_idx = triangulate(pts_a, pts_b, alpha, IMAGE_SHAPE)
        assert aug_a.shape == (N + 8, 2)
        assert aug_b.shape == (N + 8, 2)
        assert aug_m.shape == (N + 8, 2)

    def test_tri_indices_dtype_and_columns(self) -> None:
        _, _, _, tri_idx = triangulate(_landmarks(0), _landmarks(1), 0.5, IMAGE_SHAPE)
        assert tri_idx.dtype == np.int32
        assert tri_idx.ndim == 2
        assert tri_idx.shape[1] == 3

    def test_indices_within_bounds(self) -> None:
        aug_a, _, _, tri_idx = triangulate(_landmarks(0), _landmarks(1), 0.5, IMAGE_SHAPE)
        n_pts = len(aug_a)
        assert tri_idx.min() >= 0
        assert tri_idx.max() < n_pts

    def test_produces_at_least_one_triangle(self) -> None:
        _, _, _, tri_idx = triangulate(_landmarks(0), _landmarks(1), 0.5, IMAGE_SHAPE)
        assert len(tri_idx) > 0

    def test_alpha_zero_mean_equals_a(self) -> None:
        pts_a, pts_b = _landmarks(0), _landmarks(1)
        _, _, aug_m, _ = triangulate(pts_a, pts_b, 0.0, IMAGE_SHAPE)
        np.testing.assert_allclose(aug_m[:N], pts_a, atol=1e-4)

    def test_alpha_one_mean_equals_b(self) -> None:
        pts_a, pts_b = _landmarks(0), _landmarks(1)
        _, _, aug_m, _ = triangulate(pts_a, pts_b, 1.0, IMAGE_SHAPE)
        np.testing.assert_allclose(aug_m[:N], pts_b, atol=1e-4)

    def test_mean_interpolation_midpoint(self) -> None:
        pts_a, pts_b = _landmarks(0), _landmarks(1)
        _, _, aug_m, _ = triangulate(pts_a, pts_b, 0.5, IMAGE_SHAPE)
        expected = 0.5 * pts_a + 0.5 * pts_b
        np.testing.assert_allclose(aug_m[:N], expected, atol=1e-4)


# ---------------------------------------------------------------------------
# warp_triangle
# ---------------------------------------------------------------------------

class TestWarpTriangle:

    @staticmethod
    def _triangle(ox: int, oy: int) -> np.ndarray:
        """Return a small valid triangle offset by (ox, oy)."""
        return np.array(
            [[ox,      oy     ],
             [ox + 40, oy     ],
             [ox + 20, oy + 40]],
            dtype=np.float32,
        )

    def test_canvas_modified_inside_triangle(self) -> None:
        src = _solid((200, 100, 50))
        dst = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
        warp_triangle(src, dst, self._triangle(10, 10), self._triangle(30, 30))
        assert dst.any(), "Canvas must be modified inside the destination triangle."

    def test_degenerate_triangle_skipped_silently(self) -> None:
        src = _solid((200, 100, 50))
        dst = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
        degen = np.array([[20.0, 20.0]] * 3, dtype=np.float32)
        warp_triangle(src, dst, degen, degen)  # must not raise
        assert not dst.any(), "Degenerate triangle must not modify canvas."

    def test_canvas_shape_is_unchanged(self) -> None:
        src = _solid((255, 128, 0))
        dst = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
        warp_triangle(src, dst, self._triangle(5, 5), self._triangle(5, 5))
        assert dst.shape == (*IMAGE_SHAPE, 3)

    def test_pixels_outside_triangle_untouched(self) -> None:
        """Pixels far from the destination triangle should remain black."""
        src = _solid((200, 200, 200))
        dst = np.zeros((*IMAGE_SHAPE, 3), dtype=np.uint8)
        # Tiny triangle in top-left corner
        tiny = np.array([[0.0, 0.0], [5.0, 0.0], [2.5, 5.0]], dtype=np.float32)
        warp_triangle(src, dst, tiny, tiny)
        # Bottom-right quadrant should be completely black
        assert not dst[80:, 80:].any()


# ---------------------------------------------------------------------------
# warp_and_blend
# ---------------------------------------------------------------------------

class TestWarpAndBlend:

    def test_shape_mismatch_raises(self) -> None:
        img_a = np.zeros((128, 128, 3), dtype=np.uint8)
        img_b = np.zeros((64, 64, 3), dtype=np.uint8)
        pts = _landmarks()
        with pytest.raises(ValueError, match="same shape"):
            warp_and_blend(img_a, img_b, pts, pts, 0.5)

    @pytest.mark.parametrize("alpha", [-0.01, -1.0, 1.01, 2.0])
    def test_out_of_range_alpha_raises(self, alpha: float) -> None:
        img = _solid((100, 100, 100))
        pts = _landmarks()
        with pytest.raises(ValueError, match="alpha"):
            warp_and_blend(img, img.copy(), pts, pts, alpha)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_output_shape_and_dtype(self, alpha: float) -> None:
        img_a, img_b = _solid((200, 100, 50)), _solid((50, 100, 200))
        morphed = warp_and_blend(img_a, img_b, _landmarks(0), _landmarks(1), alpha)
        assert morphed.shape == img_a.shape
        assert morphed.dtype == np.uint8

    def test_output_is_not_all_black(self) -> None:
        img_a, img_b = _solid((200, 100, 50)), _solid((50, 100, 200))
        morphed = warp_and_blend(img_a, img_b, _landmarks(0), _landmarks(1), 0.5)
        assert morphed.any(), "Morphed image must not be entirely black."

    def test_alpha_zero_dominated_by_image_a(self) -> None:
        img_a = _solid((200, 0, 0))   # strong blue (BGR channel 0)
        img_b = _solid((0, 0, 200))   # strong red  (BGR channel 2)
        morphed = warp_and_blend(img_a, img_b, _landmarks(0), _landmarks(1), 0.0)
        assert morphed[:, :, 0].mean() > morphed[:, :, 2].mean(), (
            "At alpha=0, image A (blue) should dominate."
        )

    def test_alpha_one_dominated_by_image_b(self) -> None:
        img_a = _solid((200, 0, 0))   # strong blue
        img_b = _solid((0, 0, 200))   # strong red
        morphed = warp_and_blend(img_a, img_b, _landmarks(0), _landmarks(1), 1.0)
        assert morphed[:, :, 2].mean() > morphed[:, :, 0].mean(), (
            "At alpha=1, image B (red) should dominate."
        )

    def test_identical_images_produce_similar_output(self) -> None:
        """Morphing an image with itself should leave non-black pixels near
        the source colour."""
        img = _solid((128, 128, 128))
        pts = _landmarks(0)
        morphed = warp_and_blend(img, img.copy(), pts, pts.copy(), 0.5)
        non_black = morphed[morphed.any(axis=2)]
        if len(non_black) > 0:
            assert np.abs(non_black.astype(np.int16) - 128).max() < 10