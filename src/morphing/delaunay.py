"""
src/morphing/delaunay.py
=========================

Delaunay triangulation, per-triangle affine warping, and alpha blending.

This module is the mathematical core of the morphing pipeline.  Given two
sets of corresponding face landmarks it:

1. Computes a Delaunay triangulation of the *mean* landmark positions so that
   the same triangle topology is shared by both source images.
2. For each triangle, computes the affine transform that maps the source
   triangle vertices to the morphed (mean) triangle vertices.
3. Warps each triangular patch from each source image into a blank canvas.
4. Alpha-blends the two warped canvases to produce the final morph.

All operations run on CPU using NumPy and OpenCV only.

Design decisions
----------------
* The triangulation is computed on the *interpolated* (alpha-blended)
  landmark positions rather than on either source individually, which
  guarantees that every triangle is geometrically valid in both images.
* Eight border landmarks (corners + edge midpoints) are appended before
  triangulation so that the entire image rectangle is covered and no border
  pixels are left black.
* ``warp_and_blend`` stores the augmented mean array returned by
  ``triangulate`` so that the per-triangle destination position is looked up
  rather than recomputed inside the loop.
* All public helpers are independently importable and testable.

Public API
----------
triangulate(points_a, points_b, alpha, image_shape)
    Compute interpolated points and Delaunay triangle indices.
warp_triangle(src_image, dst_canvas, src_tri, dst_tri)
    Warp a single triangular patch from src_image into dst_canvas in-place.
warp_and_blend(image_a, image_b, landmarks_a, landmarks_b, alpha)
    Full pipeline: warp both images and alpha-blend them.
"""

from __future__ import annotations

import logging
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Float32 point array of shape (N, 2).
Points = NDArray[np.float32]

#: Int32 index array of shape (M, 3).
TriIndices = NDArray[np.int32]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of synthetic border landmarks added per image to ensure full
#: coverage of the image rectangle during triangulation.
_NUM_BORDER_POINTS: Final[int] = 8

#: Decimal precision used when rounding coordinates in the lookup table
#: built by ``_triangles_to_indices``.  Two decimal places tolerate the
#: floating-point drift introduced by ``cv2.Subdiv2D.getTriangleList``.
_COORD_ROUND_DECIMALS: Final[int] = 2


# ---------------------------------------------------------------------------
# Private: border landmark helpers
# ---------------------------------------------------------------------------

def _border_points(image_shape: tuple[int, int]) -> Points:
    """Return the eight corner and edge-midpoint coordinates for an image.

    Adding these synthetic landmarks before triangulation ensures that
    Delaunay triangles cover the full image rectangle, so every output pixel
    receives colour from the warp rather than defaulting to black.

    Parameters
    ----------
    image_shape : tuple[int, int]
        ``(height, width)`` of the image.

    Returns
    -------
    NDArray[np.float32]
        Shape ``(8, 2)``.  All coordinates lie on or inside the image boundary.
    """
    h, w = image_shape
    h1, w1 = float(h - 1), float(w - 1)
    return np.array(
        [
            [0.0,    0.0   ],   # top-left
            [w1 / 2, 0.0   ],   # top-centre
            [w1,     0.0   ],   # top-right
            [0.0,    h1 / 2],   # mid-left
            [w1,     h1 / 2],   # mid-right
            [0.0,    h1    ],   # bottom-left
            [w1 / 2, h1    ],   # bottom-centre
            [w1,     h1    ],   # bottom-right
        ],
        dtype=np.float32,
    )


def _add_border_points(
    points: Points,
    image_shape: tuple[int, int],
) -> Points:
    """Append the eight border points to ``points`` and return the result.

    Parameters
    ----------
    points : Points
        Shape ``(N, 2)`` landmark coordinates.
    image_shape : tuple[int, int]
        ``(height, width)`` of the image.

    Returns
    -------
    Points
        Shape ``(N + 8, 2)``.  The first ``N`` rows are the original points
        unchanged; the last 8 rows are the border landmarks.
    """
    return np.vstack([points, _border_points(image_shape)])


# ---------------------------------------------------------------------------
# Private: coordinate-to-index lookup
# ---------------------------------------------------------------------------

def _build_coord_lookup(points: Points) -> dict[tuple[float, float], int]:
    """Build a rounded-coordinate → index mapping for ``points``.

    ``cv2.Subdiv2D.getTriangleList`` returns vertex coordinates rather than
    indices.  This function pre-computes the reverse mapping so that
    ``_triangles_to_indices`` can do O(1) lookups instead of O(N) searches.

    Parameters
    ----------
    points : Points
        Shape ``(N, 2)`` point array.

    Returns
    -------
    dict mapping ``(round(x, 2), round(y, 2))`` → row index.
    """
    d = _COORD_ROUND_DECIMALS
    return {
        (round(float(p[0]), d), round(float(p[1]), d)): i
        for i, p in enumerate(points)
    }


def _triangles_to_indices(
    triangles_raw: NDArray[np.float32],
    points: Points,
    image_shape: tuple[int, int],
) -> TriIndices:
    """Convert ``cv2.Subdiv2D.getTriangleList`` output to point indices.

    Parameters
    ----------
    triangles_raw : NDArray[np.float32]
        Shape ``(M, 6)`` as returned by ``cv2.Subdiv2D.getTriangleList``.
        Each row is ``[x1, y1, x2, y2, x3, y3]``.
    points : Points
        Shape ``(N, 2)`` augmented point array (includes border points).
    image_shape : tuple[int, int]
        ``(height, width)`` used to discard triangles whose vertices lie
        outside the image rectangle.

    Returns
    -------
    TriIndices
        Shape ``(K, 3)``, dtype ``int32``, where ``K ≤ M`` after filtering
        out-of-bounds triangles and unresolvable vertices.
    """
    h, w = image_shape
    d = _COORD_ROUND_DECIMALS
    lookup = _build_coord_lookup(points)
    indices: list[tuple[int, int, int]] = []

    for tri in triangles_raw:
        x1, y1, x2, y2, x3, y3 = tri
        verts = ((x1, y1), (x2, y2), (x3, y3))

        # Discard triangles with any vertex outside the image boundary.
        if any(x < 0 or x >= w or y < 0 or y >= h for x, y in verts):
            continue

        idx: list[int] = []
        valid = True
        for x, y in verts:
            key = (round(float(x), d), round(float(y), d))
            if key not in lookup:
                valid = False
                break
            idx.append(lookup[key])

        if valid:
            indices.append((idx[0], idx[1], idx[2]))

    if not indices:
        return np.empty((0, 3), dtype=np.int32)

    return np.array(indices, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public: triangulation
# ---------------------------------------------------------------------------

def triangulate(
    points_a: Points,
    points_b: Points,
    alpha: float,
    image_shape: tuple[int, int],
) -> tuple[Points, Points, Points, TriIndices]:
    """Compute a shared Delaunay triangulation for morphing two point sets.

    The triangulation is performed on the interpolated mean positions
    ``points_m = (1 - alpha) * points_a + alpha * points_b`` so that the
    resulting triangle indices are simultaneously valid for ``points_a``,
    ``points_b``, and ``points_m``.

    Eight border landmarks are appended to all three arrays before
    triangulation to ensure complete coverage of the image rectangle.

    Parameters
    ----------
    points_a : Points
        Shape ``(N, 2)`` landmarks from image A, in pixel coordinates.
    points_b : Points
        Shape ``(N, 2)`` landmarks from image B, in pixel coordinates.
        Must have the same shape as ``points_a``.
    alpha : float
        Blend weight in ``[0, 1]``.  ``alpha=0`` reproduces image A;
        ``alpha=1`` reproduces image B.
    image_shape : tuple[int, int]
        ``(height, width)`` of the image, used for border landmarks and
        the ``cv2.Subdiv2D`` bounding rectangle.

    Returns
    -------
    aug_a : Points
        Augmented source-A landmarks, shape ``(N + 8, 2)``.
    aug_b : Points
        Augmented source-B landmarks, shape ``(N + 8, 2)``.
    aug_m : Points
        Augmented mean (morphed) landmarks, shape ``(N + 8, 2)``.
    tri_indices : TriIndices
        Shape ``(M, 3)`` integer indices into the augmented arrays defining
        ``M`` Delaunay triangles.

    Raises
    ------
    ValueError
        If ``points_a.shape != points_b.shape``.
    """
    if points_a.shape != points_b.shape:
        raise ValueError(
            f"points_a and points_b must have the same shape, "
            f"got {points_a.shape} vs {points_b.shape}."
        )

    h, w = image_shape

    points_m: Points = (1.0 - alpha) * points_a + alpha * points_b

    aug_a = _add_border_points(points_a, image_shape)
    aug_b = _add_border_points(points_b, image_shape)
    aug_m = _add_border_points(points_m, image_shape)

    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for pt in aug_m:
        subdiv.insert((float(pt[0]), float(pt[1])))

    triangles_raw: NDArray[np.float32] = subdiv.getTriangleList()
    tri_indices = _triangles_to_indices(triangles_raw, aug_m, image_shape)

    logger.debug(
        "Triangulation: %d triangles from %d landmarks (alpha=%.3f).",
        len(tri_indices),
        len(aug_m),
        alpha,
    )
    return aug_a, aug_b, aug_m, tri_indices


# ---------------------------------------------------------------------------
# Public: per-triangle affine warp
# ---------------------------------------------------------------------------

def warp_triangle(
    src_image: NDArray[np.uint8],
    dst_canvas: NDArray[np.uint8],
    src_tri: Points,
    dst_tri: Points,
) -> None:
    """Warp a triangular patch from ``src_image`` into ``dst_canvas`` in-place.

    Steps
    -----
    1. Compute axis-aligned bounding rectangles for both triangles.
    2. Compute the affine transform mapping the source-local vertices to the
       destination-local vertices.
    3. Apply the warp to the source patch.
    4. Create a filled triangular mask and composite the warped patch into
       ``dst_canvas`` only where the mask is non-zero.

    The operation modifies ``dst_canvas`` in-place and returns nothing.
    Degenerate triangles (zero-area bounding rectangles) are silently skipped.

    Parameters
    ----------
    src_image : NDArray[np.uint8]
        Source BGR image.
    dst_canvas : NDArray[np.uint8]
        Destination BGR canvas, modified in-place.  Must have the same
        spatial dimensions as ``src_image``.
    src_tri : Points
        Shape ``(3, 2)`` source triangle vertices in pixel coordinates.
    dst_tri : Points
        Shape ``(3, 2)`` destination triangle vertices in pixel coordinates.
    """
    src_rect = cv2.boundingRect(src_tri)
    dst_rect = cv2.boundingRect(dst_tri)

    sx, sy, sw, sh = src_rect
    dx, dy, dw, dh = dst_rect

    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return  # degenerate triangle – skip silently

    src_patch = src_image[sy : sy + sh, sx : sx + sw]

    # Shift vertices to bounding-rect-local coordinates.
    src_offset = np.array([sx, sy], dtype=np.float32)
    dst_offset = np.array([dx, dy], dtype=np.float32)
    src_tri_local = (src_tri - src_offset).astype(np.float32)
    dst_tri_local = (dst_tri - dst_offset).astype(np.float32)

    M = cv2.getAffineTransform(src_tri_local, dst_tri_local)

    warped = cv2.warpAffine(
        src_patch,
        M,
        (dw, dh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Build a single-channel triangular mask, then broadcast to 3 channels.
    mask = np.zeros((dh, dw), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_tri_local.astype(np.int32), 255)
    mask_3ch = cv2.merge([mask, mask, mask])

    dst_patch = dst_canvas[dy : dy + dh, dx : dx + dw]
    dst_patch[:] = np.where(mask_3ch > 0, warped, dst_patch)


# ---------------------------------------------------------------------------
# Public: full morph pipeline
# ---------------------------------------------------------------------------

def warp_and_blend(
    image_a: NDArray[np.uint8],
    image_b: NDArray[np.uint8],
    landmarks_a: Points,
    landmarks_b: Points,
    alpha: float,
) -> NDArray[np.uint8]:
    """Produce a morphed image by warping and alpha-blending two face images.

    Algorithm
    ---------
    1. Compute a shared Delaunay triangulation on the mean landmark positions.
    2. For each triangle, warp the corresponding patch from image A into a
       blank canvas ``warped_a``, and the patch from image B into ``warped_b``.
       The destination for both warps is the mean (morphed) triangle.
    3. Return ``cv2.addWeighted(warped_a, 1 - alpha, warped_b, alpha, 0)``.

    Parameters
    ----------
    image_a : NDArray[np.uint8]
        BGR source image A.
    image_b : NDArray[np.uint8]
        BGR source image B.  Must have the same shape as ``image_a``.
    landmarks_a : Points
        Shape ``(N, 2)`` pixel-space landmarks for image A.
    landmarks_b : Points
        Shape ``(N, 2)`` pixel-space landmarks for image B.
    alpha : float
        Blend weight in ``[0, 1]``.  ``alpha=0`` reproduces image A;
        ``alpha=1`` reproduces image B.

    Returns
    -------
    NDArray[np.uint8]
        Morphed image, same shape and dtype as ``image_a``.

    Raises
    ------
    ValueError
        If ``image_a.shape != image_b.shape`` or ``alpha`` is outside
        ``[0, 1]``.
    """
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"image_a and image_b must have the same shape, "
            f"got {image_a.shape} vs {image_b.shape}."
        )
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    h, w = image_a.shape[:2]

    # Step 1: triangulate on mean positions; keep aug_m for dst_tri lookup.
    aug_a, aug_b, aug_m, tri_indices = triangulate(
        landmarks_a, landmarks_b, alpha, (h, w)
    )

    # Step 2: warp each triangle from both source images into blank canvases.
    warped_a = np.zeros_like(image_a)
    warped_b = np.zeros_like(image_b)

    for i0, i1, i2 in tri_indices:
        src_tri_a = aug_a[[i0, i1, i2]].astype(np.float32)
        src_tri_b = aug_b[[i0, i1, i2]].astype(np.float32)
        # Use the stored mean positions rather than recomputing.
        dst_tri = aug_m[[i0, i1, i2]].astype(np.float32)

        warp_triangle(image_a, warped_a, src_tri_a, dst_tri)
        warp_triangle(image_b, warped_b, src_tri_b, dst_tri)

    # Step 3: alpha-blend the two warped canvases.
    morphed: NDArray[np.uint8] = cv2.addWeighted(
        warped_a, 1.0 - alpha, warped_b, alpha, 0.0
    )

    logger.debug(
        "warp_and_blend: alpha=%.3f, %d triangles, output shape=%s.",
        alpha,
        len(tri_indices),
        morphed.shape,
    )
    return morpheds