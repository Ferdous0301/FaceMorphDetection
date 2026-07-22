"""
src/morphing
============

Face morph generation pipeline for the Face Morph Attack Detection project.

Modules
-------
mediapipe_landmarks
    MediaPipe Face Mesh initialisation and landmark detection.
delaunay
    Delaunay triangulation, affine warping, and triangle blending.
morph_generator
    End-to-end pair selection, morphing, saving, and CLI entry-point.
metadata
    Append-safe CSV metadata writer for generated morph images.

Typical usage
-------------
>>> from src.morphing.mediapipe_landmarks import LandmarkDetector
>>> from src.morphing.delaunay import triangulate, warp_and_blend
>>> from src.morphing.morph_generator import MorphGenerator
>>> from src.morphing.metadata import MetadataWriter
"""

from src.morphing.mediapipe_landmarks import LandmarkDetector, detect_landmarks
from src.morphing.delaunay import triangulate, warp_and_blend
from src.morphing.morph_generator import MorphGenerator
from src.morphing.metadata import MetadataWriter

__all__ = [
    "LandmarkDetector",
    "detect_landmarks",
    "triangulate",
    "warp_and_blend",
    "MorphGenerator",
    "MetadataWriter",
]