"""Plot ViT self-attention maps as standalone heatmaps.

Consumes the layer-averaged attention weights produced by
:meth:`vit.models.vit_model.ViTMorphClassifier.get_attention_maps` and
renders the class token's attention over image patches as a heatmap. Per
project convention, this shows the raw attention distribution only — it
does not overlay the heatmap on the original input image.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

__all__ = ["plot_attention_rollout"]

# Index of the class token within the token sequence (torchvision ViT
# prepends the class token, so it is always token 0).
_CLASS_TOKEN_INDEX = 0


def plot_attention_rollout(attention: Tensor, output_path: Path) -> None:
    """Render the class token's attention over image patches as a heatmap.

    Args:
        attention: Attention weights for a single image, shape ``(N, N)``
            where ``N`` is the number of tokens (patch tokens + 1 class
            token), as returned per-sample by
            :meth:`vit.models.vit_model.ViTMorphClassifier.get_attention_maps`
            (index a single batch element with ``attn[i]`` before calling
            this function). Row/column 0 is assumed to be the class token.
        output_path: Destination PNG path. Parent directories are created
            if needed.

    Raises:
        ValueError: If ``attention`` is not square, has fewer than 2 tokens,
            or the number of patch tokens (``N - 1``) is not a perfect
            square (i.e. does not correspond to a square patch grid).
    """
    attn_np = attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.asarray(attention)

    if attn_np.ndim != 2 or attn_np.shape[0] != attn_np.shape[1]:
        raise ValueError(f"attention must be a square 2-D matrix, got shape {attn_np.shape}")
    if attn_np.shape[0] < 2:
        raise ValueError(f"attention must have at least 2 tokens (class + 1 patch), got {attn_np.shape[0]}")

    num_patch_tokens = attn_np.shape[0] - 1
    grid_size = int(math.isqrt(num_patch_tokens))
    if grid_size * grid_size != num_patch_tokens:
        raise ValueError(
            f"Number of patch tokens ({num_patch_tokens}) is not a perfect square; "
            "cannot reshape into a square patch grid"
        )

    # Attention paid *by* the class token *to* each patch token, reshaped
    # into the original spatial patch grid.
    class_to_patch_attention = attn_np[_CLASS_TOKEN_INDEX, 1:].reshape(grid_size, grid_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(class_to_patch_attention, cmap="viridis")
    ax.set_title("Class-Token Attention Rollout")
    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)