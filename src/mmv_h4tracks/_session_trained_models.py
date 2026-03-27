"""
Track Cellpose models trained in the current session (mask export dir + frames) for prediction prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Values: training_masks_dir, layer_prefix (sanitized raw layer name at export), frames
SessionTrainedModelsMap = dict[str, dict[str, Any]]


def register_session_trained_model(
    store: SessionTrainedModelsMap,
    display_name: str,
    training_masks_dir: str | Path,
    layer_prefix: str,
    frames: tuple[int, ...],
) -> None:
    store[display_name] = {
        "training_masks_dir": str(Path(training_masks_dir).resolve()),
        "layer_prefix": layer_prefix,
        "frames": tuple(frames),
    }


def update_layer_name_in_store(
    store: SessionTrainedModelsMap, old_name: str, new_name: str
) -> None:
    from ._train import _sanitize_model_name_fragment

    old_p = _sanitize_model_name_fragment(old_name)
    new_p = _sanitize_model_name_fragment(new_name)
    for meta in store.values():
        if meta.get("layer_prefix") == old_p:
            meta["layer_prefix"] = new_p


def remove_entries_for_layer(store: SessionTrainedModelsMap, layer_name: str) -> None:
    from ._train import _sanitize_model_name_fragment

    p = _sanitize_model_name_fragment(layer_name)
    to_del = [k for k, v in store.items() if v.get("layer_prefix") == p]
    for k in to_del:
        del store[k]


def overlap_training_frames_with_stack(
    training_frames: tuple[int, ...],
    n_frames_in_data: int,
    is_single_frame_2d: bool,
) -> list[int]:
    """
    Training frame indices are global (0 .. T-1 for the image layer time axis).
    ``data_squeezed`` is the array passed to segmentation (preview uses the first slices only);
    index ``i`` along time always matches global frame ``i`` in this codebase.
    """
    if is_single_frame_2d:
        return sorted(set(training_frames) & {0})
    if n_frames_in_data < 1:
        return []
    represented = set(range(n_frames_in_data))
    return sorted(set(training_frames) & represented)
