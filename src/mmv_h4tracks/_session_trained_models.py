"""
Track Cellpose models trained in the current session (layer + frames) for prediction prompts.
"""

from __future__ import annotations

from typing import Any

# Values: {"layer_name": str, "segmentation_layer_name": str, "frames": tuple[int, ...]}
SessionTrainedModelsMap = dict[str, dict[str, Any]]


def register_session_trained_model(
    store: SessionTrainedModelsMap,
    display_name: str,
    layer_name: str,
    segmentation_layer_name: str,
    frames: tuple[int, ...],
) -> None:
    store[display_name] = {
        "layer_name": layer_name,
        "segmentation_layer_name": segmentation_layer_name,
        "frames": tuple(frames),
    }


def update_layer_name_in_store(
    store: SessionTrainedModelsMap, old_name: str, new_name: str
) -> None:
    for meta in store.values():
        if meta["layer_name"] == old_name:
            meta["layer_name"] = new_name
        if meta.get("segmentation_layer_name") == old_name:
            meta["segmentation_layer_name"] = new_name


def remove_entries_for_layer(store: SessionTrainedModelsMap, layer_name: str) -> None:
    to_del = [
        k
        for k, v in store.items()
        if v["layer_name"] == layer_name
        or v.get("segmentation_layer_name") == layer_name
    ]
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
