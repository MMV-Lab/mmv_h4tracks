"""Tests for Cellpose training export helpers."""

import time

import numpy as np
import pytest
from unittest.mock import Mock

from mmv_h4tracks._session_trained_models import overlap_training_frames_with_stack
from mmv_h4tracks._train import (
    export_cellpose_training_pairs,
    find_cellpose_cli_weights,
    parse_use_frames,
    prune_cellpose_training_export_dir,
    classify_mmvh4tracks_training_dir,
    parse_layer_prefix_and_frames_from_masks_dir,
    _sanitize_model_name_fragment,
    _cellpose_training_raw_cyx,
    _frame_2d,
    _get_array_from_layer,
    _n_frames,
    _safe_rmtree,
    _validate_frame_pairs,
)


def test_parse_use_frames_valid():
    assert parse_use_frames("0, 1, 2") == [0, 1, 2]
    assert parse_use_frames("2, 1 , 1") == [1, 2]
    assert parse_use_frames("42") == [42]


def test_parse_use_frames_invalid_or_empty():
    assert parse_use_frames("") is None
    assert parse_use_frames("   ") is None
    assert parse_use_frames("1,,2") is None
    assert parse_use_frames("1, x") is None
    assert parse_use_frames("1.5") is None


def test_n_frames_and_frame_2d():
    single = np.zeros((10, 10))
    assert _n_frames(single) == 1
    assert _frame_2d(single, 0).shape == (10, 10)
    with pytest.raises(ValueError, match="out of range"):
        _frame_2d(single, 1)

    stack = np.zeros((5, 8, 9))
    assert _n_frames(stack) == 5
    assert _frame_2d(stack, 3).shape == (8, 9)
    with pytest.raises(ValueError, match="out of range"):
        _frame_2d(stack, 10)


def test_overlap_training_frames_with_stack():
    assert overlap_training_frames_with_stack((1, 2, 10), 5, False) == [1, 2]
    assert overlap_training_frames_with_stack((0,), 1, True) == [0]
    assert overlap_training_frames_with_stack((5,), 1, False) == []


def test_safe_rmtree(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "f.txt").write_text("x")
    _safe_rmtree(d)
    assert not d.exists()


def test_cellpose_training_raw_cyx():
    yx = np.arange(12, dtype=np.float32).reshape(3, 4)
    cyx = _cellpose_training_raw_cyx(yx)
    assert cyx.shape == (2, 3, 4)
    np.testing.assert_array_equal(cyx[0], yx)
    np.testing.assert_array_equal(cyx[1], 0)


def test_validate_frame_pairs():
    raw = np.zeros((3, 4, 4))
    seg = np.zeros((3, 4, 4))
    _validate_frame_pairs(raw, seg, [0, 2])

    raw_bad = np.zeros((3, 4, 4))
    seg_bad = np.zeros((3, 5, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_frame_pairs(raw_bad, seg_bad, [0])


def test_get_array_from_layer_mock():
    layer = Mock()
    layer.data = np.zeros((1, 5, 5))
    out = _get_array_from_layer(layer)
    assert out.ndim == 2
    assert out.shape == (5, 5)


def test_export_cellpose_training_pairs_writes_tiffs(tmp_path, monkeypatch):
    """Export creates Cellpose-style ``img.tif`` + ``img_masks.tif`` in one folder."""
    raw_vol = np.random.rand(2, 16, 16).astype(np.float32)
    seg_vol = np.zeros((2, 16, 16), dtype=np.uint16)
    seg_vol[:, 4:12, 4:12] = 1

    class V:
        layers = []

    viewer = V()

    def fake_grab(v, name):
        if name == "R":
            return Mock(data=raw_vol)
        if name == "S":
            return Mock(data=seg_vol)
        raise AssertionError(name)

    def fake_prepare(model_name: str):
        d = tmp_path / f"mmv_h4tracks_train_{_sanitize_model_name_fragment(model_name)}"
        _safe_rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        return d

    monkeypatch.setattr("mmv_h4tracks._train.grab_layer", fake_grab)
    monkeypatch.setattr(
        "mmv_h4tracks._train.prepare_cellpose_training_export_dir", fake_prepare
    )

    result = export_cellpose_training_pairs(
        viewer, "R", "S", [0, 1], "my model"
    )
    assert result.export_dir == tmp_path / (
        "mmv_h4tracks_train_" + _sanitize_model_name_fragment("my model")
    )
    assert (result.export_dir / "R_frame_00000.tif").is_file()
    assert (result.export_dir / "R_frame_00000_masks.tif").is_file()
    assert (result.export_dir / "R_frame_00001.tif").is_file()
    assert (result.export_dir / "R_frame_00001_masks.tif").is_file()

    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not installed")
    raw0 = tifffile.imread(result.export_dir / "R_frame_00000.tif")
    assert raw0.ndim == 3 and raw0.shape[0] == 2


def test_find_cellpose_cli_weights_picks_newest(tmp_path):
    root = tmp_path / "exp"
    md = root / "models"
    md.mkdir(parents=True)
    old = md / "cellpose_0.0"
    new = md / "cellpose_1.1"
    old.write_bytes(b"1")
    time.sleep(0.02)
    new.write_bytes(b"2")
    picked = find_cellpose_cli_weights(root)
    assert picked.resolve() == new.resolve()


def test_find_cellpose_cli_weights_no_models_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="models/"):
        find_cellpose_cli_weights(tmp_path)


def test_find_cellpose_cli_weights_empty_models(tmp_path):
    (tmp_path / "models").mkdir()
    with pytest.raises(FileNotFoundError, match="No cellpose_"):
        find_cellpose_cli_weights(tmp_path)


def test_prune_cellpose_training_export_dir_keeps_only_masks(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    (root / "L_frame_00000.tif").write_bytes(b"r")
    (root / "L_frame_00000_masks.tif").write_bytes(b"m")
    (root / "L_frame_00000_flows.tif").write_bytes(b"f")
    (root / "L_frame_0.tif").write_bytes(b"x")
    (root / "junk.txt").write_text("x")
    junk_dir = root / "other"
    junk_dir.mkdir()
    (junk_dir / "a").write_text("b")
    models = root / "models"
    models.mkdir()
    (models / "cellpose_0.0").write_bytes(b"w")

    prune_cellpose_training_export_dir(root)

    assert (root / "L_frame_00000_masks.tif").is_file()
    assert not (root / "L_frame_00000.tif").exists()
    assert not (root / "L_frame_00000_flows.tif").exists()
    assert not (root / "models").exists()
    assert not (root / "junk.txt").exists()
    assert not (root / "L_frame_0.tif").exists()
    assert not junk_dir.exists()


def test_prepare_cellpose_training_export_dir_removes_existing(tmp_path, monkeypatch):
    from mmv_h4tracks._train import prepare_cellpose_training_export_dir

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    d0 = prepare_cellpose_training_export_dir("alpha")
    (d0 / "stale.txt").write_text("x")
    d1 = prepare_cellpose_training_export_dir("alpha")
    assert d0 == d1
    assert not (d1 / "stale.txt").exists()


def test_classify_mmvh4tracks_training_dir(tmp_path):
    root = tmp_path / "mmv_h4tracks_train_x"
    root.mkdir()
    assert classify_mmvh4tracks_training_dir(root) == "empty"

    (root / "L_frame_00000_masks.tif").write_bytes(b"m")
    assert classify_mmvh4tracks_training_dir(root) == "masks_only"

    root_inc = tmp_path / "mmv_h4tracks_train_inc"
    root_inc.mkdir()
    (root_inc / "L_frame_00000_masks.tif").write_bytes(b"m")
    (root_inc / "stale.txt").write_text("x")
    assert classify_mmvh4tracks_training_dir(root_inc) == "incomplete"

    (root / "L_frame_00000.tif").write_bytes(b"r")
    assert classify_mmvh4tracks_training_dir(root) == "interrupted"

    (root / "L_frame_00000_flows.tif").write_bytes(b"f")
    assert classify_mmvh4tracks_training_dir(root) == "interrupted"


def test_parse_layer_prefix_and_frames_from_masks_dir(tmp_path):
    root = tmp_path / "d"
    root.mkdir()
    (root / "GFP_frame_00010_masks.tif").write_bytes(b"a")
    (root / "GFP_frame_00002_masks.tif").write_bytes(b"b")
    pfx, frames = parse_layer_prefix_and_frames_from_masks_dir(root)
    assert pfx == "GFP"
    assert frames == (2, 10)


def test_is_custom_model_display_name_taken_key_only():
    from mmv_h4tracks._processing import is_custom_model_display_name_taken

    class W:
        def __init__(self):
            self.custom_models = {}

    w = W()
    assert not is_custom_model_display_name_taken(w, "fresh")
    w.custom_models["used"] = {}
    assert is_custom_model_display_name_taken(w, "used")


def test_persist_custom_model_entry_json_key_matches_filename(tmp_path, monkeypatch):
    """``custom_models`` key and ``filename`` field are both the canonical basename."""
    import mmv_h4tracks._processing as proc

    fake_pkg = tmp_path / "mmv_h4tracks"
    fake_pkg.mkdir(parents=True)
    (fake_pkg / "models" / "custom_models").mkdir(parents=True)
    monkeypatch.setattr(proc, "__file__", str(fake_pkg / "_processing.py"))
    source = tmp_path / "w.pth"
    source.write_bytes(b"x")
    widget = Mock()
    widget.custom_models = {}
    proc.persist_custom_model_entry(widget, "a b", source, {})
    canonical = proc.custom_model_weights_basename("a b")
    assert canonical == "a_b"
    assert widget.custom_models[canonical]["filename"] == canonical
    assert (fake_pkg / "models" / "custom_models" / canonical).is_file()
