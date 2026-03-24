"""Tests for Cellpose training export helpers."""

import types

import numpy as np
import pytest
from unittest.mock import Mock

from mmv_h4tracks._session_trained_models import overlap_training_frames_with_stack
from mmv_h4tracks._train import (
    export_cellpose_training_pairs,
    list_training_pair_paths,
    parse_use_frames,
    run_cellpose_training,
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
    """Export creates paired raw/masks with matching names."""
    raw_vol = np.random.rand(2, 16, 16).astype(np.float32)
    seg_vol = np.zeros((2, 16, 16), dtype=np.uint16)
    seg_vol[:, 4:12, 4:12] = 1

    class V:
        layers = []

    viewer = V()
    export_dir = tmp_path / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    def fake_grab(v, name):
        if name == "R":
            return Mock(data=raw_vol)
        if name == "S":
            return Mock(data=seg_vol)
        raise AssertionError(name)

    def fake_mkdtemp(prefix=""):
        return str(export_dir)

    monkeypatch.setattr("mmv_h4tracks._train.grab_layer", fake_grab)
    monkeypatch.setattr("mmv_h4tracks._train.tempfile.mkdtemp", fake_mkdtemp)

    result = export_cellpose_training_pairs(
        viewer, "R", "S", [0, 1], "my model"
    )
    assert result.export_dir == export_dir
    assert (result.export_dir / "raw" / "frame_00000.tif").is_file()
    assert (result.export_dir / "masks" / "frame_00000.tif").is_file()
    assert (result.export_dir / "raw" / "frame_00001.tif").is_file()
    assert (result.export_dir / "masks" / "frame_00001.tif").is_file()

    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not installed")
    raw0 = tifffile.imread(result.export_dir / "raw" / "frame_00000.tif")
    assert raw0.ndim == 3 and raw0.shape[0] == 2


def test_list_training_pair_paths(tmp_path):
    raw = tmp_path / "raw"
    masks = tmp_path / "masks"
    raw.mkdir()
    masks.mkdir()
    (raw / "frame_00001.tif").write_bytes(b"a")
    (masks / "frame_00001.tif").write_bytes(b"b")
    (raw / "frame_00000.tif").write_bytes(b"a")
    (masks / "frame_00000.tif").write_bytes(b"b")
    tf, lf = list_training_pair_paths(tmp_path)
    assert len(tf) == 2
    assert Path(tf[0]).name == "frame_00000.tif"
    assert Path(lf[0]).name == "frame_00000.tif"


def test_list_training_pair_paths_missing_mask(tmp_path):
    raw = tmp_path / "raw"
    masks = tmp_path / "masks"
    raw.mkdir()
    masks.mkdir()
    (raw / "a.tif").write_bytes(b"x")
    with pytest.raises(ValueError, match="Missing matching mask"):
        list_training_pair_paths(tmp_path)


def test_run_cellpose_training_mocked(tmp_path, monkeypatch):
    """run_cellpose_training calls train_seg with min_train_masks=1 and returns weights path."""
    raw = tmp_path / "export" / "raw"
    masks = tmp_path / "export" / "masks"
    raw.mkdir(parents=True)
    masks.mkdir(parents=True)
    (raw / "frame_00000.tif").write_bytes(b"r")
    (masks / "frame_00000.tif").write_bytes(b"m")

    ctr = tmp_path / "ctr"
    ctr.mkdir()
    (ctr / "models").mkdir()
    wp = ctr / "models" / "fin.pth"
    wp.write_bytes(b"w")

    def fake_mkdtemp(prefix=""):
        return str(ctr)

    fake_net = Mock()
    fake_net.diam_labels = Mock(mean=Mock(return_value=Mock(item=Mock(return_value=22.5))))

    def fake_cellpose_model(**kwargs):
        m = Mock()
        m.net = fake_net
        return m

    captured = {}

    def fake_train_seg(net, **kwargs):
        captured["kwargs"] = kwargs
        return (str(wp), np.array([1.0]), np.array([2.0]))

    monkeypatch.setattr("mmv_h4tracks._train.tempfile.mkdtemp", fake_mkdtemp)
    monkeypatch.setattr("cellpose.models.CellposeModel", fake_cellpose_model)
    monkeypatch.setattr("mmv_h4tracks._train.use_gpu_for_cellpose", lambda: False)
    monkeypatch.setattr(
        "mmv_h4tracks._train._import_cellpose_train",
        lambda: types.SimpleNamespace(train_seg=fake_train_seg),
    )

    result = run_cellpose_training(tmp_path / "export", "my model", None)
    assert result.weights_path.resolve() == wp.resolve()
    assert result.diam_mean == 22.5
    assert captured["kwargs"].get("min_train_masks") == 1
