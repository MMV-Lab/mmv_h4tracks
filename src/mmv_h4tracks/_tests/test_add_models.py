"""Tests for the "Add custom model" dialog."""

from mmv_h4tracks.add_models import ModelWindow


def test_add_model_with_blank_diameter_omits_it(create_widget, tmp_path):
    """
    Cellpose falls back to the diameter it was trained with when none is
    given (as already relied on for the built-in "cpsam" model), so a blank
    diameter field must not be forced into a float and must not raise.
    """
    widget = create_widget
    seg = widget.segmentation_window

    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")

    window = ModelWindow(seg)
    window.lineedit_name.setText("blank_diameter_model")
    window.model_path = str(source)
    window.lineedit_diameter.setText("")

    window.add_model()

    entry = seg.custom_models["blank_diameter_model"]
    assert "diameter" not in entry["params"]


def test_add_model_with_diameter_keeps_it(create_widget, tmp_path):
    widget = create_widget
    seg = widget.segmentation_window

    source = tmp_path / "weights.bin"
    source.write_bytes(b"weights")

    window = ModelWindow(seg)
    window.lineedit_name.setText("explicit_diameter_model")
    window.model_path = str(source)
    window.lineedit_diameter.setText("17.5")

    window.add_model()

    entry = seg.custom_models["explicit_diameter_model"]
    assert entry["params"]["diameter"] == 17.5
