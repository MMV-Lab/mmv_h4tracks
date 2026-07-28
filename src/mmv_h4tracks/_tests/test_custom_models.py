"""Tests for the user-scoped custom model store."""

from __future__ import annotations

import json

from mmv_h4tracks._custom_models import (
    CustomModelStore,
    get_custom_model_store,
    set_custom_model_store,
)


def test_store_persist_and_load(tmp_path):
    store = CustomModelStore(tmp_path / "data")
    set_custom_model_store(store)
    try:
        models: dict = {}
        source = tmp_path / "weights.bin"
        source.write_bytes(b"abc")
        dest = store.persist(
            "My Model",
            source,
            {"diameter": 12},
            models,
            canonical="My_Model",
        )
        assert dest.is_file()
        assert dest.read_bytes() == b"abc"
        assert models["My_Model"]["filename"] == "My_Model"
        assert store.json_path.is_file()

        set_custom_model_store(None)
        set_custom_model_store(CustomModelStore(tmp_path / "data"))
        loaded = get_custom_model_store().load()
        assert "My_Model" in loaded
        assert loaded["My_Model"]["params"]["diameter"] == 12
    finally:
        set_custom_model_store(None)


def test_migrate_legacy_package_models(tmp_path, monkeypatch):
    """Entries still under the package tree are copied into the user store once."""
    fake_pkg = tmp_path / "pkg"
    pkg_weights = fake_pkg / "models" / "custom_models"
    pkg_weights.mkdir(parents=True)
    (pkg_weights / "legacy").write_bytes(b"w")
    pkg_json = fake_pkg / "custom_models.json"
    pkg_json.write_text(
        json.dumps({"legacy": {"filename": "legacy", "params": {"diameter": 9}}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "mmv_h4tracks._custom_models.PACKAGE_ROOT",
        fake_pkg,
    )

    store = CustomModelStore(tmp_path / "user")
    models = store.load()
    assert "legacy" in models
    assert store.weights_path("legacy").is_file()
    assert store.weights_path("legacy").read_bytes() == b"w"
    # Package copy left in place.
    assert (pkg_weights / "legacy").is_file()


def test_env_user_data_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MMV_H4TRACKS_USER_DATA", str(tmp_path / "env_root"))
    set_custom_model_store(None)
    store = get_custom_model_store()
    try:
        assert store.root == tmp_path / "env_root"
    finally:
        set_custom_model_store(None)
