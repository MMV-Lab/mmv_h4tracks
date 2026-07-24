"""User-scoped custom Cellpose model registry (JSON + weights).

Bundled models stay under the package ``models/`` directory. User-added /
trained custom models are stored outside the install tree so writable
installs and upgrades do not fight site-packages.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent

_default_store: "CustomModelStore | None" = None


def package_models_dir() -> Path:
    """Directory of shipped (hardcoded) Cellpose weights."""
    return PACKAGE_ROOT / "models"


def default_user_data_dir() -> Path:
    """
    Root for user custom models.

    Override with env ``MMV_H4TRACKS_USER_DATA`` (useful in tests).
    """
    env = os.environ.get("MMV_H4TRACKS_USER_DATA")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".mmv_h4tracks"


class CustomModelStore:
    """Load/save ``custom_models.json`` and weight files under a data root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_user_data_dir()
        self.json_path = self.root / "custom_models.json"
        self.weights_dir = self.root / "custom_models"

    def ensure_dirs(self) -> None:
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def weights_path(self, filename: str) -> Path:
        return self.weights_dir / filename

    def list_weight_filenames(self) -> set[str]:
        if not self.weights_dir.is_dir():
            return set()
        return {p.name for p in self.weights_dir.iterdir() if p.is_file()}

    def load(self) -> dict:
        """Return the registry dict, migrating legacy package entries if needed."""
        self.ensure_dirs()
        models = self._read_json(self.json_path)
        models = self._migrate_legacy_package_models(models)
        return models

    def save(self, models: dict) -> None:
        self.ensure_dirs()
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(models, file, indent=2)

    def persist(
        self,
        display_name: str,
        source_weights: Path,
        params: dict,
        models: dict,
        *,
        canonical: str,
    ) -> Path:
        """
        Copy weights into the store and update ``models`` + JSON.

        ``canonical`` is both the JSON key and the on-disk basename.
        """
        self.ensure_dirs()
        dest_path = self.weights_path(canonical)
        shutil.copy2(source_weights, dest_path)
        models[canonical] = {"filename": canonical, "params": params}
        self.save(models)
        return dest_path

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read custom models JSON at %s: %s", path, exc)
            return {}

    def _migrate_legacy_package_models(self, models: dict) -> dict:
        """
        Copy custom models that still live under the package tree into the user store.

        Existing user keys are left unchanged. Package files are not deleted.
        """
        pkg_json = PACKAGE_ROOT / "custom_models.json"
        pkg_weights = PACKAGE_ROOT / "models" / "custom_models"
        pkg_models = self._read_json(pkg_json)
        if not pkg_models:
            return models

        changed = False
        for key, entry in pkg_models.items():
            if key in models:
                continue
            if not isinstance(entry, dict):
                continue
            filename = entry.get("filename", key)
            src = pkg_weights / filename
            if not src.is_file():
                continue
            dest = self.weights_path(filename)
            try:
                if not dest.is_file():
                    shutil.copy2(src, dest)
                models[key] = {
                    "filename": filename,
                    "params": dict(entry.get("params") or {}),
                }
                changed = True
            except OSError as exc:
                logger.warning(
                    "Failed to migrate custom model %s from package: %s", key, exc
                )
        if changed:
            self.save(models)
            logger.info("Migrated legacy custom models into %s", self.root)
        return models


def get_custom_model_store() -> CustomModelStore:
    global _default_store
    if _default_store is None:
        _default_store = CustomModelStore()
    return _default_store


def set_custom_model_store(store: CustomModelStore | None) -> None:
    """Replace the process-wide store (tests) or reset with ``None``."""
    global _default_store
    _default_store = store
