"""Convert test TIFF/TIF images to .npy for fast pytest loading.

Not collected by pytest (filename does not match ``test_*.py`` / ``*_test.py``).

Canonical visual assets remain the ``.tif`` / ``.tiff`` files (open in napari).
After editing a TIFF, regenerate the sibling ``.npy``::

    python src/mmv_h4tracks/_tests/data/generate_numpy.py

Arrays are saved in BioImage ``ZYX`` order so they match previous test loads.
Existing ``tracks/*.npy`` files are left unchanged (already numpy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bioio import BioImage

DATA_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".tif", ".tiff"}
TARGET_SUBDIRS = ("images", "segmentation")


def convert_tree() -> list[Path]:
    written: list[Path] = []
    for sub in TARGET_SUBDIRS:
        folder = DATA_ROOT / sub
        if not folder.is_dir():
            print(f"Skip missing dir: {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            out = path.with_suffix(".npy")
            arr = np.asarray(BioImage(path).get_image_data("ZYX"))
            np.save(out, arr)
            written.append(out)
            print(f"Wrote {out.relative_to(DATA_ROOT)} shape={arr.shape} dtype={arr.dtype}")
    return written


if __name__ == "__main__":
    paths = convert_tree()
    print(f"Done: {len(paths)} file(s).")
