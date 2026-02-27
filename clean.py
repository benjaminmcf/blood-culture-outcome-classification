"""Remove all pipeline outputs so the project can be tested end-to-end.

Deletes the contents of:
  - datasets/          (training & testing CSVs)
  - models/            (trained model .sav, .json files)
  - features/          (selected feature lists)
  - results/           (CV results, training report)
  - predictions/       (inference outputs, report)
  - exports/           (LR coefficients, DT rules, validation)

Usage
-----
    python clean.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRS_TO_CLEAN = [
    "datasets",
    "models",
    "features",
    "results",
    "predictions",
    "exports",
]


def main() -> None:
    print("Cleaning pipeline outputs...\n")
    for d in DIRS_TO_CLEAN:
        path = ROOT / d
        if path.exists():
            n_files = sum(1 for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path)
            print(f"  ✓ Removed {d}/ ({n_files} files)")
        else:
            print(f"  - {d}/ (not found, skipping)")
    print("\nDone! To start fresh, run:")
    print("  bcoc-train")
    print("  bcoc-infer")


if __name__ == "__main__":
    main()
