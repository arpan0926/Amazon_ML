"""Compatibility entry point for the packaged entity-resolution pipeline."""

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SOURCE_DIR))

from pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()