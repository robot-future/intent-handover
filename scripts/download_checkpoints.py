"""Compatibility wrapper; installed users can run intent-handover download-weights."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from intent_handover.cli import main

if __name__ == "__main__":
    main(["download-weights", *sys.argv[1:]])
