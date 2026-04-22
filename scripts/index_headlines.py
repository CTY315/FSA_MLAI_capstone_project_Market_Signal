"""One-time script to index S&P 500 headlines into ChromaDB.

Run from the project root:
    python scripts/index_headlines.py [--force]

--force  Drop the existing collection and rebuild from scratch.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import CHROMA_DIR, index_headlines, is_indexed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index S&P 500 headlines into ChromaDB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop the existing collection and rebuild from scratch.",
    )
    args = parser.parse_args()

    if is_indexed() and not args.force:
        logger.info(
            "ChromaDB collection already exists at %s. "
            "Pass --force to rebuild.",
            CHROMA_DIR,
        )
        return

    logger.info("Starting headline indexing (this may take a few minutes)…")
    t0 = time.time()
    n = index_headlines(force=args.force)
    elapsed = time.time() - t0
    logger.info("Done. Indexed %d headlines in %.1fs.", n, elapsed)
    logger.info("ChromaDB stored at: %s", CHROMA_DIR.resolve())


if __name__ == "__main__":
    main()
