#!/usr/bin/env python3
"""
Index / reindex the session RAG base (BGE-M3 + Qdrant, contextual chunking)
===========================================================================
Builds or updates the local hybrid RAG over every .md / .txt file under
.sessions/ (recursive, technical dirs excluded), with section-aware contextual
chunking. Embeddings + ColBERT run on GPU when available.

Usage:
    PY = .sessions/venv/Scripts/python.exe

    # Full (re)build from scratch  -- run once, or to rebuild everything
    PY .sessions/tools/index_sessions.py --full

    # Incremental update           -- add/update/remove only changed files
    PY .sessions/tools/index_sessions.py --update

    # Show index status
    PY .sessions/tools/index_sessions.py --stats

Notes:
- Set HF_HOME=F:\\HuggingFace so models load from the shared cache.
- Qdrant local mode is single-writer: stop the `sessions` MCP server before
  running this (and vice-versa).
"""

import argparse
import io
import os
import sys

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, TOOLS_DIR)

import hybrid_rag  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Index/reindex the session RAG base")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--full", "--rebuild", action="store_true",
                   help="Full rebuild from scratch")
    g.add_argument("--update", action="store_true",
                   help="Incremental update (changed files only)")
    g.add_argument("--stats", action="store_true", help="Show index status")
    args = parser.parse_args()

    if args.stats:
        st = hybrid_rag.index_status()
        print("Index status:")
        print(f"  chunks:    {st['chunks']}")
        print(f"  documents: {st['docs']}")
        print(f"  model:     {st['hash']}")
        print(f"  fresh:     {st['is_fresh']}")
        return

    if args.full:
        print("=== FULL REBUILD ===")
        res = hybrid_rag.build_full()
    else:
        # default action is an incremental update (safe + fast)
        print("=== INCREMENTAL UPDATE ===")
        res = hybrid_rag.update_incremental()

    print("\nDone:")
    print(f"  chunks:  {res['chunks']}  docs: {res['docs']}")
    print(f"  added:   {res['added']}  updated: {res['updated']}  removed: {res['removed']}")


if __name__ == "__main__":
    main()
