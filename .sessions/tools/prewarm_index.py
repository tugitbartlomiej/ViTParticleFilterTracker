#!/usr/bin/env python3
"""
Pre-warm Hybrid Sessions Index (BGE-M3 + Qdrant + reranker)
===========================================================
Loads models (GPU if available) and rebuilds/refreshes the Qdrant index.
Run once before heavy use to avoid first-query latency.

Usage:
    .sessions/venv/Scripts/python.exe .sessions/tools/prewarm_index.py
"""

import os
import sys

# Project root + importable tools dir
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, TOOLS_DIR)

import hybrid_rag  # noqa: E402

print("=" * 60)
print("PRE-WARMING HYBRID SESSIONS INDEX")
print("=" * 60)

print("\n[1/3] Loading embedder (BGE-M3, dense+sparse, GPU if available)...")
hybrid_rag.get_embed_model()
print("      Embedder loaded!")

print("\n[2/3] Loading reranker (bge-reranker-v2-m3)...")
hybrid_rag.get_reranker()
print("      Reranker loaded!")

print("\n[3/3] Checking/rebuilding Qdrant index...")
hybrid_rag.ensure_fresh()

state = hybrid_rag.index_status()
print("\n" + "=" * 60)
print("INDEX READY!")
print("=" * 60)
print(f"  Sessions: {state.get('sessions', '?')}")
print(f"  Chunks:   {state.get('chunks', '?')}")
print(f"  Hash:     {str(state.get('hash', '?'))[:16]}...")
print(f"  Fresh:    {state.get('is_fresh')}")
print("=" * 60)
