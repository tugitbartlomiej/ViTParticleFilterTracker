#!/usr/bin/env python3
"""
Pre-warm Sessions MCP Index
============================
Run this once before using MCP semantic search to avoid timeout.

Usage:
    .sessions/venv/Scripts/python.exe .sessions/tools/prewarm_index.py
"""

import sys
import os

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("=" * 60)
print("PRE-WARMING SESSIONS MCP INDEX")
print("=" * 60)

# Import server module
sys.path.insert(0, '.sessions/tools')
import sessions_mcp_server as srv

print("\n[1/3] Loading embedding model (bge-m3, ~30s)...")
model = srv.get_model()
print("      Model loaded!")

print("\n[2/3] Loading ChromaDB collection...")
collection = srv.get_collection()
print(f"      Collection ready ({collection.count()} chunks)")

print("\n[3/3] Checking/rebuilding index...")
srv.ensure_index_fresh()

# Final status
state = srv.get_index_state()
print("\n" + "=" * 60)
print("INDEX READY!")
print("=" * 60)
print(f"  Sessions: {state.get('sessions', '?')}")
print(f"  Chunks:   {state.get('chunks', '?')}")
print(f"  Hash:     {state.get('hash', '?')[:16]}...")
print("\nMCP semantic search is now ready to use!")
print("=" * 60)
