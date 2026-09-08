#!/usr/bin/env python3
"""
Hybrid Semantic + Keyword Search over Sessions (BGE-M3 + Qdrant + reranker)
==========================================================================
Standalone CLI - bypasses MCP timeout issues. Delegates to hybrid_rag.

Pipeline: BGE-M3 dense + sparse -> RRF fusion -> bge-reranker-v2-m3 -> top_k.

Usage:
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "query"
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "DETR training" --top 10
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "IEEE article" --json

Note: Qdrant local mode is single-writer. Do not run this while the `sessions`
MCP server is actively holding the qdrant_db lock (use one or the other).
"""

import argparse
import io
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Resolve project root and make the tools dir importable
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TOOLS_DIR))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, TOOLS_DIR)

import hybrid_rag  # noqa: E402


def run(query: str, top_k: int = 5, output_json: bool = False):
    if not output_json:
        print("Loading BGE-M3 + reranker (hybrid)...", file=sys.stderr)
        print(f"Searching: {query}", file=sys.stderr)

    results = hybrid_rag.search(query, top_k=top_k)

    if output_json:
        print(json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2))
        return

    if not results:
        print("No results found.")
        return

    print("\n" + "=" * 70)
    print(f"WYNIKI WYSZUKIWANIA (hybryda): {query}")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        print(f"""
{i}. {r['title'][:70]}   [score {r['score']}]
   Sesja_folder: {r['session_id']}
   Plik:         {r['source']}{('   |   Sekcja: ' + r['section'][:55]) if r['section'] else ''}
   Wyniki: Rerank {r['score']} | ColBERT {r['colbert_score']} | RRF {r['rrf_score']}   (Data: {r['date']} | Type: {r['type'] or '-'})
   Tagi:   {r['tags'][:60] or '-'}
   Fragment: {r['snippet']}""")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Hybrid search over project sessions")
    parser.add_argument("query", help="Search query (Polish or English)")
    parser.add_argument("--top", "-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    run(args.query, args.top, args.json)


if __name__ == "__main__":
    main()
