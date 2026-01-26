#!/usr/bin/env python3
"""
Semantic Search over Sessions (bge-m3)
======================================
Standalone script for semantic search - bypasses MCP timeout issues.

Usage:
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "query"
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "DETR training" --top 10
    .sessions/venv/Scripts/python.exe .sessions/tools/search_sessions.py "IEEE article" --json
"""

import argparse
import io
import json
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Set working directory to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, '.sessions/tools')

import sessions_mcp_server as srv


def search(query: str, top_k: int = 5, output_json: bool = False):
    """Run semantic search and print results."""

    # Load model and collection
    if not output_json:
        print("Loading bge-m3 model...", file=sys.stderr)
    model = srv.get_model()
    collection = srv.get_collection()

    # Ensure index is fresh
    srv.ensure_index_fresh()

    # Embed query
    if not output_json:
        print(f"Searching: {query}", file=sys.stderr)
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2,  # Get more to deduplicate
        include=['documents', 'metadatas', 'distances']
    )

    if not results['ids'][0]:
        if output_json:
            print(json.dumps({"query": query, "results": []}))
        else:
            print("No results found.")
        return

    # Deduplicate by session_id
    seen = {}
    for i, doc_id in enumerate(results['ids'][0]):
        meta = results['metadatas'][0][i]
        session_id = meta['session_id']
        distance = results['distances'][0][i]
        score = 1 - distance

        if session_id not in seen or score > seen[session_id]['score']:
            seen[session_id] = {
                'session_id': session_id,
                'title': meta.get('title', 'Unknown'),
                'date': meta.get('date', 'Unknown'),
                'type': meta.get('type', 'Unknown'),
                'tags': meta.get('tags', ''),
                'score': round(score, 3),
                'snippet': results['documents'][0][i][:200] + '...'
            }

    # Sort by score
    sorted_results = sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:top_k]

    if output_json:
        print(json.dumps({"query": query, "results": sorted_results}, ensure_ascii=False, indent=2))
    else:
        print("\n" + "=" * 70)
        print(f"WYNIKI WYSZUKIWANIA: {query}")
        print("=" * 70)

        for i, r in enumerate(sorted_results, 1):
            print(f"""
{i}. {r['title'][:65]}
   Session: {r['session_id']}
   Date: {r['date']} | Type: {r['type']} | Score: {r['score']}
   Tags: {r['tags'][:50]}""")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Semantic search over project sessions')
    parser.add_argument('query', help='Search query (Polish or English)')
    parser.add_argument('--top', '-n', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')

    args = parser.parse_args()
    search(args.query, args.top, args.json)


if __name__ == '__main__':
    main()
