#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sessions RAG Agent
==================
Interactive agent that answers questions about project sessions using RAG.

Features:
- Semantic search over all sessions
- Context-aware answers with citations
- Supports Polish and English
- TF-IDF based (lightweight, no GPU needed)

Usage:
    python sessions_rag_agent.py "Jakie problemy naprawiałem z YOLO?"
    python sessions_rag_agent.py "What was done in IEEE article sessions?"
    python sessions_rag_agent.py --interactive
"""

import argparse
import io
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Text processing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Configuration
# =============================================================================

SESSIONS_PATH = Path(__file__).parent.parent  # .sessions/
INDEX_CACHE_PATH = SESSIONS_PATH / "vectordb" / "tfidf_index.json"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SessionChunk:
    """A chunk of session content with metadata."""
    session_id: str
    session_name: str
    date: str
    session_type: str
    content: str
    chunk_index: int
    topics: list


@dataclass
class SearchResult:
    """Search result with relevance score."""
    session_id: str
    session_name: str
    date: str
    score: float
    snippet: str
    topics: list


# =============================================================================
# Session Parser
# =============================================================================

def parse_session(session_dir: Path) -> Optional[dict]:
    """Parse a session directory and extract metadata + content."""
    summary_path = session_dir / "SESSION_SUMMARY.md"
    readme_path = session_dir / "README.md"

    content = ""
    if summary_path.exists():
        content = summary_path.read_text(encoding='utf-8', errors='ignore')
    elif readme_path.exists():
        content = readme_path.read_text(encoding='utf-8', errors='ignore')
    else:
        return None

    # Extract metadata
    folder_name = session_dir.name

    # Parse date from folder name
    date_match = re.search(r'Session_(\d{4}-\d{2}-\d{2})_(\d{6})', folder_name)
    date = date_match.group(1) if date_match else "Unknown"

    # Extract description from folder name
    desc_match = re.search(r'Session_\d{4}-\d{2}-\d{2}_\d{6}_(.+)$', folder_name)
    name = desc_match.group(1).replace('_', ' ') if desc_match else folder_name

    # Parse type
    session_type = "Mixed"
    type_match = re.search(r'\*\*Type:\*\*\s*(.+?)(?:\n|$)', content)
    if type_match:
        type_str = type_match.group(1).strip()
        for t in ["Training", "Benchmark", "Analysis", "SSH", "Writing"]:
            if t in type_str:
                session_type = t
                break

    # Extract objective
    objective = ""
    obj_match = re.search(r'## Objective\n(.+?)(?=\n##|\Z)', content, re.DOTALL)
    if obj_match:
        objective = obj_match.group(1).strip()[:500]

    # Extract topics
    topics = set()
    topic_patterns = [
        r'\b(DETR|YOLO|DINO|SAM|EL2N|Fourier|K-Means|K-Center)\b',
        r'\b(IEEE|Article|Paper)\b',
        r'\b(Dataset|Selection|Pipeline|Training)\b',
        r'\b(Benchmark|SSH|Eden|GPU)\b',
    ]
    for pattern in topic_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        topics.update([m.upper() if len(m) <= 5 else m.title() for m in matches])

    return {
        "session_id": folder_name,
        "name": name,
        "date": date,
        "type": session_type,
        "objective": objective,
        "content": content,
        "topics": list(topics)
    }


def load_all_sessions() -> list[dict]:
    """Load all sessions from the sessions directory."""
    sessions = []
    for session_dir in SESSIONS_PATH.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith('Session_'):
            session = parse_session(session_dir)
            if session:
                sessions.append(session)
    return sorted(sessions, key=lambda s: s["date"], reverse=True)


# =============================================================================
# RAG Engine
# =============================================================================

class SessionsRAG:
    """RAG engine for session search and retrieval."""

    def __init__(self):
        self.sessions: list[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._loaded = False

    def load(self):
        """Load sessions and build TF-IDF index."""
        if self._loaded:
            return

        print("Loading sessions...", file=sys.stderr)
        self.sessions = load_all_sessions()
        print(f"Loaded {len(self.sessions)} sessions", file=sys.stderr)

        if not self.sessions:
            self._loaded = True
            return

        # Build TF-IDF index
        print("Building TF-IDF index...", file=sys.stderr)
        documents = []
        for s in self.sessions:
            # Create rich document for indexing
            doc = f"{s['name']}. {s['objective']}. Topics: {', '.join(s['topics'])}. {s['content'][:2000]}"
            documents.append(doc)

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        print("Index ready!", file=sys.stderr)
        self._loaded = True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic search over sessions."""
        self.load()

        if not self.sessions or self.vectorizer is None:
            return []

        # Transform query
        query_vec = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum threshold
                s = self.sessions[idx]
                results.append(SearchResult(
                    session_id=s["session_id"],
                    session_name=s["name"],
                    date=s["date"],
                    score=round(float(similarities[idx]), 3),
                    snippet=s["objective"][:200] if s["objective"] else s["content"][:200],
                    topics=s["topics"][:5]
                ))

        return results

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get full session by ID."""
        self.load()
        for s in self.sessions:
            if s["session_id"] == session_id or session_id in s["session_id"]:
                return s
        return None

    def get_context(self, query: str, max_sessions: int = 3) -> str:
        """Get context from relevant sessions for RAG."""
        results = self.search(query, top_k=max_sessions)

        if not results:
            return "Nie znaleziono powiązanych sesji."

        context_parts = []
        for r in results:
            session = self.get_session(r.session_id)
            if session:
                context_parts.append(f"""
### Sesja: {r.session_name} ({r.date})
**Typ:** {session['type']} | **Tematy:** {', '.join(r.topics)}
**Score:** {r.score}

{session['content'][:1500]}

---
""")

        return "\n".join(context_parts)

    def answer(self, question: str) -> str:
        """Generate an answer with context from sessions."""
        self.load()

        # Search for relevant sessions
        results = self.search(question, top_k=5)

        if not results:
            return """
## Wynik wyszukiwania

Nie znaleziono sesji pasujących do zapytania.

Spróbuj użyć innych słów kluczowych lub sprawdź dostępne tematy:
- DETR, YOLO, DINO, SAM
- IEEE, Article, Paper
- Training, Benchmark, Pipeline
- Dataset Selection, EL2N, Fourier
- SSH, Eden, GPU
"""

        # Build response with context
        response = f"""
## Znalezione sesje dla: "{question}"

"""
        for i, r in enumerate(results, 1):
            session = self.get_session(r.session_id)
            type_marker = {
                "Training": "[TRAIN]",
                "Analysis": "[ANALYSIS]",
                "Writing": "[WRITE]",
                "Benchmark": "[BENCH]",
                "SSH": "[SSH]",
                "Mixed": "[MIX]"
            }.get(session["type"] if session else "Mixed", "[?]")

            response += f"""
### {i}. {type_marker} {r.session_name}
**Data:** {r.date} | **Typ:** {session['type'] if session else 'Unknown'} | **Score:** {r.score}
**Tematy:** {', '.join(r.topics)}

> {r.snippet}...

**Pełna ścieżka:** `.sessions/{r.session_id}/SESSION_SUMMARY.md`

"""

        # Add summary of key findings
        all_topics = set()
        for r in results:
            all_topics.update(r.topics)

        response += f"""
---

## Podsumowanie

Znaleziono **{len(results)} sesji** powiązanych z zapytaniem.

**Główne tematy:** {', '.join(sorted(all_topics))}

**Aby zobaczyć szczegóły sesji:**
```
Read .sessions/<session_id>/SESSION_SUMMARY.md
```

**Aby wyszukać więcej:**
```python
python .sessions/tools/sessions_rag_agent.py "twoje zapytanie"
```
"""

        return response


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sessions RAG Agent - Search and answer questions about project sessions"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question or search query (Polish or English)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode - keep asking questions"
    )
    parser.add_argument(
        "--context", "-c",
        action="store_true",
        help="Return raw context instead of formatted answer"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    rag = SessionsRAG()

    if args.interactive:
        print("=" * 60)
        print("  SESSIONS RAG AGENT - Interactive Mode")
        print("=" * 60)
        print("Zadaj pytanie o sesje projektu (wpisz 'quit' aby wyjść)")
        print()

        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nDo widzenia!")
                break

            if not query or query.lower() in ['quit', 'exit', 'q']:
                print("Do widzenia!")
                break

            print()
            print(rag.answer(query))
            print()

    elif args.query:
        if args.json:
            results = rag.search(args.query, top_k=args.top_k)
            output = {
                "query": args.query,
                "results": [
                    {
                        "session_id": r.session_id,
                        "name": r.session_name,
                        "date": r.date,
                        "score": r.score,
                        "snippet": r.snippet,
                        "topics": r.topics
                    }
                    for r in results
                ]
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))

        elif args.context:
            print(rag.get_context(args.query))

        else:
            print(rag.answer(args.query))

    else:
        # No query - show stats
        rag.load()
        print(f"""
Sessions RAG Agent
==================
Załadowane sesje: {len(rag.sessions)}
Zakres dat: {rag.sessions[-1]['date'] if rag.sessions else 'N/A'} - {rag.sessions[0]['date'] if rag.sessions else 'N/A'}

Użycie:
  python {Path(__file__).name} "zapytanie"
  python {Path(__file__).name} --interactive
  python {Path(__file__).name} --json "query"

Przykłady zapytań:
  "Jakie problemy naprawiałem z YOLO?"
  "Co robiono w sesjach IEEE article?"
  "Kiedy trenowałem DETR na Eden?"
  "Pokaż sesje o dataset selection"
""")


if __name__ == "__main__":
    main()
