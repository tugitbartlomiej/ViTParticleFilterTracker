#!/usr/bin/env python3
"""
Sessions RAG MCP Server
========================
Semantic search over project sessions using embeddings and vector database.

Features:
- Hybrid search: BGE-M3 dense + sparse -> RRF -> bge-reranker-v2-m3 (see hybrid_rag.py)
- Multilingual (Polish & English), GPU-accelerated when available
- Auto-rebuild index when sessions change
- Lazy loading for fast startup
- Qdrant (local/embedded) for persistent vector storage

Note: the legacy ChromaDB helpers (get_model/get_collection/_do_rebuild_index)
are retained but no longer used; the active path delegates to hybrid_rag.

Usage:
    # Add to Claude Code
    claude mcp add --transport stdio sessions -- python .sessions/tools/sessions_mcp_server.py

    # Or use .mcp.json in project root

Requirements:
    pip install -r .sessions/tools/requirements_mcp.txt
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Configuration
SESSIONS_PATH = Path(__file__).parent.parent  # .sessions/
PROJECT_ROOT = SESSIONS_PATH.parent  # ViTParticleFilterTracker/
MODELS_CACHE = PROJECT_ROOT / "External" / "Models"  # legacy local models cache (fallback only)

# Embedding model configuration
# bge-m3: 2.3GB, 1024-dim, multilingual (PL+EN), loads in ~60s
# Note: MCP has timeout issues, use /szukaj-sesje command instead
#
# Default: load from the shared HuggingFace cache (HF_HOME, e.g. F:\HuggingFace)
# by repo id. "BAAI/bge-m3" resolves to <HF_HOME>\hub\models--BAAI--bge-m3 with no
# re-download. Override with the SESSIONS_EMBED_MODEL env var (repo id or local path).
EMBEDDING_MODEL = os.environ.get("SESSIONS_EMBED_MODEL", "BAAI/bge-m3")
# Legacy local copy in External/Models; used only if the HF-cache load fails.
LEGACY_LOCAL_MODEL_PATH = MODELS_CACHE / "BAAI_bge-m3"

CHROMA_PATH = SESSIONS_PATH / "vectordb"  # ChromaDB index location
INDEX_STATE_FILE = CHROMA_PATH / "index_state.json"

# Chunk settings
CHUNK_SIZE = 400  # words per chunk
CHUNK_OVERLAP = 50  # overlapping words

# FastMCP server
mcp = FastMCP(
    "Sessions RAG Server",
    instructions="Semantic search over ViTParticleFilterTracker project sessions"
)

# Lazy-loaded globals
_model = None
_collection = None
_initialized = False


# =============================================================================
# Pydantic Models
# =============================================================================

class SessionSearchResult(BaseModel):
    """Result from semantic search."""
    session_id: str = Field(description="Session folder name")
    title: str = Field(description="Session objective/title")
    score: float = Field(description="Similarity score (0-1, higher is better)")
    date: str = Field(description="Session date")
    snippet: str = Field(description="Relevant text snippet")


class SessionMetadata(BaseModel):
    """Basic session metadata."""
    session_id: str = Field(description="Session folder name")
    date: str = Field(description="Session date")
    type: str = Field(description="Session type (Training, Analysis, etc.)")
    status: str = Field(description="Session status (Completed, In Progress, etc.)")
    tags: list[str] = Field(default_factory=list, description="Auto-detected tags")


class IndexStatus(BaseModel):
    """Status of the vector index."""
    total_chunks: int = Field(description="Total indexed chunks")
    total_sessions: int = Field(description="Total indexed sessions")
    last_hash: str = Field(description="Last content hash")
    is_fresh: bool = Field(description="Whether index is up to date")


# =============================================================================
# Lazy Loading Functions
# =============================================================================

def get_model():
    """Lazy load the bge-m3 embedding model (multilingual: PL+EN).

    Loads from the shared HuggingFace cache (HF_HOME) by repo id, so the same
    on-disk model is reused across projects. Falls back to the legacy local copy
    in External/Models only if the HF-cache load fails.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        hf_home = os.environ.get("HF_HOME", "<HF default>")
        try:
            print(f"Loading embedding model '{EMBEDDING_MODEL}' from HF cache (HF_HOME={hf_home})")
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as exc:
            if LEGACY_LOCAL_MODEL_PATH.exists():
                print(f"HF-cache load failed ({exc}); using legacy local copy: {LEGACY_LOCAL_MODEL_PATH}")
                _model = SentenceTransformer(str(LEGACY_LOCAL_MODEL_PATH))
            else:
                raise
        print("Model loaded!")
    return _model


def get_collection():
    """Lazy load ChromaDB collection."""
    global _collection
    if _collection is None:
        try:
            import chromadb
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_PATH))
            _collection = client.get_or_create_collection(
                name="sessions",
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            raise ImportError(
                "chromadb not installed. "
                "Run: pip install chromadb"
            )
    return _collection


# =============================================================================
# Index Management
# =============================================================================

def get_sessions_hash() -> str:
    """Calculate hash of all session files to detect changes."""
    files = sorted(SESSIONS_PATH.glob("Session_*/SESSION_SUMMARY.md"))
    content_parts = []
    for f in files:
        try:
            stat = f.stat()
            content_parts.append(f"{f.name}:{stat.st_mtime}:{stat.st_size}")
        except OSError:
            continue
    return hashlib.md5("|".join(content_parts).encode()).hexdigest()


def get_index_state() -> dict:
    """Load saved index state."""
    if INDEX_STATE_FILE.exists():
        try:
            return json.loads(INDEX_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_index_state(state: dict):
    """Save index state to disk."""
    INDEX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    INDEX_STATE_FILE.write_text(json.dumps(state, indent=2))


def ensure_index_fresh():
    """Ensure the hybrid Qdrant index is fresh (delegates to hybrid_rag)."""
    global _initialized
    if _initialized:
        return
    import hybrid_rag
    hybrid_rag.ensure_fresh()
    _initialized = True


def _do_rebuild_index():
    """Perform the actual index rebuild."""
    model = get_model()
    collection = get_collection()

    # Clear existing data
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    documents = []
    metadatas = []
    ids = []

    session_dirs = list(SESSIONS_PATH.glob("Session_*"))
    print(f"Indexing {len(session_dirs)} sessions...")

    for session_dir in session_dirs:
        summary_path = session_dir / "SESSION_SUMMARY.md"
        if not summary_path.exists():
            continue

        try:
            content = summary_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Extract rich metadata
        title = extract_title(content)
        date = extract_field(content, "Date") or "Unknown"
        session_type = extract_field(content, "Type") or "Unknown"
        tags = extract_tags(content)
        problems = extract_problems_solved(content)
        lessons = extract_lessons_learned(content)

        # Create enriched content for better RAG
        # Prepend important sections for better retrieval
        enriched_header = f"Session: {session_dir.name}\n"
        enriched_header += f"Date: {date}\n"
        enriched_header += f"Type: {session_type}\n"
        if tags:
            enriched_header += f"Tags: {', '.join(tags)}\n"
        if problems:
            enriched_header += f"Problems Solved: {'; '.join(problems[:3])}\n"
        if lessons:
            enriched_header += f"Lessons: {'; '.join(lessons[:3])}\n"
        enriched_header += "\n"

        # Chunk the content
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            doc_id = f"{session_dir.name}__chunk_{i}"
            # Add enriched header to first chunk only
            doc_content = (enriched_header + chunk) if i == 0 else chunk
            documents.append(doc_content)
            metadatas.append({
                "session_id": session_dir.name,
                "title": title,
                "date": date,
                "type": session_type,
                "tags": ",".join(tags),
                "chunk_index": i
            })
            ids.append(doc_id)

    if not documents:
        print("No sessions found to index!")
        return

    # Embed all documents
    print(f"Embedding {len(documents)} chunks...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        normalize_embeddings=True
    ).tolist()

    # Store in ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Indexed {len(documents)} chunks from {len(session_dirs)} sessions")


# =============================================================================
# Text Processing Helpers
# =============================================================================

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def extract_title(content: str) -> str:
    """Extract title/objective from session content."""
    # Try TL;DR section first (best for RAG)
    tldr_match = re.search(
        r"## TL;DR\s*\n(.+?)(?:\n\n|\n##)",
        content,
        re.DOTALL
    )
    if tldr_match:
        tldr = tldr_match.group(1).strip()
        # Skip HTML comments
        tldr = re.sub(r"<!--.*?-->", "", tldr, flags=re.DOTALL).strip()
        if tldr:
            return tldr[:200]

    # Try Objective section
    obj_match = re.search(
        r"## Objective\s*\n(.+?)(?:\n\n|\n##)",
        content,
        re.DOTALL
    )
    if obj_match:
        return obj_match.group(1).strip()[:150]

    # Try title from header
    title_match = re.search(r"^# .+?:\s*(.+)$", content, re.MULTILINE)
    if title_match:
        return title_match.group(1).strip()[:150]

    return "Unknown"


def extract_field(content: str, field: str) -> Optional[str]:
    """Extract a metadata field value."""
    # Try YAML frontmatter first
    yaml_match = re.search(rf"^{field.lower()}:\s*(.+?)$", content, re.MULTILINE)
    if yaml_match:
        return yaml_match.group(1).strip()

    # Try markdown bold format
    pattern = rf"\*\*{field}:\*\*\s*(.+?)(?:\n|$)"
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None


def extract_yaml_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter if present."""
    frontmatter_match = re.match(r"^---\s*\n(.+?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return {}

    frontmatter = {}
    for line in frontmatter_match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Parse lists
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
            frontmatter[key] = value

    return frontmatter


def extract_tags(content: str) -> list[str]:
    """Extract tags from YAML frontmatter and auto-detect from content."""
    tags = set()

    # First, try YAML frontmatter tags
    frontmatter = extract_yaml_frontmatter(content)
    if "tags" in frontmatter and isinstance(frontmatter["tags"], list):
        tags.update(frontmatter["tags"])

    # Extract explicit Keywords section
    keywords_match = re.search(r"## Keywords\s*\n(.+?)(?:\n\n|\n##|$)", content, re.DOTALL)
    if keywords_match:
        keywords_text = keywords_match.group(1)
        # Extract backtick-wrapped keywords
        explicit_keywords = re.findall(r"`([^`]+)`", keywords_text)
        tags.update(kw.lower() for kw in explicit_keywords)

    # Auto-detect common keywords
    auto_keywords = [
        "detr", "yolo", "training", "benchmark", "ssh", "eden",
        "ieee", "article", "pipeline", "dataset", "analysis",
        "k-means", "clustering", "el2n", "dino", "sam",
        "cataract", "error", "fix", "bug", "checkpoint",
        "slurm", "gpu", "cuda", "memory", "oom"
    ]

    content_lower = content.lower()
    for kw in auto_keywords:
        if kw in content_lower:
            tags.add(kw)

    return sorted(tags)


def extract_problems_solved(content: str) -> list[str]:
    """Extract problems solved section for better RAG."""
    problems = []

    # Try "Problems Solved" section
    section_match = re.search(
        r"## Problems Solved\s*\n(.+?)(?:\n##|$)",
        content,
        re.DOTALL
    )
    if section_match:
        section = section_match.group(1)
        # Extract numbered or bulleted items
        items = re.findall(r"(?:^|\n)\d+\.\s*\*\*(.+?)\*\*", section)
        problems.extend(items)

    return problems


def extract_lessons_learned(content: str) -> list[str]:
    """Extract lessons learned section."""
    lessons = []

    section_match = re.search(
        r"## Lessons Learned\s*\n(.+?)(?:\n##|$)",
        content,
        re.DOTALL
    )
    if section_match:
        section = section_match.group(1)
        # Extract list items
        items = re.findall(r"(?:^|\n)-\s*(.+?)(?:\n|$)", section)
        lessons.extend(item.strip() for item in items if item.strip())

    return lessons


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
def search_sessions(query: str, top_k: int = 5) -> list[SessionSearchResult]:
    """Semantic search over all project sessions.

    Use natural language to find relevant sessions, e.g.:
    - "sessions where I fixed DETR pipeline"
    - "kiedy pracowałem nad artykułem IEEE"
    - "problems with K-Means clustering"
    - "SSH sessions on Eden cluster"

    Args:
        query: Natural language search query (Polish or English)
        top_k: Maximum number of results to return (default: 5)

    Returns:
        List of matching sessions with relevance scores and snippets
    """
    import hybrid_rag
    results = hybrid_rag.search(query, top_k=top_k)
    return [
        SessionSearchResult(
            session_id=r["session_id"],
            title=r["title"],
            score=r["score"],
            date=r["date"],
            snippet=r["snippet"],
        )
        for r in results
    ]


@mcp.tool()
def get_session_content(session_id: str) -> str:
    """Get the full SESSION_SUMMARY.md content for a specific session.

    Args:
        session_id: Session folder name (e.g., "Session_2026-01-12_041738_IEEE_Pipeline_Rewrite")

    Returns:
        Full markdown content of the session summary
    """
    summary_path = SESSIONS_PATH / session_id / "SESSION_SUMMARY.md"

    if not summary_path.exists():
        # Try partial match
        matches = list(SESSIONS_PATH.glob(f"*{session_id}*"))
        if matches:
            summary_path = matches[0] / "SESSION_SUMMARY.md"

    if summary_path.exists():
        return summary_path.read_text(encoding="utf-8")

    return f"Session not found: {session_id}"


@mcp.tool()
def list_sessions(
    tag: Optional[str] = None,
    session_type: Optional[str] = None,
    limit: int = 20
) -> list[SessionMetadata]:
    """List all sessions with optional filtering.

    Args:
        tag: Filter by auto-detected tag (e.g., "detr", "training", "ieee")
        session_type: Filter by session type (e.g., "Training", "Analysis", "SSH")
        limit: Maximum number of results (default: 20)

    Returns:
        List of session metadata, sorted by date (newest first)
    """
    sessions = []

    session_dirs = sorted(
        SESSIONS_PATH.glob("Session_*"),
        key=lambda x: x.name,
        reverse=True
    )

    for session_dir in session_dirs:
        summary_path = session_dir / "SESSION_SUMMARY.md"
        if not summary_path.exists():
            continue

        try:
            content = summary_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Extract metadata
        date = extract_field(content, "Date") or "Unknown"
        stype = extract_field(content, "Type") or "Unknown"
        status = extract_field(content, "Status") or "Unknown"
        tags = extract_tags(content)

        # Apply filters
        if tag and tag.lower() not in [t.lower() for t in tags]:
            continue
        if session_type and session_type.lower() not in stype.lower():
            continue

        sessions.append(SessionMetadata(
            session_id=session_dir.name,
            date=date,
            type=stype,
            status=status,
            tags=tags
        ))

        if len(sessions) >= limit:
            break

    return sessions


@mcp.tool()
def get_index_status() -> IndexStatus:
    """Get the current status of the vector index.

    Returns:
        Index statistics including freshness status
    """
    import hybrid_rag
    st = hybrid_rag.index_status()
    return IndexStatus(
        total_chunks=st["chunks"],
        total_sessions=st["sessions"],
        last_hash=str(st["hash"])[:12] + "...",
        is_fresh=st["is_fresh"],
    )


@mcp.tool()
def rebuild_index() -> str:
    """Force rebuild the vector index.

    Use this after adding many new sessions or if search results seem stale.

    Returns:
        Status message with index statistics
    """
    global _initialized
    _initialized = False
    import hybrid_rag
    st = hybrid_rag.build_full()
    _initialized = True
    return f"Rebuilt index: {st.get('chunks', 0)} chunks from {st.get('docs', 0)} documents"


@mcp.tool()
def update_index() -> str:
    """Incrementally update the index (only new / changed / removed files).

    Faster than rebuild_index; use after saving new sessions. Adds and links
    new embeddings to the existing base without rebuilding everything.

    Returns:
        Status message with add / update / remove counts.
    """
    import hybrid_rag
    st = hybrid_rag.update_incremental()
    return (f"Updated index: +{st['added']} ~{st['updated']} -{st['removed']} "
            f"({st['chunks']} chunks, {st['docs']} docs)")


# =============================================================================
# MCP Resources
# =============================================================================

@mcp.resource("sessions://{session_id}/summary")
def get_session_resource(session_id: str) -> str:
    """Get session summary as MCP resource."""
    return get_session_content(session_id)


@mcp.resource("sessions://index/status")
def get_index_status_resource() -> str:
    """Get index status as MCP resource."""
    status = get_index_status()
    return json.dumps(status.model_dump(), indent=2)


# =============================================================================
# Pre-warm Model (avoid timeout on first query)
# =============================================================================

def _warmup():
    """Pre-load embedding model on server startup to avoid first-query timeout."""
    import sys
    print("=" * 50, file=sys.stderr)
    print("Sessions MCP Server - Warming up...", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    try:
        import hybrid_rag
        print("Loading embedder + reranker (GPU if available)...", file=sys.stderr)
        hybrid_rag.get_embed_model()
        hybrid_rag.get_reranker()
        print("Models loaded!", file=sys.stderr)

        print("Checking index freshness...", file=sys.stderr)
        hybrid_rag.ensure_fresh()
        print("Index ready!", file=sys.stderr)

        print("=" * 50, file=sys.stderr)
        print("Server ready to handle queries!", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
    except Exception as e:
        print(f"Warmup failed: {e}", file=sys.stderr)
        # Don't crash - allow lazy loading as fallback


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # _warmup()  # Disabled - causes MCP timeout. Model loads lazily on first query.
    mcp.run()
