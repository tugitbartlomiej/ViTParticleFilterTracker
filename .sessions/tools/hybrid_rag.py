#!/usr/bin/env python3
"""
Hybrid RAG engine for project sessions (BGE-M3 + Qdrant + reranker + ColBERT)
============================================================================
Deluxe hybrid retrieval over the .sessions corpus (recursive Markdown/text):

    query
      -> BGE-M3 dense  (semantic)   -> Qdrant top-N
      -> BGE-M3 sparse (lexical)    -> Qdrant top-N
      -> Reciprocal Rank Fusion (RRF, in Python)              -> candidates
      -> bge-reranker-v2-m3 cross-encoder                     -> precision rerank
      -> BGE-M3 ColBERT MaxSim (computed on candidates only)  -> auxiliary signal
      -> top_k results (deduplicated per session)

Key design choices
------------------
- Corpus: every .md / .txt under .sessions/ (recursive), excluding technical dirs
  (venv, tools, vectordb, qdrant_db, ...). Mainly the saved session notes.
- Contextual chunking: markdown is split section-aware (by headings); every chunk
  is prefixed with "Document / Session / Section" context before embedding, so no
  chunk is anonymous (Anthropic "Contextual Retrieval", deterministic/local flavour).
- One embedder: BGEM3FlagModel gives dense + sparse + ColBERT in a single pass.
- ColBERT is computed at query time on the ~40 fused candidates only (no storage,
  no Qdrant multivector dependency) -> scales cleanly.
- Models load from the shared HuggingFace cache (HF_HOME) by repo id, on GPU when
  available (RTX 3070). Vector store: Qdrant local/embedded (no Docker).
- Incremental updates: per-file signature tracking -> add/update/delete only the
  changed files (see update_incremental / index_sessions.py --update).

NOTE: Qdrant local mode is single-writer. Only one process may hold qdrant_db at a
time (the MCP server OR a CLI, not both simultaneously).
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

# --- Paths / config ---------------------------------------------------------
SESSIONS_PATH = Path(__file__).parent.parent          # .sessions/
QDRANT_PATH = SESSIONS_PATH / "qdrant_db"             # local Qdrant storage
STATE_FILE = QDRANT_PATH / "index_state.json"

EMBEDDING_MODEL = os.environ.get("SESSIONS_EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("SESSIONS_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

COLLECTION = "sessions"
DENSE_DIM = 1024
DENSE_NAME = "dense"
SPARSE_NAME = "sparse"

# Corpus selection
INDEX_EXTENSIONS = {".md", ".txt"}     # add ".json" here if you want raw json indexed
# Technical dirs + generated/scaffolding content (obsidian_vault is a re-export of
# sessions -> would duplicate; templates are scaffolding; analysis is generated HTML).
EXCLUDE_DIRS = {"venv", "tools", "vectordb", "qdrant_db", "__pycache__",
                ".git", ".ipynb_checkpoints", "node_modules",
                "obsidian_vault", "templates", "analysis"}
MAX_FILE_BYTES = 2_000_000             # skip very large files

# Chunking
CHUNK_WORDS = 400
CHUNK_OVERLAP = 50

# Retrieval knobs
PREFETCH_LIMIT = 80      # candidates pulled from each of dense / sparse
CANDIDATE_LIMIT = 40     # candidates kept after RRF, fed to reranker / colbert
RRF_K = 60               # RRF dampening constant
MAX_EMBED_LEN = 1024     # max tokens per chunk when embedding
ENABLE_COLBERT = True    # compute ColBERT MaxSim on candidates
DEFAULT_SCORE_MODE = "rerank"   # 'rerank' | 'colbert' | 'blend'
BLEND_RERANK_W = 0.7     # weight of cross-encoder in 'blend' mode

# Lazy globals
_embed_model = None
_reranker = None
_client = None


def _log(*args):
    """Diagnostics go to stderr (stdout is reserved for JSON output / MCP protocol)."""
    sys.stderr.write(" ".join(str(a) for a in args) + "\n")


# =============================================================================
# Model / client loaders (lazy, GPU-aware, HF cache via HF_HOME)
# =============================================================================

def get_embed_model():
    """Load BGE-M3 (dense + sparse + colbert) from the HF cache, fp16 on GPU."""
    global _embed_model
    if _embed_model is None:
        import torch
        from FlagEmbedding import BGEM3FlagModel
        use_gpu = torch.cuda.is_available()
        hf_home = os.environ.get("HF_HOME", "<HF default>")
        _log(f"Loading embedder '{EMBEDDING_MODEL}' (GPU={use_gpu}, HF_HOME={hf_home})")
        _embed_model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=use_gpu)
    return _embed_model


def get_reranker():
    """Load bge-reranker-v2-m3 cross-encoder from the HF cache, GPU when available."""
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"Loading reranker '{RERANKER_MODEL}' (device={device})")
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
    return _reranker


def get_client():
    """Open the local Qdrant store (embedded, on-disk, no server)."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


# =============================================================================
# Embedding helpers
# =============================================================================

def _to_sparse_vector(lexical_weights: dict):
    """Convert BGE-M3 lexical_weights {token_id: weight} -> Qdrant SparseVector."""
    from qdrant_client import models
    indices, values = [], []
    for tok, w in lexical_weights.items():
        w = float(w)
        if w <= 0:
            continue
        indices.append(int(tok))
        values.append(w)
    if not indices:                      # avoid empty sparse vectors
        indices, values = [0], [1e-6]
    return models.SparseVector(indices=indices, values=values)


def _encode_dense_sparse(texts: list[str]):
    """Encode texts -> (dense_list[list[float]], sparse_list[SparseVector])."""
    model = get_embed_model()
    out = model.encode(
        texts, return_dense=True, return_sparse=True, return_colbert_vecs=False,
        max_length=MAX_EMBED_LEN,
    )
    dense = [vec.tolist() for vec in out["dense_vecs"]]
    sparse = [_to_sparse_vector(lw) for lw in out["lexical_weights"]]
    return dense, sparse


def _encode_colbert(texts: list[str]):
    """Encode texts -> list of ColBERT token-vector matrices (np arrays)."""
    model = get_embed_model()
    out = model.encode(
        texts, return_dense=False, return_sparse=False, return_colbert_vecs=True,
        max_length=MAX_EMBED_LEN,
    )
    return out["colbert_vecs"]


def _maxsim(q, d) -> float:
    """ColBERT MaxSim: sum over query tokens of max similarity to any doc token."""
    import numpy as np
    if q is None or d is None or getattr(q, "size", 0) == 0 or getattr(d, "size", 0) == 0:
        return 0.0
    sim = np.asarray(q) @ np.asarray(d).T          # (Tq, Td)
    return float(sim.max(axis=1).sum())


# =============================================================================
# Corpus + contextual chunking
# =============================================================================

def _iter_files():
    """Yield (rel_path, abs_path) for every indexable file under .sessions/."""
    for abs_path in SESSIONS_PATH.rglob("*"):
        if not abs_path.is_file():
            continue
        if abs_path.suffix.lower() not in INDEX_EXTENSIONS:
            continue
        rel = abs_path.relative_to(SESSIONS_PATH)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            if abs_path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield rel.as_posix(), abs_path


def _doc_meta(rel_path: str, abs_path: Path, content: str) -> dict:
    """Derive session_id / title / type / tags for a document."""
    parts = rel_path.split("/")
    # Prefer a "Session_*" ancestor (handles archived/nested sessions); else the
    # immediate parent dir; else the file stem.
    session_id = next((p for p in parts if p.startswith("Session_")), None)
    if not session_id:
        session_id = parts[-2] if len(parts) >= 2 else Path(parts[-1]).stem

    if abs_path.name == "SESSION_SUMMARY.md":
        # Reuse the rich extractors from the MCP server for session summaries
        import sessions_mcp_server as sms
        title = sms.extract_title(content)
        stype = sms.extract_field(content, "Type") or "Unknown"
        tags = ",".join(sms.extract_tags(content))
        date = sms.extract_field(content, "Date") or "Unknown"
    else:
        m = re.search(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)
        title = (m.group(1).strip() if m else abs_path.stem)[:160]
        stype = ""
        tags = ""
        date = "Unknown"
    return {"session_id": session_id, "title": title, "type": stype, "tags": tags, "date": date}


def _split_sections(content: str):
    """Split markdown into (breadcrumb, section_text) by headings, keeping a
    hierarchical breadcrumb (e.g. 'Results > Key Findings')."""
    sections = []
    stack = []          # list of (level, title)
    cur_lines, cur_crumb = [], ""

    def flush():
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append((cur_crumb, text))

    for line in content.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            cur_lines.clear()
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_crumb = " > ".join(t for _, t in stack)
        else:
            cur_lines.append(line)
    flush()
    if not sections:                         # no headings -> single section
        text = content.strip()
        if text:
            sections.append(("", text))
    return sections


def _word_chunks(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    chunks = []
    for i in range(0, len(words), size - overlap):
        piece = words[i:i + size]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def _contextual_chunks(rel_path: str, abs_path: Path, content: str):
    """Yield (orig_id, context_text, payload) for each contextual chunk of a file."""
    meta = _doc_meta(rel_path, abs_path, content)
    idx = 0
    for crumb, sec_text in _split_sections(content):
        for chunk in _word_chunks(sec_text):
            crumb_part = f"\nSekcja: {crumb}" if crumb else ""
            context_text = (
                f"Dokument: {rel_path}\nSesja: {meta['session_id']}{crumb_part}\n\n{chunk}"
            )
            orig_id = f"{rel_path}__chunk_{idx}"
            payload = {
                "source": rel_path,
                "session_id": meta["session_id"],
                "title": meta["title"],
                "type": meta["type"],
                "tags": meta["tags"],
                "date": meta["date"],
                "section": crumb,
                "chunk_index": idx,
                "orig_id": orig_id,
                "text": context_text,
            }
            yield orig_id, context_text, payload
            idx += 1


def _point_id(orig_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, orig_id))


# =============================================================================
# Index state (per-file signatures for incremental updates)
# =============================================================================

def _file_sig(abs_path: Path) -> str:
    st = abs_path.stat()
    return f"{st.st_mtime}:{st.st_size}"


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"files": {}, "chunks": 0, "docs": 0}


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# =============================================================================
# Collection management
# =============================================================================

def _ensure_collection(recreate: bool = False):
    from qdrant_client import models
    client = get_client()
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={DENSE_NAME: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_NAME: models.SparseVectorParams()},
        )


def _index_file(rel_path: str, abs_path: Path) -> list[str]:
    """Embed + upsert all chunks of one file. Returns the list of point ids."""
    from qdrant_client import models
    try:
        content = abs_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    triples = list(_contextual_chunks(rel_path, abs_path, content))
    if not triples:
        return []

    texts = [t[1] for t in triples]
    dense, sparse = _encode_dense_sparse(texts)

    points, ids = [], []
    for (orig_id, _text, payload), d, s in zip(triples, dense, sparse):
        pid = _point_id(orig_id)
        ids.append(pid)
        points.append(models.PointStruct(
            id=pid, vector={DENSE_NAME: d, SPARSE_NAME: s}, payload=payload,
        ))
    get_client().upsert(collection_name=COLLECTION, points=points)
    return ids


def _delete_orphans(keep_ids) -> int:
    """Remove points whose ids are not in keep_ids.

    Qdrant local-mode delete_collection does not reliably purge on-disk points,
    so builds reconcile by explicitly dropping anything they did not write.
    """
    from qdrant_client import models
    client = get_client()
    keep = set(keep_ids)
    existing, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=1000, offset=offset,
            with_payload=False, with_vectors=False,
        )
        existing.update(p.id for p in points)
        if offset is None:
            break
    orphans = list(existing - keep)
    if orphans:
        client.delete(collection_name=COLLECTION,
                      points_selector=models.PointIdsList(points=orphans))
        _log(f"Reconcile: removed {len(orphans)} orphan points")
    return len(orphans)


def build_full() -> dict:
    """(Re)build the whole index from scratch. Returns stats."""
    _ensure_collection(recreate=True)

    files_state, total_chunks, docs = {}, 0, 0
    all_written: set = set()
    for rel_path, abs_path in _iter_files():
        ids = _index_file(rel_path, abs_path)
        if ids:
            files_state[rel_path] = {"sig": _file_sig(abs_path), "ids": ids}
            all_written.update(ids)
            total_chunks += len(ids)
            docs += 1
            _log(f"  indexed {rel_path} ({len(ids)} chunks)")

    removed = _delete_orphans(all_written)     # defend against stale on-disk points

    state = {"files": files_state, "chunks": total_chunks, "docs": docs,
             "model": EMBEDDING_MODEL}
    _save_state(state)
    _log(f"Full build: {total_chunks} chunks from {docs} documents "
         f"(reconcile removed {removed} orphans)")
    return {"chunks": total_chunks, "docs": docs, "added": docs,
            "updated": 0, "removed": removed}


def update_incremental() -> dict:
    """Add/update/delete only changed files vs the saved state. Returns stats."""
    from qdrant_client import models
    client = get_client()
    if not client.collection_exists(COLLECTION) or client.count(COLLECTION).count == 0:
        return build_full()
    _ensure_collection(recreate=False)

    state = _load_state()
    files_state = state.get("files", {})
    current = {rel: abs_path for rel, abs_path in _iter_files()}

    added = updated = removed = 0

    # Added / changed
    for rel_path, abs_path in current.items():
        sig = _file_sig(abs_path)
        prev = files_state.get(rel_path)
        if prev and prev.get("sig") == sig:
            continue                                   # unchanged
        if prev and prev.get("ids"):                   # changed -> drop old points
            client.delete(collection_name=COLLECTION,
                          points_selector=models.PointIdsList(points=prev["ids"]))
            updated += 1
        else:
            added += 1
        ids = _index_file(rel_path, abs_path)
        files_state[rel_path] = {"sig": sig, "ids": ids}
        _log(f"  {'updated' if prev else 'added'} {rel_path} ({len(ids)} chunks)")

    # Removed files
    for rel_path in list(files_state.keys()):
        if rel_path not in current:
            ids = files_state[rel_path].get("ids", [])
            if ids:
                client.delete(collection_name=COLLECTION,
                              points_selector=models.PointIdsList(points=ids))
            del files_state[rel_path]
            removed += 1
            _log(f"  removed {rel_path}")

    # Reconcile so the live DB always matches the tracked state exactly
    all_ids = {i for f in files_state.values() for i in f.get("ids", [])}
    _delete_orphans(all_ids)

    total_chunks = sum(len(f["ids"]) for f in files_state.values())
    state = {"files": files_state, "chunks": total_chunks, "docs": len(files_state),
             "model": EMBEDDING_MODEL}
    _save_state(state)
    _log(f"Incremental update: +{added} ~{updated} -{removed} "
         f"({total_chunks} chunks total)")
    return {"chunks": total_chunks, "docs": len(files_state),
            "added": added, "updated": updated, "removed": removed}


def ensure_fresh():
    """Guarantee the index exists (build once if missing). Updates are manual."""
    client = get_client()
    if not client.collection_exists(COLLECTION) or client.count(COLLECTION).count == 0:
        _log("Index missing/empty - building...")
        build_full()


def index_status() -> dict:
    client = get_client()
    exists = client.collection_exists(COLLECTION)
    count = client.count(COLLECTION).count if exists else 0
    state = _load_state()
    return {
        "chunks": count,
        "docs": state.get("docs", 0),
        "sessions": state.get("docs", 0),     # back-compat alias
        "hash": state.get("model", "n/a"),
        "is_fresh": exists and count > 0,
    }


# =============================================================================
# Hybrid search
# =============================================================================

def _rrf_merge(result_lists, k: int = RRF_K):
    """Reciprocal Rank Fusion over ranked id lists -> [(id, fused_score), ...]."""
    scores: dict = {}
    for results in result_lists:
        for rank, p in enumerate(results):
            scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def search(query: str, top_k: int = 5, score_mode: str | None = None,
           use_colbert: bool | None = None) -> list[dict]:
    """Hybrid search: dense+sparse -> RRF -> rerank (+ColBERT) -> dedupe per session."""
    score_mode = score_mode or DEFAULT_SCORE_MODE
    use_colbert = ENABLE_COLBERT if use_colbert is None else use_colbert
    ensure_fresh()
    client = get_client()

    dense, sparse = _encode_dense_sparse([query])
    qdense, qsparse = dense[0], sparse[0]

    dense_hits = client.query_points(
        collection_name=COLLECTION, query=qdense, using=DENSE_NAME,
        limit=PREFETCH_LIMIT, with_payload=False,
    ).points
    sparse_hits = client.query_points(
        collection_name=COLLECTION, query=qsparse, using=SPARSE_NAME,
        limit=PREFETCH_LIMIT, with_payload=False,
    ).points

    fused = _rrf_merge([dense_hits, sparse_hits])[:CANDIDATE_LIMIT]
    if not fused:
        return []

    cand_ids = [pid for pid, _ in fused]
    rrf_by_id = dict(fused)
    records = client.retrieve(collection_name=COLLECTION, ids=cand_ids, with_payload=True)
    payloads = {r.id: r.payload for r in records}
    cand = [(pid, payloads[pid]) for pid in cand_ids if pid in payloads]
    texts = [p.get("text", "") for _, p in cand]

    # Cross-encoder reranking
    rerank_scores = [0.0] * len(cand)
    if score_mode in ("rerank", "blend"):
        reranker = get_reranker()
        rerank_scores = [float(s) for s in reranker.predict([(query, t) for t in texts])]

    # ColBERT MaxSim (computed on candidates only)
    colbert_scores = [0.0] * len(cand)
    if use_colbert or score_mode == "colbert":
        q_cb = _encode_colbert([query])[0]
        d_cb = _encode_colbert(texts)
        colbert_scores = [
            (_maxsim(q_cb, d) / max(1, len(q_cb))) for d in d_cb
        ]

    # Final score per mode
    finals = []
    for i, (pid, payload) in enumerate(cand):
        if score_mode == "colbert":
            score = colbert_scores[i]
        elif score_mode == "blend":
            score = BLEND_RERANK_W * rerank_scores[i] + (1 - BLEND_RERANK_W) * colbert_scores[i]
        else:
            score = rerank_scores[i]
        finals.append((payload, score, rerank_scores[i], colbert_scores[i], rrf_by_id[pid]))

    finals.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate per session, keep best chunk
    seen: dict = {}
    for payload, score, rr, cb, rrf in finals:
        sid = payload.get("session_id", "?")
        if sid not in seen:
            seen[sid] = {
                "session_id": sid,
                "source": payload.get("source", ""),
                "title": payload.get("title", "Unknown"),
                "date": payload.get("date", "Unknown"),
                "type": payload.get("type", ""),
                "tags": payload.get("tags", ""),
                "section": payload.get("section", ""),
                "score": round(float(score), 4),
                "rerank_score": round(float(rr), 4),
                "colbert_score": round(float(cb), 4),
                "rrf_score": round(float(rrf), 5),
                "snippet": payload.get("text", "").split("\n\n", 1)[-1][:300] + "...",
            }
    return list(seen.values())[:top_k]
