# `.sessions/` — narzędzia sesji projektu VIT (główny indeks)

Jeden punkt wejścia do całego oprzyrządowania sesji: **wyszukiwanie (RAG)**, **indeksowanie**,
**zapis sesji** i **wizualizacje**. Szczegóły RAG → [`RAG_README.md`](RAG_README.md).

> **Dla innej sesji Claude Code** (np. z folderu artykułu): żeby przeszukać sesje VIT,
> przejdź do sekcji [Szybki start](#szybki-start). Nic nie instalujesz — wszystko jest w venv.

---

## Spis treści
- [Szybki start](#szybki-start)
- [Ścieżki i środowisko](#ścieżki-i-środowisko)
- [1. Wyszukiwanie (RAG)](#1-wyszukiwanie-rag)
- [2. Indeksowanie bazy](#2-indeksowanie-bazy)
- [3. Zapis sesji](#3-zapis-sesji)
- [4. Wizualizacje](#4-wizualizacje)
- [5. Narzędzia MCP](#5-narzędzia-mcp-dla-claude-code)
- [6. Slash-komendy](#6-slash-komendy-claudecommands)
- [Ważne zasady](#ważne-zasady)
- [Mapa plików](#mapa-plików)

---

## Szybki start

```powershell
$env:HF_HOME = "F:\HuggingFace"
$PY    = "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\venv\Scripts\python.exe"
$TOOLS = "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\tools"

# WYSZUKIWANIE (semantyka + słowa kluczowe)
& $PY "$TOOLS\search_sessions.py" "fine-tuning YOLO learning rate" --top 5

# AKTUALIZACJA bazy po dodaniu sesji
& $PY "$TOOLS\index_sessions.py" --update

# STATUS bazy
& $PY "$TOOLS\index_sessions.py" --stats
```

---

## Ścieżki i środowisko

| Co | Ścieżka |
|---|---|
| Python (venv) | `…\.sessions\venv\Scripts\python.exe` |
| Narzędzia | `…\.sessions\tools\` |
| Baza Qdrant (RAG) | `…\.sessions\qdrant_db\` |
| Modele (cache HF) | `F:\HuggingFace\hub\` — `HF_HOME=F:\HuggingFace` |

Stack RAG: **BGE-M3** (dense+sparse+ColBERT) + **bge-reranker-v2-m3**, GPU (RTX 3070), Qdrant local.

---

## 1. Wyszukiwanie (RAG)

Hybrydowe wyszukiwanie po znaczeniu **i** słowach kluczowych. Pełny opis: [`RAG_README.md`](RAG_README.md).

```powershell
& $PY "$TOOLS\search_sessions.py" "twoje zapytanie" --top 5     # --json dla JSON
& $PY "$TOOLS\prewarm_index.py"                                  # załaduj modele + odśwież
```

| Plik | Rola |
|---|---|
| `search_sessions.py` | CLI wyszukiwania (`--top N`, `--json`) |
| `prewarm_index.py` | załadowanie modeli na GPU + odświeżenie indexu |
| `hybrid_rag.py` | silnik: build/update/search, contextual chunking, RRF, reranker, ColBERT |

Tryby oceny w `hybrid_rag.search(score_mode=...)`: `rerank` (domyślny) / `colbert` / `blend`.

---

## 2. Indeksowanie bazy

```powershell
& $PY "$TOOLS\index_sessions.py" --full     # pełny rebuild od zera
& $PY "$TOOLS\index_sessions.py" --update   # inkrementalnie (tylko zmienione pliki)
& $PY "$TOOLS\index_sessions.py" --stats    # status
```

- Indeksuje wszystkie `.md`/`.txt` pod `.sessions/` (rekursywnie; pomija `venv`, `tools`,
  `qdrant_db`, `obsidian_vault`, `templates`, `analysis`).
- `--update` dowiązuje nowe embeddingi do istniejącej bazy; `--full` przebudowuje całość.
- Indeksowanie **nie jest automatyczne** — uruchamiasz je po dodaniu/edycji sesji.

---

## 3. Zapis sesji

Tworzą plik `SESSION_SUMMARY.md` nowej sesji. Najwygodniej przez slash-komendy
([sekcja 6](#6-slash-komendy-claudecommands)); skrypty bazowe:

| Plik | Rola |
|---|---|
| `save_session.py` / `.ps1` / `.sh` | zapis sesji (Python / PowerShell / Bash) |
| `list_sessions.sh` | listowanie sesji |

Zasady nazewnictwa/struktury sesji: `..\SESSION_RULES.md` oraz `..\..\GENERATE_SESSION_HELP.md`.

> Po zapisie nowej sesji uruchom `index_sessions.py --update`, żeby trafiła do RAG-a.

---

## 4. Wizualizacje

Generują interaktywne pliki HTML / eksport do Obsidian. Najwygodniej przez
`/regeneruj-wizualizacje`; skrypty bazowe:

| Plik | Rola |
|---|---|
| `generate_graph_view.py` / `generate_graph_view_3d.py` | graf powiązań między sesjami (HTML 2D / 3D) |
| `generate_hybrid_timeline.py` | oś czasu sesji (HTML) |
| `generate_tree_data.py` + `visualize_sessions_tree.py` | drzewo sesji |
| `visualize_sessions_3d.py` | wizualizacja 3D |
| `export_to_obsidian.py` | eksport do Obsidian vault (`..\obsidian_vault\`) |

Wyniki lądują w `..\analysis\` (graph_view.html, hybrid_timeline.html) i `..\obsidian_vault\`.

---

## 5. Narzędzia MCP (dla Claude Code)

Serwer `sessions` w `…\.mcp.json` (z `HF_HOME` w env). Narzędzia:

| Narzędzie | Opis |
|---|---|
| `search_sessions(query, top_k=5)` | hybrydowe wyszukiwanie |
| `get_session_content(session_id)` | pełna treść `SESSION_SUMMARY.md` |
| `list_sessions(tag, session_type, limit)` | lista sesji z filtrami |
| `get_index_status()` | status indexu |
| `rebuild_index()` | pełna przebudowa |
| `update_index()` | inkrementalny update |

> Po zmianie kodu narzędzi przeładuj serwer: `/mcp` → reconnect `sessions`.

---

## 6. Slash-komendy (`.claude/commands/`)

Działają, gdy odpalasz Claude Code **w folderze VIT**. Szczegóły każdej: odpowiedni plik `.md`.

| Komenda | Opis |
|---|---|
| `/szukaj-sesje` | wyszukiwanie w RAG |
| `/indeksuj` | indeksowanie sesji |
| `/prewarm-sesje` | załadowanie modeli + indexu |
| `/regeneruj-wizualizacje` | odśwież graf + timeline + Obsidian |
| `/zapisz-sesje`, `/zapisz-sesje-ssh` | zapis sesji (zwykłej / SSH) |
| `/zapisz-notatke`, `/zapisz-plan` | zapis notatki / planu |
| `/wczytaj-ostatnia-sesje`, `/wczytaj-ostatnia-sesje-ssh` | wczytanie ostatniej sesji |
| `/commit` | git commit |

---

## Ważne zasady

1. **Qdrant local = single-writer.** Tylko jeden proces naraz może otworzyć `qdrant_db/` —
   **albo** serwer MCP `sessions`, **albo** CLI. Inaczej drugi dostanie błąd locka.
2. **Zawsze `HF_HOME=F:\HuggingFace`** — żeby modele ładowały się z lokalnego cache (bez pobierania).
3. **Indeksowanie jest ręczne** — po zapisaniu sesji uruchom `index_sessions.py --update`.
4. **Nie kasuj `qdrant_db\index_state.json`** — pamięta, co już zindeksowano (inaczej wymusisz pełny rebuild).
5. Komunikaty `QdrantClient.__del__ ... sys.meta_path is None` i `DeprecationWarning: swigvarlink`
   przy zamykaniu są **nieszkodliwe** (dane już zapisane).

---

## Mapa plików

```
.sessions/
├── tools/
│   ├── README.md            ← TEN PLIK (główny indeks)
│   ├── RAG_README.md        ← szczegóły RAG
│   ├── hybrid_rag.py        ← silnik RAG (BGE-M3 + Qdrant + reranker + ColBERT)
│   ├── search_sessions.py   ← CLI wyszukiwania
│   ├── index_sessions.py    ← CLI indeksowania (--full/--update/--stats)
│   ├── prewarm_index.py     ← prewarm modeli + indexu
│   ├── sessions_mcp_server.py ← serwer MCP
│   ├── save_session.*       ← zapis sesji (py/ps1/sh)
│   ├── list_sessions.sh     ← listowanie
│   ├── generate_*.py / visualize_*.py / export_to_obsidian.py ← wizualizacje
│   └── sessions_rag_agent.py ← legacy (TF-IDF, odłączony)
├── qdrant_db/               ← baza wektorowa RAG (+ index_state.json)
├── venv/                    ← izolowany Python (torch+CUDA, qdrant, FlagEmbedding, ...)
├── analysis/                ← wygenerowane wizualizacje (HTML)
├── obsidian_vault/          ← eksport Obsidian
└── Session_*/               ← zapisane sesje (SESSION_SUMMARY.md, README.md, podfoldery)
```
