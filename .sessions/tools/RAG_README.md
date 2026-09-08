# Sessions RAG — instrukcja użycia (dla Claude Code i ludzi)

Lokalny, hybrydowy RAG nad **sesjami projektu VIT** (pliki `.md`/`.txt` w `.sessions/`).
Pozwala wyszukiwać "co się działo w sesjach" po **znaczeniu (semantyka)** i po
**słowach kluczowych (leksyka)** jednocześnie. Wszystko działa **lokalnie** (GPU RTX 3070),
bez chmury.

> **Dla innej sesji Claude Code:** jeśli czytasz to z innego projektu (np. folderu
> artykułu) i chcesz przeszukać sesje VIT — wykonaj komendę z sekcji
> [Szybki start](#szybki-start). Nie musisz nic instalować; wszystko jest gotowe w venv.

---

## Architektura (co siedzi pod spodem)

```
zapytanie
  -> BGE-M3 dense  (semantyka, 1024-dim)   -> Qdrant top-80
  -> BGE-M3 sparse (słowa kluczowe)        -> Qdrant top-80
  -> RRF fusion (łączenie obu list)        -> ~40 kandydatów
  -> bge-reranker-v2-m3 (cross-encoder)    -> rerank precyzyjny
  -> BGE-M3 ColBERT MaxSim (na kandydatach)-> sygnał pomocniczy
  -> top_k wyników (deduplikacja per sesja)
```

- **Embedder:** `BAAI/bge-m3` (dense + sparse + ColBERT z jednego modelu), GPU/fp16.
- **Reranker:** `BAAI/bge-reranker-v2-m3` (cross-encoder), GPU.
- **Baza wektorowa:** Qdrant w trybie **lokalnym/embedded** (`.sessions/qdrant_db/`, bez Dockera).
- **Modele:** ładowane ze wspólnego cache HuggingFace (`HF_HOME=F:\HuggingFace`) po repo-id.
- **Contextual chunking:** każdy chunk dostaje prefiks `Dokument / Sesja / Sekcja`
  (section-aware, po nagłówkach markdown) — żaden fragment nie jest "anonimowy".

---

## Ścieżki i środowisko

| Co | Ścieżka |
|---|---|
| Python (venv) | `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\venv\Scripts\python.exe` |
| Narzędzia | `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\tools\` |
| Baza Qdrant | `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\qdrant_db\` |
| Modele (cache HF) | `F:\HuggingFace\hub\` (ustaw `HF_HOME=F:\HuggingFace`) |

**Zawsze** ustaw `HF_HOME=F:\HuggingFace` przed uruchomieniem (żeby modele ładowały się
z lokalnego cache, bez pobierania).

---

## Szybki start

PowerShell (zmienne dla czytelności):

```powershell
$env:HF_HOME = "F:\HuggingFace"
$PY     = "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\venv\Scripts\python.exe"
$TOOLS  = "F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.sessions\tools"

# WYSZUKIWANIE (semantyka + słowa kluczowe naraz)
& $PY "$TOOLS\search_sessions.py" "fine-tuning YOLO learning rate" --top 5
& $PY "$TOOLS\search_sessions.py" "benchmark DETR mAP" --json     # wynik jako JSON
```

Pierwsze uruchomienie ładuje modele na GPU (~15–25 s) i, jeśli baza jest pusta,
zbuduje index automatycznie.

---

## Komendy

### 1. Wyszukiwanie — `search_sessions.py`
```powershell
& $PY "$TOOLS\search_sessions.py" "twoje zapytanie" --top 5
& $PY "$TOOLS\search_sessions.py" "twoje zapytanie" --json
```
- `--top N` — liczba wyników (domyślnie 5)
- `--json` — wynik strukturalny (do dalszego przetwarzania)

Wynik zawiera per sesję: `session_id`, `source` (plik), `title`, `date`, `type`,
`section`, oraz wyniki: `score` (finalny), `rerank_score`, `colbert_score`, `rrf_score`.

### 2. Indeksowanie / przebudowa — `index_sessions.py`
```powershell
# Pełny build od zera (pierwszy raz albo gdy chcesz przebudować wszystko)
& $PY "$TOOLS\index_sessions.py" --full

# Inkrementalny update (dodaje/aktualizuje/usuwa TYLKO zmienione pliki)
& $PY "$TOOLS\index_sessions.py" --update

# Status indexu
& $PY "$TOOLS\index_sessions.py" --stats
```
`--update` dowiązuje nowe embeddingi do istniejącej bazy (nie przebudowuje całości) —
używaj po zapisaniu nowych sesji.

### 3. Prewarm (opcjonalnie) — `prewarm_index.py`
```powershell
& $PY "$TOOLS\prewarm_index.py"
```
Ładuje modele na GPU i odświeża index — przydatne, by uniknąć opóźnienia pierwszego zapytania.

---

## Jakie zapytania zadawać

Hybryda obsługuje oba style w jednym zapytaniu:

- **Semantyczne / opisowe** (po znaczeniu): `"jak naprawiono problem z pamięcią GPU"`,
  `"dobór datasetu metodą klastrowania"`, `"uruchomienie treningu na klastrze"`.
- **Słowa kluczowe / nazwy własne / liczby** (leksyka): `"epoch 170"`, `"mAP 85.74"`,
  `"SLURM job hopper"`, `"EL2N"`, `"DETR 20k"`. Tu pomaga komponent sparse/leksykalny.
- **Mieszane** (najlepsze): `"benchmark DETR 20k ep270 mAP external"`.

Wskazówki:
- Pisz po **polsku lub angielsku** — model jest multilingual, sesje są mieszane PL/EN.
- Dłuższe, opisowe zapytania działają lepiej niż jedno słowo.
- Wyniki są **deduplikowane per sesja** (jedna sesja = jeden wynik, najlepszy fragment).

---

## Serwer MCP (dla Claude Code)

Skonfigurowany w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\.mcp.json` jako serwer
`sessions` (z `HF_HOME` wpiętym w env). Narzędzia MCP:

| Narzędzie | Opis |
|---|---|
| `search_sessions(query, top_k=5)` | hybrydowe wyszukiwanie (zwraca sesje + score) |
| `get_session_content(session_id)` | pełna treść `SESSION_SUMMARY.md` danej sesji |
| `list_sessions(tag, session_type, limit)` | lista sesji z filtrami |
| `get_index_status()` | status indexu |
| `rebuild_index()` | pełna przebudowa |
| `update_index()` | inkrementalny update (tylko zmienione pliki) |

> Jeśli serwer MCP trzyma starą wersję kodu, przeładuj go: `/mcp` → reconnect `sessions`.

---

## WAŻNE: ograniczenie single-writer (Qdrant local)

Tryb lokalny Qdrant **blokuje katalog `qdrant_db/` na jeden proces naraz**. Czyli:

- **Albo** serwer MCP `sessions`, **albo** CLI (`search_sessions.py` / `index_sessions.py`)
  — **nie oba jednocześnie** (drugi dostanie błąd locka).
- Praktyka: jeśli używasz CLI, najpierw zatrzymaj serwer MCP (i odwrotnie).

---

## Co jest indeksowane

- **Włączone:** wszystkie `.md` i `.txt` pod `.sessions/` rekursywnie (sesje + ich podfoldery
  + `archive/`).
- **Wyłączone** (w `hybrid_rag.py` → `EXCLUDE_DIRS`): `venv`, `tools`, `vectordb`, `qdrant_db`,
  `obsidian_vault` (re-eksport sesji = duplikaty), `templates` (szablony), `analysis`
  (generowane HTML), `__pycache__`, `.git`.
- **JSON** domyślnie **off** (mogą być duże/zaszumione). Włączysz dodając `".json"` do
  `INDEX_EXTENSIONS` w `hybrid_rag.py`.

Strojenie (`hybrid_rag.py`): `CHUNK_WORDS`, `CHUNK_OVERLAP`, `PREFETCH_LIMIT`,
`CANDIDATE_LIMIT`, `ENABLE_COLBERT`, `DEFAULT_SCORE_MODE` (`rerank` | `colbert` | `blend`).

---

## Pliki

| Plik | Rola |
|---|---|
| `hybrid_rag.py` | silnik (modele, Qdrant, contextual chunking, build/update, search) |
| `search_sessions.py` | CLI wyszukiwania |
| `index_sessions.py` | CLI indeksowania (`--full` / `--update` / `--stats`) |
| `prewarm_index.py` | załadowanie modeli + odświeżenie indexu |
| `sessions_mcp_server.py` | serwer MCP (deleguje do `hybrid_rag`) |
