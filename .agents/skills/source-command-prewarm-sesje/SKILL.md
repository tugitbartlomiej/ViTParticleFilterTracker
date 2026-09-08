---
name: "source-command-prewarm-sesje"
description: "Migrated source command `prewarm-sesje`"
---

# source-command-prewarm-sesje

Use this skill when the user asks to run the migrated source command `prewarm-sesje`.

## Command Template

# Pre-warm Sessions Index

Załaduj model bge-m3 i przebuduj indeks ChromaDB dla MCP semantic search.

## Instrukcje

Uruchom skrypt pre-warm:

```bash
.sessions/venv/Scripts/python.exe .sessions/tools/prewarm_index.py
```

## Co robi skrypt

1. **Ładuje model bge-m3** (~2.3GB, 30-60s pierwsze uruchomienie)
2. **Sprawdza indeks ChromaDB** - czy jest aktualny
3. **Przebudowuje indeks** jeśli dodano nowe sesje
4. **Potwierdza gotowość** - ile sesji/chunków zindeksowano

## Po wykonaniu

Poinformuj użytkownika:
- Indeks jest gotowy
- MCP `search_sessions()` będzie teraz działać szybko
- Kolejne wyszukiwania będą trwać <1s

## Kiedy używać

- Po dodaniu nowych sesji (`/zapisz-sesje`)
- Przed intensywnym wyszukiwaniem w sesjach
- Gdy MCP `search_sessions` daje timeout

## Uwagi

- Pierwsze uruchomienie: 1-2 minuty (ładowanie modelu)
- Kolejne uruchomienia: 5-10s (model w cache, sprawdza tylko hash)
- Wymaga venv: `.sessions/venv/`
