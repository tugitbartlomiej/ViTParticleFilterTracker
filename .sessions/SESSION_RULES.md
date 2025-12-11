# 📋 Session Management System v2.0
**Project:** ViTParticleFilterTracker
**Version:** 2.0
**Updated:** 2025-10-31

---

## 🎯 Purpose

System zarządzania sesjami roboczymi w projekcie. Każda sesja jest zapisywana w osobnym folderze oznaczonym datą i czasem (z dokładnością do sekundy).

---

## 📁 Directory Structure

```
.sessions/
├── SESSION_RULES.md                    # Dokumentacja (ten plik)
├── templates/                          # Szablony dla sesji
│   ├── ssh_session_template.md
│   ├── benchmark_session_template.md
│   ├── analysis_session_template.md
│   └── training_session_template.md
├── Session_2025-10-20_221215/         # Sesja z 20 października 2025, 22:12:15
│   ├── ssh/                           # Podfolder SSH (opcjonalny)
│   ├── README.md                      # Szybki opis sesji
│   └── SESSION_SUMMARY.md             # Pełna dokumentacja
├── Session_2025-10-29_142909/         # Sesja z 29 października 2025, 14:29:09
│   ├── benchmark/                     # Podfolder benchmark (opcjonalny)
│   ├── README.md
│   └── SESSION_SUMMARY.md
├── Session_2025-10-31_173025/         # Sesja z 31 października 2025, 17:30:25
│   ├── training/                      # Podfolder training (opcjonalny)
│   │   ├── configs/
│   │   └── logs/
│   ├── analysis/                      # Podfolder analysis (opcjonalny)
│   ├── README.md
│   └── SESSION_SUMMARY.md
└── archive/                           # Stare, zarchiwizowane sesje
    └── eden_2025-10/
```

---

## 🏷️ Session Naming Convention

### Format
```
Session_YYYY-MM-DD_HHMMSS
```

### Components
- **Session_** - Static prefix
- **YYYY** - Rok (np. 2025)
- **MM** - Miesiąc (01-12)
- **DD** - Dzień (01-31)
- **HH** - Godzina (00-23)
- **MM** - Minuta (00-59)
- **SS** - Sekunda (00-59)

### Examples
- `Session_2025-10-28_193015` - Sesja z 28 października 2025 o 19:30:15
- `Session_2025-10-29_200045` - Sesja z 29 października 2025 o 20:00:45
- `Session_2025-10-31_184512` - Sesja z 31 października 2025 o 18:45:12

### Time Zone
**Zawsze używaj czasu lokalnego** (nie UTC) - łatwiejsza identyfikacja sesji.

---

## 📂 Session Structure

### Główny folder sesji
Każda sesja ma:
- **Folder główny:** `Session_YYYY-MM-DD_HHMMSS/`
- **README.md** - Szybki opis (co, dlaczego, kluczowe info)
- **SESSION_SUMMARY.md** - Pełna dokumentacja sesji

### Opcjonalne podfoldery
W obrębie sesji możesz tworzyć dowolne podfoldery, np.:

#### `analysis/`
Pliki analizy danych:
- Wykresy, statystyki
- Jupyter notebooks
- CSV/JSON z wynikami
- Raporty markdown

#### `archive/`
Tymczasowe backupy z sesji:
- Stare wersje plików
- Zarchiwizowane logi
- Kopie zapasowe

#### `benchmark/`
Wyniki benchmarków:
- JSON/YAML z metrykami
- Porównania modeli
- Konfigurace testów
- Raporty wydajności

#### `ssh/`
Pliki z pracy na clusterze:
- Skrypty SLURM
- Logi zadań
- Status clustera
- Checkpointy (linki, nie pliki!)

#### `templates/`
Szablony użyte w sesji:
- Konfiguracje
- Boilerplate code
- Przykładowe struktury

#### `tools/`
Narzędzia/skrypty stworzone w sesji:
- Python scripts
- Bash scripts
- PowerShell scripts
- Helper functions

#### `training/`
Treningi modeli:
- Konfigurace treningowe
- Logi treningu
- Checkpointy (linki!)
- Metryki (loss, accuracy)

**UWAGA:** Nie musisz tworzyć wszystkich podfolderów - tylko te, których potrzebujesz!

---

## 📝 Required Files

### 1. README.md
Szybki opis sesji (1-2 akapity).

**Template:**
```markdown
# Session: Session_YYYY-MM-DD_HHMMSS

**Created:** YYYY-MM-DD HH:MM:SS
**Type:** [Analysis | Benchmark | Training | SSH | Mixed]
**Status:** [Active | Completed | Paused | Archived]

## Quick Summary
[2-3 zdania opisujące co było robione]

## Key Results
- [Główny wynik 1]
- [Główny wynik 2]

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `subdir/` - [Opis podfolderu]

## Related Sessions
- Previous: `Session_2025-10-29_142909`
- Next: `Session_2025-11-01_103000`
```

---

### 2. SESSION_SUMMARY.md
Pełna dokumentacja sesji.

**Template:**
```markdown
# Session Summary: Session_YYYY-MM-DD_HHMMSS

## Metadata
- **Date:** YYYY-MM-DD
- **Time:** HH:MM:SS
- **Duration:** [czas trwania]
- **Status:** [Completed | In Progress | Failed | Paused]
- **Type:** [Analysis | Benchmark | Training | SSH | Mixed]

## Objective
[Cel sesji - co chciałeś osiągnąć?]

## Context
[Kontekst - dlaczego ta sesja była potrzebna? Co było wcześniej?]

## Actions Taken
1. [Krok 1]
2. [Krok 2]
3. [Krok 3]
...

## Results

### Key Findings
- [Finding 1]
- [Finding 2]
- [Finding 3]

### Metrics/Data
[Tabele, liczby, statystyki]

### Issues Encountered
- **Problem 1:** [Opis] → **Rozwiązanie:** [Jak naprawione]
- **Problem 2:** [Opis] → **Rozwiązanie:** [Jak naprawione]

## Conclusions
[Wnioski - co się udało, co nie, czego się nauczyłeś]

## Next Steps
- [ ] [Akcja 1]
- [ ] [Akcja 2]
- [ ] [Akcja 3]

## Files Generated
- `file1.py` - [Opis]
- `subdir/file2.yaml` - [Opis]
- `results/data.json` - [Opis]

## Commands Used
```bash
# Przykładowe komendy użyte w sesji
command1 --arg value
command2 --flag
```

## Related Work
- **Previous session:** `Session_2025-10-29_142909` - [Krótki opis]
- **Referenced files:** `path/to/file.py`
- **External resources:** [Linki]
```

---

## 🛠️ Creating New Sessions

### Manual Creation
```bash
# Utwórz folder z aktualną datą i czasem
mkdir .sessions/Session_$(date +%Y-%m-%d_%H%M%S)

# Przejdź do folderu
cd .sessions/Session_$(date +%Y-%m-%d_%H%M%S)

# Stwórz strukturę
mkdir -p analysis benchmark ssh tools training
touch README.md SESSION_SUMMARY.md
```

### Using Tools (TODO: Zaktualizuj narzędzia)
```bash
# Bash
.sessions/tools/save_session.sh

# Python
py -3.11 .sessions/tools/save_session.py

# PowerShell
.\.sessions\tools\save_session.ps1
```

---

## 🕐 Timestamp Generation

### Bash (Linux/Mac/Git Bash)
```bash
echo "Session_$(date +%Y-%m-%d_%H%M%S)"
# Output: Session_2025-10-31_173025
```

### Python
```python
from datetime import datetime
timestamp = f"Session_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
print(timestamp)  # Session_2025-10-31_173025
```

### PowerShell
```powershell
"Session_$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"
# Output: Session_2025-10-31_173025
```

---

## 🗂️ Archiving Old Sessions

### Kiedy archiwizować?
- Sesje starsze niż 3 miesiące
- Sesje zakończone i nieaktywne
- Sesje które już nie są potrzebne

### Jak archiwizować?
```bash
# Przenieś do archive z opisową nazwą
mkdir -p .sessions/archive/2025-10_training
mv .sessions/Session_2025-10-15_* .sessions/archive/2025-10_training/

# Lub skompresuj
cd .sessions
tar -czf archive/2025-10-training-sessions.tar.gz Session_2025-10-15_* Session_2025-10-16_*
rm -rf Session_2025-10-15_* Session_2025-10-16_*
```

---

## 🔧 Git Integration

### .gitignore Rules
```gitignore
# Large files nie są commitowane
.sessions/**/*.log
.sessions/**/logs/
.sessions/**/*.pth
.sessions/**/*.pt
.sessions/**/*.ckpt

# Commitowane są:
!.sessions/SESSION_RULES.md
!.sessions/**/README.md
!.sessions/**/SESSION_SUMMARY.md
!.sessions/templates/
```

### Co commitować?
✅ README.md i SESSION_SUMMARY.md
✅ Konfiguracje (.yaml, .json)
✅ Małe pliki wynikowe (<100KB)
✅ Skrypty i narzędzia
✅ Raporty markdown

❌ Logi
❌ Duże dane
❌ Model checkpoints
❌ Pliki tymczasowe

---

## 📚 Best Practices

1. **Twórz sesję od razu** gdy zaczynasz pracę
2. **Używaj opisowych nazw podfolderów** (np. `benchmark_yolo_vs_detr`)
3. **Aktualizuj SESSION_SUMMARY.md na bieżąco**, nie na koniec
4. **Linkuj powiązane sesje** w README
5. **Nie duplikuj dużych plików** - używaj linków symbolicznych lub referencji
6. **Archiwizuj regularnie** (co miesiąc)
7. **Dokumentuj problemy i rozwiązania** - będą przydatne później
8. **Commituj często** README i SUMMARY

---

## 🎓 Examples

### Example 1: Training Session
```
Session_2025-10-31_173025/
├── README.md
├── SESSION_SUMMARY.md
├── training/
│   ├── configs/
│   │   └── detr_config.yaml
│   ├── logs/
│   │   └── training.log
│   └── checkpoints/
│       └── checkpoint_links.txt
└── analysis/
    └── loss_curves.png
```

### Example 2: SSH Cluster Work
```
Session_2025-10-28_193015/
├── README.md
├── SESSION_SUMMARY.md
└── ssh/
    ├── job_1226363_status.txt
    ├── cluster_resources.txt
    └── slurm_script.slurm
```

### Example 3: Benchmark Session
```
Session_2025-10-29_142909/
├── README.md
├── SESSION_SUMMARY.md
├── benchmark/
│   ├── results/
│   │   ├── yolo_metrics.json
│   │   └── detr_metrics.json
│   └── configs/
│       └── benchmark_config.yaml
└── analysis/
    ├── comparison_report.md
    └── visualizations/
        └── performance_plot.png
```

### Example 4: Mixed Session
```
Session_2025-10-30_120000/
├── README.md
├── SESSION_SUMMARY.md
├── ssh/
│   └── job_commands.txt
├── training/
│   └── configs/
├── benchmark/
│   └── results/
└── tools/
    └── helper_script.py
```

---

## 📞 Migration from Old Structure

Stare sesje (z podziałem `ssh/`, `benchmark/`, etc.) są migrowane do archive.

### Process:
1. Przenieś do `archive/` z opisową nazwą
2. Dodaj `MIGRATION_LOG.md` w archive
3. Zaktualizuj referencje w dokumentacji

---

## 📝 Changelog

### Version 2.0 (2025-10-31)
- **BREAKING:** Nowy format nazewnictwa `Session_YYYY-MM-DD_HHMMSS`
- **BREAKING:** Wszystkie sesje w głównym `.sessions/`, nie w podfolderach według typu
- **NEW:** Podfoldery tematyczne wewnątrz sesji (analysis, benchmark, ssh, training, tools, etc.)
- **NEW:** Elastyczna struktura - twórz tylko te podfoldery których potrzebujesz
- Uproszczone zasady i dokumentacja
- Migracja starych sesji do archive

### Version 1.0 (2025-10-28)
- Początkowy system z podziałem na typy (ssh/, benchmark/, analysis/, training/)
- Cztery typy sesji
- Trzy narzędzia (bash, python, powershell)

---

**System Status:** Active
**Maintainer:** Project Team
**Last Updated:** 2025-10-31
