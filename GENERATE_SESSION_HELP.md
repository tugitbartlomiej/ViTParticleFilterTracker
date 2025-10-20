# Generate Session Summary - Instrukcja Użytkownika

## Funkcja

Skrypt `generate-session-summary.ps1` automatycznie:
1. Łączy się z Eden-cluster przez SSH
2. Pobiera dane o jobach i statusie klastra
3. Dynamicznie znajduje logi dla różnych JobId
4. Zapisuje podsumowanie lokalnie w `Eden\ClaudeSshSession\sesja_YYYY-MM-DD_HH-MM\`

## Obsługa Różnych JobId

Skrypt obsługuje **dowolne JobId** i **dynamicznie znajduje logi**:
- Szuka logów automatycznie w katalogu logs/
- Obsługuje różne formaty nazw logów
- Jeśli log nie będzie znaleziony, pokazuje listę dostępnych

## Użycie

### Podstawowy Usage
```powershell
.\generate-session-summary.ps1
```

Pobierze aktualne statusy i utworzy folder sesji z automatyczną datą i czasem.

### Z ID Joba
```powershell
.\generate-session-summary.ps1 -JobId 1190351
```

Pobierze szczegółowe informacje dla konkretnego joba i automatycznie znajdzie log.

### Z Opisem
```powershell
.\generate-session-summary.ps1 -JobId 1190351 -Description "DETR training 500 epochs on 2x GPU"
```

Dodaj opis do sesji dla lepszej dokumentacji.

### Z Niestandardową Ścieżką Logu
```powershell
.\generate-session-summary.ps1 -JobId 1190351 -LogPath "/mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190351.log"
```

Jeśli skrypt nie znajdzie logu automatycznie, możesz podać ścieżkę ręcznie.

### Kompleksowy Przykład
```powershell
.\generate-session-summary.ps1 -JobId 1190351 -Description "DETR training checkpoint after fixing torch.compile bug"
```

## Co Pobiera Ze SSH

Skrypt automatycznie pobiera:
- `squeue -u bpiotrowski` - Aktualne joby
- `scontrol show job <ID>` - Szczegóły konkretnego joba
- `sfree` - Status zasobów klastra
- `find /logs -name '*<JobId>*'` - Szuka loga automatycznie
- `tail -200 <log>` - Ostatnie 200 linii loga treningu
- `nvidia-smi -L` - Lista GPUs
- `sinfo` - Status węzłów SLURM
- `du -sh` - Użycie dysku
- `find /logs -name '*.log'` - Lista dostępnych logów

## Dynamiczne Wyszukiwanie Logów

Skrypt automatycznie:
1. Szuka logu zawierającego JobId w nazwie
2. Jeśli nie znaleziony - sprawdza domyślne lokalizacje:
   - `/logs/detr_ddp_2gpu_500ep_<JobId>.log`
   - `/logs/detr_ddp_3gpu_dgx3_<JobId>.log`
   - `/logs/*.log`
3. Jeśli nadal nie znaleziony - pokazuje listę dostępnych logów w podsumowaniu

## Struktura Wygenerowanej Sesji

```
Eden\ClaudeSshSession\
└── sesja_2025-10-20_22-12\
    ├── SESSION_SUMMARY.md        (kompletne podsumowanie)
    ├── README.md                 (szybki przewodnik)
    ├── cluster_status.txt        (sfree output)
    ├── jobs_status.txt           (squeue output)
    ├── disk_usage.txt            (du output)
    ├── logs_available.txt        (lista logów)
    └── job_details.txt           (scontrol output - jeśli podany JobId)
```

## Naming Convention

Każda sesja automatycznie otrzymuje nazwę:
```
sesja_YYYY-MM-DD_HH-MM
```

Gdzie:
- YYYY-MM-DD = data UTC
- HH-MM = godzina i minuta UTC

**Przykłady:**
- `sesja_2025-10-20_22-12` (20 października 2025 o 22:12)
- `sesja_2025-10-21_09-45` (21 października 2025 o 09:45)

## Wymagania

1. SSH access do eden-cluster (configured)
2. PowerShell 5.0+
3. ssh command available in PATH
4. Foldery: `Eden\ClaudeSshSession\` - utworzony

## Typowe Scenariusze

### Scenario 1: Monitorowanie Treningu (Job 1190351)
```powershell
.\generate-session-summary.ps1 -JobId 1190351 -Description "DETR 500 epochs training started"
```

### Scenario 2: Inny Job ID (np. 1190352)
```powershell
.\generate-session-summary.ps1 -JobId 1190352 -Description "New training run"
```

### Scenario 3: Wielokrotne Joby
```powershell
# Job 1
.\generate-session-summary.ps1 -JobId 1190351 -Description "First job"

# Job 2
.\generate-session-summary.ps1 -JobId 1190352 -Description "Second job"

# Job 3
.\generate-session-summary.ps1 -JobId 1190353 -Description "Third job"
```

Każdy job będzie w osobnej sesji z różnym czasem (jeśli minie 1 minuta).

### Scenario 4: Kontrola Statusu
```powershell
# Periodycznie sprawdzić postęp dla dowolnego joba
.\generate-session-summary.ps1 -JobId <CURRENT_JOB_ID>
```

### Scenario 5: Bez Konkretnego JobId
```powershell
# Pokaż tylko ogólny status klastra
.\generate-session-summary.ps1 -Description "Cluster overview"
```

## Output Przykład

```
Session Summary Generator for Eden Cluster
===========================================
Timestamp: 2025-10-20_22-12 (UTC)
Session: sesja_2025-10-20_22-12
Job ID: 1190351

[OK] Created session folder
[FETCH] Retrieving data from Eden cluster...
[OK] Retrieved current jobs
[OK] Retrieved cluster status
[OK] Retrieved hardware status
[OK] Retrieved logs list
[OK] Retrieved job details for ID: 1190351
[OK] Found log automatically: /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/detr_ddp_2gpu_500ep_1190351.log
[OK] Retrieved training log
[OK] Retrieved disk usage

Generating files...
[OK] Created SESSION_SUMMARY.md
[OK] Created README.md
[OK] Created cluster_status.txt
[OK] Created jobs_status.txt
[OK] Created disk_usage.txt
[OK] Created job_details.txt
[OK] Created logs_available.txt

Session Summary Generated Successfully
======================================

Session Folder: F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\sesja_2025-10-20_22-12
Folder Name: sesja_2025-10-20_22-12

Files created:
  - SESSION_SUMMARY.md (3.5 KB)
  - README.md (2.1 KB)
  - cluster_status.txt (0.6 KB)
  - jobs_status.txt (0.4 KB)
  - disk_usage.txt (1.2 KB)
  - logs_available.txt (2.8 KB)
  - job_details.txt (4.2 KB)

View summary:
  F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\sesja_2025-10-20_22-12\SESSION_SUMMARY.md

Command to generate next session:
  .\generate-session-summary.ps1 -JobId <NEW_JOB_ID> -Description <text>
```

## Troubleshooting

### Problem: SSH connection failed
```
Error: No matching host key type found
```
**Rozwiązanie:** Upewnij się, że SSH do eden-cluster jest skonfigurowany: `ssh eden-cluster "echo ok"`

### Problem: Permission denied
```
Error: permission denied
```
**Rozwiązanie:** Sprawdź dostęp do Eden: `ssh eden-cluster "whoami"`

### Problem: Log nie znaleziony
```
Log file not readable
```
**Rozwiązanie:**
1. Sprawdź JobId: `ssh eden-cluster "squeue -u bpiotrowski"`
2. Podaj ścieżkę ręcznie: `.\generate-session-summary.ps1 -JobId 1190351 -LogPath "..."`
3. Sprawdź dostępne logi: `ssh eden-cluster "ls -lah /mnt/evafs/faculty/home/bpiotrowski/DETR/logs/"`

### Problem: Folder już istnieje
```
Error: Item already exists
```
**Rozwiązanie:** Zmień godzinę - każda sesja ma unikalny czas (co 1 minutę)

### Problem: JobId nie rozpoznany
```
scontrol: error: Invalid job id specified
```
**Rozwiązanie:** Sprawdzam dostępne joby: `ssh eden-cluster "squeue -u bpiotrowski"`

## Automation (Optional)

### Windows Task Scheduler
1. Otwórz Task Scheduler
2. Create Basic Task
3. Name: "Generate Session Summary Hourly"
4. Trigger: Hourly (co godzinę)
5. Action: Run PowerShell script
6. Script: `generate-session-summary.ps1`
7. Arguments: `-JobId 1190351`

## Session Rules

Dla pełnych instrukcji zobacz:
- `Eden\ClaudeSshSession\SESSION_NAMING_RULE.md`

## Quick Commands

```powershell
# Wygeneruj sesję z automatycznym statusem
.\generate-session-summary.ps1

# Wygeneruj dla konkretnego joba
.\generate-session-summary.ps1 -JobId 1190351

# Wygeneruj dla innego joba
.\generate-session-summary.ps1 -JobId 1190352 -Description "New training"

# Wygeneruj z pełnym kontekstem
.\generate-session-summary.ps1 -JobId 1190351 -Description "DETR training checkpoint"

# Wygeneruj z niestandardową ścieżką logu
.\generate-session-summary.ps1 -JobId 1190351 -LogPath "/path/to/custom/log.log"
```

## Dynamiczne Szukanie Logów - Jak Działa

1. Szuka: `/logs/*1190351*` - wszystkie pliki z JobId
2. Jeśli nie znaleziono, szuka domyślnych:
   - `detr_ddp_2gpu_500ep_1190351.log`
   - `detr_ddp_3gpu_dgx3_1190351.log`
3. Jeśli nadal nie znaleziono, pokazuje listę dostępnych logów

---

**Script Location:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\generate-session-summary.ps1`

**Session Archive:** `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\ClaudeSshSession\`

**Version:** 2.0 - Obsługuje różne JobId i dynamiczne szukanie logów
