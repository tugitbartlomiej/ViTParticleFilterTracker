# Eden Cluster - Kompletny przewodnik

## 🔧 Konfiguracja połączenia

### Architektura połączenia
```
Komputer lokalny → piotrowskib2@ssh.mini.pw.edu.pl (jump host) → bpiotrowski@eden (klaster)
```

### SSH Config (już skonfigurowany)
Plik: `~/.ssh/config`
```
Host eden-jump
    HostName ssh.mini.pw.edu.pl
    User piotrowskib2
    ServerAliveInterval 60

Host eden-cluster
    HostName eden
    User bpiotrowski
    ProxyJump eden-jump
    ServerAliveInterval 60
```

### SSH Keys (już zainstalowany)
- ✅ Klucze SSH wygenerowane i skopiowane na oba serwery
- ✅ Brak potrzeby wpisywania haseł
- ✅ Automatyczna autoryzacja

## 🖥️ Łączenie się z klastrem

### Interaktywna sesja SSH
```bash
# Główna komenda:
ssh eden-cluster

# Alternatywnie (pełna komenda):
ssh -J piotrowskib2@ssh.mini.pw.edu.pl bpiotrowski@eden

# Przez skrypt:
.\connect_to_eden.bat
```

### Pojedyncze komendy
```bash
# Wykonaj komendę na klastrze:
ssh eden-cluster "ls -la"
ssh eden-cluster "pwd"
ssh eden-cluster "df -h"
```

## 📁 Struktura folderów na Eden

### Lokalizacje główne
```
/mnt/evafs/faculty/home/bpiotrowski/          # Katalog domowy (200GB)
├── models/                                   # Modele ML/AI (utworzony)
├── datasets/                                 # Zbiory danych
│   └── DETR_BACKGROUNG/                     # Dane DETR (utworzony)
├── DETR/                                    # Projekt DETR
│   └── logs/                                # Logi treningów
└── slurm_sync/                              # Synchronizowane pliki SLURM
```

### Miejsce na dysku
- **Przydzielone**: 200 GB
- **Używane**: 72 GB (36%)
- **Dostępne**: 129 GB (64%)

## 🔄 Synchronizacja plików

### Lokalne SLURM → Eden (automatyczna)
```bash
# Z lokalnego Windows:
F:\RAG\SLURM> auto_sync_simple.bat
```
Synchronizuje: `F:\RAG\SLURM\` → `/home2/faculty/bpiotrowski/slurm_sync/`

### Przesyłanie plików (SCP)
```bash
# Z lokalnego na klaster:
scp "lokalny_plik" eden-cluster:/sciezka/na/klastrze/

# Foldery (rekurencyjnie):
scp -r "lokalny_folder" eden-cluster:/sciezka/na/klastrze/

# Z klastra na lokalny:
scp -r eden-cluster:/sciezka/na/klastrze/folder "C:\lokalna\sciezka\"

# Przykłady:
scp "E:\Cataract\train22.mp4" eden-cluster:/home2/faculty/bpiotrowski/datasets/DETR_BACKGROUNG/
scp -r eden-cluster:/home2/faculty/bpiotrowski/DETR/logs "F:\Eden_Logs\"
```

## 🛠️ Zainstalowane narzędzia

### Git + Git LFS
- ✅ **Git**: version 2.43.0
- ✅ **Git LFS**: version 3.4.0 (zainstalowany lokalnie)
- ✅ **Lokalizacja**: `~/bin/git-lfs`
- ✅ **PATH**: dodany do `~/.bashrc`

### Inicjalizacja Git LFS (wykonane w `~/models`)
```bash
cd ~/models
git lfs install  # Wykonane
```

### Użycie Git LFS
```bash
# Klonowanie repo z LFS:
git clone https://huggingface.co/microsoft/DialoGPT-medium

# Pobieranie dużych plików:
git lfs pull

# Śledzenie dużych plików:
git lfs track "*.bin" "*.safetensors" "*.h5"

# Status LFS:
git lfs ls-files
```

## 🎯 SLURM - system kolejkowy

### Status klastra
```bash
# Info o partycjach:
sinfo

# Twoje zadania:
squeue -u bpiotrowski

# Dostępne GPU:
sinfo -o '%P %.5a %.10l %.6D %.6t %N %G'
```

### Dostępne zasoby
**GPU dostępne:**
- A100 (8x na węzłach dgx-1, dgx-3, dgx-4)
- H100 (8x na węźle hopper) 
- Tesla (4x na węźle pascal)

**Partycje:**
- `short` (1 dzień)
- `long` (5 dni)
- `experimental` (5 dni)
- `debug` (infinite)

### Twoje konto SLURM
- **Account**: `transform+`
- **QOS**: `normal`
- **Status**: aktywny

## 🔧 Przydatne komendy na klastrze

### Nawigacja i pliki
```bash
pwd                           # Aktualna lokalizacja
ls -lth                       # Pliki wg daty (najnowsze na górze)
du -sh *                      # Rozmiary folderów
df -h                         # Miejsce na dysku
tree -D                       # Struktura z datami
watch -n 2 'ls -lth'         # Monitorowanie na żywo
```

### System
```bash
hostname                      # Nazwa węzła
whoami                        # Nazwa użytkownika
free -h                       # RAM
nvidia-smi                    # GPU (jeśli dostępne)
module list                   # Dostępne moduły
```

### Środowiska Python
```bash
# Aktualnie aktywne:
(yolo_py310) bpiotrowski@eden:~$

# Lista środowisk:
conda env list
```

## 🚀 Workflow sesji roboczej

### 1. Synchronizacja SLURM (lokalnie)
```bash
F:\RAG\SLURM> auto_sync_simple.bat
```

### 2. Połączenie z klastrem
```bash
ssh eden-cluster
```

### 3. Praca na klastrze
```bash
cd ~/models                   # Przejdź do modeli
git clone repo_url            # Klonuj repo z modelami
cd ~/datasets                 # Pracuj z danymi
sbatch job_script.sh          # Wyślij zadanie SLURM
```

### 4. Pobieranie rezultatów (lokalnie)
```bash
scp -r eden-cluster:/home2/faculty/bpiotrowski/results "F:\Results\"
```

## 📧 Kontakt z administratorami

**W przypadku problemów:**
- **Sebastian Korsak**: s.korsak@datascience.edu.pl
- **Email**: hpc@mini.pw.edu.pl
- **Web**: https://hpc.mini.pw.edu.pl/

## ⚡ Szybkie polecenia

```bash
# Łączenie:
ssh eden-cluster

# Synchronizacja:
auto_sync_simple.bat

# Przesyłanie:
scp plik eden-cluster:~/

# Status klastra:
sinfo && squeue -u bpiotrowski

# Miejsce na dysku:
df -h /mnt/evafs

# Git LFS:
git lfs --version
```

## 🔐 Bezpieczeństwo

- ✅ SSH Keys zainstalowane (brak haseł)
- ✅ Połączenie przez jump host (bezpieczne)
- ✅ Hasła nie są przechowywane w plikach
- ✅ Automatyczna autoryzacja

---

**Ostatnia aktualizacja**: 2025-08-08
**Status**: Wszystko skonfigurowane i działające ✅