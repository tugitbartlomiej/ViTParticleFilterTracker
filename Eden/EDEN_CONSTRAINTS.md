# Eden Cluster — Ograniczenia i zasady (2026-03-30)

## ❌ ZAKAZ: partycja `debug`
- Dostęp do partycji `debug` został zablokowany przez administratora
- NIE używać `#SBATCH -p debug` ani `--partition=debug` w żadnym skrypcie

## ✅ Dostępne partycje GPU (sprawdzone 2026-03-30)

```
PARTITION    NODELIST   GPU                   MaxTime   UWAGI
short*       dgx-[1-4]  gpu:a100:8 per node   1-00:00   domyślna, krótkie joby
long         dgx-[1-4]  gpu:a100:8 per node   5-00:00   długie joby (A100)
hopper       hopper     gpu:h100:8            5-00:00   H100, 8 GPU ✅ ZALECANE dla ablacji
hopper-2     hopper-2   gpu:h200:4            1-00:00   ⚠️ MaxTime=24h — za krótkie dla 130 epok!
experimental pascal     gpu:tesla:4           ?         stary node, ograniczony
```

**Dla treningu ablacji (3x GPU, 130 epok ~3 dni): używaj `hopper` (MaxTime=5 dni).**
**NIE używaj `hopper-2` dla długich treningów — MaxTime=24h!**

```bash
#SBATCH -A transformers_vsc
#SBATCH -p hopper           # H100, MaxTime=5 dni ✅
#SBATCH --nodelist=hopper
#SBATCH --gres=gpu:3
#SBATCH --time=3-00:00:00
```

## ✅ Zasady pisania skryptów SLURM

### Sprzątanie po sobie
Każdy skrypt MUSI zawierać cleanup na końcu i przy błędzie:
```bash
cleanup() {
    echo "Cleaning up TMPDIR..."
    rm -rf "$TMPDIR/data" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
```

### Dane tymczasowe na node
- **NIE używaj `/tmp` bezpośrednio** — ma tylko ~62GB na hopper-2 (za mało na 91k dataset)
- **Używaj `/raid`** na hopper-2: 14TB wolnego, lokalny, szybki
- Fallback: `/dev/shm` (1TB tmpfs in-memory)
- SLURM_TMPDIR **nie jest ustawiany automatycznie** na Eden (stan: 2026-03-30)
- NIE zostawiaj dużych plików w `~/` ani `/mnt/evafs` po zakończeniu
- Po zakończeniu treningu: zachowaj tylko `best_model.pth`, usuń `checkpoint_epoch_*.pth`

### TMPDIR selection (wzorzec)
```bash
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    TMPDIR="$SLURM_TMPDIR"
elif [[ -d "/raid" ]] && [[ "$(df /raid --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/raid/${USER}_${SLURM_JOB_ID:-$$}_jobname"
elif [[ "$(df /dev/shm --output=avail 2>/dev/null | tail -1)" -gt 52428800 ]]; then
    TMPDIR="/dev/shm/${USER}_${SLURM_JOB_ID:-$$}_jobname"
else
    TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}_jobname"
fi
```

## Historia
- 2026-03-30: Admin zablokował dostęp do partycji `debug` po nieautoryzowanym użyciu
- 2026-03-30: Odkryto brak partycji `standard` — pierwsza próba na `hopper-2`
- 2026-03-30: Odkryto że `hopper-2` MaxTime=24h — za krótkie dla treningu ablacji (3 dni)
- 2026-03-30: Zmieniono na partycję `hopper` (H100, MaxTime=5 dni) — joby 1641365–1641370
- 2026-03-30: `/tmp` na hopper/hopper-2 niewystarczające; `/raid` 14TB dostępny na obu węzłach
- 2026-03-30: `SLURM_TMPDIR` nie ustawiany automatycznie — skrypt sam wybiera `/raid`
