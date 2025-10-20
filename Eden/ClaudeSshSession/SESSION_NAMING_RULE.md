# Reguła Nazewnictwa Sesji SSH

## Konwencja Nazewnictwa

Każda sesja SSH z Eden-cluster powinna być zapisana w osobnym folderze z następującą konwencją nazewnictwa:

### Format
```
sesja_YYYY-MM-DD_HH-MM
```

### Komponenty
- **sesja_** - prefiks stały
- **YYYY** - rok (np. 2025)
- **MM** - miesiąc (01-12)
- **DD** - dzień (01-31)
- **HH** - godzina UTC (00-23)
- **MM** - minuta (00-59)

### Przykłady
- `sesja_2025-10-20_22-12` - 20 października 2025 o 22:12 UTC
- `sesja_2025-10-21_09-45` - 21 października 2025 o 09:45 UTC
- `sesja_2025-11-15_14-30` - 15 listopada 2025 o 14:30 UTC

## Zawartość Każdej Sesji

Każdy folder sesji powinien zawierać:

### Dokumentacja
- `README.md` - Przewodnik po zawartości sesji
- `SESSION_SUMMARY.md` - Kompletne podsumowanie zdarzeń, napraw i statusu

### Konfiguracja & Skrypty
- `*.slurm` - SLURM job scripts (wszystkie wersje)
- `*.py` - Fragmenty lub całe skrypty treningowe

### Logi & Status
- `*.log` - Wszystkie logi (training, error, etc.)
- `*_log.txt` - Wyjście z komend SSH
- `*_status.txt` - Status zasobów klastra (sfree, squeue)
- `*_fix.txt` - Sekcje kodu pokazujące dokonane naprawy

## Procedura Dodawania Nowej Sesji

1. Określ bieżący czas w UTC (godzina i minuta)
2. Utwórz nowy folder z nazwą: `sesja_YYYY-MM-DD_HH-MM`
3. Zapisz wszystkie pliki wygenerowane w tej sesji do nowego folderu
4. Utwórz `SESSION_SUMMARY.md` z opisem wszystkich zdarzeń
5. Utwórz `README.md` z instrukcją przeglądania zawartości

## Przykład Struktury

```
ClaudeSshSession/
├── SESSION_NAMING_RULE.md (ten plik)
├── sesja_2025-10-20_22-12/
│   ├── README.md
│   ├── SESSION_SUMMARY.md
│   ├── cluster_status.txt
│   ├── training_log.txt
│   ├── torch_compile_fix.txt
│   ├── detr_train_optimized_excerpt.py
│   └── run_detr_2gpu_500epochs.slurm
├── sesja_2025-10-21_09-45/
│   ├── README.md
│   ├── SESSION_SUMMARY.md
│   └── ...
└── sesja_2025-10-22_14-30/
    ├── README.md
    ├── SESSION_SUMMARY.md
    └── ...
```

## Oznaczenie Czasu

Zawsze używaj **UTC** dla konsystencji:
- SSH session start time → czas w UTC
- Job submission time → czas w UTC
- Log timestamps → wszystkie w UTC

## Przydatne Komendy

### Pobranie aktualnego czasu UTC
```bash
date -u +"%Y-%m-%d_%H-%M"
```

### Nazwa folderu dla bieżącego czasu
```bash
echo "sesja_$(date -u +"%Y-%m-%d_%H-%M")"
```

## Notatka

Ta konwencja ułatwia:
- Szybkie znalezienie sesji po dacie
- Automatyzację zarządzania sesjami
- Historię zmian i napraw
- Debugowanie problemów
