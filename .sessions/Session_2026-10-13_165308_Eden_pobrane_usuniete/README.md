# Session: Eden — co pobrane lokalnie, co usunięte na Edenie

**Created:** 2026-10-13 16:53
**Type:** SSH
**Status:** Active

## Quick Summary
Dysk na Edenie (`/mnt/evafs`, kwota 200 GB) był zajęty w 96 % (192 GB, 8,1 GB wolne). Polecenie użytkownika (2026-10-13):
pobierać lokalnie, a na Edenie usuwać to, co już zostało pobrane, i zapisywać w sesji, co pobrane i co usunięte.
Zasada: na Edenie kasuję plik tylko wtedy, gdy lokalna kopia ma tę samą sumę SHA-256 policzoną na nowo po obu stronach
w dniu usunięcia. Zbiory danych używane przez biegi na Edenie (imagenette, cataract_lmm) zostają.

## Files
- `REJESTR.md` — rejestr: co pobrane (skąd, dokąd, ile), co usunięte na Edenie (ścieżka, liczba plików, GB, sumy).
