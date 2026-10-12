# Session: Szkło — 60 epok, wspólny napęd, test „768 wiązek trzyma 600 koloidów” (Eden)

**Created:** 2026-10-11 17:33:59
**Type:** Mixed (Analysis + SSH/Eden)
**Status:** Active (dwa zadania Eden w toku: 1826876 napęd, 1826881 wiązki)

## Quick Summary
Kontynuacja badania „ruch tokenów ViT jak w szkle”. Wyeksportowano trajektorie 100 zdjęć po 806 checkpointach (60 epok) w rzucie 128D,
policzono na Edenie test Markowa / operator piDMD / jądro pamięci Moriego–Zwanziga w skali kroku oraz ruch w skali epok; uruchomiono
odjęcie wspólnego napędu i test hipotezy użytkownika „768 wiązek lasera (wiersze W fc1) trzyma 600 koloidów (u_i)”.
Liczenie przeniesione z lokalnego komputera na Eden na prośbę użytkownika.

## Key Results
- 60 epok: tokeny relaksują do centrum (operator symetryczny, brak obrotu, prawie izotropowo, τ ~125–215 kroków), MSD α 0,2–0,4,
  sąsiedztwo zamrożone (Jaccard 0,56–0,59 po 4000 krokach) — jak ciało stałe z drgającymi cząstkami.
- Ruchliwość zdjęcia przewiduje jego błąd detekcji od 5. epoki: Spearman ruch–IoU −0,43…−0,68, także na VALID.
- Skala kroku: pamięć 3–8 kroków (pęd AdamW), VAMP niestacjonarny (dryf centrów), centra skaczą (kurtoza 1,3–1,8× gauss).
- Wspólny napęd: 30 składowych z pola INNYCH zdjęć tłumaczy 56 % energii ruchu tokenu (kontrola 0–5 %).

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
