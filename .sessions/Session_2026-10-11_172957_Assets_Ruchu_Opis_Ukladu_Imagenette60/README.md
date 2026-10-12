# Session: Assets ruchu cząsteczek, opis układu, ViT od zera na Imagenette (60 epok)

**Created:** 2026-10-11 17:29:57
**Type:** Mixed (Analysis + SSH + Training)
**Status:** Active (bieg Eden 1826880 trwa)

## Quick Summary
Zebrano wszystkie zapisy ruchu tokenów badania „szkło” w jednym assecie `SmallViTAblation/assets/RuchCzasteczek_ViT_szklo_2026-10`
(lokalnie twarde dowiązania, z Edena pobieranie tar + SHA-256; pobieranie przerwane przez użytkownika w połowie). Napisano szczegółowy
opis układu i mapę danych 60 epok w `SmallViTAblation/OpisUkładu/Claude` z modułem wczytującym. Uruchomiono na Edenie trening ViT od zera
na Imagenette przez 60 epok z pełną historią wag.

## Key Results
- Asset: 8 podfolderów (01 gęsty zapis 35k, 02 osie aplikacji, 03 pomiar 60 epok, 04 szkło BD, 05 pokrętła, 06 analizy, 07 kod, 08 literatura), README generowane `opisz.py`.
- Opis układu: cząsteczka = token, położenie = 768 GELU MLP ostatniego bloku, czas = krok; siły: pęd AdamW 64–68 %, paczka 31–35 %, BN ~1 %, decay 0 %; werdykt: kulki w pułapkach sprzężone wspólnymi wagami.
- Nowe trajektorie 60 epok `ruch60_rp128_50tr_50val` (806 ckpt × 100 zdjęć, rp128 + 768D) powstały na Edenie (zadanie 1826867, inna sesja).
- Imagenette: `trening_klasyfikacja.py --historia_wag`, próba 1826879 OK, bieg 1826880 (60 epok, 24 sondy, gęsto 0–2188 i 16576–17760).

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
- `training/` - parametry biegu Imagenette
