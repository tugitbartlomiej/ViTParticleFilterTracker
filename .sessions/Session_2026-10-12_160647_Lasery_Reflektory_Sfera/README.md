# Session: Lasery, reflektory i szklana sfera (szklo3_3D)

**Created:** 2026-10-12 16:06:47
**Type:** Analysis (wizualizacje + literatura)
**Status:** Completed

## Quick Summary
Sprawdzono, czy obraz „sfera laserów i koloidy w pułapkach” jest fizycznie poprawnym modelem uczenia MałegoViT (Undermind: deep search + pełne teksty 10 prac), spisano zasady modelu (Z1–Z14) i diagram architektury, zmierzono na wagach „przeciwlasery” i obrót reflektora, a potem zbudowano serię filmów 8–12. Najlepsza wizualizacja według użytkownika: **film 12**.

## Najlepsza wizualizacja
- Film: `SmallViTAblation/outputs/porownanie_szklo_20261010/film_krysztal/lasery_12_reflektory_stale.mp4` (77 s, 1080p)
- Skrypt: `.../film_krysztal/film_lasery_12_reflektory_stale.py` (commit `8b4fe1eb`, gałąź `worktree-laser-koloid-zasady`)
- Układ: 768 stałych reflektorów (neurony bloku 6) na zewnętrznej sferze Fibonacciego, przypisanych węgierskim algorytmem do −d_c; każdy ma 600 wiązek (cząsteczki = patche) sterowanych galwami; szklana sfera załamuje wiązkę raz na powierzchni, w środku wiązki biegną prosto w kierunku d_c = P v̂_c (wspólnym dla reflektora); położenia cząsteczek = P W₂ (z − z̄).

## Key Results
- Model „koloid nadąża za pułapką” w czasie uczenia jest błędny (token nie ma własnego stanu: z = f(x; θ_t)); token jest cząstką wzdłuż GŁĘBOKOŚCI, neurony są cząstkami w czasie UCZENIA.
- Blok MLP = 768 sił: Δx = Σ_c z_c v_c, z_c = GELU(w_c·LN(x)+b_c) (dokładnie).
- Uczenie tworzy przeciwlasery: pary v_c z cos < −0,5 rosną 0 % → 38 % do 21. epoki; cos(w_c, v_c) → −0,16; |v_c| ×2,2.
- Reflektor nie obraca się jako całość: wspólny obrót wyjaśnia 10–14 % zmiany wobec 7–12 % zera.
- τ ∝ η^−0,5 (nie η^−1): prosty obraz „lr = temperatura” ilościowo się nie zgadza.
- MałyViT ma 6 bloków (nie 12).

## Files
- `SESSION_SUMMARY.md` - pełna dokumentacja
