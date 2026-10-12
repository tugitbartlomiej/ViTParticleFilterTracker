# Animacje ruchu cząsteczek w latencie ViT — pamięć projektu (skill video-use)

## Sesja 1 — 2026-10-10

**Strategia:** dwa źródła obrazu. (1) Nagrania z aplikacji (zakładka „Ruch zbioru”, nowa metoda
`ZakladkaRuchu.nagraj_film`): sześć animacji A1–A6 na prawdziwych pomiarach (historia K1000, 1500 zdjęć × 600
cząsteczek, 101 checkpointów co 10 kroków; historia 35 000 klatek, 100 zdjęć, 22 końce epok). (2) Animacje z danych
policzonych bezpośrednio z gęstego zapisu co 1 krok (`edit/dane/licz_trajektorie.py` → `trajektorie_wlasnosci.npz`,
`PODSUMOWANIE.json`), budowane równolegle przez podagentów w `edit/animations/slot_1` (MSD: trzy reżimy w τ) i
`slot_2` (zegar uczenia: serie czasowe z kursorem). Bez transkrypcji i dźwięku (brak nagrań mówionych).

**Decyzje:**
- Silnik dla punktów: karta w aplikacji (900 000 cząsteczek na klatkę, 30 kl./s, H.264 przez ffmpeg).
- Silnik dla wykresów: matplotlib + ffmpeg (Manim odrzucony: brak LaTeX, kerning Pango; PIL/matplotlib pewniejsze).
- Paleta jak w aplikacji: tło #10171c, obszary oka: tło #8a8f99, rogówka #4cd964, źrenica #4a8cff, narzędzie
  #ff4433; akcent #ffd166; czcionka Segoe UI.
- A1 w trybie „przesunięcie od kroku 0” (przepływ kolektywny widać jako wspólny wektor), A2 i A6 w położeniach
  (A6: kondensacja widoczna tylko w położeniach; w przesunięciach chmura pozornie rośnie).
- A3 i A5: skala osi Z z bieżącego stanu (kadr 2), inaczej wczesne kroki są płaskie.
- A4: dwa widoki obok siebie (tło | narzędzie) z tym samym zakresem osi.

**Dziennik rozumowania:**
- Odporny promień chmury (mediana) zamiast RMS: RMS zawyżały pojedyncze dalekie cząsteczki.
- Pierwsza łatka ustawień nie weszła (asercja w heredoku, wynik ukryty) — zawsze sprawdzać `grep` po łatce.
- Wyłączenie komputera zaplanowane przez użytkownika na 03:58 (shutdown /s /t 10800; anulowanie: shutdown /a).

**Wynik sesji:** A1–A6 (aplikacja, 2216 × 746, A4 4442 × 746), slot_1 `render.mp4` (30,5 s) i slot_2 `render.mp4`
(39,2 s) po 1920 × 1080, `final.mp4` 168 s (119 MB). Testy aplikacji: 143 zaliczone, 1 pominięty; 1249 kontrolek z
dymkami. Poprawka interpretacji po slocie 2: przewaga temperatury narzędzia słabnie po ~2 epokach (PCA32 1,04 od
5. epoki; w 768D 0,7–0,8), źrenica zostaje cieplejsza — wpisane do README.

**Zaległe:** wersja z narracją (brak nagrań); to samo na innych ziarnach biegu (hipoteza stygnięcia narzędzia).
