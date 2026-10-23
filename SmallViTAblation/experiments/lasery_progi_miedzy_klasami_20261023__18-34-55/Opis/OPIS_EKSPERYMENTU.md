# Progi laserów MLP między klasami: czy lasery stają się wybiórcze klasowo i jaką część skoku Fishera robią

**Data:** 2026-10-23, 18:34 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach, bez treningu, bez Edena.
**Rodzic:** `plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41` (moduł `Kod_eksperymentu/wspolne.py`, klatki `Wyniki/Dane/klatki.json`).
Wzór struktury: `lodyga_cnn_wplyw_na_czasteczki_20261023__17-33-38`.

## Cel
Pytanie użytkownika: „uwaga ustawia koloidy tak, że MLP 4 je rozdziela; dowodem bezpośrednim byłby odsetek zapalonych laserów osobno dla
narzędzia i tła". Mierzymy, czy po uczeniu progi laserów MLP (768 neuronów ukrytych po GELU) leżą MIĘDZY tokenami narzędzia a tokenami
tła (lasery wybiórcze klasowo), oraz jaką część skoku Fishera (narzędzie vs tło) w MLP b robi wkład laserów wybiórczych.

## Hipotezy
H1. Po uczeniu odsetek zapalonych laserów (z > 0,05) różni się między tokenami narzędzia i tła (nie jest równy na starcie, krok 0).
H2. Rośnie liczba laserów „tylko narzędzie” i „tylko tło” (p_T ≥ 0,5 i p_B ≤ 0,1 i odwrotnie), a rozkład |p_T − p_B| przesuwa się
względem zera z przetasowania etykiet.
H3. W MLP 3–5 większość skoku Fishera pochodzi z wkładu laserów wybiórczych (|p_T − p_B| > 0,5), a nie niewybiórczych; kierunki v_c
laserów „tylko narzędzie” są dodatnio, „tylko tło” ujemnie skorelowane z μ_T − μ_B po MLP.

## Model i dane
Seed 3047, 46 checkpointów 0–437 500 (`skae_replikacja_mapy_vit3047_20261022__17-14-27/Wyniki/Dane/Wejscia/wagi`, SHA256 w
`Kod_eksperymentu/pochodzenie.json`), 10 klatek TEST z `klatki.json`. Kod modelu `maly_vit_od_zera_35000_21epok_20261009/kod/model.py`.
Etykieta tokenu (łata 16 px, 600 tokenów): narzędzie (cystotom) = udział ≥ 0,25 (po 20 na klatkę), tło = udział == 0 (560–578 na klatkę),
tokeny pośrednie pomijane.

## Procedura
`Kod_eksperymentu/lasery_klasy.py Wyniki/Dane` (GPU, float32 przejście, dekompozycja w float64). Numeracja bloków 1-based (b = 1..6,
`net.bloki[b-1]`): stan przed MLP b = X[2b−1], po MLP b = X[2b] (zgodnie z `lodyga.py`, Fisher „m4” = X[8]).
Z = wyjście GELU (hak na `mlp.act`, wartości PO aktywacji), „zapalony” = z > 0,05 (jak `PROG_Z` w `pomiar_3047.py`). Bloki timm bez LayerScale
(`ls2` = Identity), więc wkład MLP do tokenu j = Σ_c z_jc v_c + fc2.bias, v_c = `fc2.weight[:, c]`; odtworzenie X[2b] − X[2b−1] sprawdzane.
Dla każdego checkpointu, klatki i bloku:
1. `lit_narz`, `lit_tlo`: mediana po tokenach klasy odsetka laserów z z > 0,05.
2. p_T, p_B = odsetek tokenów narzędzia / tła, w których laser jest zapalony; klasy: obie (p_T ≥ 0,5 i p_B ≥ 0,5), tylko narzędzie
   (p_T ≥ 0,5 i p_B ≤ 0,1), tylko tło (p_B ≥ 0,5 i p_T ≤ 0,1), ciemne (oba ≤ 0,1), reszta mieszane. Rozkład |p_T − p_B| (mediana, odsetek > 0,5).
   Zero: to samo po losowym przetasowaniu etykiet narzędzie/tło między 20 + ~570 etykietowanymi tokenami (1 przetasowanie na klatkę, seed 0,
   te same przetasowania dla wszystkich checkpointów).
3. Fisher (‖μ_T − μ_B‖² / średnia ważona suma wariancji wewnątrzklasowych, jak w `lodyga.py`) dla: stanu przed MLP, po MLP, „przed + tylko wkład
   laserów wybiórczych” (|p_T − p_B| > 0,5), „przed + tylko wkład niewybiórczych (z biasem)”. Zero z przetasowanych etykiet (dla etykiet
   przetasowanych żaden laser nie jest wybiórczy, więc to zero jest zdegenerowane) oraz kontrola dodatkowa: „przed + losowy podzbiór laserów
   o tej samej liczności co wybiórcze” (seed 0).
4. cos(v_c, μ_T − μ_B po MLP b) – mediana po laserach klas „tylko narzędzie”, „tylko tło” i po laserach niewybiórczych (zero). Liczone dla
   wszystkich bloków (żądany jest blok 4).
5. Surowe p_T, p_B bloku 4 na krokach 0, 1000, 10 000, 65 640, 437 500 (i wersja po przetasowaniu) w npz.
`Wyniki/Code_Interpretujacy_Wyniki/odczyt_lasery.py Wyniki/Dane` drukuje tabele do `Wyniki/Dane/tabele_odczyt.txt`.

## Metryki
lit_narz, lit_tlo; liczby laserów w 5 klasach; mediana i odsetek > 0,5 z |p_T − p_B|; Fisher przed/po/tylko wybiórcze/tylko niewybiórcze;
udział_sel = (F_sel − F_przed)/(F_po − F_przed) (Fisher nie jest addytywny); mediana cos; błąd odtworzenia Δ MLP
‖(X_po − X_przed) − (Σ z v + bias)‖ / ‖X_po − X_przed‖.

## Ograniczenia
10 klatek TEST, jeden seed (3047); mediany po klatkach. Tylko 20 tokenów narzędzia na klatkę, więc p_T ma rozdzielczość 0,05 i jest zaszumione;
bardzo rzadko zapalone lasery (lit ~0,04–0,10) dają małe liczby wybiórczych. Progi 0,5 / 0,1 / 0,05 ustalone a priori, bez analizy czułości.
Fisher na 192 wymiarach ze 20 próbkami narzędzia jest zawyżony szumem (ten sam estymator co w rodzicu). Udziały Fishera przy małym mianowniku
(F_po − F_przed) są niestabilne (bloki 1–2, kroki wczesne). Zero z przetasowania etykiet daje 0 laserów wybiórczych, więc nie wnosi porównania
dla Fishera; zastępuje je kontrola losowym podzbiorem. Zależność, nie przyczynowość. Bez wykresów, raportu i wniosku (nie zamówione).

## Stan (2026-10-23, 18:45): pomiar ukończony; `Wyniki/Dane/lasery_klasy.npz`, `lasery_klasy_tabele.json`, `tabele_odczyt.txt`, `log.txt`

Mediany po 10 klatkach, bloki 1..6 (jeśli nie napisano inaczej: blok 4 = MLP 4).
- Odtworzenie Δ MLP: błąd względny max 2,9·10⁻⁷, mediana 1,3·10⁻⁷ (46 × 10 × 6), czyli Δ = Σ z v + bias dokładne; Z to wartości po GELU.
- H1 częściowo: na kroku 0 lit narzędzie = tło (0,42–0,45, różnica ≤ 0,007). Po uczeniu laserów jest dużo mniej zapalonych i w MLP 2–6
  tło zapala więcej niż narzędzie. Blok 4: 0,070 vs 0,131 (65 640), 0,049 vs 0,147 (437 500); blok 6: 0,117 vs 0,230 i 0,096 vs 0,167. Narzędzie
  jest więc „ciemniejsze” od tła, nie jaśniejsze. Na pojedynczych klatkach (blok 4, 437 500) lit narzędzia 0,04–0,12, tła 0,12–0,18.
- H2 częściowo: odsetek laserów z |p_T − p_B| > 0,5 w bloku 4: 0,029 (krok 0) → 0,059 (65 640) → 0,062 (437 500) (48 z 768), po przetasowaniu
  0 w każdym bloku; mediana |p_T − p_B| 0,036 / 0,033 vs 0,016 (zero). Wybiórcze są głównie lasery „tylko tło”: blok 4 na 65 640: obie 11,0 /
  tylko narzędzie 5,0 / tylko tło 34,5 / ciemne 460,5 / mieszane 258,0; na 437 500: 9,5 / 2,0 / 40,5 / 459,0 / 251,0. Zero (przetasowane):
  obie 47,0 i 61,0, tylko narzędzie 0 i 0, tylko tło 0 i 0, ciemne 498,0 i 485,5. Blok 3 (65 640): 14,5 / 2,0 / 34,0 / 456,5 / 250,5; blok 5: 8,0 / 13,5 / 23,0 /
  453,5 / 268,5; blok 6: 18,5 / 10,0 / 77,0 / 340,5 / 315,0. Laserów „tylko narzędzie” jest 1–13, czyli tak mało, że nie ma zbioru laserów
  zapalanych przez narzędzie i milczących na tle; lasery „obie” topnieją z ~300 (krok 0) do ~10.
- H3 częściowo: Fisher przed / po / przed+tylko wybiórcze / przed+tylko niewybiórcze / przed+losowy podzbiór tej samej liczności:
  65 640, MLP 3: 0,92 / 1,60 / 1,13 / 1,30 / 0,94; MLP 4: 1,71 / 3,61 / 3,09 / 2,02 / 1,73; MLP 5: 3,63 / 3,58 / 3,65 / 3,54 / 3,61.
  437 500, MLP 3: 0,77 / 1,47 / 1,03 / 1,15 / 0,79; MLP 4: 1,54 / 3,36 / 3,19 / 1,59 / 1,53; MLP 5: 3,35 / 3,90 / 3,86 / 3,32 / 3,34.
  W MLP 4 lasery wybiórcze (48 z 768) odtwarzają udział 0,73 (65 640) i 0,91 (437 500) skoku Fishera, niewybiórcze 0,16 i 0,03; losowy
  podzbiór tej samej liczności prawie nic (≈ 0). W MLP 3 udziały 0,31 i 0,37 (niewybiórcze 0,56 i 0,54), czyli skok mniej skupiony; w MLP 5
  Fisher już przed MLP jest ≈ po (skok ≈ 0), udziały bez znaczenia; w MLP 1–2 mały mianownik, niestabilne.
- cos(v_c, μ_T − μ_B po MLP 4): „tylko tło” −0,276 (65 640), −0,226 (437 500); „tylko narzędzie” −0,046 i −0,024; niewybiórcze −0,006 i −0,003.
  Hipoteza dodatniego cos dla laserów „tylko narzędzie” NIE potwierdzona (≈ 0); lasery „tylko tło” pchają tło ujemnie wzdłuż osi narzędzie − tło
  (−0,23…−0,31 w MLP 1–4), co oddala tło od narzędzia. Mechanizm to więc głównie „tło zapala lasery, które odsuwają je od narzędzia”.
- Dodatkowo: wybiórczość pojawia się już w pierwszych 1000 krokach (odsetek wybiórczych 0,115 w bloku 4 na kroku 1000), potem maleje do ~0,06
  razem z zanikaniem zapalonych laserów.
