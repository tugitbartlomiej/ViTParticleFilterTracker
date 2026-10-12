# Wyniki etapu 1: czy Depth Anything V2 Small widzi igłę (2026-09-30)

Skrypt `glebia_proba.py` uruchomiony przez `czekaj_i_licz.ps1` na GPU (RTX 3070) po zakończeniu `horyzont.py`
z sąsiedniej sesji: start 17:34, 46 s. Wynik `wyniki/GLEBIA_PROBA.json` (także wartości dla każdej klatki), podgląd
lokalny `wyniki/GLEBIA_PODGLAD.png` (nie w repo, *.png jest ignorowane), log `wyniki/glebia_proba.log`.

Trzy filmy, po 300 kolejnych klatek wokół okien z testu zawiasu. Głębia względna: większa wartość = bliżej kamery.

## Wynik

| film (klatki) | igła bliżej niż tło wokół niej | nachylenie głębi igły ~ nachylenie tła (korelacja) | R² prostej wzdłuż igły | R² prostej wzdłuż tła | korelacja nachyleń kolejnych klatek |
|---|---|---|---|---|---|
| TR_0002 (6075–6374) | 1,3 % klatek | 0,992 | 0,945 | 0,891 | 0,88 |
| TR_0006 (1830–2129) | 15,7 % | 0,979 | 0,503 | 0,509 | 0,71 |
| TR_0020 (945–1244) | 0,4 % | 0,959 | 0,989 | 0,972 | 0,84 |

„Igła bliżej niż tło”: mediana głębi igły wyższa od mediany pierścienia 6 px wokół niej o więcej niż jeden odporny
rozrzut tego pierścienia. Kontrast w medianie −0,02 do 0,14 rozrzutu, czyli zero.

## Werdykt

- Kryterium etapu 1 niespełnione. Depth Anything V2 Small nie oddziela igły od tła: w 84–100 % klatek głębia igły
  jest taka sama jak głębia tła tuż obok.
- Wysokie R² i płynność w czasie są pozorne. Wzdłuż tej samej osi tło ma prawie tak samo liniową głębię,
  a nachylenia igły i tła są skorelowane w 0,96–0,99. Model powtarza kształt oka, a nie igłę.
- Na podglądzie mapa głębi to gładka misa: źrenica najdalej, powieki i brzegi kadru najbliżej. Igły nie widać
  w mapie; tylko w jednej klatce TR_0006 (igła na tle czerwonego odblasku) pojawia się garb głębi.
- Etap 2 w wersji opartej na tej mapie głębi nie ma podstaw: nachylenie igły w głąb odczytane z mapy byłoby
  nachyleniem powierzchni oka.

## Zgodność z literaturą (przegląd Undermind 2026-09-30, 200 prac)

Najlepiej udokumentowana droga to dopasowanie znanej geometrii narzędzia do obrysu, osi i końcówki z ograniczeniem
punktu wkłucia (np. Hasan i in. 2021, Li i in. 2024, Doignon i in. 2007), a nie mapa głębi jako rozwiązanie.
Monokularne modele głębi wymagają dostosowania do sceny chirurgicznej (Surgical Depth Anything, Surgical-DINO,
EndoDAC). W mikroskopie rzut jest prawie równoległy, więc głębia i obrót igły wokół własnej osi są słabo
obserwowalne; obrót staje się widoczny dopiero przez zagięty koniec.

## Czego to NIE dowodzi

- Tylko najmniejszy wariant (Small); wariantów Base i Large nie ma lokalnie. Większy model może lepiej oddać cienkie
  obiekty, ale nie usunie ograniczenia prawie równoległego rzutu.
- Trzy filmy po 300 klatek; to próba, nie pełny pomiar.
- Nie sprawdzono modeli dostrojonych do chirurgii ani syntetycznego zbioru głębi z operacji zaćmy (Zhou i in. 2023).
