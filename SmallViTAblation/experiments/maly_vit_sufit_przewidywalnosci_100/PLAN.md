# Sufit przewidywalności latentu małego ViT na 100 kroków uczenia

Data: 2026-10-03. Polecenie użytkownika: „zmierz sufit dla małego ViT na 100 kroków”. Cel użytkownika:
prognoza na 100 kroków uczenia lepsza o ponad 85 procent od prognozy „bez zmian”.

## Pytanie

Czy w ruchu latentu jest dość powolnego, ciągłego dryfu, żeby **jakakolwiek** prognoza na 100 kroków mogła
zmniejszyć błąd prognozy „bez zmian” o 85 procent? Dla której wielkości (pojedynczy token, średnia po tokenach
narzędzia, rzut na kierunek tożsamości) i w którym okresie uczenia?

Powód pomiaru: NNDMD na pojedynczych tokenach w krokach 900–1000 przegrał z prognozą „bez zmian” o 15 procent
(`../maly_vit_nndmd_lokalne_przyrosty_hopper/run1819918/RAPORT_KONCOWY.md`). Jeśli sufit jest niski, winne są
dane, nie model.

## Dane

Gęsta historia `../maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy`: bieg Eden 1818732,
ziarno 47, kroki 0..1000 co 1 aktualizację, 12 obrazów, 600 tokenów, 768 współrzędnych (MLP ostatniego bloku po
GELU, tryb eval). Maski narzędzia z `maski.npz` (218 tokenów narzędzia łącznie). Bez nowego treningu, bez Edenu.

## Aparat

Ruch rozkładam na gładką ścieżkę `m_t` (średnia ruchoma o szerokości `w`, wyśrodkowana) i drgania `e_t = x_t - m_t`.

- **Sufit (wyrocznia):** prognoza `m_{t+h}`, czyli dokładna gładka ścieżka przyszłości. Błąd to same drgania
  w punkcie docelowym. `S = 1 - błąd wyroczni / błąd prognozy bez zmian`. To górna granica dla każdej prognozy,
  która traktuje drgania jako nieprzewidywalne.
- **Kontrola wzoru:** `S ≈ 1 - σ²/(D² + 2σ²)`, gdzie `D²` to kwadrat przesunięcia gładkiej ścieżki przez h kroków,
  a `σ²` to kwadrat drgań.
- **Osiągalne bez uczenia (przyczynowo):** C0 = średnia z ostatnich `w` stanów; C1 = C0 plus przedłużenie
  przesunięcia średniej z ostatnich h kroków; C2 = C0 plus połowa tego przesunięcia. Nic nie jest dopasowywane,
  więc nie ma przecieku z przyszłości.
- **Trwałość dryfu:** cosinus między przesunięciem gładkiej ścieżki w przyszłych h krokach a w przeszłych h krokach.

Horyzonty h = 10, 50, 100 (główny: 100). Szerokości `w` = 5, 11, 21, 51. Okresy według kroku startu prognozy:
0–100, 100–300 (koniec rozgrzewki lr w kroku 300), 300–600, 600–koniec.

Metryki: łączna (suma błędów po wszystkich jednostkach) oraz mediana po jednostkach. Dwie skale: MSE (`1 - MSE/MSE₀`)
i RMSE (`1 - RMSE/RMSE₀`). Cel 85 procent jest w nich bardzo różny: 85 procent w RMSE to 97,75 procent w MSE.

## Wielkości

1. `tok_narz`: pojedyncze tokeny narzędzia, 768 liczb (jak NNDMD, ale tylko narzędzie).
2. `tok_tlo`: 100 losowych tokenów tła na obraz, 768 liczb.
3. `sr_narz`: średnia po tokenach narzędzia w obrazie, 768 liczb.
4. `sr_obraz`: średnia po wszystkich 600 tokenach obrazu, 768 liczb.
5. `wekt_tozs`: średnia narzędzia minus średnia tła, 768 liczb (wektor tożsamości).
6. `rzut_tok`: rzut pojedynczego tokenu narzędzia na kierunek LDA (narzędzie przeciw tłu, krok 1000, Ledoit-Wolf).
7. `sila_obrazu`: średni rzut tokenów narzędzia obrazu na kierunek LDA (siła tożsamości).
8. `rzut_tok_los`, `sila_obrazu_los`: to samo dla losowego kierunku (kontrola: czy LDA jest szczególny).

## Podłoga

`S_drgania`: ta sama wyrocznia na samych drganiach `e_t` (dryf usunięty). Pokazuje, ile daje samo odszumienie
punktu startowego bez żadnego dryfu. Dla białych drgań wychodzi około 0,5 w MSE. Sufit blisko tej podłogi znaczy:
dryfu nie ma, jest tylko szum.

## Co obala cel 85 procent

Sufit wyroczni poniżej 0,85 (w wybranej skali) dla danej wielkości i okresu. Wtedy żaden model nie osiągnie 85
procent dla tej wielkości w tym okresie, bo nawet znajomość całej gładkiej przyszłości nie wystarcza.

## Zastrzeżenia ustalone przed pomiarem

- Średnia ruchoma zalicza do „gładkiej ścieżki” także powolne błądzenie losowe (dyfuzję). Sufit może więc być
  zawyżony: wolne błądzenie wygląda jak dryf, ale nie da się go przewidzieć. Rozstrzyga to trwałość dryfu i C1.
- Mała `w` zostawia część drgań w ścieżce (sufit zawyżony), duża `w` wycina zakręty dryfu (sufit zaniżony).
  Dlatego raportuję cały zakres `w`.
- Kierunek LDA liczony w kroku 1000 korzysta z końca biegu. Do sufitu to bez znaczenia, bo sufit nie uczy modelu,
  ale prognoza w praktyce musiałaby mieć kierunek z przeszłości.
- Jedno ziarno, 12 obrazów treningowych. Ruch tokenów zawiera też zmianę statystyk BatchNorm.
