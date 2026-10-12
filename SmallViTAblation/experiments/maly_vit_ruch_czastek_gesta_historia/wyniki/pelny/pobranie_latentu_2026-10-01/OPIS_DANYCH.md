# Co zawiera gęsta historia

Data: 2026-10-01. Źródło: istniejący, zakończony bieg Eden `1818732`; nie uruchomiono nowego treningu. Status transferu jest zapisany osobno w [STATUS.json](STATUS.json). Ten opis nie zastępuje potwierdzenia ukończenia transferu i lokalnego SHA256.

Plik `../dane/LATENT_GESTY.npy` zawiera rzeczywiste pełne reprezentacje tokenów mierzone podczas uczenia. Nie jest dziennikiem samych aktualizacji wag.

| Oś | Rozmiar | Znaczenie |
|---|---:|---|
| Krok | 1001 | Stan początkowy 0 i pomiar po każdej aktualizacji 1..1000 |
| Obraz | 12 | Te same, niezmienne obrazy RGB 480×320 przy każdym pomiarze |
| Token | 600 | Siatka 20×30, kolejność `i=y*30+x` |
| Współrzędna | 768 | Pełna aktywacja ukrytej MLP ostatniego bloku po GELU |

Format: NumPy, float32 little-endian, C-order. Dokładny rozmiar: **22 140 518 528 bajtów**, w tym 128 bajtów nagłówka. To 22,1405 GB w jednostkach dziesiętnych, około 20,62 GiB. W tablicy nie wykonano uśredniania tokenów.

Każdy pomiar wykonywano w `eval`, bez gradientów, w float32, bez autocast i TF32. Obrazy uint8 dzielono przez 255, a model stosował normalizację ImageNet. Wagi i biegnące statystyki BatchNorm zmieniały się wskutek treningu; pomiar `eval` nie aktualizował BN. Dlatego różnica dwóch pomiarów obejmuje wpływ zmiany wag **oraz** stanu BN na reprezentacje tych samych obrazów.

Ruch cząsteczki otrzymuje się jako `Z[t+1, obraz, token, :] - Z[t, obraz, token, :]`. Plik zawiera położenia; różnica jest późniejszą operacją analityczną. Token tego samego indeksu innego obrazu nie oznacza automatycznie tego samego materialnego fragmentu narzędzia.

Ten plik nie zawiera gradientów, momentów AdamW, pełnej historii wykonanych minibatchy ani dalszych stanów po kroku 1000. Osobne 101 checkpointów co 10 kroków zawiera wagi i bufory BN. Nie pobierano pełnego stanu optymalizatora.

## Pliki potrzebne do interpretacji

- `../dane/metadata.json`: metadane tego biegu, kroki, nazwy obrazów, hash latentu, wersje i warstwa pomiarowa.
- `../dane/obrazy.npy`: dokładne wejścia pomiaru; 12×320×480×3, uint8.
- `../dane/maski.npz`: oryginalne maski i udziały pikseli dla tokenów. Łącznie 218 tokenów przecina maskę narzędzia co najmniej jednym pikselem.
- `../dane/images.json`: identyfikatory obrazów, filmów i pochodzenie masek.

Trzy ostatnie pliki były już lokalnie w eksperymencie `maly_vit_ruch_czastek_maska_1px`. Ich SHA256 porównano z odpowiednikami na Edenie i skopiowano identyczne małe pliki do katalogu gęstej historii. Metadanych starszego przebiegu nie podstawiono: zachowano właściwy `metadata.json` biegu gęstego. Szczegóły: [METADANE_ZDALNE_2026-10-01.json](METADANE_ZDALNE_2026-10-01.json).

Do analizy należy używać `np.load(path, mmap_mode='r', allow_pickle=False)` i pobierać wybrane kroki/tokeny. Wczytanie całej tablicy do RAM ani tworzenie drugiej kopii na dysku nie jest potrzebne.
