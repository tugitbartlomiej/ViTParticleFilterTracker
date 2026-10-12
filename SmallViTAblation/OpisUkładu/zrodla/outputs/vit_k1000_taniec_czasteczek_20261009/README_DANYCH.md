# Dane do lokalnego podglądu tańca cząsteczek K1000

Eksport dotyczy rzeczywistych edycji aktywacji przy zamrożonych wagach K1000. Nie jest trajektorią uczenia wag. Eksporter nie uruchamia modelu, treningu ani generowania lub odczytu obrazów. Aplikację, serwer, registry i próbkowanie podglądu prowadzi osobny wykonawca.

## Pliki i status

Docelowy plik: `data/projection.npz`. Kontrakt opisuje `data/metadata.json`, a `data/BUNDLE_MANIFEST.json` zawiera rozmiary i SHA256 wszystkich składników. Obecność samego NPZ nie oznacza ukończenia: wymagany jest bundle manifest i pozytywny wynik lokalnej weryfikacji. Eksport i lokalny transfer są zakończone. Gate LOCAL_TRANSFER.json ma status complete, ready=true, bitwise_verified=true oraz schema_verified=true. Wyniki i rozmiary są poniżej.

Nazwa finalna została doprecyzowana podczas trwającego eksportu. Pierwsze zadanie na Edenie zapisuje `particles.npz`; po weryfikacji transferu lokalny plik jest jednorazowo przemianowany na `projection.npz`, bez zmiany bajtów tablic. `BUNDLE_MANIFEST_EDEN.json` zachowuje pierwotny manifest, a `export_projection_executed.py` dokładną wersję eksportera uruchomioną na Edenie. `LOCAL_TRANSFER.json` dokumentuje zgodność hashów i nazwę finalną.

## Stabilny schemat NPZ

Wczytywać przez `numpy.load(path, allow_pickle=False)`. Wszystkie dane są tablicami liczbowymi; brak obiektów wymagających pickle. Identyfikatory tekstowe i komplet metadanych obrazów są w JSON. Kopia schematu w NPZ to `schema_json_utf8`, bajty UTF-8 jako uint8.

| Klucz | Kształt | Typ | Znaczenie |
|---|---|---|---|
| positions | 1500 × 3 × 600 × 3 | float32 | Wszystkie obrazy, milestone, tokeny, trzy osie wspólnego PCA |
| movement_norm | 1500 × 3 × 600 | float32 | L2 różnicy względem before, obliczone w oryginalnych 768D |
| milestone_steps | 3 | int32 | Dokładnie 0, 100, 500; 0 pochodzi z before |
| image_indices | 1500 | int32 | Indeksy manifestu 0–1499, bez zmiany kolejności |
| token_indices | 600 | int32 | Tokeny 0–599; row = token // 30, col = token % 30 |
| control_positions | 16 × 501 × 8 × 3 | float32 | Wszystkie 128 rzeczywistych trajektorii kontrolnych, kroki 0–500 |
| control_norm | 16 × 501 × 8 | float32 | L2 w 768D względem początkowego stanu danego tokenu |
| control_step_norm | 16 × 501 × 8 | float32 | L2 w 768D względem poprzedniego rzeczywistego kroku; dla 0 wartość 0 |
| control_images | 16 | int32 | Globalny indeks obrazu dla każdej pozycji panelu |
| control_tokens | 16 × 8 | int32 | Prawdziwy globalny indeks tokenu w danym obrazie |
| control_steps | 501 | int32 | Dokładnie 0–500, bez interpolacji |
| confidence | 1500 × 501 | float32 | Rzeczywiste confidence top-1 z telemetrii oracle |
| iou | 1500 × 501 | float32 | Rzeczywiste IoU top-1 względem GT z telemetrii oracle |
| after_positions | 1500 × 600 × 3 | float32 | Projekcja najlepszego rzeczywiście odwiedzonego celu oracle |
| after_norm | 1500 × 600 | float32 | L2 after−before w pełnych 768D |
| pca_mean | 768 | float64 | Jedna wspólna średnia |
| pca_components | 768 × 3 | float64 | Jedna wspólna baza; kierunki w kolumnach |
| pca_explained_variance | 3 | float64 | Trzy największe wartości własne kowariancji |
| pca_explained_variance_ratio | 3 | float64 | Udział wariancji trzech osi w próbce fit |
| pca_fit_image_indices | 64 | int32 | Jawna, deterministyczna próbka fit |

`after_positions` nie jest kolejnym milestone i nie musi odpowiadać krokowi 500. Właściwy krok jest w `metadata.images[i].after_step`. Nie wolno wstawiać after do osi czasu 500 ani interpolować brakujących stanów wszystkich 600 tokenów.

Przykład mapowania panelu: `control_positions[k, s, j]` opisuje obraz `control_images[k]` i token `control_tokens[k,j]` w rzeczywistym kroku s. To 16 obrazów z ośmioma wybranymi tokenami każdy, nie 128 obrazów.

## Wspólna projekcja

Projekcja każdego stanu to `(z.astype(float64) - pca_mean) @ pca_components`, następnie zapis float32. Nie ma standaryzacji kanałów, whitening, osobnego PCA per krok ani osobnego PCA per obraz. Nie należy ponownie dopasowywać bazy w aplikacji.

Fit obejmuje 64 obrazy × 600 tokenów × trzy rzeczywiste stany 0/100/500, czyli 115 200 punktów. Próbka jest warstwowana: 64 rozłączne pasma uporządkowanych identyfikatorów filmów, w każdym jeden film i jedna klatka wybierane deterministycznie z seed 20261009. Jest 118 filmów źródłowych; nie twierdzimy, że próbka obejmuje wszystkie. Wagi punktów są jednakowe; wszystkie 600 tokenów wybranych obrazów jest użyte.

Średnia i kowariancja są obliczone strumieniowo w float64, z centrowaniem partii i łączeniem Chan, dzielnik N−1. Kierunki to top-3 `numpy.linalg.eigh`. Znak każdego kierunku jest ustalony tak, aby współczynnik o największej wartości bezwzględnej był dodatni. Gotowa baza i jej hash są zapisane. Udziały wariancji dotyczą tej próbki, a nie skuteczności detekcji ani jakości rekonstrukcji pełnego ruchu.

## Metadane i provenance

`metadata.images` ma 1500 rekordów w kolejności indeksu. Zawiera identyfikatory obrazu, klatki i adnotacji, GT `bbox_xyxy_480`, docelowy token, status, `first_success_step` (może być null), `after_step`, hash obrazu źródłowego, checkpointa, manifestu i każdego źródłowego NPZ. Nie zawiera pikseli.

Źródło na Edenie:

```
/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/results/production/latents/000000.npz
...
/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008/results/production/latents/001499.npz
```

Manifest danych wspólnych ma rzeczywistą nazwę `MANIFEST_1500.jsonl`, nie `MANIFEST1500.json`. Jego SHA256: `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`. SHA256 `CONTROL_PANEL.json`: `f764ffb966e24e81e9e52257be1006c5365369f7ae382d181c8c23860930287f`. Obie wartości odpowiadają lokalnym plikom w provenance istniejącej rodziny wyników.

Wszystkie 1500 NPZ są zahashowane przed dopasowaniem PCA. Każde wczytanie fit/eksport odbywa się z tych samych bajtów, których hash sprawdzono. Po obliczeniach ponownie sprawdzane są wszystkie hashe źródeł i oba pliki provenance. Dopiero potem powstaje manifest gotowy do transferu. Do lokalnego katalogu trafiają wyłącznie projekcje, JSON-y, mała kopia manifestu/panelu i raporty; surowe 11 GB latentów pozostają na Edenie.

`VALIDATION.json` zapisuje skończoność, schemat, zgodność surowych stanów panelu z milestone, błędy niezależnego ponownego liczenia projekcji na 16 obrazach spoza próbki fit, zgodność hashów i środowisko CPU. Eksport korzysta z zadania Slurm z dwoma CPU i bez GPU. Nie ingeruje w inne joby.

## Kontrakt podglądu

- Domyślny podgląd: etykieta **8000 / 900000 cząsteczek (próbka)**. Próbkowanie wykonuje aplikacja.
- Pojedynczy obraz: pełne 600 tokenów w krokach 0/100/500.
- Tryb dokładny: wyłącznie 128 wskazanych trajektorii panelu, wszystkie 501 kroków.
- Pełne 900 000 cząsteczek jest dostępne lokalnie w tablicy positions dla rzeczywistych milestone.
- Norma ruchu pochodzi z pełnych 768D; odległość na ekranie PCA nie jest jej zamiennikiem.

Weryfikacja lokalna, bez obrazów i bez połączenia z Edenem:

```powershell
& F:/Python/python.exe experiments/vit_k1000_taniec_czasteczek_20261009/export_projection.py verify `
  --output outputs/vit_k1000_taniec_czasteczek_20261009/data
```

## Wynik końcowy — pomierzone dane

- Finalizacja CPU: Slurm 1825064, COMPLETED, exit 0; bazę i projekcje wykonał wcześniejszy job 1825060.
- Pierwszy job zakończył obliczenia i kontrolę źródeł, lecz zgłosił błąd serializacji int64 do JSON. Finalizacja naprawiła raport, ponownie sprawdziła świeże projekcje i wszystkie hashe, bez ponownego fit i bez zmiany bajtów tablic.
- projection.npz: 49,429,943 bajtów; około 49.430 MB.
- metadata.json: 2,539,035 bajtów.
- 900 000 cząsteczek, trzy rzeczywiste milestone; 128 trajektorii dokładnych po 501 stanów; komplet telemetrii 1500 × 501.
- Wszystkie wymagane kształty są zgodne z ustaleniami; nie było potrzeby interpolacji ani zmiany schematu.
- Lokalna kopia manifestu i panelu została dodatkowo porównana z istniejącym lokalnym provenance: oba hashe zgodne.
- PCA: 69.601360, 7.232019, 2.455562 procent wariancji; razem 79.288941 procent.
- Maksymalny błąd świeżej kontroli projekcji milestone: 3.79996038191e-06.
- Maksymalny błąd świeżej kontroli norm 768D: 6.18263928231e-06.
- Maksymalny błąd kontroli projekcji wszystkich trajektorii panelu: 3.81381462944e-06.
- Różnice są błędami zapisu float32 względem niezależnego liczenia float64; kontrola skończoności oraz zgodności panelu z milestone przeszła.

SHA256 projection.npz:

```
95d61d173ddb404abf7deadde10f112a3e3fc9d12fa3e63abb74a3ca7321f36a
```

SHA256 wspólnej bazy (mean, potem components, little-endian float64):

```
8d3cb28ad2b1581efeca89fab659b9d6fa518bffe4d452b64c68ea7dcd1b9cca
```

SHA256 zamrożonego drzewa wszystkich źródeł:

```
5294a6c5447ac89f9d9e5de2f68c6c2aec1bb2edf5ad44e7da87a31e8a274ff5
```

Nie zmieniono plików UI, serwera, planu ani registry. Nie uruchomiono GPU i nie pobrano obrazów lub pełnych latentów.
