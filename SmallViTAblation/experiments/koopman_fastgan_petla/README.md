# Koopman–FastGAN: pętla kontrolowanego doboru prawdziwych klatek

Stan: implementacja pilota zatwierdzonego przez użytkownika. Wszystkie wejścia i wyniki są jawnie manifestowane.
Plan: PLAN.md. Oryginalne ustalenia: ../fastgan_narzedzie/PIPELINE_LATENT_KOOPMAN_FASTGAN.md.

Moduły: dane.py (podziały i parowane próbki), sonda.py i czastki.py (kontrolowane obrazy i odczyt),
dynamika.py (kierunki i prognoza), wyszukiwanie.py (osobny DINOv2–FAISS), generowanie.py (zamrożony FastGAN),
trening.py (pełne stany i gęste pomiary), petla.py (etapy i raporty).

## Uruchamianie i sprawdzenie

W katalogu eksperymentu, Python 3.11:

```powershell
py -3.11 -m pytest -q test_dane.py test_ocena.py test_dynamika.py test_wyszukiwanie.py test_petla.py
py -3.11 -m pytest -q test_granica.py
py -3.11 -m pytest -q tests
```

Dwa procesy testowe rozdzielają zależności OpenMP FAISS i lokalnego PyTorch.
Nie uruchamiają treningu detektora ani generatora. Testy numeryczne obejmują:
przeciek przyszłych checkpointów, przeciek operacji, przestawienie tokenów,
parowanie strumienia danych, przypadek zerowego przyrostu, odłączony gradient,
niezmienność wag generatora i wyszukiwanie z wykluczeniem użytych klatek.

Eden: katalog `~/smallvit_runs/2026-09-24_koopman_fastgan_petla`, środowisko
`~/venvs/rfdetr_nano_clean_1_10_1/bin/python`. Osobny `deps` zawiera wyłącznie FAISS;
nie zmieniamy wspólnego torch ani bibliotek RF-DETR. Oficjalne wagi COCO pobiera
`przygotuj_wagi.py`. Wytrenowany FastGAN i dane pochodzą z istniejących archiwów.

```bash
sbatch --export=NIL --output=smoke_%j.log code/run_eden.slurm smoke
sbatch --export=NIL --output=run_%j.log code/run_eden.slurm run
```

Drugie polecenie jest do uruchomienia po zaliczeniu testu rzeczywistych modeli.
Skrypt kopiuje kod i rozpakowuje wejścia do własnego katalogu węzła; trap sprząta
tylko ten katalog. Wyniki trafiają na współdzielony dysk. Nie używamy środowiska
powłoki logowania na węźle. Pierwsza próba z takim środowiskiem została zatrzymana
przez SLURM przed wykonaniem kodu.

`smoke.py` wykonuje krótki trening, ciągły odpowiednik i dwie identyczne kontynuacje,
a następnie gałąź z nową klatką. Sprawdza wagi, optimizer, scheduler i RNG także
przy akumulacji gradientów. Następnie sprawdza rzeczywisty gradient generatora,
mały indeks DINOv2–FAISS i ocenę detekcji. To test inżynierski, nie wynik naukowy.

## Jak czytać wyniki

- `STATUS.json`: bieżący etap; `FAILURE.json`: błąd wykonania; `REPORT.json`: koniec.
- `pretrain/FINAL.ckpt`: pełny stan do kontynuacji. `FINAL.pth`: same wagi do odczytu.
- `LATENT_*.npz`: stała sonda w kolejnych chwilach uczenia. Cztery bloki deskryptora:
  średnia na masce, średnia poza maską, moment wzdłuż x i moment wzdłuż y.
- `DIRECTION_QUALITY.json`: rzeczywista jakość odczytu etykiet na odłożonych operacjach.
  Nie nazywamy automatycznie całej znalezionej bazy pełną tożsamością narzędzia.
- Ten pilot mierzy dostępność informacji przez nadzorowany odczyt; nie certyfikuje
  przyczynowej kompletności przez usuwanie kierunków wewnątrz całego detektora.
  Takie interwencje z szerszego dokumentu pozostają oddzielnym etapem badania.
- `FORECAST.json`: błąd na późniejszych checkpointach, porównanie z brakiem zmiany i trendem.
  Dodatkowo zapisuje kontrolę z potasowaną kolejnością checkpointów.
- `generation_*/report.json`: cel przyszły `target`, cel obecny `current`, losowa zmiana
  o tej samej normie `random_norm`. Są to obrazy diagnostyczne, nie nowe etykiety.
- `retrieval_*.json`: podobne wizualnie prawdziwe klatki z treningowej puli.
- `control`, `prim`, `random`: ten sam stan początkowy, tyle samo aktualizacji; różni się
  jedna zapisana prezentacja. `LATENT_EFFECT_*` oddziela dodatkowy efekt od całej epoki.
- `test/REPORT.json`: mAP, pominięcia i fałszywe alarmy na ustalonym progu.
  Test następuje dopiero po zamrożeniu wszystkich decyzji rund.

Na starcie baza kierunków jest wyznaczana z pierwszych dwóch snapshotów i potem
zamrożona. Koopman jest ponownie dopasowywany tylko do dostępnej przeszłości.
Horyzont prognozy odpowiada całej kolejnej epoce, a nie pojedynczemu zapisowi.
Sonda obejmuje też przesunięcia, zmianę jasności samego narzędzia oraz łagodną
perspektywę 2D. Te przekształcenia nie są nowymi, rzeczywistymi widokami 3D.

## Granice pilota

Maksymalnie trzy rundy, jedno ziarno, jedna nowa klatka i jeden jej pokaz na rundę.
Nowa klatka zastępuje pokaz starej przy równym budżecie; nie jest dodatkowym krokiem.
Koopman musi pokonać obie proste prognozy z dodatnią dolną granicą bootstrapu operacji.
Zawiedzioną bramkę zapisujemy jako wynik negatywny. Nie obniżamy progów po zobaczeniu wyniku.

Obrazy sondy to transformacje płaskiego wycinka, nie odtworzenie pełnej geometrii 3D.
Wynik kanału maski FastGAN-a nie dowodzi, że RGB zawiera poprawny metal. Progi wizualne
są heurystykami pilota. Dopiero porównanie gałęzi sprawdza użyteczność klatki.
Zgodność z przyrostem Koopmana nie jest równoważna poprawie wykrywania.

Pierwszy test wykrył różnice pomiędzy identycznymi powtórzeniami treningu CUDA.
`deterministyka.py` wymusza ścisłe algorytmy i matematyczny wariant uwagi SDPA.
Operację `grid_sample` wykonuje na CPU z zachowaniem gradientu przez transfery.
Dotyczy to pretreningu i wszystkich kontynuacji. Jest to wolniejszy wariant
diagnostyczny, bez zmian w instalacji RF-DETR. Powtarzalność potwierdza dopiero
test ciągły trening–wznowienie oraz powtórzenie; sama flaga jej nie dowodzi.
Wcześniejsze sesje używały tej partycji testowej: nie nazywamy jej nowym, nigdy nieoglądanym testem.

`pobierz.py --job ID --mode run --out NOWY_KATALOG` pakuje i pobiera końcowe wagi,
raporty, obrazy i latent oraz sprawdza SHA256 każdego pliku. Nie usuwa źródeł.

## Wykonanie lokalne i Eden, zmiana z 2026-09-24

Na prośbę użytkownika analiza Koopmana, dobór kierunków, sterowanie zamrożonym
FastGAN-em oraz indeks DINOv2–FAISS i wyszukiwanie działają na lokalnym RTX 3070.
Trening RF-DETR i jego ocena pozostają na Edenie. Zapisy latentu przy checkpointach
powstają przy treningu na Edenie i są pobierane do lokalnej analizy.

Pretrening zadania 1810448 nie jest powtarzany. Zakończył cztery epoki i uzyskał
około 0,6574 mAP na walidacji. Wcześniejszy orkiestrator został wstrzymany przed
przejściem do analizy; samo uczenie i jego ocena mogły zakończyć się normalnie.
`handoff_eden.py` pakuje historię, pełny stan optymalizatora, wagi, sondę oraz
sumy rzeczywistych obrazów użytych przez trening. `hybryda_lokalna.py` sprawdza
sumy pobranych plików przed zwolnieniem starego orkiestratora. Zakończenie jego
procesu po przekazaniu jest celowe, nawet jeśli SLURM oznaczy zadanie jako FAILED.
Dowodem ukończenia uczenia jest `pretrain/STATUS.json`, a przekazania `HANDOFF.json`.

`LOCAL_ENV.json` zawiera ścieżki i weryfikację lokalnego środowiska.
`HYBRID_STATE.json` wskazuje katalog lokalnych wyników i aktualny etap.
`hybryda_zdalne.py` wysyła na Eden jedynie zatwierdzone w planie dalsze treningi
oraz ocenę. `przenosny_checkpoint.py` przepisuje wyłącznie ścieżki kontraktu po
sprawdzeniu niezmienionych obrazów, adnotacji i sondy; nie zmienia stanu uczenia.
Każdy worker usuwa tylko własny katalog tymczasowy węzła.

Lokalna pula pochodzi z tego samego archiwum co ukończony trening, sprawdzonego
sumą SHA256. Generator pozostaje istniejącym modelem po 50000 krokach.
Poszczególne etapy uruchamiają się kolejno, aby nie zajmować GPU jednocześnie.
Czas lokalnego generowania trzeba zmierzyć; dotychczasowy test na A100 nie jest
wiarygodną prognozą szybkości RTX 3070.

Istotne ograniczenie znalezione przy przenoszeniu: bieżący manifest puli ma
10637 obrazów train i 12043 val, ale nie ma osobnej partycji test. W tej wersji
`ocena.py` zapisuje dla pustego testu `state=unavailable`, zamiast podawać metrykę
lub wykorzystywać walidację pod nazwą test. Wynik walidacyjny nie jest zatem
potwierdzeniem poprawy na niezależnym teście. Nie zmieniamy podziału po wynikach.
