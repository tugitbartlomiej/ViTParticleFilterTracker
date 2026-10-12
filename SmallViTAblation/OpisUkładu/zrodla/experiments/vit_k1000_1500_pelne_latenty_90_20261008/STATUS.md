# Stan końcowy

- Dokładny manifest ma 1500 obrazów i SHA-256
  `33caf6f005b3db0d091d6e9ba9a58cec141a859c66fcd7bd777d31e6fe1a80d7`.
- Przed wdrożeniem wykryto historyczny błąd skalowania ramek: 203 obrazy nie
  mają rozmiaru 720×480, chociaż loader kampanii mnożył każdą ramkę przez 2/3.
  Manifest zachowuje te same 1500 klatek, ale IoU liczy względem prawidłowo
  przeskalowanego GT. Stara ramka kampanii pozostaje osobnym polem audytowym.
- Checkpoint K1000 ma SHA-256
  `3df42a6adc1c47075a52f85641772e19be430280a7d27e0f6635b6753db4649e`.
- Horyzont to 500 rzeczywistych aktualizacji dla każdego obrazu. Pełne wspólne
  stany FP32 są zachowane w `k=0`, `k=100` i `k=500`; `after_step` wskazuje
  rzeczywisty krok wybranego stanu oracle.
- Surowy budżet dużych tensorów wynosi 14 610 825 216 bajtów. Nie przyznano
  kredytu za kompresję. Preflight wymaga dodatkowo 800 MB na smoke i małe
  artefakty oraz 5 GB rezerwy operacyjnej.
- Audyt Nasha z 8 października widział 21 029 957 632 bajty wolne. Rozszerzony
  audyt zakończył się statusem `PASS` i planem odzyskania 127 652 560 896
  bajtów bloków plików. Użytkownik zatwierdził usuwanie i przekazał, że Nash
  wykonuje teraz wyłącznie `PLAN_USUNIECIA.csv`.
- Sprzątanie zakończyło się usunięciem 35 728 pozycji bez pominięć. Niezależna
  kontrola wykazała 148 682 518 528 bajtów wolnego, brak procesu kasowania,
  pustą kolejkę `k1k1500`, brak zdalnego katalogu tej rodziny i zgodne SHA-256
  wszystkich czterech chronionych wejść. Użytkownik zwolnił blokadę wdrożenia.
- Kod wdrożono do
  `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_1500_pelne_latenty_90_20261008`.
  Właściwy smoke `1824869` zakończył się `COMPLETED`, `exit 0:0`, na GPU H100.
  Ukończył 8/8 obrazów, poprawił funkcję celu na 8/8 i osiągnął oba progi na
  2/8; pozostałe 6/8 zapisano uczciwie jako wyczerpanie budżetu 500 kroków.
  Pierwszy techniczny smoke `1824864` zakończył się przed obliczeniami z powodu
  ścieżki skryptu w katalogu bufora SLURM; przyczynę naprawiono i zachowano jego
  marker jako historię próby.
- Produkcja `1824873` zakończyła 8/8 niezależnych shardów H100 z `exit 0:0`.
  Tak samo zakończyły się merge `1824877`, movement `1824878`, regimes
  `1824879` i DMDc `1824880`. Pierwsze evaluation `1824881` zatrzymało się
  przed oceną obrazów, ponieważ informacyjne pole provenance wymagało metadanych
  nieobecnego, opcjonalnego `timm`. Po teście regresji poprawiono wyłącznie ten
  punkt; lokalnie przeszło 18/18 testów, a na Edenie 17/17 z jednym pominiętym
  testem referencyjnym `timm`. Wznowione evaluation `1824979` i final
  `1824980` zakończyły się `COMPLETED`, `exit 0:0`; stary final `1824882` został
  anulowany po `DependencyNeverSatisfied`.
- Oracle zakończył 1500/1500 unikalnych obrazów: 1283 osiągnęły jednocześnie
  confidence i IoU co najmniej 0,90, a 217 uczciwie wyczerpało 500 kroków.
  Mediany baseline/after wyniosły odpowiednio 0,771/0,938 dla confidence oraz
  0,666/0,923 dla IoU. Mediana pierwszego sukcesu to 58 kroków, a 90. percentyl
  156 kroków. Wszystkie artefakty przeszły ponowne wczytanie; wagi i bufory
  miały niezmieniony hash
  `443ef243f956ead49a7b9119cf3d454d93251476115006d1416a88d45415246e`.
- Wspólna metoda A nie spełniła obu progów dla żadnego obrazu i obniżyła
  mAP50:95 z 0,245496 do 0,239411 in-sample oraz do 0,239096 w holdoucie po
  filmach. Metoda B również miała 0 przypadków obu progów i mAP50:95 0,244481
  in-sample oraz 0,242897 w holdoucie. Żadna wspólna reguła nie pomaga wszystkim.
  Oracle osiągnął mAP50:95 0,861131, ale jest dopasowaniem per-image na tym
  samym zbiorze i nie dowodzi generalizacji ani osiągalności z pikseli.
- DMDc użył 128 prawdziwych trajektorii i 64 000 przejść, z gradientem i
  momentami Adama sprzed aktualizacji jako sterowaniem. Na odłożonych całych
  trajektoriach persistence miało RMSE 0,080697, a DMD i DMDc około 0,16673;
  DMDc nie poprawił DMD. Klasyfikator reżimów z samego naturalnego `z0` i
  kontekstu odtworzył 36,06 procent etykiet reżimu na danych uczących.
- Zdalny root ma 11 474 808 890 bajtów; po zakończeniu pozostało
  134 170 943 488 bajtów wolnego. Nie zapisano plików obrazu. Raporty,
  analizy i dwa samodzielne pakiety skopiowano lokalnie: 1547 plików,
  355 813 711 bajtów, hash drzewa
  `d179b873f883386aed86a23726f6da06ddcca1803c81d2a3477a91c8332e4131`.
- Chronione zależności: dokładne wagi K1000, pełny stan AdamW `pelny/k1000.pt`,
  dokładny COCO 1500 oraz `tracking_static36.tar`. Pipeline nie usuwa i nie
  modyfikuje tych plików. Preflight sprawdza teraz także pełne SHA-256 stanu
  AdamW i archiwum źródłowego, a nie tylko ich rozmiary.
- Wybrane środowisko `rfdetr_nano_clean_1_10_1` i jego wymagane ścieżki bazowe
  również są w `ZACHOWAJ_MAPA.json`. Nie wykryto żadnej zależności pipeline'u
  poza tą mapą.
- Po sprzątaniu środowisko GPU nie zawiera `timm` ani `pytest`. Kod ma
  samodzielny blok ViT zgodny bitowo z lokalnym `timm` i dokładnym K1000, a
  testy używają zachowanego `nndmd_expanded_20261002`. Oba środowiska są w
  `ZACHOWAJ_MAPA.json`; niczego w nich nie instalowano.

Skrypty SLURM obejmują smoke, 8 niezależnych shardów GPU, scalanie, analizę
ruchu, reżimy, DMDc, budowę pakietów A/B, fresh-load ewaluację i finalizację.
Każdy worker oracle wznawia pracę per obraz dopiero po sprawdzeniu metadanych i
SHA istniejącego artefaktu. Markery i atomowe locki blokują przypadkowe
podwójne zgłoszenie.

Walidacja lokalna: 18 testów kontraktu, parsowanie wszystkich plików Python,
kontrola składni Bash/SLURM i PowerShell. Test techniczny fast tail na dokładnym
K1000 dał maksymalny błąd bezwzględny `1.1920928955078125e-07`; nie wykonywał
optymalizacji ani nie powtarzał pilota użytkownika.
