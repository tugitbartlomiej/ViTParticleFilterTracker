# Stan przygotowania

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
- Ponowny odczyt tylko do wglądu potwierdził ten sam stan miejsca, brak jobów
  `k1k1500` oraz brak zdalnego katalogu tej rodziny. Nie wdrożono kodu i nie
  zgłoszono smoke ani produkcji.
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

Walidacja lokalna: 14 testów kontraktu, parsowanie wszystkich plików Python,
kontrola składni Bash/SLURM i PowerShell. Test techniczny fast tail na dokładnym
K1000 dał maksymalny błąd bezwzględny `1.1920928955078125e-07`; nie wykonywał
optymalizacji ani nie powtarzał pilota użytkownika.
