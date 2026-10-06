# Podsumowanie sesji 2026-10-07 00:21:39

## Cel sesji

Użytkownik chciał mieć kilka małych ViT o różnej wielkości latentu, wytrenowanych na zmniejszonym zbiorze
(około 5000 zdjęć), żeby mieć podgląd na różne wielkości latentu, w tym wersję z 8 tokenami (TokenLearner).
Przygotowanie wszystkiego na Edenie i pytanie przed uruchomieniem; zgoda padła przed wysłaniem zadań.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, gałąź `codex/eden-nano-training-20260914`.
- Poprzednie części rozmowy: `.sessions\Session_2026-10-06_171603_...` i
  `.sessions\Session_2026-10-06_220807_Aplikacja_PCA_Czubek_GUI_Ustawienia_Odpinanie` (plan TokenLearner
  `experiments\maly_vit_tokenlearner_8\PLAN.md`, pomiar RTX 3070).
- Wzorzec kodu i SLURM: `experiments\maly_vit_gesta_historia_paczka` (MalyViT, trening.Uczen, dane, bieg_paczka.slurm
  ze sprzątaniem /dev/shm pułapką EXIT).
- Eden przed biegiem: kwota 165,8 / 200 GB, 110 tys. / 200 tys. plików, brak zadań w kolejce; wejścia
  z `2026-09-28_maly_vit_od_zera/wejscia`, `deps/timm`, `2026-10-01_..._gesta_historia/wejscia/obrazy.npy`,
  `tracking_static36.tar` obecne.

## Wykonane kroki

1. Lista znanych ViT z wielkością latentu (ViT-B/S/Ti, MobileViT, TinyViT, LeViT, TokenLearner, TiTok) w PLAN.md,
   oznaczona jako z pamięci, do sprawdzenia przed cytowaniem.
2. Nowy eksperyment `SmallViTAblation\experiments\maly_vit_rozne_latenty_5000` (kopia kodu, zgodnie z zasadą
   „eksperyment = osobny folder”):
   - `kod\model.py`: MalyViT z parametrami szerokości, głów i opcjonalnym TokenLearnerem (wersja 1.1: MLP z GELU,
     sigmoid, średnia ważona) po bloku 3 plus TokenFuser; słownik RAMIONA; latent tokeny x (4 x szerokość).
   - `kod\trening.py`: Uczen(ziarno, konfig); reszta przepisu bez zmian.
   - `kod\dane.py`: `zbior_n(5000)` = baza 1500 + 3500 klatek z puli (split train, z ramką, bez 1501 wykluczonych)
     równo po 118 filmach; nadpisywanie ścieżek wejść zmiennymi środowiska (uruchomienie lokalne).
   - `kod\bieg.py`: jedno ramię, ocena mAP@0,5 na 3000 klatkach val co 500 kroków, zapis wag i latentu 12 klatek
     pomiaru co 500 kroków (przy TokenLearnerze także map uwagi), MODEL_<ramię>.pt z konfiguracją.
   - `bieg_ramiona.slurm`: zadanie tablicowe 0..6 z `%1` (jedno ramię naraz), sprzątanie /dev/shm i /tmp.
3. Sprawdzenia lokalne: dobór 5000 (3500 unikalnych, 19 do 30 na film, 0 wspólnych z bazą); próba 20 kroków
   femto, tiny_tl8, small (kształty latentu 600 x 128, 8 x 768 z mapami 600 x 8, 600 x 1536); tiny ma budowę
   MODEL_VIT_K1000 (wczytanie strict). Błąd w `ksztalt_latentu` złapany próbą i poprawiony.
4. Wysłanie na Eden do `~/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000` (sumy SHA256 zgodne, py_compile).
5. Uruchomienie po zgodzie: próba `1823464` (7 x 20 kroków, wszystkie zakończone), bieg `1823465` z
   `--dependency=afterok` (7 x 5000 kroków, COMPLETED, 2:27 do 3:03 na ramię, H100 PCIe, 24 do 28 ms na krok).
6. Wyniki (mAP@0,5 na 3000 klatkach val, koniec / najlepszy): femto 600 x 128 0,559; pico 600 x 256 0,585;
   nano 600 x 384 0,612; tiny 600 x 768 0,654; small 600 x 1536 0,589 / 0,624; tiny_tl8 8 x 768 0,595 / 0,628;
   nano_tl8 8 x 384 0,542 / 0,608. Mapy TokenLearnera: w nano_tl8 token 4 ma średnią mapę 0,01 i 15,6 razy większą
   masę na patchach narzędzia niż mapa równa; w tiny_tl8 mapy szerokie (najwyżej 1,41 x).
7. Ściągnięcie wyników (126 plików) do `wyniki\pelny_<ramię>`, raport `WYNIKI.md`, deklaracja `WYNIKI.json`.
8. Fałszywy alarm pętli obserwacji (chwilowy brak odpowiedzi SSH wzięty za pustą kolejkę): sprawdzone w logu,
   pętla poprawiona (znacznik SSH_OK).

## Pliki zmienione / utworzone

### W repozytorium ViT (`F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`)

- `SmallViTAblation\experiments\maly_vit_rozne_latenty_5000\PLAN.md`, `WYNIKI.md`, `WYNIKI.json`, `bieg_ramiona.slurm`
- `SmallViTAblation\experiments\maly_vit_rozne_latenty_5000\kod\model.py`, `dane.py`, `trening.py`, `bieg.py`
- `SmallViTAblation\experiments\maly_vit_rozne_latenty_5000\wyniki\pelny_<ramię>\dane\*.json` (w gicie),
  `MODEL_<ramię>.pt`, `wagi\`, `pelny\`, `dane\LATENT_KONTROLNY.npz` (poza gitem: `*.pt`, `*.npz`)
- Commity: `f2addbd1` (eksperyment), `77b5dec6` (wyniki).

### Zewnętrzne (poza repozytorium)

- Eden: `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-06_maly_vit_rozne_latenty_5000` (kod, logi, wyniki,
  2,2 GB, 253 pliki); kwota po biegu 167,9 / 200 GB.
- Pamięć podręczna danych próby lokalnej w katalogu roboczym sesji (`%TEMP%\claude\...\scratchpad\rozne_latenty`).
- Nic nie zmieniano w projekcie grantowym ani w `external/`.

## Otwarte problemy

- Na węźle Edenu `hopper` w `/dev/shm` leżą dwa stare katalogi użytkownika z zadań `kont245` (1814538, 2026-09-25)
  i `plarge` (1817335, 2026-09-27); nie pochodzą z tego biegu, czekają na decyzję o usunięciu.
- Jedno ziarno, szum ocen 0,03 do 0,05: kolejność small i ramion _tl8 względem tiny niepewna.
- Aplikacja obsługuje tylko latent 600 x 768; modele innej wielkości wymagają przeróbki widoków.

## Następne kroki

- Na życzenie: wybór modelu w aplikacji i widoki dla dowolnej wielkości latentu (8 tokenów z mapami uwagi).
- Ewentualnie drugie ziarno albo dłuższe uczenie small, żeby rozstrzygnąć jego położenie.
- Sprawdzenie tokenu narzędzia nano_tl8 na sekwencjach filmu.
