# Podsumowanie sesji: równanie latentu małego ViT na Edenie i Stable Diffusion uczone na całych filmach

Data: 2026-09-29 (14:52) - 2026-09-30 (12:43).
Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, commity od `3ce17b1c` do `056127e9`.
Rozmowa prowadzona z katalogu projektu grantowego `F:\Studia\MCHTR\Grant\Projekt` (gałąź
`feature/koopman-tft-experiments`), ale cała praca merytoryczna dotyczy repozytorium ViT.

## Cel sesji

1. Zmierzyć, na ile kroków naprzód Koopman i Langevin przewidują siłę tożsamości narzędzia lepiej niż „bez zmiany”.
2. Przejrzeć literaturę o generowaniu obrazów, które trafiają w zadany latent sieci (Stable Diffusion, prowadzenie
   gradientem), i przeczytać najważniejsze prace w całości.
3. Na prośbę użytkownika przerobić równanie Langevina z TFT na równanie ruchu latentu małego ViT w trakcie uczenia,
   ze sterowaniem danymi, aż do co najmniej 80 procent trafności na jeden krok; potwierdzić na Edenie.
4. Sprawdzić pomysł użytkownika z TS: duże, uczone macierze (jądro pamięci, wymuszenie, dryf, szum) w równaniu dla
   latentu i wektorów tożsamości.
5. Posprzątać Eden, wysłać tam całe filmy Cataract-LMM i uczyć na nich Stable Diffusion na tylu kartach, ile się da.

## Kontekst

- Poprzednia sesja: `Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek` (pętla siły
  tożsamości na małym ViT; zysk z wyszukiwania podobnych klatek, nie z prognozy).
- Model cząstkowy użytkownika: token = cząstka w 768-wymiarowej przestrzeni latentu; tożsamość narzędzia to wąska
  wiązka kierunków, osobna dla każdej sieci.
- Stałe reguły: plan przed kampanią GPU i czekanie na „tak”; nie podmieniać metody użytkownika bez nazwania każdej
  różnicy; bez usuwania plików bez wyraźnej zgody; foldery równoległych sesji tylko do odczytu; na Edenie pułapka
  sprzątająca, `sed` na końce linii, sprawdzenie wejść i kwoty `lfs quota -p 1299 /mnt/evafs` przed `sbatch`.
- Nowa reguła użytkownika z tej sesji (2026-09-29): **do prognozy nie wchodzą przyszłe klatki**. Dozwolone są
  stany do chwili t, paczki do t-1 i pęd optymalizatora. Doprecyzowanie z 2026-09-30: **wyjaśnianie bieżącego
  latentu klatkami, które go w tej chwili stworzyły, jest dozwolone** (to tryb wyjaśnienia, nie prognoza).

## Wykonane kroki

1. **Horyzont prognozy** (`SmallViTAblation\experiments\maly_vit_horyzont_prognozy`, `3ce17b1c`). Na danych z Edenu
   Koopman i Langevin, każdy osobno, biją „bez zmiany” aż do 1000 kroków, ale błąd kwadratowy spada tylko o
   10-25 procent. Langevin krok po kroku najlepszy na 100-250 kroków, Koopman równo około 13 procent, model
   dopasowany wprost na docelowy horyzont trzyma 15-25 procent do 1000 kroków.
2. **Literatura** (`335c8528`, `ddacbc74`): notatka `fastgan_odwrotnosc_mlp\NOTATKA_LITERATURA.md` i pełne teksty
   10 prac przeczytanych przez Undermind (`literatura\pelne_teksty_sd_2026-09-29.md`; Universal Guidance nie dało się
   otworzyć). Wnioski dla użytkownika: metoda Farida (lekkie zaszumienie kotwicy i filtr zgodności podpowiedzi
   sieci z generatorem) nie przesuwa narzędzia, więc odpada; różne położenia dają ReNO (dobór szumu startowego,
   pasuje do FastGAN-a) i FreeDoM (dyfuzja od zera prowadzona małym ViT). Jedna liczba z wklejki użytkownika
   (Shirahama, cosinus powyżej 0,99) okazała się zawyżona.
3. **Równanie latentu ze sterowaniem danymi** (`maly_vit_rownanie_latentu`, `019030f4`). Nagranie uczenia małego
   ViT z latentem sond i siłą paczki S(t) (zmiana latentu sond po kroku wag tylko z bieżącej paczki). T1 bez
   sterowania: pełne równanie z TFT 60 procent na krok. T2 ze sterowaniem: `Δz(t+1) = 0,83 Δz(t) + 0,99 S(t)`,
   85,8 procent na krokach 2000-3000 niewidzianych przy dopasowaniu; placebo (siła sprzed 250 kroków) spada do
   poziomu bez sterowania.
4. **Kampania Eden 1818331** (10 ziaren, 47-56; `5a2ac3b1`, `zbierz_eden.py`, `WYNIKI_EDEN.md`). Mediana 85,4
   (81,3-89,4), proste równanie 85,0 (80,5-89,2), a = 0,816-0,846, b = 0,977-0,990, placebo najwyżej 0,01 ponad T1.
   Każde ziarno powyżej 80 procent. `nagranie.py` dostało `--sily-od` i `--wagi-co`.
5. **Ujawnienie użytkownikowi**: wynik 85 procent korzysta z siły bieżącej paczki, czyli z klatek kroku, który
   właśnie się dzieje. Po regule o przyszłych klatkach opisane w `WYNIKI_T2.md` i `WYNIKI_EDEN.md` jako tryb
   wyjaśnienia, a po doprecyzowaniu użytkownika (`4a0c9a92`) jako dozwolony.
6. **Uczone macierze jak w TS** (`maly_vit_rownanie_macierze`, `c85c75c6`, `37a022cd`). Przejrzany kod TS
   (`tft_langevin_latent_56\model.py`): tam wymuszenie miało do 3,2 mln parametrów, dryf do 1,8 mln. Tutaj człony D
   (sieć MLP), JP (jądro pamięci, okno 32), W (wymuszenie jako uwaga od sondy do 32 obrazów z 16 opóźnieniami),
   N (szum kolorowy), razem około 11 mln parametrów, uczone osobno na każde ziarno; W widzi tylko paczki do t-1.
   Wyniki: prognoza z siłą liczoną z samego pędu optymalizatora 64-67 procent, sama bezwładność 0,76 Δz(t) 57
   procent, jądra pamięci z T1 59 procent, uczone człony jak w TS około 40 procent (uczą się okna dopasowania na
   pamięć), W z przeszłych paczek równe placebo, pełne macierze 768 x 768 (regresja grzbietowa) 52-54 procent.
   Wektory tożsamości różnych ziaren mają cosinus 0,085, więc równania nie da się przenieść między sieciami.
   Znaleziony błąd w TS: `SzumKolorowy` przez dopełnienie OKNO-1 i dodatkowe przesunięcie widzi reszty od t-2.
7. **Zapis metody** (`1a85dfbc`): `SmallViTAblation\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md` (dwa tryby, równania,
   liczenie siły, tabela trafności, co nie działa, ograniczenia), odnośnik w `docs\README.md`.
8. **Dyfuzja pikselowa wycinków narzędzia** (`dyfuzja_narzedzie`, `62f5a7ed`, `fc7ae09e`). Plan i kod modelu
   dyfuzyjnego uczonego od zera z prowadzeniem małym ViT (FreeDoM); wycinanie okna wokół narzędzia w trakcie uczenia
   z klatek `context_train.tar` na Edenie plus wysłane obrysy (23 MB); uczenie 1818354 (0,38 s na krok). Wątek
   przejęła dalej równoległa sesja; w tej sesji tylko diagnoza złych kolorów: pierwszy krok DDIM przy t = 999, gdzie
   pierwiastek z ᾱ wynosi 5e-5 i wzmacnia błąd x0; start od t = 979 to naprawia (poprawka należy do
   `siatka_krokow` w `model_dyfuzji.py`, nie wprowadzona, bo folder należy do tamtej sesji).
9. **Eden: miejsce i sprzątanie** (`13166fa0`, `1f4de9d1`, `SmallViTAblation\docs\EDEN_SPRZATANIE_2026-09-30.md`).
   Analiza bez usuwania, potem po „Usuwaj” użytkownika: 117 -> 57 GiB. Limit konta `transformers_vsc` sprawdzony:
   5 x H100, 17 x A100, 20 kart razem dla wszystkich użytkowników.
10. **Wysyłka Cataract-LMM** (`SmallViTAblation\eden\wyslij_cataract_lmm.sh`). Cały `3_Object_Tracking`: 170 filmów MP4
    (5,6 GB) i 170 archiwów klatek z obrysami cystotomu, rogówki i źrenicy (42 GB) do
    `~/smallvit_data/cataract_lmm/3_Object_Tracking/`. Trzy strumienie scp, około 5 MB/s, 3 h, sumy SHA256 zgodne.
11. **Stable Diffusion 1.5 na całych filmach** (`dyfuzja_sd_filmy`, `2bf184b0`, `056127e9`). Latentna dyfuzja z VAE
    sd-vae-ft-mse, obraz 576 x 384, latent 72 x 48; wejście UNet rozszerzone z 4 do 7 kanałów o maski narzędzia,
    rogówki i źrenicy (nowe wagi od zera, maski wyłączane w 20 procentach przykładów); stały pusty tekst, offset
    noise 0,05, AdamW 2e-5, EMA 0,9999, bf16, DDP przez torchrun z paczką globalną około 256; zapis, wznowienie i plik
    STOP 15 minut przed limitem czasu. Tylko 118 filmów `train` z `pool_manifest.jsonl`. Lokalny test na małym UNet
    (RTX 3070) i próby na Edenie: przygotowanie 73 klatki/s na H100, uczenie 110 obrazów/s na 2 x H100, podgląd po
    200 krokach już realistyczny.
12. **Pełne uczenie** po „tak wysyłaj i trenuj”: przygotowanie latentów wszystkich klatek `train` (1818494,
    4 x H100) i co 10. klatki `val` (1818495), uczenie 1818496 (4 x H100, 32 na kartę x akumulacja 2, 20 000 kroków).
    Tempo 209 obrazów/s, 1,22 s na krok, 57,1 GB pamięci na karcie. W chwili zapisu: krok 5050, strata 0,107
    (start około 0,17), zero błędów w logu. Podgląd po 4000 krokach: 16 par (u góry prawdziwa klatka walidacyjna po
    koderze i dekoderze VAE, pod nią obraz wygenerowany z jej masek; poprawka 30.09: wcześniej opisane jako „32 klatki
    wygenerowane”), wygenerowane klatki to realistyczne oko w trakcie operacji z narzędziem w miejscu obrysu; kilka klatek
    w dolnym rzędzie jeszcze zaszumionych. Koniec przewidziany około 17:45.
13. Odpowiedzi użytkownikowi: uczenie idzie na latentach VAE (obraz ściśnięty 8 razy w każdym wymiarze), bo to
    zasada Stable Diffusion i przyspiesza uczenie wielokrotnie; latenty powstały z klatek filmów wysłanych na Eden.

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

Brak zmian w plikach merytorycznych projektu grantowego. Zmieniona tylko pamięć Claude Code (poza repozytorium):

- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-bez-przyszlych-klatek-w-prognozie.md` (nowy)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\metoda-przewidywania-latentu-vit.md` (nowy)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\eden-po-sprzataniu-2026-09-28.md` (zaktualizowany)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\eden-zbiory-klatek-bez-obrysow.md` (zaktualizowany)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\MEMORY.md` (indeks)

### Zewnętrzne (poza projektem)

Repozytorium ViT, `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\`:

- `experiments\maly_vit_horyzont_prognozy\` (`horyzont.py`, `WYNIKI.md`, `wyniki\HORYZONT.json`)
- `experiments\fastgan_odwrotnosc_mlp\NOTATKA_LITERATURA.md`, `literatura\pelne_teksty_sd_2026-09-29.md`
- `experiments\maly_vit_rownanie_latentu\` (`PLAN.md`, `dane.py`, `nagranie.py`, `t1_bez_sterowania.py`,
  `t2_sterowanie.py`, `rownanie_eden.slurm`, `zbierz_eden.py`, `WYNIKI_T0_T1.md`, `WYNIKI_T2.md`, `WYNIKI_EDEN.md`,
  `wyniki\T1.json`, `wyniki\T2.json`, `wyniki_eden\T2_s47..56.json`, `wyniki_eden\ZBIORCZO.json`, `.gitignore`)
- `experiments\maly_vit_rownanie_macierze\` (`PLAN.md`, `dane.py`, `model.py`, `uczenie.py`, `liniowe.py`,
  `WYNIKI.md`, `wyniki\BIEG_s47_*.json`, `wyniki\BIEG_s48_*.json`, `wyniki\LINIOWE.json`, `.gitignore`)
- `experiments\dyfuzja_narzedzie\` (plan, kod i skrypty Eden do commitu `fc7ae09e`; dalej prowadzi inna sesja)
- `experiments\dyfuzja_sd_filmy\` (`PLAN.md`, `README.md`, `.gitignore`, `kod\wspolne.py`,
  `kod\przygotuj_latenty.py`, `kod\trenuj_sd.py`, `eden\wyslij.sh`, `eden\przygotuj_sd.slurm`, `eden\trenuj_sd.slurm`)
- `docs\23_METODA_PRZEWIDYWANIA_LATENTU.md`, `docs\README.md`, `docs\EDEN_SPRZATANIE_2026-09-30.md`
- `eden\wyslij_cataract_lmm.sh`

Lokalnie poza repozytorium:

- `F:\Studia\PhD_projekt\VIT\eden_archiwum\wysylka_cataract_lmm\` (log wysyłki i `SHA256SUMS`)

Eden (`/home2/faculty/bpiotrowski`):

- `smallvit_runs/2026-09-29_rownanie_latentu/` (kampania 1818331)
- `smallvit_data/cataract_lmm/3_Object_Tracking/` (filmy i obrysy), `sd576_train`, `sd576_val_co10`, zbiory próbne
  `sd576_train_proba2`, `sd576_val_co10_proba2`
- `smallvit_runs/2026-09-30_dyfuzja_sd_filmy/` (kod, `deps` z diffusers 0.39.0, wagi SD 1.5 i VAE, `wyniki/proba`,
  `wyniki/trening`)

## Otwarte problemy

- Uczenie 1818496 trwa; wynik końcowy (podglądy, krzywa straty, zgodność z maskami) jeszcze nieoceniony.
- Brak miary jakości generatora: nie ma jeszcze skryptu FD-DINOv2 ani porównania z najbliższymi sąsiadami w zbiorze
  (czy model nie kopiuje klatek).
- Prognoza latentu bez przyszłych klatek kończy się na 57-67 procent; równanie działa tylko na jeden krok, a na
  dalsze kroki potrzebna jest siła przyszłych paczek zależna od przyszłych wag.
- Kwota Edenu: około 139 z 200 GiB po latentach; `wyniki/proba/ostatni.pt` (13,75 GB) i migawki EMA z próby czekają
  na zgodę na usunięcie, a każda migawka EMA pełnego uczenia to 1,7 GB.
- Poprawka `siatka_krokow` (start DDIM od t = 979) w `dyfuzja_narzedzie` niewprowadzona, bo folder prowadzi inna
  sesja.

## Następne kroki

1. Po zakończeniu 1818496: pobrać podglądy, `LOG.jsonl`, `STATUS.json` i w razie potrzeby `ema_020000`, obejrzeć
   obrazy, dopisać wyniki do `dyfuzja_sd_filmy\README.md`, zacommitować i zdać raport; sprawdzić czystość `/dev/shm`
   na hopperze. Przy przerwaniu przez limit czasu zgłosić ponownie z tymi samymi argumentami (wznowienie).
2. Za zgodą użytkownika: ocena generatora (FD-DINOv2, najbliżsi sąsiedzi) i generowanie z prowadzeniem małym ViT
   (FreeDoM) w stronę zadanej siły tożsamości.
3. Za zgodą: usunięcie artefaktów próby na Edenie.
4. Za zgodą: prognoza wielokrokowa (T4) w `maly_vit_rownanie_macierze` na zapisanych wagach ziarna 48.
