# Inwentarz SmallViTAblation: kod, dane, reguły, przepis uczenia (stan 2026-10-05, tylko odczyt)

Niczego nie uruchamiałem ani nie zmieniałem. Stan Edenu znam wyłącznie z lokalnych README i logów. Dziś o 11:57 inna sesja założyła `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\maly_vit_odpowiedz_latentu_na_paczke\` (`PLAN.md`, `kod\odpowiedz.py`, `wyniki\proba`): funkcja odpowiedzi latentu na pchnięcie paczki, liczona z nagrań s47 i s48.

Skrót: `E` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments`, `R` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`.

## 1. KOD

**`E\maly_vit_rownanie_latentu\`**
- `nagranie.py`: uczenie jak rozgrzewka plus ramię `bez`, z zapisem po każdym kroku. Model importuje z `E\maly_vit_od_zera` albo ze ścieżki w `MALY_VIT_KOD`.
  - Argumenty: `--ziarno 47 --kroki 3000 --sily --sily-od 900 --wagi-co N --tag --out`.
  - Stałe: `KROKI_OBR=500`, `ROZGRZEWKA_KROKI=1000`, `CO_ROZKLAD=10`, `SILY_OD=900`.
  - `kierunki(opt, nazwy)`: trzy części kroku AdamW przed `opt.step()`, czyli `nowa` (gradient paczki), `ped` (pamięć `exp_avg`) i `zanik` (zanik wag).
  - `z_fp32(net_f, nag_f, params, bufs, X, tn, dev)`: latent narzędzia 192 sond w float32 dla podstawionych wag, przez `torch.func.functional_call`.
  - `Nagrywacz`: haki na wyjściu GELU i wejściu `fc1` ostatniego bloku. Do tego `srednie`, `sondy_pomiar`.
  - Zapisuje: `sondy_lat` (T+1,192,2,768) f16, `sondy_h`, `paczka_idx` (T,32), `paczka_odb`, `paczka_a` (T,32,2), `paczka_lat`, `paczka_g`, `paczka_h`, `paczka_ma_narz`, `strata`, `norma_grad`, `lr`, `rozklad_{param,bufory,ostatni}` co 10 kroków, `sila_{nowa,ped,zanik,suma}` (T−sily_od,192,768) f16, opcjonalnie `wagi\kNNNNN.pt` (sam `state_dict`), `META.json`.
  - Indeksy paczek i augmentację zapisuje. Stanu optymalizatora nie zapisuje.
- `dane.py`: `eden(s, ramie, kanal)`, `nagranie(s)`, `obecnosc(s)`, `umiejetnosc(y,p)` = 100·(1 − SSE/SS).
- `t1_bez_sterowania.py`: równanie z samej historii latentu (`cechy`, `filtry`, `dopasuj_diag`), wynik w `wyniki\T1.json`.
- `t2_sterowanie.py`: warianty `t1`, `t1+paczka`, `t1+paczka_placebo` (przesunięcie o 250 kroków), `t1+jadro`, `t1+paczka+ped`, `t1+wszystko`, `same_sily`, `rownanie_proste`. Argumenty: `--katalog --ziarno --fit 1000 2000 --test 2000 2999 --wynik`.
- `zbierz_eden.py`, `rownanie_eden.slurm` (tablica ziaren 47–56).

**`E\maly_vit_petla_sd_rownanie\kod\`** (pętla 4.3; opis kanoniczny w `R\docs\26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md`)
- `dane.py`: `baza()`, `walidacja()`, `pula(wykluczone)`, `zbior_z_puli()`, `sondy()`, klasa `Zbior` z `dolacz()`. Ścieżki bierze ze zmiennych `MALY_VIT_WEJSCIA`, `MALY_VIT_DANE`, `MALY_VIT_CACHE`.
- `model.py`, `trening.py`: opis w sekcji 4. `Uczen.kroki(..., obserwator=)` przyjmuje obserwatora z metodami `czy`, `przed`, `po`.
- `sila.py`: `wektor(Lk, obecn)` (kierunek LDA ze ściąganiem Ledoit-Wolf, narzędzie przeciw tłu), `sila(nar, wek)`, `prognozy(Sb, H)` (Koopman albo Langevin wersji 3, wybór testem wstecznym), plus sterowanie FastGAN-em.
- `rownanie.py`: `gradient_paczki`, `gradient_tensorow`, `czesci_kroku(u, g)` (odpowiednik `kierunki`), `Kontrola` (strata i gradient zbioru kontrolnego), `ocen_grupe`, `zysk_bezposredni`, `latent_narzedzia_sond(u, sondy, d)`, `dryf_pedu`.
- `prognoza.py`: `bloki` (uśrednianie po 5 zapisów), `pca` (8 wymiarów), `dopasuj`, `rollout`, `test_wsteczny` dla modeli `koopman` i `langevin` (pamięć 4, ridge 1e-2).
- `prognoza_gle.py`: `prognoza(S_hist, H, p=20)`, Langevin z jądrem pamięci dopasowany wprost na horyzont obrotu (okno poziomu 50 zapisów).
- `prognoza_krok.py`: Δz(t+1) = S_pęd + S_paczka. Funkcje `pchniecie_pedu`, `pchniecie_paczki`, `prognoza_bez_nowych`, klasa `Sprawdzian` (pomiar równania na prawdziwych krokach).
- `prognoza4.py`: równanie z bezwładnością, a=0,83 i b=0,98 (wersja 4.1).
- `prowadzenie.py`: `Prowadzony.rysuj_prowadzony(...)` (DDIM 30 kroków, CFG 2, prowadzenie energią latentu ViT), `latent_obrazu`, `pokrycie`.
- `generator_sd.py`: `GeneratorSD.rysuj(maski_u8, ziarno)`. Wagi wskazują `MALY_VIT_SD_UNET`, `MALY_VIT_SD_VAE`, `MALY_VIT_SD_TEKST`.
- `wybor.py`: `zapytaj(png, wykluczone, out_json)` (DINOv2, iloczyn skalarny po całym indeksie), `koopman(listy,…)` (na przemian z list), `losowe`, `kolejne`, `Zajete`. Stałe `OKNO=30`, `TOP_K=90`.
- `wybor4.py`: grupy po 8 klatek, bezpiecznik, limit 20 klatek na film.
- `uklady.py`: wielokąty z `annotation.csv` w archiwach zip, `przeksztalc` (obrót ±35°, skala 0,85–1,15, przesunięcie ±20 px), `maski`, `wycinek_zapytania`, `wycinek_176`.
- `petla4.py`:
  - Argumenty: `--ziarno` (wymagany), `--ramiona`, `--proba`, `--obroty` (domyślnie 7; kampania 14), `--top-k` (domyślnie 90; kampania 180).
  - Stałe: `ROZGRZEWKA=2`, `KROKI_ROZG=500`, `OBROTY=7`, `KROKI_OBR=500`, `NOWYCH=134`, `KOTWIC=6`, `CO_OCENA=100`, `WERSJI=8`, `N_PACZEK_F=8`, `N_SD=48`, `MAX_SYNT=40`, `N_KONTROLA=512`, `N_FILMOW_KONTROLI=10`, `SPR_NA_POCZATKU=10`, `SPR_CO=50`.
  - Prowadzenie: `sila_startu=0.5`, `eta=0.15`, `iteracje=3`, `koniec_prowadzenia=2`.
  - Ramiona domyślne: `metoda`, `metoda_kl3`, `sd_bez_prognozy`, `losowe`, `kolejne`, `bez`. Opcjonalne: `krok_sd`, `sd_strata`, `sd_strata_losowo`, `kotwice_strata`.
  - Wyjście w `$MALY_VIT_WYNIKI\petla43b_s<ziarno>\`: `wspolny\k01..k02` (`LATENT.npz`, `kNN.pt` z wagami i AdamW), a w każdym ramieniu `kNN\OBROT.json`, `wagi_fp16.pth`, `OCENY.json`, `SILA.npy`, `wagi_koncowe.pth`, `KONIEC.json`.
- Poza `kod\`: `analiza43.py`, `diagnoza_prognozy.py`, `uruchom_lokalnie.cmd`, `eden\petla43_eden.slurm`, `eden\wyslij43.sh`, `eden\pobierz43.sh`.

**`E\maly_vit_ruch_czastek_gesta_historia\`**
- `kod\bieg.py`: argumenty `--wyniki --ziarno 47 --krokow 1000 --zapis-wag 10 --zapis-pelny 100 --co-ocena 100 --proba`.
- Obserwator `Pomiar` po każdej aktualizacji przepuszcza 12 pełnych klatek 480×320 (eval, fp32, bez autocast i TF32).
- Zapisuje: `dane\LATENT_GESTY.npy` (1001,12,600,768) float32, `wagi\kNNNN.pt` (sieć z buforami BatchNorm) co 10 kroków, `pelny\kNNNN.pt` (wagi i AdamW) co 100 kroków, `LATENT_SOND.npz`, `OCENY.json`, `LOG.json`, `RUCH_KROKOW.json`, `metadata.json`.
- Indeksów paczek nie zapisuje. Odmawia nadpisania istniejącego pliku.
- Odczyt z kodu, niesprawdzony uruchomieniem: `kroki()` wołane raz na 1000 kroków, więc kolejka permutacji nie zeruje się w kroku 500. W `nagranie.py` i w rozgrzewce pętli zeruje się co 500.
- `kod\analiza.py` (`--dane --wyniki --co N | --kroki 60,120`), `kod\wykres_w_czasie.py`, `bieg_eden.slurm proba|pelny`.

**`E\maly_vit_gesta_historia_paczka\kod\`**
- `bieg.py`: kopia powyższego z dodatkami `--paczka 32` (ustawia `trening.PACZKA`), `--beta1 0.9` (ustawia `trening.BETA1`) i `--tokeny-na-obraz N` (wszystkie tokeny narzędzia plus losowe tło do N, lista w `TOKENY_WYBRANE.json`, ziarno 20261003).
- `trening.py` różni się od poprzednika tylko stałą `BETA1`.
- `porownaj_paczki.py` (`--wyniki --biegi`), `tabela.py`.
- `..\bieg_paczka.slurm <proba|pelny> <paczka> [beta1]`. Ramię `--beta1` jest przygotowane, nieuruchomione.

**`E\maly_vit_sufit_przewidywalnosci_100\kod\`** (bez treningu, czyta gęstą historię)
- `dane.py`: mmap i maski.
- `sufit.py`: sufit wyroczni (gładka ścieżka), podłoga z samych drgań, prognozy przyczynowe C0–C2, model liniowy krzyżowo między połowami obrazów. Zawiera `kierunek_lda`.
- `tabela.py`: tabele z `METRYKI.json`.
- `prognoza_900.py`, `wykres_900.py`: równanie Ornsteina-Uhlenbecka od kroku 900 do 1000, porównanie z NNDMD.
- `co_jeszcze_z_historii.py`: bogatsze modele z historii, wyrocznia ruchu wspólnego.
- `autokorelacja_przyrostow.py`: autokorelacja do opóźnienia 150.
- `wczesna_faza_10_ziaren.py`: kroki 0–500, ziarna 47–56, latent sond co 10 kroków.
- `jadro_pamieci_10.py`: jądro J=5..40 na horyzonty 1..10.
- `rownanie_z_pedem_10.py`: historia plus `S_ped` i `S_zanik` na nagraniach s47 i s48, z granicami „paczki znane”.

**Kanały J128 i 400 par**
- `E\maly_vit_test_orientacji_narzedzia_400\`: buduje 400 par. Skrypty `przygotuj.py` (wybór 400 klatek train, 116 filmów, 12 przedziałów kąta), `pobierz_lama.py`, `lama_inpaint.py [--all]` (LaMa ONNX na CPU, dylatacja 5 px), `sprawdz_pary.py`, `zastosuj_poprawke_qc.py`.
- `E\maly_vit_tozsamosc_czasteczki_pary_400\`: `dane.py` (podział filmów 69/23/24, ziarno 20261001, zapis w `PODZIAL.json`), `pomiar.py` (pełne tokeny par na `k1000`), `identyfikacja.py` (podprzestrzeń SVD i LDA), `interwencje.py`, `kontrola_fc2.py`, `raport*.py`.
- `E\maly_vit_langevin_kierunki_tozsamosci\identyfikacja500.py`: średnie ważone pokryciem na `k0500`, klucz `pooled400` w `IDENTYFIKACJA_K500.npz`.
- `E\maly_vit_tozsamosc90_bez_pca_langevin\`: `wybor_kanalow.py --stage validation|test` (ranking kanałów tylko na train, K z listy 1..768, logistyczny L2; wybrane K=128, C=0,01), `przygotuj_dynamike.py`, `dynamika.py --stage validation|test`, `model_native.py`, `metryki.py`, `reszty.py`, `raport.py`.

**`E\maly_vit_pykoopman_30_obrazow_128\`**
- `dobor.py`: 34 obrazy, 30 do dopasowania i 4 odłożone.
- `ekstrakcja.py [--pilot]`: J128 na 101 checkpointach. `ekstrakcja_768.py`: pełne 768.
- `pykoopman_fit.py --stage fit|evaluate|descriptive`: ważony afiniczny EDMD, operator 128×128.
- `nndmd_fit.py --stage smoke|train|evaluate`: PyKoopman 1.2.1 NNDMD, stan 768.
- `report_pykoopman.py`, `report_nndmd.py`, `audit_nndmd.py`, testy `test_*.py`. `online_edmd.py` to sam nagłówek.
- Środowisko `.venv-pykoopman`.

**`R\pipeline\latent_dynamics\`** (czas filmu przy zamrożonych wagach, nie czas uczenia)
- `gle.py::diagnose_gle(rows, features, seed=42, dim=24, lags=3, folds=3, …)`: krzyżowo dopasowane kontrasty czterech predyktorów następnej klatki (persystencja, Markow, nieliniowy, historia).
- `diagnose.py` (`--pool --features --out …`), `synthetic_benchmark.py`.

**`R\pipeline\active_learning\`**
- `acquisition.py::select_batch(...)`: strategie `random`, `uniform_time`, `coverage`, `coverage_uncertainty|residual|history|reducible`. Dokładny budżet, tylko train, bez czytania etykiet.
- `simulate.py` (`--pool --embeddings --scores --out --strategy --static --budgets --seed --min-gap-frames`), `merge_scores.py`, `uncertainty.py`.
- Obok leży `R\pipeline\koopman_langevin\` (linia RF-DETR, m.in. `latent_langevin.py`, `kalman_state.py`).

**`R\analysis\formalizacja_doboru_klatek\`**
- Pliki: `README.md`, `sprawdz_wyprowadzenia.py`, `WERYFIKACJA.json`, `sprawdz_lean.ps1`, `LEAN_WERYFIKACJA.json`, `LEAN_WERYFIKACJA.log`, `sprawdz_reszte_kalmana.py`, `RESZTA_KALMAN_WERYFIKACJA.json`, `zbuduj_pdf_langevin.py`.
- Twierdzeń w tym katalogu nie ma. Są w `R\docs\Analiza_ponownej_publikacji\44_FORMALIZACJA_DOBORU_KLATEK_KOOPMAN_LANGEVIN.md`:
  - **T1**: zmiana obserwabli Koopmana w kroku jest ograniczona przez |λ−1|‖φ‖ + ‖r‖; trwałość nie mówi, czego obserwabla dotyczy.
  - **T2**: dla straty kwadratowej deficyt informacji reprezentacji to E‖E[Y|X] − E[Y|Z]‖².
  - **T3**: eliminacja ukrytych współrzędnych daje dokładne równanie z jądrem pamięci i wymuszeniem; zerowe jądro nie oznacza domknięcia.
  - **T4**: historia zmniejsza błąd o nieujemną lukę (liniowo: dopełnienie Schura).
  - **T5**: poprawa straty po kroku różni się od η⟨g,d⟩ najwyżej o Lη²‖d‖²/2.
  - **T6**: przy ograniczonych błędach gradientu i kierunku poprawa leży w przedziale â ± E; przewaga partii S nad B wymaga â_S − E_S > â_B + E_B; wybór maksimum traci najwyżej 2E_max.
  - **T7**: błąd cech przenosi się na błąd gradientu liniowego odczytu ze stałą 2MR_z + Y_max.
  - **T7a**: łańcuch od równania pamięci przez cechy i gradient do przedziału poprawy.
  - **T8**: ważone pokrycie przybliża gradient puli; bez wag giną proporcje.
  - **T9**: samo widmo nie wyznacza użyteczności klatki.
  - **T10**: przy gładkości, warunku Polyaka–Łojasiewicza i względnym błędzie kierunku zbieżność jest geometryczna.
  - **T11**: jednoczesne ograniczenie Hoeffdinga dla m procedur na n niezależnych operacjach.
- W Lean 4.28 sprawdzono 9 twierdzeń w `F:\Studia\MCHTR\Grant\Projekt\Research\Math\lean4_spm\SpmProofs\ViTFrameSelection.lean`: `alignment_error`, `direction_norm`, `direction_norm_sq`, `gain_error` (T6 przy założonej reszcie Taylora), `certified_improvement`, `certified_comparison`, `selection_regret`, `particle_error`, `particle_gradient_error` (przy założonej wrażliwości mapy).
- Poza Lean pozostają: T5 z gładkości, stała T7, tożsamość Moriego–Zwanziga, T10, T11. T1–T4, T8 i T9 mają kontrolę SymPy/NumPy; dokument 44 powołuje się też na wcześniejsze dowody w `MoriZwanzig.lean` i `MzLogdetMacierz.lean`.

## 2. ARTEFAKTY DANYCH

Kształty podaję tylko tam, gdzie stoją w README, JSON albo kodzie obok pliku. Na dysku F: jest 106 GB wolnego.

| Co | Gdzie | Rozmiar, kształt |
|---|---|---|
| Gęsta historia, paczka 32, s47 | `E\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny\dane\LATENT_GESTY.npy` | 22,1 GB, (1001,12,600,768) f32, SHA256 zgodne; obok `obrazy.npy`, `maski.npz`, `images.json`, `metadata.json`, `OCENY.json`, `LATENT_SOND.npz` (54 MB) |
| Wagi tego biegu | `E\maly_vit_ruch_czastek_gesta_historia\wyniki\pelny\wagi\k0000.pt … k1000.pt` | 101 plików co 10 kroków, 1,18 GB, klucze `net` i `krok` |
| Stany AdamW tego biegu (`pelny\k0000..k1000`, co 100) | lokalnie nie znaleziono | według README na Edenie: `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny` |
| Pochodna telemetria wag i AdamW | `E\maly_vit_nndmd_1500_expandedstate_eden\AUX_HISTORY.npz` | (101,59) |
| Gęsta historia, paczka 128 i 320 | `E\maly_vit_gesta_historia_paczka\wyniki\pelny_p128\dane\`, `…\pelny_p320\dane\` | `LATENT_GESTY.npy` po 2,21 GB, (1001,12,60,768); `TOKENY_WYBRANE.json`, `LATENT_SOND.npz` |
| Nagranie s47 bez sił | `E\maly_vit_rownanie_latentu\nagranie_s47\` | około 3,4 GB, 3000 kroków |
| Nagranie s47 z siłami | `E\maly_vit_rownanie_latentu\nagranie_s47_sily\` | około 5,9 GB, siły od kroku 900 (4 × 619 MB) |
| Nagranie s48 z siłami | `E\maly_vit_rownanie_latentu\nagranie_s48_sily\` | około 4,9 GB, siły od kroku 2000, `wagi\k01000..k03000.pt` co 100 (21 plików) |
| Równanie na 10 ziarnach | `E\maly_vit_rownanie_latentu\wyniki_eden\T2_s47..s56.json`, `ZBIORCZO.json` | same JSON-y; nagrania skasowane na węźle |
| Latent sond co 10 kroków, ziarna 47–56 | `E\maly_vit_od_zera\wyniki_eden\petla_s47..56\` (5,3 GB), `E\maly_vit_wektor_tozsamosci\wyniki_eden\petla_s47..56\` (5,7 GB) | `wspolny\kNN\LATENT.npz` i ramiona `bez`, `koopman`, `losowe`; format (192,2,768) f16 na zapis |
| Stan po rozgrzewce z AdamW | `E\maly_vit_petla_sd_rownanie\wyniki\diagnoza_s47\k02.pt` (s47, krok 1000), `E\maly_vit_od_zera\wyniki\petla_s42\wspolny\k01..k03.pt` | po 35 MB |
| Telemetria kampanii 4.3 | `E\maly_vit_petla_sd_rownanie\wyniki_eden43\` | `WYNIKI_EDEN43.json`, `.png`, `petla43b_s47..s56\` (około 6 MB każdy: `OBROT.json`, `OCENY.json`, `KONIEC.json`, `KOTWICE_I_OBRAZY.jpg`), logi `.out` |
| Wagi po obrotach i `SILA.npy` kampanii 4.3 | lokalnie nie znaleziono | według `pobierz43.sh` zostają na Edenie: `~/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s*` |
| J128 | `E\maly_vit_tozsamosc90_bez_pca_langevin\KANALY_MODEL.npz`, `WYBOR_WALIDACJA.json`, `RANKING_KANALOW.csv`; kopia `E\maly_vit_pykoopman_30_obrazow_128\KANALY_J128.npz` | 128 indeksów, wybrane na `k0500` |
| Latenty 34 obrazów × 101 checkpointów | `E\maly_vit_pykoopman_30_obrazow_128\LATENT_J128.npy`, `LATENT_768.npy` | 1,05 GB, (101,34,600,128); 6,33 GB, (101,34,600,768); obok `MANIFEST.json`, `MASKI.npz`, `OBRAZY.npy`, `DANE*.json` |
| Poprawka pięciu obrazów | `E\maly_vit_pykoopman_30_obrazow_128\BASE1500_CORRECTED\LATENT_NOWE5_768.npy` | 0,93 GB, (101,5,600,768) |
| Operatory | `E\maly_vit_pykoopman_30_obrazow_128\MODEL_PREFIX_500.npz`, `MACIERZ_KOOPMANA_128.npy`, `NNDMD_*.pt`, `NNDMD_*_A_LEARNED_768.npy` | |
| 400 par | `E\maly_vit_test_orientacji_narzedzia_400\` | `obrazy_oryginalne` (400 JPG), `obrazy_bez_narzedzia` (400 PNG), trzy katalogi masek, `TOKEN_MASKI.npz`, `manifest.json`, `PARY.json` |
| Tokeny par | `E\maly_vit_tozsamosc_czasteczki_pary_400\TOKENY_PARY.npz` (70 MB), `PODPRZESTRZEN.npz`, `PODZIAL.json`; `E\maly_vit_langevin_kierunki_tozsamosci\IDENTYFIKACJA_K500.npz` | `pooled400` 400×2×768 |
| Stable Diffusion | `F:\Studia\PhD_projekt\VIT\eden_archiwum\sd_modele\` | `ema_010000`, `ema_015000`, `ema_020000` (po 1,72 GB), `sd-vae-ft-mse`, `pusty_tekst.pt`, `scheduler_config.json`, `deps\` (diffusers 0.39.0, peft) |
| Reszta archiwum | `F:\Studia\PhD_projekt\VIT\eden_archiwum\` | `smallvit_runs_raporty_2026-09-28.tar.gz` (1,3 GB), `wysylka_cataract_lmm\` (logi) |
| Indeks DINOv2 puli | `E\maly_vit_od_zera\eden\wejscia\index\` | `features.npy` 10 637 × 384 (wycinki narzędzia, ViT-S/14), `manifest.jsonl`, `metadata.jsonl`, `config.json`; obok `dinov2_vits14_pretrain.pth`, `dino\`, `wyszukiwanie\wyszukiwanie.py` |
| Indeks DINOv2 pełnych klatek | nie znaleziono | plan z 2026-10-02 mówi wprost, że nie jest zbudowany |
| Pula klatek | `E\koopman_fastgan_petla\local_inputs\data\` | `images\` 22 680 JPG 720×480 (10 637 train, 12 043 val), `pool_manifest.jsonl`, `split_manifest.json`; archiwum `…\local_inputs\tracking_static36.tar` (2,1 GB); kopia w `R\feature_cache\lokalny_static36\` |
| Tablice w RAM | `E\maly_vit_od_zera\cache\baza1500.npy` (691 MB, (1500,320,480,3) uint8), `wal3000.npy` (1,38 GB) | |
| Wejścia pętli | `E\maly_vit_od_zera\eden\wejscia\` | `pretrain\{train,valid}\_annotations.coco.json`, `probe\` (192 obrazy i maski, `manifest.json`), `wykluczone.json` |
| Surowy zbiór | `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Dataset\Cataract-LMM\3_Object_Tracking\` | `annotations\` (170 zip), `videos\` |

Według `E\maly_vit_gesta_historia_paczka\PLAN.md` kopię `LATENT_GESTY.npy` (22 GB) usunięto z Edenu 2026-10-03. Lokalny plik jest wtedy jedyną kopią.

## 3. REGUŁY I KONWENCJE

**Źródła reguł**
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\AGENTS.md` to ogólna instrukcja Task Master. Reguł eksperymentów nie zawiera.
- `CLAUDE.md` obok mówi o Pythonie 3.11 i zakazie `du -sh ~`.
- `R\PLAN.md`, `R\docs\01_PROTOCOL.md` i `R\docs\05_OPEN_QUESTIONS.md` dotyczą linii RF-DETR (ponowna publikacja).
- Linia małego ViT ma konwencje w samych katalogach eksperymentów i w `R\docs\23_METODA_PRZEWIDYWANIA_LATENTU.md` oraz `R\docs\26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md`.

**Struktura eksperymentu** (wzór: `E\maly_vit_gesta_historia_paczka\`)
- Katalog `experiments\maly_vit_<opis>` małymi literami, ASCII, jeden pomiar na katalog.
- `PLAN.md`: data, cytat polecenia, Pytanie, Co obala, Bieg albo Aparat, Podłoga, Pomiar po biegu.
- `README.md`: skąd skopiowano kod i nazwane zmiany, polecenia Eden i lokalne.
- `WYNIKI.md`: Werdykt, tabela, Co z tego wynika, Zastrzeżenia, Czego to NIE dowodzi.
- `WYNIKI.json`: lokalna deklaracja z kluczami `rodzina`, `klasa`, `status`, `pytanie`, `werdykt`, `confirmatory`, `primary_replications`, `training_new_vit`, `slurm`, `plan`, `raport`, `metryki`, `dane`.
- `kod\` z kopiami `dane.py`, `model.py`, `trening.py`; `wyniki\<proba|pelny…>\`; plik `.slurm`; sumy SHA256.
- Ciężkie pliki wyłącza `.gitignore`. Rejestru projektu grantowego to repozytorium nie używa.
- Plany przyszłych pomiarów: `R\plan_na_przyszle_eksperymenty\RRRR-MM-DD_GG-MM-SS_opis.md` plus wpis w `README.md`. Zrealizowane dostają własny katalog w `experiments\`.
- Poprawki zostają jako osobne pliki (`AMENDMENT_*.md`, stary JSON z przyrostkiem), bez nadpisywania.

**Przed uruchomieniem**
- Plan zapisany przed kodem; kryteria (K1…) i podłogi ustalone przed wynikiem (brak zmiany, placebo, kontrola losowa).
- Każda różnica wobec metody użytkownika nazwana wprost.
- Etapy: próba kodu (`--proba`), potem pełne ziarno lokalnie albo kampania na Edenie, każde po osobnej zgodzie z liczbą ziaren, kart i czasu.
- Wybór modelu zamrożony i zahaszowany przed testem; test czytany raz.

**Podział danych**
- Cataract-LMM dzielony po grupach operacji: 117 train, 25 val, 25 test (ziarno 42, `patient_disjointness_verified: false`).
- Test nie został wyeksportowany (`test_images_exported: 0`) i jest nietknięty.
- Mały ViT uczy się tylko na train (baza 1500 klatek ze 118 filmów, pula train). Ocena to próbka 3000 klatek val, więc wszystkie liczby to walidacja, nie test.
- Podziały wewnątrz train: pary 69/23/24 filmy, sondy 8 i 4 źródła, obrazy 0–5 i 6–11 na krzyż, podział w czasie (nagrania: dopasowanie 1000–2000, test 2000–2999).
- Linia RF-DETR: 50 nagrań 35/8/7 po pacjencie; zbiór zewnętrzny liczony raz, z rejestrem `external_access_ledger.jsonl`.

**Jedna zmiana, ziarna, wynik**
- `R\README.md`: „Zmieniamy jedną rzecz na raz”. Budżet klatek i kroków równy w ramionach.
- Każde ramię to osobny bieg z tego samego stanu rozgrzewki. Kontrole: `losowe`, `kolejne`, `bez`, `sd_bez_prognozy`.
- Kampanie: 10 ziaren 47–56, porównania sparowane po ziarnach (odsetek wygranych, Wilcoxon). Jednostką replikacji jest cały bieg ziarna.
- Jedno ziarno i próba kodu nie są wynikiem. Wynik negatywny opisuje się tak samo starannie.
- Słowa „wykonano” i liczby wymagają logu albo manifestu (`R\docs\README.md`).
- Prognoza nie może używać przyszłych klatek: tylko stany do t, paczki do t−1 i stan optymalizatora. Wyjaśnianie latentu paczką, która go stworzyła, jest dozwolone (`docs\23`).
- Telemetria: wagi po każdym obrocie, snapshoty latentu.

**Eden** (`R\eden\`: `run_node.slurm`, `queue.py`, `runner.py`, `submit_chain.sh` dla macierzy RF-DETR; mały ViT ma własne `.slurm` w katalogach eksperymentów)
- Konto `transformers_vsc`, partycje `hopper,short`, `--gres=gpu:1`, 8 CPU, 64–96 GB, `--no-requeue`, `--signal=B:TERM@120` albo `@180`.
- Katalog `~/smallvit_runs/<data>_<nazwa>/{kod,eden,logi,wyniki,wejscia}`.
- Python `~/venvs/rfdetr_nano_clean_1_10_1/bin/python`. `PYTHONPATH` wskazuje `…/2026-09-28_maly_vit_od_zera/deps` (timm 0.9.16) i `…/2026-09-30_dyfuzja_sd_filmy/deps` (diffusers).
- Dane rozpakowywane z `tracking_static36.tar` do `/dev/shm/...`, sprzątane pułapką `trap`. Zmienne `MALY_VIT_*`.
- Pierwszy argument `proba` albo `pelny`; ziarno to baza plus indeks tablicy (`--array=0-9%5`); skrypt odmawia nadpisania wyniku.
- Kwota sprawdzana przez `lfs quota -p 1299`, limit 200 GB i 200 tys. plików.
- Wysyłka przez `scp` z kontrolą składni, pobieranie samych JSON i JPG.

**Język i styl**
- Dokumenty po polsku z ogonkami, datowane, bez emoji, przecinek dziesiętny.
- Nazwy plików i identyfikatory po polsku w ASCII.
- Docstring skryptu zaczyna się od „co rozstrzyga” i „co obala”.
- Sekcja „Czego to NIE dowodzi” jest w każdym raporcie z wynikami.

## 4. MODEL I PRZEPIS UCZENIA (z `model.py` i `trening.py`)

**Architektura** (`MalyViT(dim=192, glebokosc=6, glowy=3, mlp=4.0)`)
- Łodyga: 4 sploty 3×3 z krokiem 2 (3→24→48→96→192), trzy pierwsze z BatchNorm i GELU. Krok siatki 16 px.
- Stałe pozycje sin-cos 2D, 6 bloków timm `Block` (`qkv_bias=True`, MLP 768), LayerNorm.
- Gęsta głowa Linear–GELU–Linear(5): logit środka i odległości l, t, r, b.
- Normalizacja ImageNet wewnątrz `forward`. Liczba parametrów: 2,93 mln (według `E\maly_vit_od_zera\PLAN.md`).
- Wejście 480×320 daje siatkę 30×20, czyli 600 tokenów. Latent to wyjście GELU `bloki[-1].mlp.act`, 768 liczb na token.
- Strata: ogniskowa CenterNet plus 2·(1 − GIoU).

**Uczenie** (`Uczen(ziarno)`)
- AdamW: lr 5e-4, betas (0,9; 0,999), zanik wag 0,05 dla parametrów o `ndim ≥ 2`.
- Liniowa rozgrzewka lr przez 300 kroków, potem stały lr.
- Paczka 32, autocast bf16, obcinanie gradientu do normy 1.
- Augmentacja: odbicie poziome z prawdopodobieństwem 0,5, kontrast i jasność z U(0,8; 1,2).
- Losowanie: kolejne permutacje zbioru z `numpy.default_rng`.
- `stan()` zwraca sieć, optymalizator i numer kroku.

**Obrót**
- Jedno wywołanie `kroki(zb, 500, …)` na bieżącym zbiorze. Przed nim ramię dokłada 134 klatki.
- Rozgrzewka: 2 × 500 kroków na bazie (`k01`, `k02`), około 21 przejść przez 1500 klatek, wspólna dla ramion danego ziarna.
- Potem 7 obrotów domyślnie, 14 w kampanii 4.3. RNG ramienia: `default_rng(1000·ziarno + nr_ramienia)`.

**Sondy**
- 12 klatek train × 16 wariantów: `original`, trzy obroty, tło `black` i `yellow`, `scale08`, `length08`, cztery przesunięcia, `foreground_dim`, `foreground_bright`, `perspective_mild`, `empty_black`.
- Razem 192 wycinki 256 px z maskami, zmniejszone do 176×176, czyli 121 tokenów.
- Token narzędzia: udział maski co najmniej 0,25. Token tła: poza maską po dylatacji 17 px.
- 180 sond z narzędziem. Latent sondy to średnia po tokenach narzędzia i osobno po tle, zapis (192,2,768) f16 co 10 kroków.

**mAP**
- `wykryj` (eval, bf16) i `dekoduj` (szczyty 3×3, do 10 ramek na obraz), potem `ap50`: jedna klasa, IoU ≥ 0,5, 101 punktów.
- Liczone na stałej próbce 3000 z 12 043 klatek val (ziarno 0), w kroku 0 i co 100 kroków.
- Miara kampanii: średnia ocen po rozgrzewce, nazywana polem pod krzywą.
