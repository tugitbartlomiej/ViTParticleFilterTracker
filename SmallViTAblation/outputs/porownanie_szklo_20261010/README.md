# Ruch tokenów ViT a szkło: porównanie, układ szkła, kontrola negatywna, model i zdjęcia walidacyjne

> **Aktualny werdykt (2026-10-11): kryształ, nie szkło.** Zobacz `WERDYKT_KRYSZTAL_2026-10-11.md` (podsumowanie, tabela „w ilu procentach”, korekty) oraz niezależny audyt `audyt_krysztalu_20261011/OCENA_KRYSZTALU.md`. Sekcje 1–6d poniżej dokumentują drogę do tego wniosku.

Data: 2026-10-10. Dane: gęsty zapis biegu 35 000 klatek (`experiments/maly_vit_od_zera_35000_21epok_20261009/wyniki/gesty/gesty`):
100 sond × 600 tokenów, PCA32 (98,6 % wariancji), co 1 krok do 2188, potem co 10 do 22 970; pomiar z Edena na 3000 zdjęciach
uczących i 1000 walidacyjnych po 341 checkpointach (`outputs/ruch_czasteczek_gui/20261010T_natural_eden_pelny_3000tr_1000val`).
Okno „stacjonarne” t ≥ 300 (koniec rozgrzewki; lr stały 5e-4 do końca, bez harmonogramu).

## Pliki

| Plik | Co robi | Wynik |
|---|---|---|
| `odciski.py` | wspólne odciski palca dynamiki szklistej dla dowolnej trajektorii i karta 14 kryteriów z literatury | moduł |
| `odciski_szkla.py`, `odciski_klatkowe.py` | odciski względem populacji i względem klatki (6 sąsiadów w latencie) | `ODCISKI_SZKLA.json`, `ODCISKI_KLATKOWE.json`, rysunki |
| `uklad_szkla_analiza.py` | cztery stopnie przeniesienia do układu szkła (S0–S4) z kartą po każdym | `UKLAD_SZKLA.json`, `uklad_szkla.png` |
| `kontrola_ou.py` | kontrola negatywna: cząstki Ornsteina–Uhlenbecka bez oddziaływań (stała T, stygnięcie, dryf, pęd, czas termiczny) | `KONTROLA_OU.json`, `kontrola_ou.png` |
| `eksport_osi_szkla.py` | oś czasu układu klatki dla 60 000 cząsteczek w formacie aplikacji | `outputs/ruch_czasteczek_gui/20261010T_natural_szklo_klatka_35k` (1362 stany, 654 MB) |
| `model_szkla.py` | symulacja szkła koloidalnego lokalnie (dynamika browna miękkich kul, 4 gęstości naraz) | wersja lokalna; wyniki właściwe z Edena |
| `eden/model_szkla_eden.py`, `eden/szklo_eden.slurm` | ta sama symulacja na Edenie (N 2048 szybko, N 4096 i 10⁷ kroków w pełnym biegu) | `eden/wyniki/MODEL_SZKLA_EDEN_*.json`, osie `20261010T_szklo_eden_*` |
| `eden/pomiar_eden.py`, `eden/pomiar_eden.slurm` | pomiar naturalny wielu zdjęć (TRAIN i VALID) po checkpointach, format aplikacji, PCA32 dla 100 + 100 zdjęć | oś `20261010T_natural_eden_pelny_3000tr_1000val` (8,4 GB) |
| `eden/pobierz_z_edena.sh` | pobieranie wyników (strumień tar, kontrola SHA) | — |
| `model_eden_rysunek.py` | rysunek i karta dla modelu z Edena obok tokenów i kontroli OU; wybór gęstości najbliższej tokenom | `model_eden_szybko.png`, `MODEL_EDEN_szybko.json` |
| `analiza_train_val.py` | zdjęcia uczące kontra walidacyjne: odciski, temperatura, trwałość sąsiedztwa, starzenie, ruchliwość 4000 zdjęć | `TRAIN_VAL.json`, `train_val.png` |
| `animacje_szkla.py`, `montaz_szkla.py` | klipy z aplikacji (G1–G5) i film końcowy | `animacje/` |
| `TEORIA_OBJETOSC_PRZEZ_WAGI.md` | jak wprowadzić objętość, gęstość i ośrodek przez wagi (NTK, zacieranie, temperatura SGD, model Langevina) | notatka |

## 1. Porównanie z literaturą (Weeks i Weitz 2000–2002, Hunter i Weeks 2012)

| Odcisk | Szkło koloidalne | Tokeny ViT (względem klatki) | Zgodność |
|---|---|---|---|
| plateau MSD | wykładnik bliski 0 | α = 0,26 (τ 35–80) | tak |
| antykorelacja c w plateau | 0,26 … ≈0,5 | 0,39 | tak, ilościowo |
| rozmiar klatki / odstęp sąsiadów | 0,07–0,15 (szkło), 0,27–0,32 (ciecz φ=0,46) | 0,34 (powłoka), 0,48 (najbliższy) | luźniej niż szkło |
| ogony rozkładu przesunięć | wykładnicze | wykładnicze, 6× więcej >3σ niż Gauss | tak |
| α₂ | maksimum na końcu plateau, ok. 1,8 | 1,7 w plateau, maksimum 3,4 przy τ=3 | wartość tak, położenie nie |
| udział cząstek w przegrupowaniu | 5–10 % w czasie Δt* | 3 % przy τ=10, 22 % przy 37 | Δt* ≈ 10–15 kroków |
| kooperatywność sąsiadów | równoległe kierunki, mniejszość przeciwna | cos 0,87 (populacja), 0,18 po odjęciu klatki | dużo silniejsza niż w szkle |
| trwałość sąsiedztwa | wolny zanik w szkle | 68 % po 1000, 57 % po 22 700 kroków (losowo 1 %) | tak, częściowo strukturalne |
| starzenie | plateau się wydłuża, drgania krótkie bez zmian | MSD ∝ t_w^−μ, μ 0,11–0,25; drgania krótkie maleją 3× | kierunek tak, mechanizm inny |
| reżim krótkoczasowy | koloid dyfuzyjny; szkło MD balistyczny | balistyczny α 1,55, VACF +0,55 (pęd Adama) | jak szkło MD |
| relaksacja strukturalna | ciecz: Fs do 1/e w czasie pomiaru | f_q 0,62, β 0,79, 1/e przy τ ≈ 1073 | ucieczka z klatki ≈ 1000 kroków |

89 % przesunięcia względem populacji to wspólny przepływ sąsiedztwa, 11 % to grzechotanie w klatce.

## 2. Cztery stopnie układu szkła (`uklad_szkla_analiza.py`, karta 14 kryteriów; po zaostrzeniu kryterium szczytu α₂)

| Stopień | Co usuwa | Karta | Lindemann | c | α₂ plateau | α krótkie | μ (τ=10) | MSD(τ=1) koniec/początek |
|---|---|---|---|---|---|---|---|---|
| S0 względem populacji | pole globalne | 5/14 | 0,96 | 0,42 | 0,30 | 1,63 | 0,36 | 0,25 |
| S1 względem klatki | przepływ 6 sąsiadów | **6/14** | 0,34 | 0,39 | 1,73 | 1,55 | 0,25 | 0,31 |
| S2 jednakowe cząstki | zamrożoną heterogeniczność (σ 0,022–0,206) | 6/14 | 0,30 | 0,39 | 0,47 | 1,45 | 0,20 | 0,34 |
| S3 czas termiczny | stygnięcie kąpieli | 5/14 | 0,31 | 0,38 | 0,40 | 1,64 | −0,06 | 2,14 |
| S4 bez pędu | bezwładność Adama (ρ = 0,59) | **6/14** | 0,32 | 0,39 | 0,38 | 1,08 | −0,08 | 3,6 |

## 3. Kontrola negatywna: ile „szkła” daje sama cząstka w dołku (`kontrola_ou.py`)

N = 6000 cząstek OU bez oddziaływań, sztywności dołków z rozrzutem 10× (jak ruchliwość tokenów), T0 skalibrowane do plateau
tokenów. Dokładny krok OU (stabilny dla każdej sztywności).

| Wariant | Karta | c | α₂ plateau | ogon ×Gauss | f_q | β | μ (τ=10) | MSD(τ=1) koniec/pocz. |
|---|---|---|---|---|---|---|---|---|
| OU-A0 stała T, dołki jednakowe | 4/14 | 0,46 | 0,00 | 1,0 | 0,49 | — | 0 | 1,0 |
| OU-A stała T, rozrzut 10× | 4/14 | 0,38 | 0,42 | 1,6 | 0,49 | — | 0 | 1,0 |
| OU-B + stygnięcie jak tokeny | **6/14** | 0,38 | 0,66 | 1,6 | 0,51 | 0,41 | 0,67 | 0,27 |
| OU-B po czasie termicznym | 4/14 | — | — | — | — | — | 0,05 (τ=100–1000: 0,23–0,25) | 1,9 |
| OU-C + dryf dołków | 6/14 | 0,38 | 0,65 | 1,6 | 0,51 | 0,66 | 0,67 | 0,27 |
| OU-D + pęd ρ = 0,59 | 4/14 | 0,36 | 0,74 | 1,8 | 0,21 | 0,93 | 0,67 | 0,27 |

Wnioski: krzywa MSD, antykorelacja c, rozciągnięty eksponens Fs i starzenie z malejącymi drganiami **wychodzą same** z równania
Langevina w stygnącym dołku; także reszta starzenia po przeskalowaniu czasem termicznym (0,23–0,25) jest w OU, więc nie jest
dowodem starzenia strukturalnego. Poza OU wykraczają u tokenów: ogony 2,9× Gaussa po wyrównaniu ruchliwości (OU 1,0×),
krótkoczasowy szczyt α₂ (3,4 przy τ=3; OU rośnie dopiero przy τ ≈ 1000) i istnienie trwałych sąsiedztw (OU ich nie ma).
Tokeny mają też luźną klatkę (0,34) przy silnej antykorelacji (0,39); w szkle BD te dwie wielkości są sprzężone
(luźna klatka → słabe c), w OU nie, więc pod tym względem tokeny są bliższe OU niż szkłu.

## 4. Model szkła koloidalnego z Edena (`eden/model_szkla_eden.py`, N = 2048, bieg szybki, H200)

Miękkie kule harmoniczne, polidyspersja 8,7 %, T = 3e-4, dt = 0,3, 150 000 kroków ściskania, 300 000 równoważenia,
700 000 zapisu (rekord co 100 kroków, τ_B = 111 rekordów). Przy T = 3e-3 kule były cieczą do φ = 0,68 (energia cieplna
15× większa od sprężystej kontaktów), stąd T = 3e-4.

| φ | Lindemann | c | α plateau | α₂ szczyt @ rekord | f_q | β | τ_1e | trwałość 1000 | karta |
|---|---|---|---|---|---|---|---|---|---|
| 0,58 | 0,36 | 0,18 | 0,65 | 0,37 @ 89 | 0,48 | 1,30 | 89 | 0,18 | 2 |
| 0,60 | 0,26 | 0,30 | 0,44 | 0,70 @ 218 | 0,66 | 1,00 | 427 | 0,38 | 5 |
| 0,62 | 0,19 | 0,43 | 0,18 | 1,79 @ 837 | 0,79 | 0,78 | ∞ | 0,57 | **8** |
| 0,64 | 0,15 | 0,46 | 0,12 | 2,49 @ 4022 | 0,86 | 0,69 | ∞ | 0,65 | **8** |

Gęstość najbliższa tokenom (po rozmiarze klatki, c, f_q i wykładniku plateau): φ = 0,60; antykorelacja i Fs tokenów
odpowiadają jednak φ = 0,62–0,64. Prawdziwe szkło (φ ≥ 0,62) spełnia to, czego ani tokeny, ani OU nie spełniają:
szczyt α₂ na końcu plateau.

Bieg stygnący (T = 0,01 (1 + t/2000)^−0,79 do 140 000 kroków, potem ∝ t^−0,30; ten sam zanik co energia tokenów):

| φ | Lindemann | c | α₂ szczyt @ rekord | f_q | β | trwałość 1000 | μ (τ = 10 / 100 / 1000) | karta |
|---|---|---|---|---|---|---|---|---|
| 0,58 | 0,35 | 0,19 | 0,61 @ 139 | 0,51 | 1,12 | 0,09 | 0,79 / 0,90 / 0,61 | 3 |
| 0,60 | 0,26 | 0,30 | 1,16 @ 341 | 0,68 | 0,90 | 0,19 | 0,86 / 1,02 / 0,81 | 6 |
| 0,62 | 0,19 | 0,40 | 2,41 @ 534 | 0,79 | 0,74 | 0,39 | 0,91 / 1,16 / 1,10 | 7 |
| 0,64 | 0,15 | 0,44 | 3,29 @ 2568 | 0,86 | 0,63 | 0,52 | 0,96 / 1,28 / 1,24 | 8 |

Szkło stygnące starzeje się w pełni (μ ≈ 1, rośnie z gęstością), cząstki OU z tym samym stygnięciem dają μ ≈ 0,67,
a tokeny ViT tylko 0,25 przy τ = 10 i ujemne μ w późnej fazie: u tokenów stygnięcie szybkich drgań nie pociąga za sobą
proporcjonalnego spowolnienia ucieczki z klatki. Pełny bieg (N = 4096, 10⁷ kroków, 8 gęstości φ 0,56–0,66, zadania
1826439/1826440 na dgx-2, ~5 h) daje te same miary z dziesięciokrotnie dłuższym oknem; uzupełnienie po pobraniu.

## 5. Zdjęcia uczące kontra walidacyjne (`analiza_train_val.py`, pomiar z Edena)

| Wielkość | TRAIN | VALID |
|---|---|---|
| IoU na końcu (mediana zdjęć) | 0,88 | 0,74 |
| odciski S1 w oknie 300–1000 (rekord = 10 kroków): Lindemann / c / α₂ / ogony | 0,57 / 0,31 / 1,34 / 5,9× | 0,56 / 0,31 / 1,55 / 5,7× |
| energia drgań w klatce, kroki 300–1000 | 8,1e-4 | 6,4e-4 (−21 %) |
| udział pola globalnego w ruchu, kroki 300–1000 | 0,34 | 0,40 |
| trwałość sąsiedztwa po 1000 / 22 974 krokach | 0,69 / 0,58 | 0,72 / 0,61 |
| ruchliwość zdjęcia w PCA3 (mediana, późna faza) | 0,0037 | 0,0037 (stosunek 1,00) |
| promień chmury zdjęcia na końcu | 0,91 | 0,88 |
| Spearman ruchliwości późnej z IoU zdjęcia | −0,22 | −0,09 |

Ruch tokenów zdjęć walidacyjnych jest prawie identyczny z uczącymi mimo luki detekcji: tokeny walidacyjne są o ok. 20 %
chłodniejsze w klatce, bardziej niesione polem globalnym i nieco trwalej zakotwiczone, a ruchliwość całych zdjęć jest
taka sama. To potwierdza obraz pola średniego: cząsteczkę porusza wspólna aktualizacja wag, nie jej własne zdjęcie
(efekt paczki 1,11). Zapamiętywanie widać w IoU, nie w ruchu; ruchliwość zdjęcia słabo (ujemnie) wiąże się z IoU.
W późnej fazie (t_w ≥ 5000) MSD względem klatki przy τ = 100–1000 rośnie z wiekiem (μ ≈ −0,2) w obu grupach: układ
„odmładza się” po plateau mAP, gdy strata ucząca dalej maleje. Uwaga: różnica 20 % energii może wynikać też z innego
rozkładu zdjęć walidacyjnych (inne filmy), nie tylko z braku własnego gradientu.


### 4b. Duży bieg stały (N = 4096, 8 gęstości, 10⁷ kroków; `model_eden_rysunek.py duze` → `model_eden_duze.png`, `MODEL_EDEN_duze.json`)

| φ | klatka / odstęp | c | α plateau | szczyt α₂ @τ | f_q | β | τ_1e | trwałość 1000 | karta |
|---|---|---|---|---|---|---|---|---|---|
| 0,56 | 0,63 | 0,07 | 0,89 | 0,26 @20 | 0,14 | – | 26 | 0,04 | 3 |
| 0,58 | 0,48 | 0,13 | 0,79 | 0,40 @34 | 0,31 | 1,42 | 56 | 0,09 | 2 |
| 0,60 | 0,32 | 0,24 | 0,57 | 0,71 @92 | 0,57 | 1,09 | 196 | 0,25 | 5 |
| 0,61 | 0,25 | 0,34 | 0,35 | 1,18 @323 | 0,69 | 0,93 | 687 | 0,40 | 8 |
| 0,62 | 0,20 | 0,43 | 0,18 | 2,79 @1135 | 0,78 | 0,70 | 10 891 | 0,56 | 9 |
| 0,63 | 0,17 | 0,47 | 0,07 | 3,99 @5125 | 0,83 | 0,66 | ∞ | 0,64 | 9 |
| 0,64 | 0,15 | 0,48 | 0,07 | 6,62 @18000 | 0,87 | 0,58 | ∞ | 0,69 | 8 |
| 0,66 | 0,12 | 0,49 | 0,03 | 4,53 @18000 | 0,91 | 0,88 | ∞ | 0,74 | 9 |

Przejście szkliste leży między φ = 0,61 a 0,62 (τ_1e skacze z 687 na 10 891 rekordów; od 0,63 brak relaksacji). Tokenom (S1)
najbliższa jest gęstość 0,61 (odległość 0,51 w czterech odciskach): ciecz przechłodzona tuż przed przejściem. Osie czasu
tego biegu mają lokalnie przyrostek `_duze`.

**Duży bieg stygnący** (1826440, N = 4096, T od 1·10⁻² do 1,9·10⁻⁴ w 10⁷ krokach, 7,2 h na A100; `MODEL_SZKLA_EDEN_stygnace.json`,
osie `20261010T_szklo_eden_szklo_BD_eden_stygnace_phi_*_duze`, 2000 klatek każda, SHA zgodne):

| φ | klatka / odstęp | c | α plateau | szczyt α₂ @τ | f_q | β | τ_1e | trwałość 1000 | μ(10) | μ(1000) | karta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0,56 | 0,55 | 0,09 | 0,85 | 0,45 @43 | 0,23 | 1,46 | 34 | 0,01 | 0,88 | 0,79 | 2 |
| 0,58 | 0,42 | 0,15 | 0,74 | 0,69 @72 | 0,42 | 1,15 | 72 | 0,03 | 0,94 | 0,93 | 3 |
| 0,60 | 0,29 | 0,27 | 0,50 | 1,74 @416 | 0,64 | 0,84 | 416 | 0,08 | 1,02 | 1,17 | 6 |
| 0,61 | 0,23 | 0,34 | 0,35 | 3,44 @687 | 0,73 | 0,69 | 2412 | 0,15 | 1,07 | 1,38 | 6 |
| 0,62 | 0,19 | 0,42 | 0,18 | 5,70 @1876 | 0,80 | 0,59 | ∞ | 0,27 | 1,11 | 1,57 | 7 |
| 0,63 | 0,16 | 0,46 | 0,12 | 6,28 @2412 | 0,85 | 0,48 | ∞ | 0,43 | 1,15 | 1,75 | 7 |
| 0,64 | 0,15 | 0,47 | 0,09 | 6,39 @2412 | 0,88 | 0,41 | ∞ | 0,54 | 1,19 | 1,82 | 9 |
| 0,66 | 0,12 | 0,49 | 0,03 | 3,65 @14001 | 0,92 | – | ∞ | 0,67 | 1,25 | 1,94 | 8 |

Stygnięcie przesuwa zatrzymanie o jedną gęstość w dół (τ_1e = ∞ już przy 0,62) i daje prawdziwe starzenie: μ rośnie z 0,9 (płyn,
brak starzenia) do 1,9 przy φ = 0,66, a szczyt α₂ jest 2× wyższy niż w biegu stałym przy tej samej gęstości. To jest wzorzec
„stygnące szkło” do porównania z treningiem: w ViT przy stałym lr μ(10) wynosi 0,2–0,5 (S1), a β spada 0,99 → 0,5 z wiekiem — tak jak tu
β spada 1,46 → 0,41 z gęstością.

## 5a. Pomiar do 60. epoki (806 wag; `TRAIN_VAL_pelny60.json`, `train_val_pelny60.png`, liczone na Edenie)

Pierwszy pomiar kończył się na kroku 22 974 (domyślny `--max_krok`, koniec 21. epoki), choć bieg ma 60 epok (65 640 kroków).
Powtórzony do końca: IoU uczące 0,86, walidacyjne 0,64 (rozjazd rośnie); ruch późny VAL/TRAIN 1,01 (bez różnicy);
trwałość sąsiedztwa niemal zamrożona od 21. epoki (TRAIN 0,58 → 0,56, VALID 0,61 → 0,60 między krokiem 22 974 a 65 640);
MSD przy τ = 100 w fazie ≥ 1000 kroków płaskie (brak starzenia przy długich τ; μ(10) = −0,23 / −0,26, lekkie młodnienie);
udział pola globalnego w energii kroku spada z 0,34 do 0,18. Korelacja Spearmana ruchu późnego z końcowym IoU zdjęcia:
TRAIN −0,37, VALID −0,08: dla zdjęć uczących ruch tokenów rośnie z resztkowym błędem (własny gradient zdjęcia), dla
walidacyjnych brak związku (tylko wspólna kąpiel). AUC rozróżniania VAL od TRAIN po samym ruchu tokenów ≈ 0,66 (po IoU 0,80).
Klatki q16 tego pomiaru (15 GB) skasowane na Edenie z braku miejsca (odtwarzalne: `pomiar_eden.slurm 3000 1000 pelny60 65640`).

## 5b. Co ucina konstrukcja: kontrola rzutu (`kontrola_rzutu.py`, `kontrola_rzutu_dh.py`, `udzial_pca32.py`)

Te same 3 sondy z pełnym zapisem 768D, okno 300–800, trzy przestrzenie z tych samych latentów:

| przestrzeń | odstęp sąsiadów | klatka | klatka / odstęp | c | α plateau | energia kroku | α₂ S0 / S1 @τ=3 | ucieczki τ=10 | C₁ CR (wzgl.) |
|---|---|---|---|---|---|---|---|---|---|
| pełne 768D | 1,96 | 1,01 | 0,52 | 0,38 | 0,27 | 0,030 | 3,7 / 10,5 | 0,15 | 0,61 |
| losowy rzut 128D | 1,93 | 1,02 | 0,53 | 0,38 | 0,27 | 0,034 | 0,7 / 4,0 | 0,16 | 0,59 |
| PCA32 z kroku 0 | 0,98 | 0,39 | 0,40 | 0,38 | 0,30 | 0,004 | 0,7 / 4,0 | 0,07 | 0,51 |

PCA32 trzyma 98,6 % wariancji statycznej, ale tylko 14 % energii ruchu w oknie 300–1000 i **4 % po kroku 15 000**
(`UDZIAL_PCA32.json`): ruch tokenów dzieje się w setkach kierunków o małej wariancji. Zgodność zbiorów 6 sąsiadów PCA32 kontra
768D: Jaccard 0,71. Odporne: plateau, jego wykładnik, antykorelacja, heterogeniczność. Zmienne: klatka względem odstępu
(0,40 → 0,52, dalej od szkła), ucieczki (2×) i parametr niegaussowski liczony po składowych (zależy od bazy; w pełnej
przestrzeni ciężkie ogony są dużo silniejsze). Wnioski z późnej fazy liczone w PCA32 z kroku 0 dotyczą 4 % ruchu, dlatego
fala 3 pokręteł zapisuje losowy rzut 128D (`--rzut rp`), a `odciski.py` liczy też α₂ niezmienniczy na obroty (`a2_inw`).
Wszystkie wyniki w PCA32 w tym README trzeba czytać z tym zastrzeżeniem.

## 5c. Heterogeniczność dynamiczna i ruch kooperatywny (`heterogenicznosc.py`, `dh.py`; `HETEROGENICZNOSC.json`, `heterogenicznosc.png`)

Test, którego nie przechodzi cząstka w dołku: korelacja (po cząstkach, w jednej chwili) fluktuacji ruchliwości cząstki
ze średnią fluktuacją ruchliwości jej 6 sąsiadów (C₁), powłok 2 i 3, skupianie 10 % najruchliwszych, cos przesunięć
sąsiadów minus pary losowe. Ruchliwość **względna** (dzielona przez średnią ruchliwość zdjęcia w tej chwili), bo w sieci
wielkość kroku skacze z paczki na paczkę wspólnie dla zdjęcia. Kontrola zerowa: sąsiedzi z innej chwili (≈ 0 wszędzie).

| układ (CR) | C₁ max @τ | powłoki 1/2/3 | skupianie | sąsiedzi z innych zdjęć | zależność od τ |
|---|---|---|---|---|---|
| ViT wczesne 300–2188 | 0,52 @3 | 0,52 / 0,41 / 0,32 | 2,65 | 0,24 | maleje z τ (0,46 → 0,35) |
| ViT późne 5000–22970 | 0,61 @10 | 0,61 / 0,47 / 0,36 | 3,10 | 0,21 | maleje z τ (0,61 → 0,44) |
| BD φ = 0,58 (ciecz) | 0,16 @100 | 0,16 / 0,03 / 0,01 | 1,29 | – | płaska |
| BD φ = 0,61 | 0,30 @400 | 0,30 / 0,20 / 0,20 | 1,65 | – | rośnie z τ |
| BD φ = 0,62 (szkło) | 0,42 @800 | 0,42 / 0,30 / 0,26 | 2,33 | – | rośnie z τ do τ_α |
| BD φ = 0,64 (szkło) | 0,43 @800 | 0,43 / 0,27 / 0,19 | 2,18 | – | rośnie z τ do τ_α |
| OU-B | 0,04 @3 | 0,04 / 0,00 / 0,00 | 0,91 | – | – |

Tokeny mają więcej korelacji ruchliwości niż szkło, ale o odwrotnej zależności od τ i o dłuższym zasięgu (trzecia powłoka,
inne zdjęcia): to **gładkie pole** aktualizacji, nie klatka z sąsiadów. W układzie lab (bez pola zdjęcia) cos sąsiedzi −
losowe wynosi 0,8 (ViT) wobec 0,15–0,23 (szkło): sąsiedzi w latencie płyną razem, bo aktualizacja jest gładką funkcją
latentu (obraz jądra / NTK). Wynik niezależny od rzutu (5b).

## 5d. Pokrętła temperatury i gęstości (`experiments/maly_vit_szklo_pokretla_20261010`; `pokretla_rysunek2.py` → `pokretla2.png`, `POKRETLA2.json`)

Ten sam przepis, 24 sondy (fale 1–2, PCA32) albo 12 sond w losowym rzucie 128D (fala 3), okno 300–2188 (S1, CR); długie
biegi 12 epok z gęstym oknem na końcu; okno gęste po 60. epoce (wznowienie ze `stan_e60.pt`). Fala 1 i ziarna:

| pokrętło | klatka / odstęp | c | τ_1e | f_q | β | szczyt α₂ @τ | E₇₆₈ (300–400) | wykł. stygnięcia | C₁ CR | mAP |
|---|---|---|---|---|---|---|---|---|---|---|
| lr 1,25e-4 | 0,175 | 0,32 | ≥ 1800 | 0,85 | 1,04 | 2,4 @5 | 0,32 | 0,69 | 0,47 | 0,44 |
| lr 2,5e-4 | 0,235 | 0,35 | ≥ 1800 | 0,77 | 0,94 | 3,8 @3 | 0,50 | 0,75 | 0,48 | 0,48 |
| lr 5e-4, ziarna 47 / 48 / 49 | 0,34 / 0,39 / 0,34 | 0,39 | 1390 / 640 / 1073 | 0,63 / 0,54 / 0,62 | 0,8 | 3,5 @3 | 0,69 / 0,67 / 0,63 | 0,62 / 0,46 / 0,51 | 0,53 / 0,50 / 0,49 | 0,53 |
| lr 1e-3 | 0,426 | 0,43 | 829 | 0,52 | 0,80 | 3,9 @3 | 0,80 | 0,39 | 0,53 | 0,55 |
| 5000 zdjęć (1,0 ogr./param) | 0,327 | 0,39 | 1073 | 0,63 | 0,82 | 3,2 @3 | 0,67 | 0,40 | 0,50 | 0,51 |
| 1500 zdjęć (0,31 ogr./param) | 0,326 | 0,38 | 829 | 0,61 | 0,82 | 2,4 @3 | 0,64 | 0,50 | 0,41 | 0,42 |
| okno po 60. epoce (PCA32 dopasowane w kroku 65 640) | 0,331 | 0,29 | 494 | 0,59 | 0,50 | 2,3 | 0,06 | – | 0,44 @10 | 0,62 |

Fala 3, **losowy rzut 128D** (12 sond; okno 300–2188; okno po 60. epoce 24 sondy):

| pokrętło (rzut 128D) | klatka / odstęp | c | τ_1e | f_q | β | szczyt α₂ inw. | ucieczki τ=10 | μ(10) | E₇₆₈ | C₁ CR @τ | inne zdjęcia | mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lr 1,25e-4 | 0,257 | 0,35 | 829 | 0,72 | 0,99 | 3,2 | 0,010 | 0,23 | 0,31 | 0,58 @5 | 0,33 | 0,47 |
| lr 2,5e-4 | 0,325 | 0,37 | 829 | 0,63 | 0,88 | 3,6 | 0,028 | 0,17 | 0,48 | 0,60 @3 | 0,32 | 0,47 |
| lr 5e-4 | 0,499 | 0,40 | 136 | 0,43 | 0,79 | 4,1 | 0,126 | 0,10 | 0,71 | 0,59 @2 | 0,27 | 0,57 |
| lr 1e-3 | 0,720 | 0,41 | 17 | 0,27 | 0,75 | 4,5 | 0,312 | −0,10 | 0,77 | 0,57 @2 | 0,26 | 0,53 |
| 5000 zdjęć | 0,542 | 0,40 | 63 | 0,37 | 0,79 | 4,0 | 0,156 | 0,14 | 0,61 | 0,58 @2 | 0,23 | 0,54 |
| 1500 zdjęć | 0,537 | 0,39 | 48 | 0,35 | 0,81 | 2,8 | 0,140 | 0,01 | 0,71 | 0,51 @2 | 0,22 | 0,40 |
| okno po 60. epoce | 0,358 | 0,30 | 295 | 0,55 | 0,54 | 2,2 | 0,008 | – | 0,058 | 0,54 @10 | 0,24 | 0,61 |

W pełnej przestrzeni (rzut 128D) tokeny przy lr 5e-4 wcześnie to **gorący płyn**: klatka równa połowie odstępu sąsiadów, Fs
spada do 1/e w 136 krokach, 13 % tokenów opuszcza klatkę w 10 krokach, starzenie słabe (μ 0,10). PCA32 (rzut na wolne mody)
zawyżała zatrzymanie (klatka 0,34, τ_1e 1390). Klatka ∝ lr^0,50 (dokładnie termicznie); przy lr ≥ 5e-4 plateau Fs schodzi
poniżej 1/e, więc τ_1e staje się czasem narastania w klatce (17–136 kroków), przy lr ≤ 2,5e-4 klatka trzyma Fs nad 1/e
(τ_1e ≥ 829): standardowy lr leży na granicy „zatrzymany / płynny”. Po 60 epokach układ jest chłodniejszy (energia kroku 12×
mniejsza) i bardziej zatrzymany: klatka 0,36, τ_1e 295, ucieczki 0,8 %, β 0,54, trwałość sąsiedztwa po 1000 krokach 0,75
(wcześnie 0,67), a szczyt korelacji ruchliwości przesuwa się z τ = 2–3 do τ = 5–10 (C₁ przy τ = 1 spada do 0,30). Na osi φ
szkła BD (po klatce): wcześnie ≈ 0,58–0,59 (ciecz), po 60 epokach ≈ 0,60 (przed przejściem przy 0,615). Długie biegi (12 epok) w rzucie 128D, **okno wczesne 300–2188 → okno końcowe 12 034–13 128** (klatka/odstęp / τ_1e / β / f_q /
ucieczki τ=10 / E₇₆₈ / C₁@τ):

| lr | wcześnie | koniec (12 epok) | mAP |
|---|---|---|---|
| 1,25e-4 | 0.258 / 829 / 0.99 / 0.72 / 0.010 / 0.310 / 0.58@5 | 0.207 / ∞ / 0.52 / 0.79 / 0.001 / 0.049 / 0.52@5 | 0.57 |
| 5e-4 | 0.490 / 176 / 0.80 / 0.43 / 0.116 / 0.669 / 0.60@2 | 0.444 / 136 / 0.45 / 0.46 / 0.046 / 0.151 / 0.62@5 | 0.62 |
| 2e-3 | 0.829 / 10 / 0.77 / 0.23 / 0.410 / 0.598 / 0.46@2 | 0.647 / 22 / 0.38 / 0.28 / 0.226 / 0.179 / 0.58@5 | 0.60 |
| 5e-4, po 60 epokach (okno60_rp) | – | 0.358 / 295 / 0.54 / 0.55 / 0.008 / 0.058 / 0.54@10 | 0,61 |

Z wiekiem treningu przy każdym lr klatka maleje, relaksacja wydłuża się i staje się silnie rozciągnięta (β 0,99 → 0,52,
0,80 → 0,45, 0,77 → 0,38), energia kroku spada 4–6× po 12 epokach i 12× po 60. Przy lr 1,25e-4 po 12 epokach układ
przestaje relaksować w oknie 1094 kroków (klatka 0,21 odstępu, τ_1e = ∞, karta S1 9/14): zatrzymanie osiąga się przez
obniżenie temperatury i starzenie, nie przez gęstość danych. Przy lr 2e-3 nawet po 12 epokach klatka 0,65 i τ_1e 22: płyn.

Temperatura: klatka ∝ lr^0,43, E₇₆₈ ∝ lr^0,45, więc klatka² ∝ E (dołek harmoniczny o sztywności niezależnej od lr); f_q
i τ_1e rosną przy chłodzeniu (jak szkło i jak OU); szczyt α₂ zawsze przy τ = 3–5 (początek plateau, nie koniec); β rośnie
przy chłodzeniu (odwrotnie niż w szkle; zastrzeżenie: to samo okno to inny etap uczenia przy innym lr). Gęstość: 23× zmiana
ograniczeń na parametr nie zmienia klatki, c ani f_q; τ_1e krótsze przy mniejszej liczbie zdjęć, sprzężenie między
zdjęciami słabsze. Rozrzut ziaren: klatka ±8 %, τ_1e ×2. Po 60 epokach: klatka jak wcześnie, c niższe (0,29), relaksacja
silnie rozciągnięta (β 0,50), energia kroku 10× mniejsza niż wcześnie. Długie biegi (okno końcowe) i fala 3 (rzut 128D):
tabela w `POKRETLA2.json`, rysunek `pokretla2.png`.

## 6. Werdykt (po pokrętłach, heterogeniczności i kontroli rzutu)

Ruch tokenów to **gorące, samostygnące cząstki w dołkach, niesione gładkim polem aktualizacji**. Temperaturę dołka ustawia lr
(klatka² ∝ energia kroku), nie gęstość danych (23× zmiana ograniczeń na parametr nie zmienia klatki). Większość odcisków szkła
(plateau, antykorelacja, rozciągnięta relaksacja, starzenie z wykładnikiem < 1) daje sama dynamika Langevina w stygnącym
dołku (kontrola OU). Korelacje przestrzenne ruchliwości są silniejsze niż w szkle, ale największe przy najkrótszym τ,
dalekozasięgowe i obecne między zdjęciami, czyli pochodzą od gładkiego, niskowymiarowego pola (obraz jądra), a nie od klatki
z sąsiadów, której korelacje narastają dopiero do czasu ucieczki. Cechy swoiste dla szkła wielu ciał (heterogeniczność
rosnąca do τ_α, szczyt α₂ na końcu plateau, zatrzymanie sterowane gęstością) nie występują we wczesnym oknie; po 60 epokach
relaksacja jest silnie rozciągnięta (β 0,5), a struktura sąsiedztw zamrożona, co wymaga potwierdzenia w rzucie 128D.
Zastrzeżenia: PCA32 z kroku 0 widzi 4–14 % ruchu (5b); jedna warstwa (ukryta MLP ostatniego bloku); kontrola dodatnia w 3D;
AdamW zamiast SGD; jedno ziarno dla większości wyników.

## 6b. Jaka to substancja? Testy rozstrzygające (2026-10-11; `dh.py`, `substancja_kontrola.py`, `substancja_rysunek.py` → `substancja.png`, `SUBSTANCJA.json`; `kod/jadro_pola.py` → `JADRO_*.json`, `jadro_tabela.py` → `JADRO_TABELA.json`)

Pytanie z 2026-10-11: skoro korelacje przestrzenne są „polowe”, a nie szkliste, to źle czy dobrze — i czy to w ogóle szkło, czy inna
substancja (huragan?). Odpowiedź: to nie wada pomiaru, tylko identyfikacja mechanizmu. Pięć testów z literatury (przegląd kandydatów
A–H: żel w trakcie sieciowania, sedymentacja/hydrodynamika, turbulencja, wspólny szum/efekt Morana, nieporządek zamrożony, odpowiedź
sprężysta ciała amorficznego, białka, stada/galaktyki, NTK) policzone dla tokenów ViT w rzucie 128D (okno 300–2188, 12 sond; biegi
lr 1,25e-4 … 2e-3, gęstości, ziarna 47–49, długie okna) i dla trzech układów odniesienia: szkła BD (duży bieg stały, φ 0,58–0,64),
OU-B (dołki zamrożone o rozrzucie sztywności 10×, szum niezależny) i OU z 50 % wspólnego szumu zdjęcia (efekt Morana).

| test (`dh.py`) | tokeny ViT (lr 5e-4; zakres po biegach) | szkło BD φ 0,58 → 0,64 | OU-B | OU + wspólny szum zdjęcia |
|---|---|---|---|---|
| trwałość rankingu ruchliwości (Spearman względny: podokna 200 kroków, odstęp 1 → 9) | 0,96 → 0,82 (0,94–0,98 → 0,75–0,94) | 0,49/0,37/0,57/0,84 → 0,46/0,22/0,19/0,51 | 0,97 → 0,95 | 0,97 → 0,95 |
| korelacja krzyżowa położeń względem klatki C_ij(0): sąsiedzi / pary losowe w zdjęciu (bez pola zdjęcia) | 0,84 / 0,00 (0,63–0,85) | 0,004 / 0,04 / 0,12 / 0,24 ; losowe 0,00 | 0,00 / 0,00 | 0,47 / 0,40 |
| to samo z polem zdjęcia: pary losowe w zdjęciu / z innych zdjęć | 0,24 / −0,02 (0,16–0,39 / ≈0) | – | – | 0,40 / −0,04 |
| energia kroku τ=1: dryf populacji / pole zdjęcia / pole sąsiedztwa / ruch własny | 0,25–0,37 / 0,11–0,22 / 0,33–0,43 / 0,10–0,15 | 0 / 0 / 0,18–0,30 / 0,83–1,16 (szum niezależny daje 0,17 / 1,17) | 0 / 0 / 0,15 / 1,15 | 0,05 / 0,44 / 0,09 / 0,59 |
| rozbieganie par sąsiadów po ≈1900 krokach, ⟨r²⟩/⟨r₀²⟩ (wykładnik wzrostu) | 0,9–1,9 (≤ 0,4); lr 2e-3: 3,6 | 3,6 / 1,7 / 1,2 / 1,0 (≈ 1, dyfuzja; nigdy t³) | 0,45 (stygnięcie) | 0,41 |
| dwupunktowa korelacja przesunięć ⟨Δz_i·Δz_j⟩/⟨Δz²⟩ | 0,5–0,75 przy r ≤ d_shell, 0 przy 3 d_shell; między zdjęciami ≈ 0,6 dla bliskich par | 0,15–0,38 przy r ≈ d_shell, 0 przy 2–3 d_shell | 0 | 0 w zdjęciu |

Co z tego wynika:
- **Huragan/turbulencja odpada**: pary sąsiadów nie rozbiegają się ani jak t³ (Richardson), ani wykładniczo (Batchelor); po 1900 krokach
  są tam, gdzie były (⟨r²⟩/⟨r₀²⟩ ≈ 1), a retencja sąsiedztwa po 60 epokach wynosi 0,75.
- **Szkło odpada jako mechanizm korelacji**: w szkle BD sąsiedzi mają C_ij(0) 0,004–0,24 i korelacje ruchliwości narastające do τ_α;
  w ViT sąsiedzi poruszają się niemal sztywno razem (C_ij(0) 0,84) od pierwszego kroku, a pary losowe z tego samego zdjęcia wcale
  (0,00). Heterogeniczność ruchliwości jest **zamrożona** (ranking tokenów trwa przez całe okno jak w OU z zamrożonymi dołkami,
  0,96 → 0,82), podczas gdy w szkle jest przemijająca (0,57 → 0,19 przy φ 0,62; część trwałości BD to polidyspersja 8,7 %).
- **Prawie cały ruch jest polem**: ruch własny w klatce to 10–15 % energii kroku, dryf populacji 25–37 %, pole zdjęcia 11–22 %, pole
  sąsiedztwa 33–43 %. W szkle i w OU ruch własny to ≥ 85 %.
- **Efekt Morana tylko częściowo**: wspólny szum zdjęcia (OU-Moran) dałby korelację także par losowych w zdjęciu (0,40); w ViT po
  odjęciu pola zdjęcia zostaje tylko lokalne pole sąsiedztwa (0,84 / 0,00). Pole jest więc gładką funkcją położenia w przestrzeni
  reprezentacji, nie wspólnym szumem zdjęcia — i sięga między zdjęcia (bliskie pary z różnych zdjęć ≈ 0,6).

**Test jądra** (`kod/jadro_pola.py`, Eden, GPU; stan `pelny/stan_e02.pt` biegu lr_5e-4_rp = krok 2188; 12 sond, 40 paczek, 100 kierunków
losowych; `JADRO_stan_e02.json`): przesunięcia tokenów Δz_i = z_i(θ+Δθ) − z_i(θ) policzone różnicami skończonymi dla rodzin Δθ, potem
TE SAME miary co dla zmierzonych przyrostów (dh.analiza τ=1 w układzie CR, rozkład energii, Spearman mapy ruchliwości |Δz_i|):

| rodzina Δθ (lr 5e-4, krok 2188) | ⟨Δz²⟩ 768D | C₁ / powłoka 2 / 3 | z innych zdjęć | dryf / zdjęcie / sąsiedztwo / własny | Spearman mapy ruchliwości (względny) |
|---|---|---|---|---|---|
| zmierzone przyrosty, kroki 1988–2188 | 0,221 | 0,85 / 0,71 / 0,56 | 0,34 | 0,25 / 0,15 / 0,46 / 0,10 | – |
| prawdziwy krok AdamW na paczce (wagi + BN, ze stanu pędu) | 0,190 | 0,89 / 0,75 / 0,58 | 0,33 | 0,30 / 0,20 / 0,40 / 0,06 | 0,73 |
| same wagi (bufory BN przywrócone) | 0,192 | 0,89 / 0,75 / 0,59 | 0,34 | 0,31 / 0,20 / 0,40 / 0,06 | 0,72 |
| same statystyki BatchNorm (forward w train bez kroku) | 0,002 | 0,83 / 0,66 / 0,50 | 0,30 | 0,11 / 0,22 / 0,50 / 0,13 | 0,57 |
| szum minibatcha (krok − krok na średnim gradiencie) | 0,037 | 0,90 / 0,76 / 0,60 | 0,40 | 0,24 / 0,18 / 0,46 / 0,07 | **0,85** |
| losowy kierunek izotropowy w wagach (norma kroku) | 0,001 | 0,80 / 0,63 / 0,46 | 0,28 | 0,25 / 0,15 / 0,48 / 0,09 | 0,61 |
| losowy kierunek przez prekondycjoner AdamW | 0,000 | 0,92 / 0,80 / 0,66 | 0,64 | 0,36 / 0,13 / 0,46 / 0,04 | −0,19 |
| kontrola: ten sam forward bez zmiany wag | 0 | – | – | – | – |

To samo przy lr 1,25e-4 (zmierzone 0,79/0,65/0,49, cross 0,40; krok 0,89/0,73/0,56, 0,51; szum: Spearman mapy 0,96) i po 12 epokach
(zmierzone 0,71/0,57/0,44, 0,23; krok 0,85/0,68/0,53, 0,26; szum 0,69). Czyli:
- cała struktura „pola” — zanik po powłokach, sprzężenie między zdjęciami, rozkład energii — **jest Jakobianem wspólnych wag**
  (δz_i = J_i δθ): odtwarza ją jeden krok optymalizatora z zapisanego stanu, a nawet losowy kierunek w przestrzeni wag, bez danych;
- **które** tokeny są ruchliwe, też jest własnością sieci: część szumu prawdziwych kroków przewiduje mapę ruchliwości ze Spearmanem
  0,85–0,96 (wczesne okna) — ruchliwość tokenu to jego „wzmocnienie” pod szumem minibatcha, stąd zamrożona heterogeniczność i korelacja
  z błędem zdjęcia; losowe kierunki prekondycjonowane mapy nie przewidują (−0,19), bo trafiają w rzadko aktualizowane parametry;
- **BatchNorm nie jest źródłem ruchu**: jitter statystyk łodygi na krok to ~1 % energii kroku (0,002 wobec 0,19), choć ma tę samą
  strukturę przestrzenną. (W pierwszej wersji testu nieprzywracane bufory BN udawały „szum numeryczny” rzędu 40 % — pliki
  `JADRO_*_bezBN.json`/`_tf32.json` zostawione jako ostrzeżenie; właściwe są `JADRO_<stan>.json`.)
- losowe kierunki o normie prawdziwego kroku dają 100–1000× mniejszą energię przesunięć: gradient trafia w czułe kierunki Jakobianu;
  porównujemy strukturę korelacji, nie wielkość.

**Werdykt po teście jądra.** Substancja to nie szkło koloidalne, nie żel i nie turbulencja, tylko **cząstki w osobnych, zamrożonych,
sztywniejących dołkach, sprzężone wspólnym ośrodkiem** — fizycznie najbliżej koloidów w pułapkach optycznych sprzężonych przez ciecz
(Meiners i Quake 1999: korelacje przez ruchliwość ośrodka, nie przez potencjał) i populacji pod wspólnym napędem (efekt Morana,
Hugueny 2006), z tą różnicą, że ośrodkiem są wspólne wagi, a sprzężenie jest lokalne w przestrzeni reprezentacji (gładkie pole).
Jednocząstkowe odciski szkła (klatka ∝ √lr, starzenie, rozciągnięta relaksacja, zatrzymanie przy niskim lr) są własnościami dołków
(krajobraz + szum kroku) i pozostają prawdziwe; zbiorowe odciski szkła (heterogeniczność narastająca do τ_α, krótkozasięgowe
korelacje) nie występują, bo tokeny nie oddziałują — ich wspólny ruch jest geometrią jednego odwzorowania. Dla artykułu to lepszy
wynik niż „szkło”: mechanizm jest nazwany, przewidywalny z Jakobianu (test jądra) i ma kontrole dodatnią i ujemną. Drugi model
(CNN z mieszaniem depthwise 5×5, `cnn_lr_*_rp`) daje to samo pole (C₁ 0,72–0,73, z innych zdjęć 0,39–0,46, trwałość 0,95 → 0,75),
ale bez modu zdjęcia (pole zdjęcia 0,02 wobec 0,11–0,22 w ViT): wspólny ruch całego zdjęcia pochodzi z uwagi, lokalne pole i
sprzężenie między zdjęciami — ze wspólnych wag.

Literatura testów (DOI zweryfikowane w Scite 2026-10-11): Meiners & Quake 1999 (10.1103/PhysRevLett.82.2211), Crocker i in. 2000
(10.1103/PhysRevLett.85.888), Levine & Lubensky 2000 (10.1103/PhysRevLett.85.1774), Segrè i in. 1997 (10.1103/PhysRevLett.79.2574),
Bourgoin i in. 2006 (10.1126/science.1121726), Hugueny 2006 (10.1111/j.2006.0030-1299.14686.x), Widmer-Cooper i in. 2004
(10.1103/PhysRevLett.93.135701), Berthier & Jack 2007 (10.1103/PhysRevE.76.041509), Lemaître 2014 (10.1103/PhysRevLett.113.245702),
Cavagna i in. 2010 (10.1073/pnas.1005766107), Fort i in. 2019 (arXiv:1901.09491), Pashakhanloo & Koulakov 2023 (10.48550/arXiv.2302.02563).

## 6c. Fala 4 (2026-10-11, noc): ziarna w 128D, SGD i paczka, drugi model, skan po warstwach, drugi zbiór, f32 (`substancja_rysunek.py` → `substancja.png`, `SUBSTANCJA.json`; `pokretla_rysunek2.py` → `pokretla2.png` ze słupkami ziaren)

22 biegi na hopper-2 (`experiments/maly_vit_szklo_pokretla_20261010/zlec_produkcje.sh`, cykl trening → analiza → kasowanie surowych gesty),
wszystkie w rzucie losowym 128D, 12 sond, okno 300–2188 (klatka = rozmiar klatki S1 / odstęp sąsiadów; τ_1e w krokach; C₁ = korelacja
ruchliwości sąsiadów w układzie CR; „z innych zdjęć” = C₁ z sąsiadami z innych zdjęć; trw = trwałość rankingu ruchliwości po 1800 krokach).

**Trzy ziarna** (47/48/49): lr 1,25e-4 — klatka 0,26/0,24/0,22, τ_1e 829/1073/1073, C₁ 0,58/0,56/0,58; lr 5e-4 — 0,50/0,57/0,50,
τ 136/48/105, C₁ 0,59/0,57/0,59, z innych zdjęć 0,28/0,30/0,33; lr 2e-3 — 0,83/0,78/1,21, τ 10/13/5. Przy lr ≤ 5e-4 rozrzut klatki
≈ 10 %, τ w granicach 2–3×; po stronie płynnej (2e-3) rozrzut duży (klatka 0,8–1,2). Struktura pola nie zależy od ziarna.

**SGD z pędem 0,9** (wd 0, obcięcie gradientu do normy 1 jak w przepisie):

| lr | paczka | klatka | τ_1e | β | f_q | C₁ | z innych zdjęć | mAP (3 epoki) |
|---|---|---|---|---|---|---|---|---|
| 0,03 | 32 | 0,37 | 382 | 0,88 | 0,53 | 0,63 | 0,42 | 0,48 |
| 0,1 | 16 | 0,39 | 1073 | 0,87 | 0,58 | 0,48 | 0,25 | 0,42 |
| 0,1 | 32 | 0,40 | 829 | 0,90 | 0,55 | 0,54 | 0,33 | 0,51 |
| 0,1 | 64 | 0,38 | 829 | 0,92 | 0,56 | 0,57 | 0,32 | 0,57 |
| 0,1 | 128 | 0,31 | 1073 | 1,05 | 0,65 | 0,55 | 0,33 | 0,61 |
| 0,3 | 32 | 0,60 | 136 | 0,90 | 0,39 | 0,45 | 0,21 | 0,36 |

Temperaturą jest **długość kroku**, nie lr/paczka: przy lr 0,1 paczka 16 → 128 (8× mniej szumu gradientu) zostawia klatkę 0,31–0,40,
bo obcięcie gradientu (jak normalizacja Adama) ustala normę kroku ≈ lr niezależnie od szumu paczki; klatka rośnie z lr jak lr^0,2
(0,03 → 0,3), słabiej niż przy AdamW (lr^0,5), bo obcięcie nasyca. Pole (C₁ 0,45–0,63, z innych zdjęć 0,21–0,42) jest takie samo
jak przy AdamW: nie zależy od optymalizatora.

**Drugi model, CNN** (`model_cnn.py`: ta sama łodyga i głowa, mieszanie depthwise 5×5 zamiast uwagi, 2,29 M parametrów; AdamW):

| lr | klatka (ViT) | τ_1e (ViT) | β | f_q | C₁ (ViT) | z innych zdjęć (ViT) | pole zdjęcia w energii kroku (ViT) | mAP (ViT) |
|---|---|---|---|---|---|---|---|---|
| 1,25e-4 | 0,28 (0,26) | 382 (829) | 1,00 | 0,64 | 0,73 (0,58) | 0,46 (0,33) | – | 0,57 (0,47) |
| 5e-4 | 0,47 (0,50) | 37 (136) | 0,69 | 0,33 | 0,72 (0,59) | 0,39 (0,28) | 0,02 (0,15) | 0,63 (0,57) |
| 2e-3 | 0,78 (0,83) | 8 (10) | 0,51 | 0,13 | 0,69 (0,46) | 0,32 (0,17) | – | 0,63 (0,60) |

To samo prawo temperatury (klatka ∝ lr^0,37), szybsza relaksacja, **silniejsze** pole sąsiedztwa i sprzężenie między zdjęciami, ale brak
modu zdjęcia (0,02 wobec 0,11–0,22 w ViT): lokalne pole i sprzężenie między zdjęciami pochodzą ze wspólnych wag (Jakobian), a wspólny
ruch całego zdjęcia — z uwagi.

**Skan po warstwach** (lr 5e-4, rzut GELU MLP każdego bloku, 4 sondy na warstwę, `warstwy_lr_5e-4_rp`):

| blok | klatka | τ_1e | f_q | C₁ | z innych zdjęć | trwałość rankingu |
|---|---|---|---|---|---|---|
| 0 | 0,34 | 640 | 0,56 | 0,67 | 0,26 | 0,89 |
| 1 | 0,38 | 382 | 0,51 | 0,61 | 0,23 | 0,84 |
| 2 | 0,47 | 136 | 0,42 | 0,56 | 0,18 | 0,83 |
| 3 | 0,47 | 295 | 0,47 | 0,62 | 0,16 | 0,82 |
| 4 | 0,51 | 228 | 0,44 | 0,60 | 0,14 | 0,81 |
| 5 | 0,51 | 228 | 0,44 | 0,61 | 0,13 | 0,81 |

Zimne dno, gorąca góra: z głębokością klatka rośnie 0,34 → 0,51, τ maleje 640 → 228, a sprzężenie między zdjęciami słabnie 0,26 → 0,13
(wczesne cechy są wspólne dla zdjęć, późne — swoiste); korelacja z sąsiadami w zdjęciu jest płaska (0,55–0,67). Wniosek, że „tokeny
ostatniej warstwy są gorącym płynem”, dotyczy więc wierzchołka sieci; dolne bloki są bliżej zatrzymania.

**Drugi zbiór: Imagenette** (10 klas ImageNet, 320 px, 9469 obrazów, ViT z głową na średniej tokenów, 400 tokenów na obraz;
`trening_klasyfikacja.py`, 8 epok = 2368 kroków; długi bieg 40 epok z oknem 10746–11840):

| bieg | klatka | τ_1e | β | f_q | C₁ | z innych zdjęć | trwałość | top-1 |
|---|---|---|---|---|---|---|---|---|
| ViT lr 1,25e-4 | 0,31 | 1390 | 0,95 | 0,66 | 0,56 | 0,17 | 0,92 | 0,75 |
| ViT lr 5e-4 | 0,73 | 105 | 0,73 | 0,40 | 0,64 | 0,16 | 0,90 | 0,75 |
| ViT lr 5e-4, okno po 36 epokach | 1,05 | 17 | 0,46 | 0,27 | 0,56 | 0,04 | 0,93 | 0,79 |
| ViT lr 2e-3 | 4,6 | 1 | – | 0,10 | 0,65 | 0,21 | 0,73 | 0,40 (niestabilny) |
| CNN lr 5e-4 | 0,94 | 10 | 0,95 | 0,28 | 0,71 | 0,21 | 0,70 | 0,73 |

Ta sama fenomenologia (klatka rośnie z lr, pole sąsiedztwa 0,56–0,71, heterogeniczność zamrożona 0,9), słabsze sprzężenie między
zdjęciami (0,16 wobec 0,3) i — co ważne — **brak samostygnięcia**: przy lr 5e-4 po 36 epokach tokeny są bardziej płynne niż na
początku (klatka 0,73 → 1,05, τ 105 → 17), choć dokładność rośnie 0,75 → 0,79. W detekcji gęsta strata przypina każdy token
(klatka maleje 0,49 → 0,44, τ rośnie); w klasyfikacji strata widzi tylko średnią tokenów, więc pojedyncze tokeny wędrują swobodnie.
Sztywność dołków jest więc własnością zadania (ile strata „trzyma” token), nie sieci.

**Ocena klasyfikacji ze stanów** (`kod/ocena_klasyfikacji.py`, `ocena_klasyfikacji.slurm` → `eden/wyniki/pokretla/OCENA_KLASYFIKACJI.json`;
walidacja 3925 obrazów, uczący 9469 bez augmentacji; mAP = średnia po 10 klasach precyzji średniej one-vs-rest na softmaksie, pewność =
max softmax, ECE w 15 przedziałach). Pole `mAP50` w OCENY.json biegów klasyfikacji to tylko kopia top-1.

| bieg (stan) | top-1 | top-5 | mAP | pewność (trafienia / pomyłki) | ECE | top-1 uczący |
|---|---|---|---|---|---|---|
| ViT lr 1,25e-4 (e08) | 0,753 | 0,974 | 0,852 | 0,79 (0,85 / 0,59) | 0,037 | 0,836 |
| ViT lr 5e-4 (e08) | 0,754 | 0,973 | 0,863 | 0,80 (0,86 / 0,60) | 0,041 | 0,822 |
| ViT lr 2e-3 (e08) | 0,403 | 0,861 | 0,446 | 0,40 (0,46 / 0,35) | 0,032 | 0,406 |
| CNN lr 5e-4 (e08) | 0,731 | 0,971 | 0,834 | 0,74 (0,81 / 0,55) | 0,018 | 0,767 |
| ViT lr 5e-4, długi (e39) | 0,816 | 0,981 | 0,897 | 0,92 (0,96 / 0,76) | 0,106 | 0,970 |
| ViT lr 5e-4, długi (e40) | 0,784 | 0,974 | 0,873 | 0,91 (0,95 / 0,78) | 0,132 | 0,938 |

Po 8 epokach modele są dobrze skalibrowane (ECE 0,02–0,04). Po 40 epokach ViT przeucza się: 97 % na zbiorze uczącym wobec 82 % na
walidacji, pewność rośnie do 0,92, także na pomyłkach (0,76), a ECE do 0,11–0,13. Najsłabsze klasy po 39 epokach: dystrybutor paliwa
(0,69) i piła łańcuchowa (0,71); najlepsze: springer spaniel (0,91) i lin (0,90).

**f32 kontra f16** (okno po 60. epoce, wznowienie ze stanu e60): klatka 0,361 wobec 0,358, τ_1e 295/295, β 0,57/0,54, f_q 0,55/0,55,
C₁ 0,538/0,538 przy τ=10, energia kroku 0,060/0,058. Zapis f16 wystarcza; różni się tylko sprzężenie między zdjęciami (0,16 wobec 0,24),
bo wznowione biegi rozchodzą się przez niedeterminizm GPU.

**Werdykt po fali 4.** Obraz z 6b utrzymuje się we wszystkich wariantach: temperatura = długość kroku optymalizatora (AdamW: klatka ∝ lr^0,5;
SGD z obcięciem: lr^0,2, paczka bez wpływu), pole przesunięć = Jakobian wspólnych wag (ten sam w ViT, CNN, na Imagenette, przy SGD),
mod zdjęcia = uwaga, zimne dolne warstwy, a sztywnienie dołków z wiekiem (starzenie, zatrzymanie) zależy od tego, czy strata przypina
pojedyncze tokeny (detekcja: tak, klasyfikacja ze średnią: nie).

## 6d. Czas relaksacji a rozmiar klatki na wszystkich biegach (`relacja_tau_klatka.py` → `RELACJA_TAU_KLATKA.json`, `relacja_tau_klatka.png`)

Pierwsze porównanie „jak w koloidach” na całym materiale z Edena: 41 okien RP128 (ViT AdamW z ziarnami i oknami późnymi, SGD, CNN,
warstwy 0–5, Imagenette) na jednym wykresie ln τ_1e kontra klatka²/odstęp². W szkłach Larini i in. 2008 (Nat. Phys., 10.1038/nphys788)
podają uniwersalną prostą ln τ_α ∝ 1/⟨u²⟩; proces OU ze stałym krokiem daje τ ∝ 1/u². Tokeny: ani jedno, ani drugie.

| rodzina | okien | τ ∝ (u²)^k, k | R² potęgowe | R² Lariniego | τ względem prawa ViT przy tej samej klatce |
|---|---|---|---|---|---|
| ViT AdamW, detekcja | 20 | −1,80 | 0,95 | 0,73 | 1 |
| ViT SGD | 6 | −1,55 | 0,71 | 0,57 | ×2,9 wolniej |
| CNN AdamW | 3 | −1,86 | 0,99 | 0,97 | ×0,43 szybciej |
| ViT warstwy 0–5 | 6 | −1,34 | 0,71 | 0,74 | ×2,0 wolniej |
| Imagenette | 6 | −1,33 | 0,90 | 0,68 | ×4,2 wolniej |

Wewnątrz rodziny ViT AdamW relacja jest potęgowa z wykładnikiem ≈ −1,8 (τ ∝ u^−3,6), a nie −1 jak w OU: przy małym lr ucieczka
z klatki zwalnia szybciej, niż wynika z samego skalowania kroku. Między rodzinami nie ma wspólnej krzywej: przy tej samej klatce SGD
relaksuje ~3× wolniej, klasyfikacja ~4× wolniej, a CNN ~2× szybciej niż ViT z detekcją. Rozmiar klatki nie wyznacza więc czasu
ucieczki, jak czyni to w szkłach (Larini); o ucieczce decyduje osobno krajobraz straty i architektura. To kolejna różnica wobec szkła
strukturalnego i argument, że „temperatura” i „sztywność dołka” to dwa niezależne pokrętła, nie jedno.

## 7. Aplikacja i animacje

Strona „Checkpoint i dane” → **oś szkła** (lista `outputs/ruch_czasteczek_gui/*szklo*`), **Odśwież**, **Wczytaj oś szkła**.
Pomiar 4000 zdjęć wczytuje się przez „zapis ruchu” (folder `20261010T_natural_eden_pelny_3000tr_1000val`, obszary oka
liczą się przy pierwszym wczytaniu ~70 s). Klipy: G1 jedno zdjęcie względem klatki ze śladami, G2 cała chmura w układzie
klatki, G3 kolor temperatury (stygnięcie), G4 symulowane szkło przy stałej T, G5 szkło stygnące; `montaz_szkla.py`.

## 8. Literatura (zweryfikowana w Undermind)

Weeks i in. 2000 doi:10.1126/science.287.5453.627 · Weeks, Weitz 2002 doi:10.1103/PhysRevLett.89.095704 i
doi:10.1016/S0301-0104(02)00667-5 · Hunter, Weeks 2012 doi:10.1088/0034-4885/75/6/066501 · Vivek i in. 2017
doi:10.1073/pnas.1607226113 · Chaudhuri, Berthier, Kob 2007 doi:10.1103/PhysRevLett.99.060604 · Weysser i in. 2010
doi:10.1103/PhysRevE.82.011504 · Löwen, Hansen, Roux 1991 doi:10.1103/PhysRevA.44.1169 · Charbonneau i in. 2012
doi:10.1073/pnas.1211825109 · Mignacco, Urbani 2022 doi:10.1088/1742-5468/ac841d · Kunin i in. 2023 doi:10.1162/neco_a_01626 ·
Sarkar i in. 2025 doi:10.1063/5.0262404 · Ottochian i in. 2009 doi:10.1063/1.3269041 · Baity-Jesi i in. 2019
doi:10.1088/1742-5468/ab3281 · Kerr Winter, Janssen 2025 doi:10.1103/physrevresearch.7.023010.

Dynamika szklista sieci (deep search Undermind 2026-10-10): Baity-Jesi i in. 2019 doi:10.1088/1742-5468/ab3281 · Geiger i in. 2019 doi:10.1103/PhysRevE.100.012115 · Herrera Segura, Montoya, Tapias 2022 doi:10.1088/2632-2153/ac8f1b · Chen, Qu, Gong 2022 doi:10.1016/j.neunet.2022.01.019 · Mignacco, Urbani 2022 doi:10.1088/1742-5468/ac841d · Kunin i in. 2023 doi:10.1162/neco_a_01626 · Pashakhanloo, Koulakov 2023 doi:10.48550/arXiv.2302.02563 · Bordelon, Pehlevan 2022 doi:10.1088/1742-5468/ad01b0 · Hertz, Tyrcha 2024 doi:10.1103/physreve.111.055307 · Kerr Winter, Janssen 2025 doi:10.1103/physrevresearch.7.023010 · Huang, Yoshino, Jin i in. 2025 arXiv:2506.06789. Metodologia szkieł: Lačević i in. 2003 doi:10.1063/1.1605094 · Berthier i in. 2005 doi:10.1126/science.1120714 · Pandit, Flenner, Castillo 2022 doi:10.1103/PhysRevE.105.014605 · Pareek i in. 2023 doi:10.1063/5.0166404 · Donati i in. 1998 doi:10.1103/PhysRevLett.80.2338 · Kob, Barrat 1997 doi:10.1103/PhysRevLett.78.4581 · Warren, Rottler 2013 doi:10.1103/PHYSREVLETT.110.025501 · Cherstvy i in. 2018 doi:10.1103/PhysRevE.98.022134.
