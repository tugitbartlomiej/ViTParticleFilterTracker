# 26. Pipeline doboru klatek: tożsamość → prognoza → Stable Diffusion → wyszukanie → uczenie (dokument żywy)

Ten plik jest kanonicznym opisem metody użytkownika. Każda zmiana kodu pętli zmienia też ten plik (sekcja „Historia
zmian” na końcu). Ostatnia aktualizacja: 2026-10-01, wersja 4.3.

Metoda słowami użytkownika (2026-10-01): „mamy poprawną analizę tego, jak wygląda tożsamość w latencie, […] które
wektory, które tokeny. Potem analizę koopmanowską albo langevinowską na ileś kroków do przodu […] i przewidzieć, jak się
latent będzie rozwijał podczas uczenia. Na podstawie tego latentu, który się rozwinął w prognozie, metodą Stable Diffusion
wygenerować obrazy, potem wyszukać podobne obrazy w zbiorze uczącym, a potem wchodzą do zbioru uczącego, uczą, i ta metoda
dawała lepsze wyniki, jeżeli chodzi o szybkość uczenia modelu w rozpoznawaniu narzędzia.” Obrazy z SD mają „pokrywać
nawet w 90 % prognozowaną część latentu, w szczególności latentu tożsamości”.

Pytanie badawcze: czy przy tej samej liczbie dokładanych klatek metoda uczy detektor szybciej niż klatki losowe i klatki
kolejne.

## Kroki

| krok | co robi | kod (pętla: `experiments/maly_vit_petla_sd_rownanie/kod/`) | stan |
|---|---|---|---|
| 0 | dane i model | `dane.py`, `model.py`, `trening.py` | gotowe |
| 1 | rozgrzewka sieci | `petla4.py` (część wspólna) | gotowe |
| 2 | tożsamość narzędzia w latencie | `sila.py` (`wektor`, `sila`) | gotowe, sprawdzone |
| 3 | prognoza latentu tożsamości na obrót | `prognoza_gle.py` (Langevin z jądrem), `sila.prognozy` (Koopman/Langevin w. 3) | gotowe (4.3) |
| 4 | cel obrazu: przewidziany latent kotwicy | `petla4.py` (4.3) | gotowe |
| 5 | Stable Diffusion prowadzone latentem tożsamości | `prowadzenie.py` | gotowe: mediana 93 % tożsamość, 92 % latent |
| 6 | wyszukanie podobnych prawdziwych klatek | `wybor.py` (`zapytaj`, DINOv2) | gotowe |
| 7 | dobór 134 klatek | `wybor.py` (`koopman`) | gotowe |
| 8 | uczenie 500 kroków i ocena | `trening.py`, `petla4.py` | gotowe |
| 9 | pomiary w każdym obrocie | `prognoza_krok.py`, `petla4.py` | gotowe w 4.2; rozszerzyć w 4.3 |
| 10 | porównanie ramion i kampania | `petla4.py`, `eden/petla43_eden.slurm` | zgoda 1.10; próba na Edenie, potem 10 ziaren |

### Krok 0. Dane i model

- Mały ViT od zera: łodyga splotowa (4 warstwy, BatchNorm), 6 bloków transformera, wymiar 192, MLP 768, głowa
  centrów i ramek; wejście 480 × 320, siatka tokenów co 16 pikseli. AdamW, lr 5e-4, paczka 32, bf16.
- Baza startowa: 1500 klatek z ramkami. Pula do dokładania: klatki treningowe Cataract-LMM z indeksem DINOv2 (bez bazy,
  wykluczonych i ±30 klatek od nich). Walidacja: 3000 klatek. Miara: mAP@0,5 (trafność wykrywania narzędzia).
- Sondy: 12 klatek × 16 wariantów (obroty, tła czarne i żółte, przesunięcia, skale, przyciemnienie, perspektywa,
  `empty_black` bez narzędzia) = 192 obrazki 176 × 176; 180 z narzędziem.

### Krok 1. Rozgrzewka

1000 kroków na bazie (2 × 500), wspólna dla wszystkich ramion tego samego ziarna; latent sond co 10 kroków.

### Krok 2. Tożsamość narzędzia w latencie

- Które tokeny: tokeny wycinka sondy, w których maska narzędzia zajmuje co najmniej 25 % pola tokenu.
- Który latent: aktywacje MLP ostatniego bloku (768 liczb na token), uśrednione po tokenach narzędzia.
- Który wektor: kierunek LDA (Ledoit-Wolf) oddzielający latent narzędzia 180 sond od latentu tła 192 sond i obszaru
  narzędzia w `empty_black`, po standaryzacji średnią i odchyleniem negatywów; liczony na końcu rozgrzewki i stały
  do końca biegu.
- Siła tożsamości s = ((z − μ_neg) / sd_neg) · w: 0 na tle, około 5-6 na narzędziu.
- Sprawdzenie (diagnoza 4.1): AUC 0,98 na odłożonych sondach dla „narzędzie kontra to samo miejsce bez narzędzia”,
  d′ 13,7 wobec tła.

### Krok 3. Prognoza latentu tożsamości

- Metoda użytkownika (wersja 3): historia siły tożsamości 180 sond w blokach po 50 kroków; Koopman (liniowy operator na
  opóźnieniach) albo Langevin (dryf + jądro pamięci), dopasowane tylko na historii do chwili t; wybór modelu testem
  wstecznym; prognoza Δs na 500 kroków (jeden obrót) dla każdej sondy.
- Zmierzona trafność (umiejętność = część zmiany przewidziana lepiej niż „nic się nie zmieni”):
  - Koopman/Langevin wersji 3, 20 ziaren: 15-25 % od 50 do 1000 kroków (`maly_vit_horyzont_prognozy/WYNIKI.md`);
  - równania z historii co krok (Koopman z opóźnieniami = Langevin z jądrem pamięci; + pęd; + przyspieszenie):
    najlepsze 69 % na t+1, 45 % na t+3, około 25 % od t+5 do t+100, 10 % na t+500
    (`maly_vit_prognoza_tozsamosci_z_historii/WYNIKI.md`);
  - równanie jednego kroku Δz(t+1) = S_pęd(t) + S_paczka(t): 95-98 % na t+1, ale wymaga klatek kroku t; przy
    zamrożonych wagach 90 % tylko przez 2 kroki (`maly_vit_horyzont_rownania_krok/WYNIKI.md`).
- Wniosek dla kroku 3: na horyzoncie obrotu (500 kroków) żadne równanie nie przekracza około 25 %. „Lepsze równanie”
  (S_pęd + S_paczka) jest lepsze tylko na jeden krok. Domyślnie: Koopman/Langevin wersji 3 (metoda użytkownika).
  Równanie jednego kroku służy do sprawdzianu w kroku 9 (decyzja użytkownika, sekcja „Otwarte decyzje”).

### Krok 4. Cel obrazu

- Kotwice: 6 z 12 klatek-sond `original` (z układem masek), o największej przewidzianej zmianie |Δs| (jak wersja 3).
- Przewidziany latent kotwicy: z* = z_kotwicy + Δs · (sd ⊙ w), czyli latent prawdziwej klatki przesunięty wzdłuż
  kierunku tożsamości o przewidzianą zmianę; s(z*) = s_kotwicy + Δs, reszta latentu bez zmian.

### Krok 5. Stable Diffusion prowadzone latentem tożsamości

- Generator: SD 1.5 dostrojone na 311 886 klatkach 118 filmów (`dyfuzja_sd_filmy`, Eden 1818496, 20 000 kroków, strata
  0,179 → 0,092), migawka `ema_020000`; lokalnie `F:\Studia\PhD_projekt\VIT\eden_archiwum\sd_modele\ema_020000`
  (sha256 safetensors `255fb1a5…738eb6`, zgodne z Edenem); na Edenie
  `~/smallvit_runs/2026-09-30_dyfuzja_sd_filmy/wyniki/trening/ema_020000`. 576 × 384, DDIM 30 kroków, CFG 2.
- Warunek SD: maski narzędzia, rogówki i źrenicy kotwicy (SD nie zna latentu ViT; „latent” SD to skompresowany obraz).
- Prowadzenie (bez nowego uczenia, w duchu FreeDoM): w każdym kroku DDIM przewidywany czysty obraz → dekoder VAE →
  wycinek wokół narzędzia jak sonda → mały ViT → latent narzędzia z; energia
  E = (s(z) − s(z*))² + ‖(z − z*)/sd‖² / 768; przewidywany obraz przesuwany kilka razy w dół gradientu E, potem
  odszumianie idzie dalej. Odpowiednik sterowania wejściem FastGAN-a wzdłuż wektora tożsamości z wersji 3.
- Wymóg użytkownika: pokrycie ≥ 90 % na końcowym obrazie (tym samym wycinkiem co w pętli):
  tożsamość 100·(1 − |s − s*|/|s*|) i cały latent narzędzia 100·(1 − ‖ẑ − ẑ*‖²/‖ẑ*‖²).
- Start jak odwrócenie FastGAN-a w wersji 3: latent VAE prawdziwej klatki kotwicy zaszumiony do połowy (siła 0,5),
  odszumiany z masek tej klatki z prowadzeniem; krok prowadzenia 0,15 · σ_t, 3 iteracje na krok, ostatnie 2 kroki bez
  prowadzenia. Baza celu: latent klatki kotwicy tym samym wycinkiem co obraz SD (nie latent sondy).
- Test (`experiments/maly_vit_sd_prowadzenie_tozsamoscia`, sieć po 21 epokach, 3 kotwice, prognoza Langevina):
  pokrycie tożsamości mediana 93,5 % (bez prowadzenia 79 %), całego latentu 91,9 % (bez 83 %); 2 z 3 obrazów ≥ 90 % w obu;
  przy celu +1 w sile 3 z 3. Obrazy bez artefaktów (zmiana pikseli około 1/255), lista DINOv2 zmienia się w połowie.
  Pierwsza próba (start z czystego szumu, krok 0,6) dawała kolorowe artefakty i 62-65 % latentu - odrzucona.

### Krok 6. Wyszukanie

Wycinek 256 px wokół narzędzia z prowadzonego obrazu → DINOv2 ViT-S/14 (ten sam co przy budowie indeksu) → podobieństwo
do 10 637 wycinków puli → 90 najbliższych z różnorodnością grup; bez klatek wykluczonych i już dodanych.

### Krok 7. Dobór

134 klatki na obrót, na przemian z 6 list (po jednej na kotwicę), reguła ±30 klatek tego samego filmu.

### Krok 8. Uczenie i ocena

500 kroków na zbiorze baza + dodane; mAP@0,5 co 100 kroków na 3000 klatkach walidacji; 7 obrotów (3500 kroków po
rozgrzewce, 938 dodanych klatek).

### Krok 9. Pomiary w każdym obrocie

- Trafność prognozy z kroku 3 po obrocie (przewidziana kontra zmierzona siła, wobec „braku zmiany”).
- Pokrycie latentu przez obrazy SD z kroku 5.
- Sprawdzian równania jednego kroku na prawdziwych krokach uczenia (`prognoza_krok.Sprawdzian`): w próbie 4.2 97-99 %.
- Przewidziane pchnięcie tożsamości przez dodane klatki (S_paczka zestawu).
- Zapisy: OBROT.json, OCENY.json, SILA.npy, wagi sieci po każdym obrocie (telemetria na każdym punkcie kontrolnym).

### Krok 10. Porównanie i kampania

- Ramiona (ta sama liczba klatek, 134 na obrót): `metoda` (kroki 1-9, prognoza Langevina z jądrem), `metoda_kl3` (to samo z
  prognozą Koopman/Langevin wersji 3: porównanie równań), `losowe`, `kolejne` (klatki następujące po
  klatkach bazy), `bez` (bez dokładania), `sd_bez_prognozy` (to samo co metoda, ale cel z* = z_kotwicy, Δs = 0:
  rozdziela zasługę prognozy od zasługi samego wyszukania).
- Miara główna: pole pod krzywą mAP@0,5 po rozgrzewce (szybkość uczenia); obok mAP końcowe.
- Kryteria ustalone przed wynikiem: K1 metoda > losowe w ≥ 70 % ziaren i Wilcoxon p < 0,05; K2 metoda > kolejne w ≥ 70 %
  ziaren; K3 metoda > sd_bez_prognozy w ≥ 70 % ziaren.
- Kolejność (zgoda użytkownika 1.10 na start na Edenie): próba kodu lokalnie (zrobiona, 504 s, 6 ramion) → próba na
  Edenie (`--array=0-0 ... 47 proba`) → 10 ziaren 47-56 (`--array=0-9%5`), 14 obrotów, listy 180, rozgrzewka 21 epok z
  wersji 2. Kod na Edenie: `~/smallvit_runs/2026-10-01_petla43_sd_prowadzenie`.
- Wcześniejszy wynik (wersja 3, FastGAN): ramię z prognozą 0,4432, bez prognozy 0,4428 (w szumie); przewagę nad losowaniem
  dawało wyszukiwanie podobnych klatek (`docs/24`, A.3).

## Eden: przechowywanie

- Kwota 2026-10-01: 152 GiB z 200 GB, 78 tys. plików z 200 tys.
- Do usunięcia, gdy kampania będzie potrzebować miejsca (zgoda użytkownika 2026-10-01 na usunięcie starych punktów
  kontrolnych): `ostatni.pt` biegu SD 1818496 (13,75 GB, stan do wznowienia uczenia), migawki EMA 5000/10 000/15 000 tego
  biegu (kopie 10 000 i 15 000 są lokalnie), artefakty próby SD (`wyniki/proba/ostatni.pt`, 13,75 GB, i jej migawki).
  Zostaje `ema_020000` (generator kampanii). Nie dotykać `~/grant_tft`, `~/venvs`, `~/.cache`.
- Kampania zapisuje na Edenie wagi sieci po każdym obrocie każdego ramienia (około 11 MB × 7 × 5 ramion × 10 ziaren).

## Otwarte decyzje

1. Krok 3: zostać przy Koopmanie/Langevinie wersji 3 (domyślnie) czy dodać ramię z równaniem z historii (Langevin +
   jądro 20 + pęd, dopasowane osobno na 500 kroków). Zmierzone trafności są podobne (około 10-25 %).
2. Krok 5: progi prowadzenia (siła kroku, liczba iteracji) zależą od testu pokrycia 90 %.

## Historia zmian

| data | wersja | zmiana |
|---|---|---|
| 2026-09 | 3 | FastGAN sterowany wzdłuż wektora tożsamości, Koopman/Langevin na 500 kroków, DINOv2, 134 klatki |
| 2026-09-30 | 4.1 | SD z maskami zamiast FastGAN-a, wybór wersji SD najbliższej prognozie; prognoza równaniem z bezwładnością (nietrafna na 500 kroków) |
| 2026-09-30 | 4.2 | prognoza jednego kroku S_pęd + S_paczka wybiera klatki (odejście od metody użytkownika, nazwane w PLAN.md) |
| 2026-10-01 | 4.3 | powrót do metody użytkownika: Langevin z jądrem pamięci (i Koopman/Langevin w. 3 w osobnym ramieniu), SD prowadzone latentem tożsamości od zaszumionej kotwicy, wyszukanie DINOv2 (listy 180), 14 obrotów; ramiona metoda, metoda_kl3, sd_bez_prognozy, losowe, kolejne, bez |
