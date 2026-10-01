# Pętla małego ViT: wersja 4.3 (aktualna) i historia 4.0-4.2

**Wersja 4.3 (2026-10-01)** jest opisana krok po kroku w `../../docs/26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md` (kanoniczny opis
metody użytkownika, aktualizowany z kodem). Uruchomienie na Edenie: `eden/wyslij43.sh`, potem
`sbatch --array=0-0 eden/petla43_eden.slurm 47 proba` i `sbatch --array=0-9%5 eden/petla43_eden.slurm 47`.
Poniżej opis wersji 4.2, zostawiony jako historia.

# (historia) Pętla małego ViT, wersja 4.2: prognoza latentu tożsamości na jeden krok i Stable Diffusion zamiast FastGAN-a

Prośby użytkownika (2026-09-30):
- „pobrać checkpoint Stable Diffusion, zastąpić FastGAN-a i zaimplementować nowe równanie; zobaczmy, co da nowa pętla”;
- „prognoza ma za zadanie przewidzieć, jak się będzie rozwijał latent w przyszłości, który identyfikuje narzędzie!”;
- „porównuję z kontrolą bez prognozy, czyli np. losowe albo następujące po sobie klatki!”;
- „zastąp FastGAN tym SD i podstaw odpowiednie równanie, które potrafi przewidywać latent tożsamości do przodu”
  (równanie Δz(t+1) = S_pęd(t) + S_paczka(t) z `../maly_vit_tozsamosc_jeden_krok/WYNIKI.md`, commit a5813796);
- „no to zróbmy przewidywanie na jeden krok do przodu”.

## Historia wersji

- **4.0** (próba 16:20): cel prognozy zmieniony na spadek błędu detekcji; użytkownik zatrzymał. Zostaje tylko jako ramiona
  dodatkowe `*_strata` (propozycja z `../../docs/25_PROPOZYCJA_POTOKU_DOBORU_KLATEK.md`), poza porównaniem głównym.
- **4.1** (commit 78cffa10): prognoza siły tożsamości na 100 i 500 kroków równaniem z bezwładnością
  Δs(t+1) = 0,83·Δs(t) + 0,98·F przy stałej średniej sile F. Diagnoza (`diagnoza_prognozy.py`,
  `wyniki/diagnoza_s47/DIAGNOZA.json`): wektor tożsamości wyznaczony dobrze (AUC 0,98 na odłożonych sondach), jeden krok
  trafny, ale prognoza wieloetapowa gorsza od „braku zmiany” (1,5-50 razy), bo średnia siła jest około 30 razy mniejsza od
  losowego rozrzutu sił kolejnych paczek. Wniosek: równanie przewiduje skutek klatek, które wejdą do kroku, a nie
  dryf na setki kroków.
- **4.2** (ten plan): prognoza na jeden krok z pędem optymalizatora zamiast bezwładności, użyta do wyboru klatek.

## Równanie (kod `kod/prognoza_krok.py`)

Δz(t+1) = S_pęd(t) + S_paczka(t), oba współczynniki równe 1, bez dopasowania.
- z: latent narzędzia sondy (768 liczb, średnia po tokenach narzędzia), 180 sond z narzędziem; siła tożsamości
  s = rzut z na wektor tożsamości (LDA na sondach z końca rozgrzewki, jak w wersji 3); s jest liniowa w z.
- S_pęd(t): zmiana latentu sond, gdy wagi przesunąć tylko o część kroku AdamW z pamięci optymalizatora (paczki do t−1).
  Znana w chwili t bez żadnych nowych klatek.
- S_paczka(t): to samo dla części kroku z gradientu paczki kroku t, czyli klatek, które tworzą latent t+1.
- Umiejętność na nagraniach s47/s48 (bez uczenia): 95,2 i 95,6 procent dla siły tożsamości. Sam pęd: 64-66 procent.
- Tylko jeden krok: pchnięcia dalszych kroków zależą od przyszłych wag.

## Obrót pętli (kod `kod/petla4.py`)

1. **Prognoza na krok t+1 bez nowych klatek** (wszystkie ramiona): s(t) + S_pęd + średnie S_paczka z 8 losowych paczek
   obecnego zbioru. W ramieniu `sd_prognoza`: 6 z 12 kotwic, w których siła tożsamości ich sond (15 wariantów tej samej
   klatki) będzie po kroku najsłabsza.
2. **Wersje SD** każdej kotwicy: 8 obrazów Stable Diffusion z masek tej samej klatki (4 × ten sam układ z innym szumem,
   4 × narzędzie przestawione: obrót do ±35°, skala 0,85-1,15, przesunięcie do ±20 px).
3. **Wyszukanie**: wycinek 256 px wokół narzędzia z każdej wersji → DINOv2 → lista podobnych prawdziwych klatek puli →
   grupa jej pierwszych 22 klatek, które wybór i tak by wziął (reguła ±30 klatek filmu).
4. **Prognoza na krok t+1 z paczką z tej grupy**: gdzie będzie siła tożsamości sond kotwicy po jednym kroku uczenia na
   tych klatkach; S_paczka liczone dla wersji z odbiciem i bez (uczenie odbija połowę klatek), różnica = szum.
   Wybór wersji o największym przewidzianym wzroście. Pęd jest wspólny dla wszystkich wersji, więc o wyborze decyduje
   S_paczka; pęd wchodzi do przewidzianego położenia latentu.
5. **Zestaw**: 134 klatki na przemian z 6 wybranych list (jak wersja 3). Uczenie 500 kroków, mAP@0,5 co 100 kroków na
   3000 klatkach walidacji.
6. **Pomiary w każdym ramieniu**:
   - sprawdzian równania na prawdziwych krokach uczenia: przed krokiem prognoza z dokładnie tą paczką, którą uczenie
     zaraz weźmie, po kroku zmierzony przyrost; pierwsze 10 kroków obrotu i co 50-ty (20 kroków na obrót); umiejętność
     dla siły tożsamości i całego latentu, obok samego pędu;
   - przewidziane pchnięcie tożsamości przez dodany zestaw (S_paczka zestawu, przed uczeniem): czy metoda wybiera
     klatki, które według równania mocniej podnoszą tożsamość niż klatki losowe i kolejne;
   - w ramionach SD także S_paczka 6 losowych grup tej samej wielkości (tło porównania dla wybranych wersji).

## Ramiona

| ramię | co robi | rola |
|---|---|---|
| `sd_prognoza` | kroki 1-6 | metoda |
| `losowe` | losowe klatki puli, reguła ±30 | kontrola bez prognozy (wskazana przez użytkownika) |
| `kolejne` | klatki następujące po klatkach bazy w tym samym filmie (jak wersja 3) | kontrola bez prognozy (wskazana przez użytkownika) |
| `bez` | bez dokładania | dolna granica |
| `sd_bez_prognozy` | kotwice według obecnej siły, wersja SD losowa, reszta jak metoda | pomocnicza: jedyna różnica wobec metody to prognoza |
| `sd_strata`, `sd_strata_losowo`, `kotwice_strata` | wybór przez przewidziany spadek błędu detekcji (docs/25) | dodatkowe, tylko przez `--ramiona` |

## Różnice wobec wersji 3 (metoda użytkownika), każda nazwana

1. **Prognoza:** Koopman albo Langevin dopasowany do historii siły, na 500 kroków (wersja 3), zastąpiony równaniem
   jednego kroku S_pęd + S_paczka liczonym przez sieć. Horyzont: 1 krok zamiast 500, bo dłuższa prognoza nie działa
   (diagnoza 4.1).
2. **Rola prognozy:** w wersji 3 prognoza wyznaczała cel dla obrazu (siła kotwicy + przewidziana zmiana), a FastGAN rysował
   obraz o tej sile. Tu prognoza ocenia kandydatów: gdzie będzie latent tożsamości po kroku uczenia na klatkach znalezionych
   przez daną wersję SD; wybierana jest wersja, która najmocniej podnosi tożsamość. Cel „siła + zmiana na jeden krok” byłby
   praktycznie równy obecnej sile (zmiana jednego kroku jest mała wobec rozrzutu wersji), więc nie niósłby informacji.
3. **Kotwice:** wersja 3 brała 6 kotwic o największej przewidzianej zmianie; tu 6 kotwic o najsłabszej przewidzianej sile
   po następnym kroku.
4. **Generator:** FastGAN sterowany wzdłuż wektora tożsamości zastąpiony przez 8 wersji Stable Diffusion z masek tej samej
   klatki (bez sterowania wewnątrz generatora).
5. **Obraz zapytania:** wycinek 256 px wokół narzędzia z pełnej klatki SD (576 × 384), a nie wycinek FastGAN-a.
6. **Pula:** bez 10 filmów zbioru kontrolnego (używanego tylko przez ramiona `*_strata`); ta sama pula w każdym ramieniu.
7. **Ramię pomocnicze:** `current` z wersji 3 zastąpione przez `sd_bez_prognozy`.
Bez zmian: model, rozgrzewka, wektor tożsamości, 7 obrotów po 500 kroków, 134 klatki na obrót, 6 kotwic, reguła ±30,
wybór na przemian z list, walidacja, miara główna.

## Kryteria ustalone przed wynikiem (kampania wieloziarnowa, pole pod krzywą mAP@0,5 po rozgrzewce)

- K1: `sd_prognoza` lepsze od `losowe` w co najmniej 70 procentach ziaren i test Wilcoxona p < 0,05.
- K2: `sd_prognoza` lepsze od `kolejne` w co najmniej 70 procentach ziaren.
- K3 (rola prognozy): `sd_prognoza` lepsze od `sd_bez_prognozy` w co najmniej 70 procentach ziaren.
- Równanie w pętli: umiejętność jednego kroku dla siły tożsamości co najmniej 85 procent w każdym ramieniu (na nagraniach
  było 95).
- Opisowo: przewidziane pchnięcie dodanych klatek w ramionach, pchnięcie wybranej wersji SD wobec grup losowych,
  mAP ramion w czasie, siła tożsamości w czasie.

## Etapy i zgody

1. Próba kodu lokalnie (`--proba`, 2 obroty × 40 kroków) - sprawdza tylko działanie kodu.
2. Pełne ziarno lokalnie i kampania wieloziarnowa na Edenie - **każde po osobnym „tak”** z liczbą ziaren, kart i czasu.

Uruchomienie lokalne: `uruchom_lokalnie.cmd kod\petla4.py --ziarno 47 [--proba] [--ramiona ...]`,
wyniki w `wyniki/petla_krok_s47` (próba: `wyniki/proba_krok_s47`).
