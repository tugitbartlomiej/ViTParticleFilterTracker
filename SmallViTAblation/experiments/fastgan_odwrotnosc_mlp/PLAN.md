# FastGAN do przewidzianego latentu: przyczyna chybiania i sieć odwrotna MLP

Data: 2026-09-28. Zgoda użytkownika: „zaprojektuj i testuj. Możesz próbować na moim GPU! Sprawdzaj i ucz MLP, żeby
ładnie tłumaczyła latent ViT (pełny latent) na wejścia do FastGAN, i potem, żeby FastGAN ładnie generował obrazy”.
Prognoza latentu: **tylko Koopman** (decyzja użytkownika).

## Problem

W pętlach `maly_vit_od_zera` i `maly_vit_wektor_tozsamosci` FastGAN sterowany gradientem (250 kroków Adama po wejściu
z, strata: latent obrazu − przewidziany latent, plus wierność obrazowi kotwicy i bliskość z) kończył średnio tak samo
daleko od przewidzianego latentu jak od stanu wyjściowego (wersja 1: reszta 2,84 przy odległości cel–obecny 2,34;
wersja 2: reszta mniejsza od tej odległości tylko w 39 % par).

## Ustawienie (wspólne dla testów, `wspolne.py`)

- ViT: mały ViT wersji 1, ziarno 42, krok 1000 (21 epok, `maly_vit_od_zera/wyniki/petla_s42/wspolny/k02.pt`).
- Pełny latent: 768 aktywacji MLP ostatniego bloku, średnia po tokenach narzędzia (wagi = udział maski w tokenie,
  próg 0,25), w jednostkach sd negatywów sond z kroku 1000.
- Cel: prognoza **Koopmana** na 500 kroków naprzód dla 12 sond „original” (kotwic), na wszystkich 768 neuronach,
  z historii latentów sond do kroku 1000 (`prognoza_bloku` z wersji 2).
- FastGAN 50k (EMA, zamrożony), stały szum warstw.
- **Postęp** = 1 − odległość(latent obrazu, cel) / odległość(latent startu, cel): 0 = brak ruchu, 100 % = trafienie.

## Krok 1: przyczyna (`diagnoza.py`)

Od inwersji kotwicy z*, trzy warianty sterowania:
- A „obecne”: jak w pętlach (250 kroków, wierność kotwicy, bliskość z*);
- B „długie bez hamulców”: 1000 kroków, tylko strata latentu;
- C „głębokie”: B plus przesunięcie cech 8×8 generatora (sterowanie głębszą warstwą).
Te same warianty na celach na pewno osiągalnych (latent innego obrazu z FastGAN-a).
Interpretacja: B dochodzi → winne hamulce; B nie dochodzi, a C tak → za mało swobody w z; nic nie dochodzi na celach
osiągalnych → problem optymalizacji.

## Krok 2: sieć odwrotna MLP (`mlp.py`)

MLP 768 → 1024 → 1024 → 256: z latentu podaje wejście z FastGAN-a. Uczona na parach z losowego z (obraz z FastGAN-a,
latent w ViT) stratą cyklu: latent(G(MLP(l))) ≈ l, z podpowiedzią ‖MLP(l) − z‖ na początku. Ocena poza zbiorem
uczącym: losowe latenty, latenty 180 prawdziwych sond, cele Koopmana (start z MLP, potem dokręcenie gradientem).

**Cel (ustalony przed wynikiem):** mediana postępu ≥ 70 % na 12 celach Koopmana po starcie z MLP i dokręceniu,
przy obrazie nadal wyglądającym jak narzędzie (ocena wzrokowa siatek).

## Wyniki kroku 1 (2026-09-28, RTX 3070 lokalnie; `wyniki/DIAGNOZA*.json`, siatki `wyniki/DIAGNOZA*.png`)

Dwie przyczyny dawnego chybiania, znalezione przed sterowaniem:
- **Cel był bezwzględny, a sieć jest pojedynczym stanem.** Prognoza Koopmana dotyczy średniej 5 zapisów (bloku);
  pojedynczy stan sieci w kroku 1000 różni się od tej średniej o 0,84 z-jednostki, a cały krok prognozy to 0,33.
- **Obraz z inwersji nie jest kotwicą dla ViT.** Inwersja FastGAN odtwarza kotwicę na oko (L1 pikseli 0,074), ale
  jej latent jest 2,19 od latentu kotwicy — prawie 7 razy dalej niż krok prognozy.
Dlatego cel liczony jest teraz jako latent obrazu startowego (inwersji) + przewidziany KROK Koopmana, a tokeny
narzędzia są stałe (z maski obrazu startowego; binarne jak w sondach — 0,16 od zapisu sond, miękkie dawały 0,24).

Postęp do celu (mediana 12 kotwic; 0 = brak ruchu, 1 = trafienie), próba 2 ze stałymi tokenami:
- A (dawne ustawienia: wierność kotwicy, bliskość z): 0,00 — wierność kotwicy blokuje ruch (w próbie 1 z ruchomą
  maską nawet −0,75: obraz uciekał od celu);
- B (samo wejście z, 400 kroków): 0,04; krok ×3: 0,05 — kierunek Koopmana leży prawie cały poza tym, co z zmienia;
- C (z + cechy 8×8, z karą za maskę): 0,20; C1 (bez kary, 800 kroków): 0,23 — połowa kroku wzdłuż kierunku (0,51),
  tyle samo w bok (0,56);
- D (z + cechy 8×8 + 16×16): **0,36**, wzdłuż 0,63, w bok 0,49, zmiana obrazu L1 0,075;
- P (same piksele, górna granica): **0,61**, wzdłuż 0,84, w bok 0,30, zmiana obrazu L1 tylko 0,022.
Cele na pewno osiągalne (latent innego obrazu FastGAN-a, odległość ~2,2): B 0,51, C 0,83–0,86.

**Wniosek:** sterowanie działa, ale kierunek przewidziany przez Koopmana wymaga zmian w głębi generatora; samo
wejście z (256 liczb) go nie wyraża. Sieć odwrotna musi więc podawać także poprawki głębszych warstw.

## Krok 2, zmieniony po wynikach kroku 1: sieć sterująca MLP (`mlp.py`)

Wejście: latent obrazu startowego l0 (768) i zadany krok Δ (768). Wyjście: poprawka wejścia δz (256) oraz
współczynniki 2 × 64 wyuczonych map poprawek cech 8×8 i 16×16 (mapy uczone razem z siecią). Strata:
‖latent(G(z0 + δz; cechy + poprawki)) − (l0 + Δ)‖² / ‖Δ‖² na stałych tokenach + małe kary za wielkość poprawek.
Dane: starty z losowego z; kroki Δ z prawdziwych zmian latentu sond w czasie uczenia (v1 s42, horyzont 500 kroków)
i z prognoz Koopmana w innych chwilach niż krok 1000 (cele oceny są wyłączone z uczenia).
Ocena na 12 celach Koopmana z kroku 1000: sama sieć (jeden przebieg) i sieć + dokręcenie gradientem (200 kroków D),
wobec samego dokręcenia D (200 kroków) od inwersji.

## Wyniki kroku 2 (2026-09-28)

**Sieć sterująca MLP (`mlp.py`, 2000 kroków, `wyniki/MLP.json`, `MLP_koopman.png`).** Błąd względny w uczeniu
spadł z ~4,4 do ~1,0 (1 = tak jak bez ruchu) i nie zszedł wyraźnie niżej. Na 12 celach Koopmana: sama sieć — postęp
0,01 (nauczyła się ostrożności: prawie nie rusza obrazu, zmiana L1 0,003); sieć + dokręcenie 200 kroków — 0,24;
samo dokręcenie 200 kroków — 0,25. **Sieć nie pomaga.**

**Stałe kierunki edycji (`kierunki.py`).** Kroki Koopmana leżą w 99,6 % w 16 kierunkach latentu (kroki oceny z kroku
1000: 99,2 %). Uczenie 16 wspólnych dla wszystkich obrazów edycji generatora rozbiegło się dwa razy (błąd względny
4–19 od początku, także przy starcie od zera i małym tempie) — przerwane; jedna edycja nie działa tak samo na różnych
obrazach.

**Wygląd obrazów** (`wyniki/DIAGNOZA2.png`: kotwica | inwersja | z + cechy 8×8 | z + 8×8 + 16×16 | piksele):
sterowanie głębszymi warstwami zostawia narzędzie, ale psuje tło (plamy, drobne jasne punkty); sterowanie pikselami
daje obraz prawie identyczny z inwersją, z drobnym szumem — latent się zmienia, a oko tego nie widzi.

**Wniosek kroku 2.** Przewidziana przez Koopmana zmiana latentu to w większości zmiana „niewidoczna”: da się ją
wymusić drobnym szumem pikseli (61 % drogi przy zmianie obrazu 2 %), ale FastGAN nie umie narysować jej jako
naturalnej zmiany wyglądu (samo wejście 4 %, głębsze warstwy do 36 % z artefaktami). Najlepsze dostępne narzędzie
do pętli: dokręcanie gradientem na warstwach 8×8 i 16×16 obrazu z inwersji, z celem względnym (start + krok).
Możliwa zmiana metody (wymaga decyzji użytkownika): szukać prawdziwych klatek wprost po latencie ViT (latent klatek
puli liczony bieżącą siecią, wybór najbliższych do przewidzianego latentu), bez pośrednictwa FastGAN-a.

## Krok 3: sterowanie po przeglądzie literatury (2026-09-28, `sterowanie3.py`, `NOTATKA_LITERATURA.md`)

Przegląd: [Rou26] „Show Me What You Don't Know” (zbiory niezmiennicze, sterowana dyfuzja; w pikselach wychodzi szum
w stylu ataku), [Vat25] (sieci nieodporne wymagają stabilizacji sterowania), [Ngu16], [San19], [Ban23c].
Test na 12 celach Koopmana (400 kroków, z + cechy 8×8 + 16×16; postęp czysty / odporny = latent na zaburzonych kopiach
obrazu: jasność, szum, rozmycie; `wyniki/STEROWANIE3.json`, `STEROWANIE3.png`):
- D0 (bez ochrony): czysty 0,29, odporny 0,15 (wzdłuż 0,54, w bok 0,57), zmiana obrazu 0,068 — połowa zmiany znika
  przy drobnym zaburzeniu obrazu;
- D_E (cel liczony na zaburzonych kopiach): odporny 0,19, czysty 0,01, zmiana 0,080 — niewielki zysk;
- D_ER (+ kara dyskryminatora): dyskryminator oszukany (wynik −1,13 → +3,43, szum ocenia wyżej niż obrazy), odporny
  0,12, zmiana 0,108, silne artefakty — **kara dyskryminatora odrzucona**;
- P_E (piksele, cel odporny): odporny 0,49 (wzdłuż 0,79, w bok 0,38), zmiana 0,028 — na obrazie widać drobne barwne
  ziarno na całej scenie, nie naturalną zmianę narzędzia.

**Wniosek kroku 3.** Żaden wariant sterowania FastGAN-em nie daje naturalnego obrazu, który realizuje krok Koopmana:
generator dochodzi do ~30 % drogi kosztem plam i drobin w tle, a odporna część zmiany to ~15–19 %. Zgodnie z [Rou26]
lepszym narzędziem jest priorytet dyfuzyjny sterowany celem (z dynamicznym celem i stabilizacją jak w [Vat25]),
co wymaga wyuczenia małego modelu dyfuzyjnego na wycinkach narzędzia (godziny GPU, osobna zgoda).
