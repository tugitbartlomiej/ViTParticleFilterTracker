# Dekoder latentu małego ViT: rysowanie obrazu wprost z latentu (propozycja, 2026-09-29)

**Status: propozycja. Nic nie uruchomiono.** Trening na Edenie ruszy dopiero po wyraźnym „tak” użytkownika.

## Prośba użytkownika

„Dla mnie najlepiej by było rysować prosto z latentu! Wtedy bym mógł rysować i widzieć różnice pomiędzy tym, co jest
aktualnie, a tym, co przewiduje Koopman i Langevin.” Wcześniej: „zrób analizę mojego pipelinu i powiedz, jak mogę ulepszać
generowanie obrazów u mnie, biorąc pod uwagę, że mogę wszystko trenować na Edenie”.

## Co dziś robi generowanie (pętla `maly_vit_petla_sila_tozsamosci`, wersja 3 i iteracja 2)

Kod: `../maly_vit_petla_sila_tozsamosci/kod/sila.py` (`_inwersja`, `_steruj`, `generuj`).

- Generator: FastGAN 256×256 RGBA (obraz + maska), uczony od zera na 1044 wycinkach narzędzia ze 116 operacji,
  50 000 kroków (`../fastgan_narzedzie`).
- Inwersja kotwicy: 4 starty × 160 kroków Adama po wejściu z, strata L1 obrazu + L1 maski.
- Sterowanie: 300 kroków Adama po z, aż siła tożsamości (bieżąca sieć, wektor z kroku 1000) = siła startu + prognoza.
  Kara za zmianę latentu poza kierunkiem tożsamości ma wagę 1/50 średniej po 768 liczbach, czyli jest słaba.

Mediany z 4200 raportów `RAPORT.json` z Edena (liczone 2026-09-29):

| | wersja 3 (420 kotwic × ramię) | iteracja 2 (1680) |
|---|---|---|
| błąd inwersji (L1 obrazu + L1 maski) | 0,104 | 0,105 |
| przewidziana zmiana siły, wartość bezwzględna (sd negatywów) | 0,55 | 0,66 |
| osiągnięta część zmiany (odporna, 4 zaburzone kopie) | 1,00 (0,93) | 1,00 (0,94) |
| zmiana obrazu „po prognozie” względem „teraz”, L1 RGB | 0,064 | 0,060 |
| zmiana latentu w bok od kierunku tożsamości (norma, sd negatywów) | 22,3 | 25,1 |
| siła tożsamości startu | 4,96 | 4,20 |

Wniosek: FastGAN trafia w przewidzianą siłę, ale zmiana latentu poza kierunkiem tożsamości jest około 40 razy
większa niż zmiana wzdłuż niego (norma 22–25 kontra 0,55–0,66). Obraz „po prognozie” różni się od obrazu „teraz”
głównie rzeczami, których prognoza nie dotyczy. To zgadza się z wynikiem kampanii: ramię `current` (obraz bez
prognozy) wypada prawie tak samo jak ramię `tozsamosc`. Druga słabość: inwersja odtwarza kotwicę z wyraźnym błędem,
bo generator widział tylko 1044 wycinki.

## Propozycja: dekoder „latent → obraz”

Warunkowy model dyfuzyjny uczony raz na Edenie. Wejście: przestrzenna mapa tokenów wycinka (siatka 11×11, 768 liczb
na token), nie uśredniony wektor narzędzia — FeatInv (Neukirch 2025) pokazuje, że warunkowanie mapą przestrzenną
trafia w cel wielokrotnie lepiej niż wektorem uśrednionym (cosinus 0,5–0,6 kontra 0,12–0,19). Wyjście: wycinek 256×256.

Rysowanie prognozy (w współrzędnych standaryzowanych jak w `sila.py`, x = (latent − mu) / sd):

    x_prognoza = x_teraz + Δs · w        (Δs z Koopmana albo z Langevina, osobno)

na tokenach narzędzia, reszta mapy bez zmian. Oba obrazy rysuje się z tym samym szumem startowym, więc różnica
pikseli pokazuje tylko przesunięcie przewidziane przez prognozę. Na każdą kotwicę i obrót: obraz teraz, obraz
Koopmana, obraz Langevina i mapy różnic. Zaleta względem FastGAN-a: na wejściu dekodera zmiana w bok wynosi zero
z definicji.

### Trudność: sieć zmienia się co obrót

Dekoder czyta latent jednej wersji sieci, a mały ViT uczy się dalej w każdym obrocie. Dwa warianty:

- **A. Most przez DINOv2 (propozycja na start).** Dekoder warunkowany tokenami zamrożonego DINOv2, uczony raz,
  wspólny dla wszystkich ziaren i obrotów. Na każdy obrót liniowy przelicznik T: tokeny małego ViT → tokeny DINOv2,
  dopasowany najmniejszymi kwadratami z grzbietem na wycinkach puli (sekundy na GPU). Prognoza:
  z_prognoza = T(x_teraz + Δs · w). Ryzyko: przelicznik liniowy może zgubić kierunek tożsamości; mierzymy to (etap 0).
- **B. Dekoder wprost na małym ViT ze stanu po rozgrzewce** (krok 1000, `wspolny/k02.pt`; z tej sieci pochodzi
  wektor tożsamości), osobno dla każdego ziarna albo tylko dla 2–3 ziaren do rysunków.

### Różnice względem metody użytkownika (nazwane wprost)

1. Generator: dyfuzja warunkowana latentem zamiast FastGAN-a z inwersją i sterowaniem gradientem.
2. Wariant A: obraz powstaje z latentu DINOv2 przeliczonego z małego ViT, a nie z gradientu małego ViT.
3. Wariant B: obraz odpowiada sieci z kroku 1000, a nie sieci bieżącej w danym obrocie.
4. Wariant A pozwala też wyszukiwać klatki wprost przesuniętymi cechami DINOv2, bez obrazu. To byłoby osobne ramię
   i wymaga osobnej zgody; w tym planie obraz dalej idzie do wyszukiwania tak jak dziś.

## Model i dane

- Model: MiniSD 256 z ControlNet (jak FeatInv) albo mała dyfuzja w pikselach 256 uczona od zera (jak RCDM i
  Frisch 2023 na CATARACTS). Wybór po etapie 0.
- Dane: wycinki 256×256 z klatek bazy (1500) i puli (9136), wiele położeń na klatkę; bez klatek walidacji,
  żeby generator nie widział walidacji.
- Koszt: FeatInv uczył ControlNet 45–60 h na dwóch L40 na ImageNet (1,28 mln obrazów, 3 epoki). Nasz zbiór jest
  kilkadziesiąt razy mniejszy; szacunek (niesprawdzony): kilka do kilkunastu godzin jednej karty `hopper`.
- Eden: diffusers i wagi w obrazie singularity (jeden plik = jeden i-węzeł, pamięć `eden-singularity-tylko-na-nodach`);
  wagi 2–5 GB; przed zgłoszeniem `lfs quota -p 1299 /mnt/evafs`; sprzątanie przez trap, nic na węzłach.

## Etapy

0. **Lokalnie, bez treningu (RTX 3070).** Dla zapisanych sieci (ziarna 47–56, obroty) dopasować przelicznik T na
   wycinkach puli i zmierzyć: R² tokenów DINOv2 na wycinkach wyłączonych, oraz ile siły tożsamości przenosi T
   (siła odtworzona z T⁻¹ kontra prawdziwa). Jeśli T gubi tożsamość, zostaje wariant B.
1. **Eden:** trening dekodera wybranego wariantu (osobna zgoda).
2. **Sprawdzenie w zamkniętej pętli** na 12 kotwicach: obraz „teraz” i „po prognozie” zakodować bieżącą siecią.
   Kryteria: osiągnięta część zmiany ≥ 0,8; zmiana w bok mniejsza niż u FastGAN-a (22–25); obraz „teraz” bliżej
   kotwicy niż inwersja FastGAN-a (L1 0,10).
3. **Rysunki:** teraz / Koopman / Langevin / mapy różnic dla każdej kotwicy i obrotu.
4. Dopiero potem ewentualna kampania pętli z ramieniem „dekoder” (osobny plan i zgoda).

## Czego to nie rozwiąże

- Koopman i Langevin przewidują jedną liczbę na sondę (siłę tożsamości), więc obraz pokaże tylko przesunięcie
  wzdłuż tożsamości. Mediana przewidzianej zmiany to około 0,6 sd przy sile narzędzia 4–5, więc różnica będzie subtelna.
- Prognoza siły jest tylko trochę lepsza od braku zmiany (błąd względem braku zmiany, mediana: Koopman 0,94,
  Langevin 0,89 dla starego wektora; `../maly_vit_tozsamosc_tla/PLAN.md`).

## Literatura

Kolejka pobierania: `F:\Studia\Articles\Zbiory\JEPA\Scripts\PdfDownloader\linki\stable-diffusion-latent-docelowy-sieci-2026-09-29.txt`
(40 linków). Undermind, workspace ViT, deep search „Stable Diffusion generating images whose representation in an
external frozen network matches a target latent” (133 prace). Najważniejsze: Neukirch 2025 FeatInv, Bordes 2021 RCDM,
Graikos 2024, Karthikeyan 2026, Shirahama 2025, Rousselot 2026, Frisch 2023.
