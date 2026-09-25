# Nowość i publikowalność: dobór klatek przez przewidziany latent (Koopman → FastGAN → DINOv2)

Data: 2026-09-25, noc. Źródła: Undermind, workspace „ViT: detekcja końcówki narzędzia w chirurgii zaćmy”
(`40294fa6-c9a3-4469-9a61-d3eddeaee893`): nowe głębokie przeszukanie „generative model synthesizes target image then
retrieves real unlabeled samples for training” (103 prace) oraz trzy wcześniejsze (Koopman na punktach kontrolnych
detektora; operator Koopmana jako kryterium selekcji klatek; diagnostyka niedomknięcia a active learning). Pełne
teksty czytane dla ASAL, DCD, LTG, CLIP-aided diffusion, ATTIC, TRACE i Swift Sampling. Scite: brak notek o wycofaniu.

## Werdykt

Sam mechanizm „wygeneruj obraz-zapytanie, potem znajdź najbliższe prawdziwe klatki w puli” nie jest nowy: istnieje od
2012–2015 i ma dopracowaną wersję z GAN-em z 2018 roku (ASAL). Nowy jest tylko cel generacji: przyszły stan
reprezentacji detektora, przewidziany operatorem Koopmana z punktów kontrolnych uczenia. Tego połączenia nie znalazło
żadne z czterech przeszukań. Nowość kombinacji nie wystarcza jednak do publikacji metody, bo w naszych danych
(sekcje 6–7e) przewidziany latent daje obraz nieodróżnialny od obecnego, więc wyszukiwanie sprowadza się do sąsiadów
kotwicy, a dobrane klatki nie wygrywają z losowymi. Dziś to materiał na wynik negatywny z diagnozą albo na osobną pracę
o przewidywaniu dynamiki reprezentacji detektora, nie na pracę metodyczną na główną konferencję.

## Najbliższe prace

| Praca | Co robi | Związek z nami |
|---|---|---|
| Wang i in. 2015, Neurocomputing, doi:10.1016/j.neucom.2014.06.042 | syntetyzuje zapytanie przy granicy decyzji, bierze najbliższy prawdziwy przykład | ten sam szkielet „synteza → najbliższy sąsiad” |
| ASAL, Mayer i Timofte, WACV 2020, doi:10.1109/WACV45572.2020.9093556 | GAN, optymalizacja z pod kątem maksymalnej entropii klasyfikatora, dopasowanie do puli (PCA 50, k-d drzewo) | najbliższy precedens; wygrywa z losowaniem w 8 z 10 zadań, przegrywa na CIFAR-10 (10 klas) i CelebA-Eyeglasses |
| Data-Centric Debugging, Singla i in., WACV 2024, doi:10.1109/WACV57701.2024.00014 | obrazy błędów jako zapytania, wyszukiwanie w puli cechami DINO ViT-S/8, dodanie do treningu | to samo wyszukiwanie cechami DINO, ale bez generatora; +9,5 pkt na trudnym podzbiorze względem losowych przy równym budżecie (klasyfikacja) |
| Distilling Model Failures as Directions, Jain i in. 2022, doi:10.48550/arXiv.2206.14754 | kierunek błędów w przestrzeni CLIP, wyszukiwanie i generacja wzdłuż niego | cel „kierunek błędów” zamiast „przewidziany dryf” |
| Longtail-Guided Diffusion, Hayden i in., ICML 2025, doi:10.48550/arXiv.2502.01980 | dyfuzja sterowana niepewnością modelu, trening na obrazach syntetycznych | obecny stan techniki generacji sterowanej modelem; tylko klasyfikacja |
| CLIP-aided diffusion, Chegini i Feizi 2023, doi:10.48550/arXiv.2312.05464 | tekstowy opis błędów, dyfuzja, trening głowy na syntetykach | generacja bez wyszukiwania prawdziwych danych |
| The Unmet Promise of Synthetic Training Images, Geng i in. 2024, doi:10.48550/arXiv.2406.05184; Burg i in. 2023, arXiv:2304.10253 | prawdziwe obrazy wyszukane w puli biją obrazy syntetyczne | wspiera nasz wybór: trenujemy na prawdziwych klatkach, nie na FastGAN |
| TiDAL, Kye i in., ICCV 2023, doi:10.1109/ICCV51070.2023.02041; TRACE, He i in., AAAI 2026, doi:10.1609/aaai.v40i37.40357 | dobór danych z dynamiki uczenia (wyjścia, zmiany aktywacji) | mierzą zaobserwowane zmiany, nie przewidują przyszłych; TRACE: agregacja wielu punktów kontrolnych nic nie daje |
| Dogra i Redman 2020, arXiv:2006.02361; Sikorski i in. 2025, doi:10.1016/j.engappai.2025.110146 | Koopman i DMD na trajektorii wag lub uczenia sieci | najbliższe dla części 1–5, ale wagi, nie aktywacje sond |
| Peng i in., MedIA 2024, doi:10.1016/j.media.2024.103246; Yang i in. 2024, doi:10.1109/CCECE59415.2024.10667104; StepAL 2025, doi:10.48550/arXiv.2507.22059 | active learning i dobór klatek w wideo chirurgicznym (w tym zaćma) | zastosowanie; żadna nie używa generatora jako zapytania |
| Mittal i in. 2019, arXiv:1912.05361; Gashi i in. 2024, doi:10.48550/arXiv.2403.14800 | rzetelna ocena active learning: przewaga nad losowaniem często znika | recenzent zapyta o te kontrole |

## Co jest nowe, a co nie

Nowe: prognoza reprezentacji detektora między punktami kontrolnymi uczenia (sondy stałych obrazów, operator Koopmana
z opóźnieniami) jako cel dla generatora i wyszukiwania. TRACE i TiDAL używają dynamiki uczenia, ale tylko zmierzonej;
ASAL, DCD i LTG celują w niepewność albo błędy obecnego modelu.

Nienowe: generacja zapytania i dopasowanie do prawdziwej puli (2012, 2015, 2018), wyszukiwanie cechami DINO (DCD
2022/2024), trening na wyszukanych prawdziwych obrazach zamiast syntetycznych (2023–2024).

Nowoczesność składników: RF-DETR (2025) i DINOv2 (2023) są aktualne. FastGAN (2021) jest starszy niż sterowana
dyfuzja, która dziś dominuje w generacji sterowanej modelem (LTG 2025). Koopman na dynamice uczenia to wąska, ale żywa
nisza (prace z 2025–2026).

## Dlaczego obecne wyniki nie niosą pracy metodycznej

1. Sekcja 6: obraz przewidzianego latentu różni się od obecnego o 0,1–0,9 procent jasności, bo ogólny dryf tożsamości
   na 100 aktualizacji to 6 procent rozrzutu między obrazami. Wyszukiwanie po takim obrazie to w praktyce wyszukiwanie
   sąsiadów kotwicy. Recenzent zażąda ablacji „wyszukaj po samej klatce-kotwicy, bez Koopmana i FastGAN” i
   najpewniej dostanie ten sam zbiór klatek.
2. Sekcja 7c: po jednej epoce dobrane +0,0007, losowe +0,0009 [mAP@0,5] względem braku dodania, w szumie ziaren.
3. Sekcja 7d: w okolicy kotwic dobrane −0,003, losowe +0,005 AP@0,5.
4. Sekcja 7e: dobrane klatki nie były trudne dla detektora (AP 0,857 na starcie, około 0,90 po epoce bez ich
   oglądania). To dokładnie mechanizm porażki ASAL na CelebA-Eyeglasses opisany przez autorów: stała przestrzeń cech
   patrzy na cały obraz, a o trudności decyduje mały szczegół (tam okulary, u nas końcówka narzędzia), więc dopasowanie
   zwraca podobne sceny bez tej cechy. ASAL naprawił to dopasowaniem w cechach samego klasyfikatora.
5. Kampania 5 epok (7f): dobrane −0,0005 mAP@0,5 względem braku dodania, losowe +0,001, trudne +0,0016, wszystko
   w szumie; w okolicy kotwic dobrane wobec losowych +0,006, −0,005, −0,013, więc kryterium niespełnione.

## Ścieżki publikacji

A. Wynik negatywny z diagnozą (warsztat MICCAI lub „I Can't Believe It's Not Better” przy NeurIPS): prognoza Koopmana
jako cel generate-then-retrieve nie bije losowania w detekcji końcówki narzędzia; pokazujemy dlaczego (mały dryf,
dopasowanie całej sceny) i mamy rzetelny protokół (pary ziaren, efekt lokalny, kontrola trudnymi klatkami).

B. Osobna praca o dynamice reprezentacji detektora (sekcje 1–5): dryf tożsamości jest wczesny i wykładniczy, a operator
z opóźnieniami przewiduje go lepiej niż brak zmian. Bezpośredniego precedensu dla aktywacji sond nie ma.

C. Metoda na główną konferencję (MICCAI, IPCAI, WACV) wymaga zmian, zanim ruszą nowe treningi:
cel o dużej amplitudzie (kierunek ku błędom detektora albo różnice między obrazami, nie przewidziany dryf);
dopasowanie w cechach detektora wokół narzędzia zamiast DINOv2 z całej klatki (poprawka ASAL);
ablacje przy równym budżecie: sąsiedzi kotwicy bez Koopmana i generatora, wyszukiwanie od klatek z błędami jak w DCD,
niepewność, trudne klatki, losowe; kilka budżetów, co najmniej 3 ziarna, podział na poziomie filmów;
efekt, który 3 ziarna odróżnią od szumu: około 0,015 AP@0,5 na top-134 val.
