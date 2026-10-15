# Czy ktoś już zrobił „tokeny ViT jako cząstki w czasie uczenia”? Nowość i wartość publikacyjna

Deep research z 2026-10-12. Pytanie użytkownika: „czy ktoś robił już taki układ i czy to ma wartość publikacyjną”.

## 1. Werdykt

**Nikt nie zrobił tego połączenia.** Nie znalazłem pracy, która spełnia jednocześnie trzy warunki:
- czasem cząstki jest czas uczenia, a nie głębokość sieci;
- cząstkami są tokeny jednego zdjęcia albo pojedyncze przykłady;
- ruch opisują statystyki koloidów i szkieł (MSD, klatka, α₂, ISF, χ₄, Lindemann, fonony), skalibrowane kontrolą dodatnią (szkło BD) i ujemną (OU).

Każdy z tych elementów osobno ma jednak bogatą literaturę:
- szkło badano na **wagach**, a najnowsza praca z PRR też mówi „brak klatek”;
- tokeny jako cząstki opisywano w **głębokości**;
- reprezentacje przykładów przez checkpointy opisano macierzą odległości (Halperin 2026, największe ryzyko wyprzedzenia).

Nowością jest więc **poziom pomiaru i synteza**, a nie sama fizyka uczenia.

**Wartość publikacyjna jest realna, ale w obecnym stanie to poziom warsztatu albo czasopisma**: TMLR, Phys. Rev. E/Research, MLST. Na główną konferencję (ICML/NeurIPS 2027) brakuje trzech rzeczy:
1. standardowego ViT na publicznym zbiorze;
2. ilościowej teorii pola z jądra (eNTK), sprawdzonej na nowych danych;
3. praktycznego zysku poza zbiorem uczącym.

Filmy nadają się na materiał dodatkowy albo blog, ale nie unoszą artykułu same.

## 2. Jak szukano

| źródło | co zrobiono | wynik |
|---|---|---|
| Undermind (workspace ViT) | 3 nowe deep searche: [prior art](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Prior%20art%20for%20tracking%20token%20representations%20across%20training%20as%20particle%20trajectories%20with%20soft%20matter%20diagnostics) (83 prace), [atencja w czasie uczenia](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Attention%20maps%20during%20training%20and%20how%20much%20individual%20token%20representations%20change) (142), [wizualizacja dynamiki uczenia](https://app.undermind.ai/projects/40294fa6-c9a3-4469-9a61-d3eddeaee893?path=/Visualizing%20and%20animating%20the%20evolution%20of%20internal%20representations%20during%20neural%20network%20training) (139); 5 wcześniejszych (szkło, kryształ, pułapki, pęseta optyczna, przypisanie ruchu); pełne teksty 10 najbliższych prac (`read_pdfs`) | w deep searchu „prior art” tylko 5 prac przekracza próg trafności r = 1; żadna nie robi tego samego |
| Scite + Consensus | ok. 30 kwerend; grafy cytowań od Baity-Jesi (≈125 cytujących), Feng & Tu (86), Aitken (52), Chen (29), Kunin, Yoshino, Pashakhanloo, Kerr Winter | żaden cytujący nie stosuje MSD/α₂/ISF/χ₄/klatki/Lindemanna do reprezentacji przykładów ani tokenów; brak wycofań i korekt |
| alphaXiv + sieć | preprinty 2025–2026, blogi, repozytoria | najbliższe: Halperin 2026 (OMD), Kerr Winter & Janssen, Pashakhanloo & Koulakov; ryzyka wyprzedzenia w p. 6 |
| sieć (CFP) | terminy i zakresy czasopism i warsztatów, strony czytane 2026-10-12 | p. 8 |

Lista 62 linków do pobrania: `literatura_szklo_tokeny_20261011/linki/nowosc-tokeny-czastki-czas-uczenia-2026-10-12.txt`.

## 3. Najbliższe prace

Bliskość w skali 0–3, gdzie 3 oznacza, że praca robi to samo. Żadna nie dostała 3.

| bliskość | praca | co robi | czego brakuje względem naszego układu |
|---|---|---|---|
| **2** | Halperin 2026, *Learning as Observable Matrix Dynamics* ([arXiv 2606.29679](https://arxiv.org/abs/2606.29679)) | 1000 stałych przykładów nazywa „cząstkami”; śledzi macierz odległości kątowych przez checkpointy; równanie Itô ze składnikiem dyfuzji z Jakobianu; animacje MDS z Procrustesem; MLP, transformer jednowarstwowy, mały CNN | brak tokenów jednego zdjęcia i tożsamości patcha, brak MSD, klatek, χ₄ i Lindemanna, brak kontroli BD/OU, brak ViT; statystyki z macierzy losowych zamiast z miękkiej materii |
| **2** | Kerr Winter & Janssen 2025, PRR ([10.1103/physrevresearch.7.023010](https://doi.org/10.1103/physrevresearch.7.023010)) | pełna diagnostyka szkła: nakładanie, superpozycja czas–temperatura, heterogeniczność, starzenie; wynik: brak dywergencji czasu relaksacji i **brak klatek** | tylko **wagi** MLP; nic o reprezentacjach ani przestrzeni |
| **2** | Baity-Jesi i in. 2018/2019, JSTAT ([10.1088/1742-5468/ab3281](https://doi.org/10.1088/1742-5468/ab3281)) | MSD i starzenie wag; szkliste tylko przy niedoparametryzowaniu | tylko wagi |
| **2** | Pashakhanloo & Koulakov 2023 ([arXiv 2302.02563](https://arxiv.org/abs/2302.02563)) | dryf reprezentacji bodźca pod szumem SGD jako fluktuacja w poprzek rozmaitości plus dyfuzja rotacyjna wzdłuż niej (D ∝ η³) | dwuwarstwowa sieć **liniowa**; brak korelacji między bodźcami, sieci głębokich i statystyk cząstek. To pojęciowy odpowiednik naszego podziału na drgania w studni i dryf studni |
| 2 | Chen, Qu, Gong 2022, Neural Networks; Herrera Segura i in. 2022, MLST; Hertz & Tyrcha 2025, PRE; Kunin i in. 2023, Neural Comput. | dyfuzja anomalna, podstarzenie, starzenie | wszystko na **wagach** |
| 1,5 | Huang, …, Yoshino, Jin 2025 ([arXiv 2506.06789](https://arxiv.org/abs/2506.06789)); Yoshino 2023, PRR | korelacje starzenia reprezentacji, „warstwy ciekłe i stałe” | maszyna termiczna (dynamika molekularna), a nie backprop; MNIST 0/1 |
| 1,5 | Liu, Tegmark i in. 2022, *Towards Understanding Grokking* ([arXiv 2205.10343](https://arxiv.org/abs/2205.10343)) | osadzenia jako oddziałujące cząstki w teorii efektywnej | brak statystyk śledzenia cząstek |
| 1,5 | Wang, Baker, Gordon, Murfet 2025, *Embryology of a Language Model* ([arXiv 2508.00331](https://arxiv.org/abs/2508.00331)); Baker i in. 2025, podatności ([arXiv 2504.18274](https://arxiv.org/abs/2504.18274)) | podatność (odpowiedź liniowa) każdego tokenu w 4 checkpointach, UMAP, język biologii rozwoju | brak trajektorii, przesunięć i kontroli; obserwablą jest strata, nie reprezentacja |
| 1 | Fort i in. 2019, *Stiffness* ([arXiv 1901.09491](https://arxiv.org/abs/1901.09491)) | zgodność gradientów dwóch przykładów; **krytyczna długość ξ** maleje w czasie uczenia i przy większym lr | dotyczy przykładów i strat, nie przesunięć tokenów na siatce. To najbliższy odpowiednik naszej korelacji ruchu sąsiadów |
| 1 | He & Su 2019 (lokalna elastyczność); Dan i in. 2021; Zhang, Wang, Su 2021, NeurIPS (LE-SDE); Ren & Sutherland ICLR 2025 ([arXiv 2407.10490](https://arxiv.org/abs/2407.10490)) | zmiana przykładu = jądro (eNTK) × reszta, sprzężenie między przykładami, „analiza sił” | wyjścia albo logity, a nie tokeny; brak przestrzeni patchy i testu losowych kierunków |
| 1 | Hazelden i in. 2026, *Global Empirical NTK* ([arXiv 2605.08746](https://arxiv.org/abs/2605.08746)) | NTK na stanach wewnętrznych, także w transformerze | brak tokenów jednego zdjęcia i korelacji ruchu; Shampoo zamiast AdamW |
| 1 | Sharon & Dar 2024 ([arXiv 2405.17377](https://arxiv.org/abs/2405.17377)) | CKA między epokami; warstwy ViT-B/16 zmieniają się synchronicznie; fazy uczenia | poziom całych warstw, bez śledzenia tokenów |
| 1 | Mannes 2026 ([arXiv 2606.00124](https://arxiv.org/abs/2606.00124)) | kodowanie pozycji kotwiczy strukturę przestrzenną tokenów ViT (SSDC, permutacja patchy) | tylko po uczeniu, wzdłuż głębokości; bez checkpointów i fononów |
| 1 | Turan, Abel, Ovsjanikov 2026, *Pattern Formation in Transformers* ([arXiv 2609.37921](https://arxiv.org/abs/2609.37921)) | relacja dyspersji tokenów z kodowania pozycji, mody Fouriera, fale; relacja dyspersji liczona też z checkpointów ConViT | czasem jest głębokość przy inicjalizacji; nie ma trajektorii tokenów w czasie uczenia |
| 1 | Järve i in. 2026 (ewolucja cech w uczeniu ViT, [10.1145/3799682.3841045](https://doi.org/10.1145/3799682.3841045)); Kapil & Upla 2026, TGO-IV ([arXiv 2608.09997](https://arxiv.org/abs/2608.09997)) | ViT przez epoki: SAE na tokenie CLS; topologia chmury 197 tokenów jednego zdjęcia | bez tożsamości tokenu i fizyki cząstek; Järve wprost zostawia patche na przyszłość |
| 0,5 | Geshkovski i in. 2023; Lu i in. 2019; Rigollet 2025; Isobe 2026; Herty & Liu 2026 | tokeny jako cząstki, gdzie czasem jest **głębokość**; Herty & Liu sprzęgają głębokość z czasem uczenia teoretycznie | przeciwna oś czasu, brak pomiarów |
| 0,5 | M-PHATE (Gigante 2019), DVI/TimeVis, SentryCam (2024, także ViT), NeuroMapper | animacje reprezentacji przez epoki | przykłady albo jednostki z przedostatniej warstwy; bez patchy, rozkładu ruchu i fizyki |

## 4. Co jest nowe, a co znane

**Nowe (nie znalezione nigdzie):**
1. Tokeny patchy **jednego zdjęcia** śledzone z tożsamością przez 806 checkpointów i co krok, jako 600 cząstek.
2. Pełny zestaw diagnostyki koloidów i szkieł (klatka, α₂, ISF, χ₄, trwałość rankingu ruchliwości, rozbieganie par, Lindemann Zahna–Mareta, widmo fononów na siatce 20×30) na **reprezentacjach**.
3. Kalibracja tych diagnostyk symulowanym szkłem koloidalnym (dodatnia) i niezależnymi cząstkami OU w stygnących dołkach (ujemna).
4. Mechanizm z kontrolą: pole wspólne δz = J δθ odtwarza korelacje. **Losowe kierunki w przestrzeni wag dają tę samą strukturę korelacji**, czyli geometrię ustala architektura, a dane tylko amplitudę.
5. Siatka z kodowania pozycji, czyli „kryształ napędzany z nieporządkiem”. Ablacja bez kodowania pozycji usuwa adres i fonony.
6. Rozkład ruchu tokenu na składową globalną, dryf centrum studni i drgania, zestawiony z atencją bloków, IoU i pewnością, z renderem na GPU.

**Znane albo częściowo znane (cytować i odróżnić):**
- „uczenie przypomina szkło, ale nim nie jest” na wagach: Baity-Jesi; Kerr Winter & Janssen („brak klatek”); Verpoort 2020, PNAS (brak wysokich barier);
- sprzężenie przykładów przez jądro: NTK, lokalna elastyczność, sztywność (długość ξ), Ren & Sutherland, OMD Halperina (dyfuzja z Jakobianu);
- dryf reprezentacji jako fluktuacja plus dyfuzja: Pashakhanloo & Koulakov; Aitken 2022; Qin 2023;
- „temperatura” SGD jako η/B: Mignacco & Urbani, Feng & Tu, Sclocchi 2024. U nas klatka nie zależy od paczki, ale **obcięcie gradientu było zawsze aktywne (norma 1,0)**, więc recenzent uzna to za artefakt obcinania. Potrzebny bieg bez obcinania;
- neurony MLP jako pamięć klucz–wartość: Geva 2021. „Lasery” to ta sama matematyka w innej szacie; dodatek z 2026-10-12 sam pokazuje, że to dobry model odczytu, a zły model sił.

## 5. Wynik „atencja a przesunięcie centrów” w świetle literatury

Deep search o atencji (142 prace) **nie znalazł pracy, która mierzy, czy patche dostające więcej atencji bardziej zmieniają reprezentację w czasie uczenia**. To luka.

Nasz słaby wynik jest zgodny z teorią. Waga atencji opisuje trasowanie, a nie wielkość zmiany cech:
- rezydua i normalizacja tłumią wpływ atencji na reprezentację (Kobayashi 2021);
- rollout przybliża przepływ informacji, a nie wielkość zmiany (Abnar & Zuidema 2020);
- ogólna krytyka: Jain & Wallace 2019.

Precedensy czasowe dotyczą samej atencji: zapadanie entropii (Zhai 2023), ujścia atencji (Gu 2024), dryf wyjaśnień w ViT (Rahman & Ottoborgo 2025).

Wartość: to dobry akapit lub wynik poboczny. Wymaga jednak setek zdjęć (Eden), bo teraz opiera się na jednym.

## 6. Ryzyko wyprzedzenia

1. **Halperin, seria OMD (arXiv 2606.29679, 29.06.2026)** — największe ryzyko. Ma „chmurę cząstek” stałych przykładów w czasie uczenia, SDE z Jakobianem, Procrustesa i animacje w repozytorium. Seria szybko przyrasta, więc krok w stronę tokenów i statystyk cząstek jest prawdopodobny.
2. **Järve i in. 2026**: patche ViT wprost jako „przyszła praca”. **TGO-IV**: zapowiada TGO-V o „trajektoriach tokenów i dryfie”, ale w głębokości.
3. **Turan i in. 2026**: wzory tokenów z kodowania pozycji. Rozszerzenie na czas uczenia dotknęłoby tezy o siatce i fononach.
4. **Han Ma (arXiv 2610.08503, 6.10.2026)**: język szkła dla uczenia transformera, ale na wyjściach.

Wniosek: warto szybko zaznaczyć pierwszeństwo, np. preprintem na arXiv lub zgłoszeniem na warsztat.

## 7. Wartość publikacyjna: mocne i słabe strony

**Mocne:**
- kontrole dodatnia i ujemna oraz losowe kierunki wag (standard pracy Antogniniego i Sohl-Dicksteina z NeurIPS 2018);
- czysty mechanizm zamiast metafory;
- wiele pokręteł już policzonych: lr, gęstość, 3 ziarna w 128D, SGD, paczka, CNN, warstwy 0–5, Imagenette, f32 i f16, ablacja kodowania pozycji, kontrola rzutu.

**Słabe:**
- jeden mały model w jednej domenie (chirurgia zaćmy);
- δz ≈ J δθ to tożsamość pierwszego rzędu, więc recenzent nazwie ją tautologią. Wartość leży w tym, **dlaczego** J daje gładkie, niskowymiarowe pole na siatce patchy (kodowanie pozycji i nakładanie pól łodygi), oraz w ilościowym przewidzeniu funkcji korelacji;
- zysk praktyczny na walidacji jest słaby (Spearman ruchliwość–błąd −0,08);
- „to nie szkło” samo nie jest nowe, bo na wagach to już wiadomo. Nowe jest: **dlaczego diagnostyka szkła myli się na reprezentacjach** (pole jądra i zamrożony nieporządek z kodowania pozycji).

**Ramy tez, które przejdą:**
- nie „uczenie to szkło”;
- tak: „diagnostyka szkła zastosowana do tokenów ViT daje fałszywe sygnały szkła; ruch jest polem jądra wspólnych wag na siatce z kodowania pozycji, a heterogeniczność jest zamrożonym nieporządkiem”. Precedensy takiego wyniku negatywnego to Kerr Winter & Janssen (PRR 2025) i Zhang i in. (NeurIPS 2025, falsyfikacja hipotezy pomiarem relaksacji).

## 8. Gdzie publikować (strony sprawdzone 2026-10-12)

| miejsce | dopasowanie | termin |
|---|---|---|
| **TMLR** (kryterium: poprawność i zainteresowanie odbiorców, nowość nie jest konieczna) | wysokie | rolling |
| **Phys. Rev. E** (dział: Computational Physics, ML and AI) / **Phys. Rev. Research** (tam Kerr Winter & Janssen) | wysokie | rolling |
| **Machine Learning: Science and Technology** (IOP) | wysokie | rolling, do 8500 słów |
| **ICLR 2027, warsztaty** (np. w stylu SciForDL: „falsyfikacja hipotez, kontrolowane eksperymenty, nowe zjawiska”) | wysokie, jeśli taki warsztat ruszy | lista warsztatów 29.11.2026, prace zwykle około 1.02.2027 |
| ICLR 2027, ścieżka blogów (dobra na filmy) | średnie | zaproszenie jeszcze nieopublikowane; w 2026 termin był 7.12.2025 |
| ICML 2027 | średnio-niskie w obecnym stanie | oficjalna strona jeszcze bez terminu (agregatory: koniec stycznia 2027, niepotwierdzone) |
| ICLR 2027, ścieżka główna | — | **minęło** (25.09.2026) |
| NeurIPS 2026 ML4PS / SciForDL / HiLD | — | **ML4PS nie ma na liście 102 warsztatów NeurIPS 2026**; terminy pokrewnych warsztatów (DynaFront, Artifacts-as-data) minęły we wrześniu |

## 9. Co już jest, a czego brakuje

| element | stan |
|---|---|
| kontrole BD i OU, losowe kierunki wag, kontrola rzutu (PCA32 vs rp128) | **jest** |
| 3 ziarna w 128D, SGD, paczka, CNN, warstwy 0–5, Imagenette, f32 | **jest** (przedziały ufności dopisać do każdej liczby z tez) |
| ablacja kodowania pozycji (`bezpoz_lr_5e-4_rp`) | **jest** (dodać PE uczone i przetasowane) |
| testy przyczynowe (pchnięcie tokenu i pikseli) | **jest** |
| standardowy ViT (DeiT-Ti/S) na publicznym zbiorze (CIFAR-10 albo ImageNet-100) | **brak**, a recenzenci zażądają tego najpierw |
| ilościowa teoria: funkcja korelacji ruchu na siatce przewidziana z widma eNTK i częstości kodowania pozycji, sprawdzona na nowych lr i nowych PE | **brak** |
| porównanie z diagnostyką OMD Halperina i z długością ξ Forta | **brak** |
| bieg bez obcinania gradientu (czy „temperatura = długość kroku” przetrwa) | **brak** |
| zysk: czy ruchliwość z wczesnego uczenia przewiduje końcowy błąd przykładu lepiej niż EL2N lub zapominanie | **brak** (łączy się z selekcją danych) |
| atencja a dryf na setkach zdjęć | **brak** (Eden) |
| publikacja kodu i trajektorii | do zrobienia |

**Pakiet minimalny** (warsztat, TMLR, PRE, MLST): to, co jest, oraz DeiT na CIFAR-10/ImageNet-100, przedziały ufności, bieg bez obcinania, porównanie z OMD i ξ, rama „wynik negatywny z mechanizmem”.

**Pakiet mocny** (ICML/NeurIPS 2027): dodatkowo 3 lub więcej architektur (ViT, DeiT/Swin, ConvNeXt, CNN), teoria eNTK z przewidywaniem na nowych danych, przemiatanie lazy/rich (μP) i zysk praktyczny lepszy od EL2N.

## 10. Roboczy tytuł i tezy

*Patch tokens as particles: soft-matter diagnostics of vision-transformer training reveal a kernel-driven, position-pinned field, not a glass.*

1. Metoda: śledzenie tokenów jednego zdjęcia w czasie uczenia, z diagnostyką koloidów kalibrowaną szkłem BD i cząstkami OU.
2. Wynik negatywny: znamiona szkła (plateau, rozciągnięta relaksacja, starzenie) odtwarza sama relaksacja w stygnącym dołku; brak cech szkła wielu ciał.
3. Mechanizm: 85–90 % energii kroku to pole wspólnych wag. Jego geometrię ustala architektura (losowe kierunki), a amplitudę gradient.
4. Porządek: kodowanie pozycji przypina tokeny do siatki (fonony tylko z PE), a heterogeniczność ruchliwości jest zamrożona.
5. (Poboczne) ruchliwość tokenu a błąd; atencja nie steruje dryfem centrów.

## 11. Najważniejsze źródła (DOI i arXiv sprawdzone narzędziami)

- Halperin 2026, *Learning as Observable Matrix Dynamics*, arXiv 2606.29679
- Kerr Winter & Janssen 2025, PRR 7, 023010, doi:10.1103/physrevresearch.7.023010
- Baity-Jesi i in. 2019, JSTAT, doi:10.1088/1742-5468/ab3281
- Pashakhanloo & Koulakov 2023, arXiv 2302.02563
- Chen, Qu, Gong 2022, doi:10.1016/j.neunet.2022.01.019
- Herrera Segura i in. 2022, doi:10.1088/2632-2153/ac8f1b
- Hertz & Tyrcha 2025, doi:10.1103/physreve.111.055307
- Fort i in. 2019, *Stiffness*, arXiv 1901.09491
- He & Su 2019, arXiv 1910.06943
- Zhang, Wang, Su 2021, arXiv 2110.05960
- Ren & Sutherland 2025, arXiv 2407.10490
- Hazelden i in. 2026, arXiv 2605.08746
- Sharon & Dar 2024, arXiv 2405.17377
- Mannes 2026, arXiv 2606.00124
- Turan, Abel, Ovsjanikov 2026, arXiv 2609.37921
- Järve i in. 2026, doi:10.1145/3799682.3841045
- Kapil & Upla 2026, arXiv 2608.09997
- Wang i in. 2025, *Embryology*, arXiv 2508.00331
- Baker i in. 2025, arXiv 2504.18274
- Liu i in. 2022, arXiv 2205.10343
- Huang i in. 2025, arXiv 2506.06789
- Geshkovski i in. 2023, doi:10.52202/075280-2493
- Gigante i in. 2019, arXiv 1908.02831
- Antognini & Sohl-Dickstein 2018, arXiv 1806.08805
- Yang i in. 2024 (SentryCam), arXiv 2405.15135
- Kobayashi i in. 2021, doi:10.18653/v1/2021.emnlp-main.373
- Abnar & Zuidema 2020, doi:10.18653/v1/2020.acl-main.385
- Geva i in. 2021, doi:10.18653/v1/2021.emnlp-main.446
- Verpoort i in. 2020, PNAS, doi:10.1073/pnas.1919995117
- Zhang i in. 2025 (grokking a relaksacja szkła), arXiv 2505.11411
