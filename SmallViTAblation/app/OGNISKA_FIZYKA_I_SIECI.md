# Ognisko wielu wiązek: co mierzy fizyka i co z tego jest nowe dla sieci neuronowych (2026-10-14)

Przegląd do okienka 11 aplikacji (ognisko jednej cząsteczki) i zasady Z25 (`app/ZASADY_DZIALANIA_UKLADU.md`).
Źródła: Undermind, workspace ViT, dwa głębokie przeszukania z 2026-10-14 — „Focal spot formed by many converging beams physics
measurable quantities shape efficiency and force” (213 prac) i „How individual MLP neuron contributions add up and cancel on a
single token in transformers and how this evolves in training” (97 prac) — oraz Scite (metadane i fragmenty; żadna z cytowanych
prac nie ma wycofania ani korekty). Wnioski oparte na streszczeniach i fragmentach, nie na pełnych tekstach.
Lista linków do pobrania: `literatura_szklo_tokeny_20261011/linki/ogniska-wiazek-wklady-neuronow-2026-10-14.txt`.

## 1. Czym jest ognisko w fizyce

- **Kształt ogniska wynika z widma kątowego wiązek**, czyli z tego, z jakich kierunków i z jaką mocą przychodzą (apertura,
  apodyzacja źrenicy). Klasyczna teoria: Richards i Wolf 1959. Miary jakości: objętość ogniska, szerokość i wydłużenie wzdłuż
  osi, energia w małym promieniu, współczynnik Strehla (Sheppard i Martínez-Corral 2008).
- **Siła na cząstkę to suma pędów pojedynczych promieni**, które w większości się znoszą (model promieni Ashkina 1992).
  Miarą jest sprawność pułapki Q (siła na jednostkę mocy, `F = Q n P / c`); zmierzone Q = −0,12 ± 0,014 dla kulki polistyrenowej
  10 µm (Sonek i Berns 1993). Q nie musi być ograniczonym ułamkiem pędu; to miara siły na moc.
- **Wiele pułapek naraz (pęsety holograficzne)** ocenia się po sprawności, jednorodności między pułapkami i mocy uciekającej do
  „pułapek-duchów” (Polin i in. 2005; teoria wektorowa: Sun, Roichman i Grier 2008).
- **Wiązki przeciwbieżne znoszą niezachowawczą (rozpraszającą) część siły** (Divitt i in. 2015).
- **Siłę dzieli się na część gradientową i rozpraszającą** (narzędzie GCforce 2019; przegląd: Pesce i in. 2020).

## 2. Co w sieciach już zmierzono (nie będzie nowe)

- Rzadkość aktywacji MLP na token i jej zmiany w uczeniu, obumieranie neuronów (Wild i Anderson 2024; Li i in. 2022 — także ViT;
  Voita i in. 2023; Luo i in. 2024).
- Cosinus wejście–wyjście neuronu (nasze κ): neurony wzmacniające i osłabiające (Gerstner i Schütze 2025, Gerstner i in. 2026).
- MLP jako pamięć klucz–wartość i promowanie pojęć (Geva i in. 2021, 2022); MLP jako pole sterujące kierunkiem tokenu, liczy się
  składowa styczna (Mudarisov i in. 2026).
- Znoszenie między komponentami: samonaprawa, wymazywanie, przeciwwagi (Hydra effect 2023; Rushing i Nanda 2024; Ahmad i in. 2026),
  sąsiednie warstwy korygują swoje wkłady (Patrawala i in., NeurIPS 2025 — modele językowe, między warstwami).
- Rdzeń kanałów krytycznych dla straty w FFN (Cherilyn i Safaai 2026).

**Luka (werdykt przeszukania Undermind):** nikt nie mierzy na JEDNYM tokenie, jak wektory wkładów neuronów jednej warstwy MLP
sumują się i znoszą (długość wypadkowej wobec sumy długości), ile neuronów naprawdę współpracuje (liczba ważona długością wkładu,
nie liczba aktywnych) ani jak to się zmienia w czasie uczenia — w ViT tym bardziej.

## 3. Pomiary, które byłyby nowe

| # | pomiar (na token, blok, checkpoint) | odpowiednik w fizyce | stan u nas |
|---|---|---|---|
| 1 | **sprawność ogniska** `η = ‖Σ_c w_c‖ / Σ_c ‖w_c‖`, `w_c = (z_jc − z̄_c) v_c` | sprawność pułapki Q | w 3D (okienko 11): blok 6 5 % na starcie, 11–23 % później; blok 4 68 %, blok 1 32 % (60. ep.); w 192D zasada Z18: 5–22 % |
| 2 | **efektywna liczba wiązek** `N = (Σ‖w_c‖)² / Σ‖w_c‖²` i **spójność** `C = η·√N` | wypełnienie apertury | nie liczone; C = 1 losowe kierunki, C > 1 współpraca, C < 1 aktywne znoszenie |
| 3 | **apertura w 192D**: liczba uczestnictwa wartości własnych tensora kierunków `T = Σ jas d dᵀ / Σ jas` | apertura numeryczna, objętość ogniska | w 3D: wydłużenie ×1,2 (blok 6), ×1,7 (blok 4) |
| 4 | **przeciwlasery a wir**: ile niezachowawczej części pola usuwają pary przeciwnych reflektorów | wiązki przeciwbieżne (Divitt 2015) | kierunek zgodny: przeciwlasery 0 → 53 %, wir 0,99 → 0,86; brak testu przyczynowego |
| 5 | **jednorodność ognisk między 600 tokenami**, „duchy”: czy narzędzie ma ostrzejsze ognisko niż tło | jednorodność i duchy w pęsetach holograficznych | nie liczone |
| 6 | **ognisko uwagi kontra ognisko MLP**: czy podkrok MLP znosi wkład uwagi tego samego bloku | korekta między warstwami (Patrawala 2025), tu w ViT i w podkrokach | dane podkroków gotowe w aplikacji |
| 7 | **wszystko powyżej w czasie uczenia** | przestawianie wiązek w czasie | 806 checkpointów dla tokenu 226 policzone |

Uwaga: procenty w okienku 11 są w rzucie 3D; publikowalna miara musi być liczona w pełnych 192 wymiarach.
Najsilniejsza propozycja: 1 + 2 + 7 (sprawność i spójność ogniska jako miary współpracy neuronów na tokenie, przez całe uczenie
ViT, z kontrolą „suma wkładów = ruch” ~10⁻⁶), uzupełnione o 4 (przewidywanie z fizyki) i 6 (podkroki).

## 4. Literatura (DOI)

Fizyka: Richards, Wolf 1959 — 10.1098/rspa.1959.0200; Ashkin 1992 (model promieni) — 10.1016/S0091-679X(08)60399-4;
Sonek, Berns 1993 — 10.1063/1.109937; Pesce i in. 2020 — 10.1140/epjp/s13360-020-00843-5; Callegari i in. (toolbox optyki
geometrycznej) — 10.1364/JOSAB.32.000B11; Sheppard, Martínez-Corral 2008 — 10.1364/OL.33.000476; objętość ogniska wielu wiązek
2009 — 10.1364/OE.17.002861; GCforce 2019 — 10.1016/J.CPC.2018.11.002; Divitt i in. 2015 — 10.1364/OL.40.001900; pułapka
wielowiązkowa 1997 — 10.1364/AO.36.006423; Polin i in. 2005 — 10.1364/OPEX.13.005831; Sun, Roichman, Grier 2008 —
10.1364/OE.16.015765; holograficzne pułapki 2005 — 10.1364/AO.45.000880; optymalne hologramy 2007 — 10.1364/OE.15.001913.
Sieci: Li i in. 2022 (lazy neuron) — arXiv 2210.06313; Wild, Anderson 2024 — 10.48550/arXiv.2407.07848; Voita i in. 2023 —
10.48550/arXiv.2309.04827; Luo i in. 2024 — 10.48550/arXiv.2411.02335; Geva i in. 2021 — 10.18653/v1/2021.emnlp-main.446;
Geva i in. 2022 — 10.48550/arXiv.2203.14680; Gerstner, Schütze 2025 — 10.48550/arXiv.2505.17936; Gerstner i in. 2026 —
arXiv 2609.18612; Mudarisov i in. 2026 — 10.48550/arxiv.2608.02071; Hydra effect 2023 — 10.48550/arXiv.2307.15771; Rushing,
Nanda 2024 — 10.48550/arXiv.2402.15390; Patrawala i in. 2025 — 10.52202/085713-4643; Ahmad i in. 2026 — arXiv 2610.02173;
Cherilyn, Safaai 2026 — 10.48550/arXiv.2604.23475.
