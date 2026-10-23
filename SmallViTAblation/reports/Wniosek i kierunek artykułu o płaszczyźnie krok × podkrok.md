# Wniosek, teza i kierunek: co wynika z pomiarów płaszczyzny krok × podkrok (2026-10-23)

Źródła: `experiments/kroki_podkroki_afinicznosc_uwagi_20261023__00-41-29` (s47, 806 checkpointów, 1 klatka TRAIN),
`experiments/plaszczyzna_krok_podkrok_replikacja_3047_20261023__02-16-41` (3047, 46 checkpointów 0–400 ep, 10 klatek TEST, predykcje
zapisane przed pomiarem), `reports/Tezy artykułu o płaszczyźnie krok × podkrok.md`, audyt nowości Undermind z 2026-10-23
(`/Novelty audit of sublayer resolved token dynamics…`, 135 prac) i wyszukiwania w sieci.

## 1. Co to znaczy, po ludzku

Sieć nie „trzyma” tokenów w żadnych pułapkach. Jedno przejście przez 6 bloków to taśma produkcyjna z trzema stanowiskami:

1. **Uwaga = rozdawanie wspólnego kontekstu.** Na starcie każda cząsteczka dostaje ten sam wektor (chmura przesuwa się jak bryła).
   Po uczeniu ten wektor pochodzi niemal wyłącznie od tokenów narzędzia (gwiazda: w pełnym jakobianie uwagi tokeny narzędzia niosą 56–77 %
   wpływu w blokach 3–6; sąsiedzi na siatce 0 %). Uwaga mówi każdemu tokenowi „gdzie i jakie jest narzędzie”, nie „co masz obok siebie”.
2. **MLP 1 = rozwinięcie, MLP 4 = decyzja, MLP 5–6 = kompresja i korekta.** MLP 1 podwaja rozrzut chmury i rozdyma objętość (log|det| do +8…+10);
   w bloku 4 następuje skok rozdziału narzędzie/tło (90 % par na 3047), a blok 4 jako jedyny rośnie do końca 400 epok (×1,03 → ×1,57); bloki 5–6
   ściskają objętość, blok 6 lekko cofa rozdział i pcha do środka. Łącznie przejście staje się mapą ściskającą: objętość ×e⁻²⁷ (narzędzie,
   400 ep), przy rozciąganiu w kilku kierunkach ×17.
3. **Uczenie najpierw „ustawia kierunek”, potem „różnicuje stanowiska”.** W 1. epoce wszystkie MLP pchają w tę samą stronę (ruch prosty,
   wskaźnik 2,8), potem każdy blok robi co innego (wskaźnik 0,9) i sąsiednie podkroki zaczynają się nawzajem poprawiać.

Dla aplikacji „lasery i koloidy”: obraz pułapki optycznej jest poprawny tylko jako kadr wokół jednej cząsteczki; obrazem całego przejścia
jest **linia optyczna** (sekwencja soczewek i kwadrupoli o losowych orientacjach: rozbieżna w części kierunków, ściskająca w innych),
a uwaga to **wiązka odniesienia** wspólna dla wszystkich cząsteczek, kierowana z narzędzia.

## 2. Teza artykułu (jedno zdanie)

**W transformerze wizyjnym uczonym do detekcji jedno przejście tokenów jest ściskającym transportem, w którym uwaga rozdaje kontekst
obiektu jako wspólny wektor, a MLP kolejno rozwija, rozdziela i kompresuje; uczenie buduje ten transport w ustalonej kolejności
(kolaps uwagi → spójna inflacja → specjalizacja bloków → korekta między podkrokami), a jego profil po głębokości zmienia się długo po
nasyceniu dokładności — co widać tylko przy jednoczesnym śledzeniu głębokości i czasu uczenia na poziomie podwarstw.**

## 3. Wniosek z pomiarów (co jest ustalone, co nie)

Ustalone (dwa seedy, klatka TRAIN + 10 klatek TEST, predykcje zapisane przed pomiarem):
- uwaga: translacja na starcie (udział 0,999–1,000), gwiazda na obiekcie po uczeniu, brak uwagi na sąsiadach (≤ 0,018 wobec 0,013 jednostajnej);
- MLP 1 rozpycha (×2), MLP 4 rośnie do końca, MLP 6 kompresuje/cofa; przejście ściska coraz mocniej (log σ_min −1 → −6,5);
- front rozdziału w bloku 4 (P5 PASS), blok 6 go nie podnosi;
- wygładzanie po siatce robi MLP, nie uwaga (P6 PASS);
- każdy podkrok to siodło, złożenie też siodło; nakładanie podprzestrzeni kurczących bliskie losowego (+0,01…+0,03).

Nieustalone / zależne od seedu: kolejność, w której bloki kolapsują (s47: 5 i 4 pierwsze; 3047: 3, 1, 4 pierwsze, blok 5 po 160 ep);
moment, gdy objętość zaczyna maleć (s47: 5–10 ep; 3047: 9–22 ep); dokładne progi (P1–P4, P7 FAIL o włos lub w czasie).
Nie wolno twierdzić: przyczynowości (gwiazda → rozdział) bez interwencji; ogólności poza tym detektorem i architekturą.

## 4. Czy ktoś to już zrobił (audyt)

Składniki są znane, złożenie nie:
- koncentracja uwagi: attention sinks / tokeny-rejestry o wysokiej normie [Dar23, Gu24b, Rus25b], kolaps entropii uwagi [Zha23au],
  uwaga globalna w ViT samonadzorowanych [Par23i]. Różnica: tu koncentracja jest **na obiekcie** (semantyczna, w 3–6 blokach), a nie na
  tokenach tła o wysokiej normie; i jest śledzona od kroku 0;
- profil po głębokości: efekt tunelu (wczesne warstwy budują, późne ściskają) [Mas23b], garb wymiaru wewnętrznego [Val23b], separacja klas
  po warstwach i epokach w ViT [Var25], migracja cech w uczeniu ViT [Jar26]. Różnica: rozdzielczość podwarstw (uwaga vs MLP), dokładny
  rozkład sił, jakobiany i 806 checkpointów;
- korekta między warstwami [Pat25d, Kob23], sprzężenie bloków i jakobiany po głębokości [Aub24, Fer26d, Li24bk, Jac25], propagacja
  sygnału [Cow24]. Różnica: tu na poziomie podkroków i w funkcji czasu uczenia;
- teoria: JoMA (uwaga najpierw rzadka, potem gęsta) [Tia23] — zgodne z blokami 1 i 6.

Podsumowanie audytu Undermind: „najmocniejsza nowość to zintegrowany protokół i jego wyniki dla detektora, nie samo istnienie ewolucji
geometrii, koncentracji uwagi czy kasowania między warstwami”. Trzeba ostrożnie sprawdzić twierdzenie „zero uwagi na sąsiadach”
względem kodowania pozycji i definicji miary (Undermind to podkreśla) — u nas sincos 2D, miara = masa A na 8 sąsiadach.

## 5. W którą stronę iść (trzy kroki, w tej kolejności)

1. **Replikacja szeroka, bez nowych pomysłów** (2–3 tygodnie): 5 seedów z 24 istniejących historii kampanii (12 checkpointów każdy) ×
   25 klatek TEST; te same skrypty (`pomiar_3047.py`, `jakobiany_3047.py`), progi przeskalowane (wzbogacenie względem maksimum 1/udział
   narzędzia; P1/P3 na siatce checkpointów). Cel: zamienić „2 seedy” na „wszystkie seedy”, z przedziałami ufności po filmach.
2. **Jedna interwencja przyczynowa** (1 tydzień): w blokach 3–5 zastąpić uwagę jej średnią po tokenach (sama translacja) albo usunąć
   wpływ tokenów narzędzia (gwiazda → jednostajna) i zmierzyć IoU i Fisher po MLP 4. Na seedzie 3047 już jest sygnał (Amean −0,009 vs
   Acentered −0,176 IoU, epoka 100); potrzebny ten sam test na 5 seedach. To zamienia korelację w mechanizm.
3. **Jedna kontrola architektoniczna** (opcjonalnie): ten sam pomiar na ViT klasyfikacyjnym (np. mały DeiT na CIFAR lub ImageNette) —
   czy gwiazda na obiekcie i front w bloku 4 to własność detekcji, czy transformerów w ogóle. Bez tego artykuł mówi tylko o detektorze.

Czego **nie** robić przed artykułem: dalszych analogii fizycznych (pułapka, szkło, temperatura) — audyty pokazały, że nie wnoszą
mierzalnych predykcji; rzutów 3D jako dowodów (R² 0,2–0,9); nowych obserwabli bez replikacji starych.

## 6. Artykuł: kształt i gdzie wysłać

Tytuł roboczy: *One forward pass, substep by substep, checkpoint by checkpoint: how a vision transformer detector builds a squeezing
transport of its tokens*. Osiem sekcji (szkielet w `Tezy artykułu…`, sekcja 5), 6–7 rysunków (F1–F9 z s47 jako przebieg, replikacja
3047 i 5 seedów jako panele z CI), 1 tabela predykcji zapisanych przed pomiarem z werdyktami.

Miejsca, od najlepiej pasującego:
1. **Transactions on Machine Learning Research (TMLR)** — ocenia poprawność i jasność, nie „hype”; lubi replikacje, pre-rejestrowane
   predykcje i negatywne wyniki (P-FAIL można pokazać). Bez terminów. Najbardziej naturalny wybór dla tej pracy.
2. **Physical Review E / Machine Learning: Science and Technology (IOP)** — jeśli akcent pójdzie na mapę ściskającą, jakobiany, siodła i
   analogię linii optycznej (Cowsik i in. 2024 wyszli w PRE z geometrią propagacji sygnału w transformerach). Wymaga mocniejszej części
   teoretycznej (np. wykładniki po głębokości, iloczyny macierzy losowych).
3. **Warsztaty ICLR/NeurIPS o dynamice uczenia i interpretowalności** (szybka ścieżka, 4–9 stron) — dobre na pierwszą wersję z 2 seedami,
   żeby dostać recenzje przed pełnym artykułem.
4. **Medical Image Analysis / MICCAI** — tylko jeśli dołożyć wątek kliniczny (okluzje, końcówka narzędzia); obecne wyniki są o sieci,
   nie o chirurgii.

Minimalny warunek wysłania do 1 lub 2: krok 5.1 (5 seedów) i 5.2 (interwencja). Bez nich warsztat (5.3).
