# Tożsamość narzędzia liczona kierunkiem odświeżanym: czy idzie w parze z mAP i co da się z niej przewidzieć

Data: 2026-10-05. Plan zapisany przed uruchomieniem kodu. Pomiar bez treningu, bez GPU i bez Edenu, na zapisanych
latentach sond z kampanii `../maly_vit_wektor_tozsamosci` (pętla v2, Eden 1817949, 10 ziaren 47-56).

## Po co

Obliczenie z 5 października na telemetrii kampanii 4.3 pokazało, że siła tożsamości liczona wzdłuż kierunku
ustalonego na końcu rozgrzewki nie idzie w parze z mAP: ramię bez dokładania klatek ma najwyższą siłę i najniższe mAP,
a zmiana siły w obrocie nie mówi nic o zmianie mAP (korelacja 0,006). To obliczenie miało jedną lukę, zapisaną przy
nim wprost: kierunek był stały. Spadek rzutu na stary kierunek może znaczyć tylko tyle, że reprezentacja się obróciła,
a narzędzie jest zakodowane równie dobrze albo lepiej, tylko gdzie indziej. Ten pomiar zamyka lukę.

Słowa: **kierunek stały** to wektor tożsamości dopasowany raz, w kroku 1000 (tak działa pętla, `docs/26` krok 2).
**Kierunek odświeżany** to ten sam przepis (`sila.wektor`: LDA ze ściąganiem Ledoit-Wolf, narzędzie przeciw tłu sond i
obszarowi narzędzia w wersji `empty_black`) zastosowany od nowa w każdej badanej chwili. **Rozdzielenie** `d'` to
odległość średnich rzutów narzędzia i negatywów podzielona przez ich wspólne odchylenie standardowe.

## Pytania

1. Czy rozdzielenie mierzone kierunkiem odświeżanym, sprawdzane na wycinkach nieużytych do dopasowania, rośnie w
   uczeniu?
2. Czy ramiona, które uczą się lepiej (wyższe mAP), mają wyższe rozdzielenie odświeżane? Dla porównania: jak
   zachowuje się w tych samych biegach siła wzdłuż kierunku stałego?
3. Czy spadek siły wzdłuż kierunku stałego to obrót kierunku (cosinus kierunku odświeżanego z kierunkiem z kroku 1000
   maleje), czy utrata rozdzielenia?
4. Co z zachowania tożsamości da się przewidzieć na 500 kroków naprzód bez podglądania przyszłości tego samego biegu:
   zmianę przewidujemy średnią zmianą w tych samych krokach w pozostałych ziarnach („krzywa innych biegów”).
5. Czy układ sond względem siebie (geometria relacyjna) jest powtarzalny między ziarnami, mimo że sam kierunek nie jest
   (cosinus między ziarnami 0,085 według `../maly_vit_wektor_tozsamosci/PLAN.md`)?

## Dane

`../maly_vit_wektor_tozsamosci/wyniki_eden/petla_s47..s56`: latent 192 sond (średnia po tokenach narzędzia i osobno po
tle, 768 liczb) co 10 kroków; rozgrzewka `wspolny` (kroki 0-1000) i trzy ramiona od kroku 1000 do 4500: `bez`
(bez dokładania klatek), `losowe`, `koopman` (klatki wyszukane pod prognozę). mAP@0,5 na 3000 klatek walidacji co
100 kroków (`OCENY.json`).

## Miary

Siatka czasu: co 50 kroków. Latent w chwili `t` to średnia z pięciu zapisów `t-40..t`, jak w pętli.

- `sila_stala(t)`: średnia po 180 sondach siła wzdłuż kierunku z kroku 1000, ze standaryzacją z kroku 1000.
- `d_staly(t)`: rozdzielenie wzdłuż tego samego stałego kierunku w chwili `t`.
- `d_odsw(t)`: rozdzielenie kierunku dopasowanego w chwili `t` na 9 wycinkach i policzone na 3 pozostałych; średnia z
  4 podziałów 12 wycinków. Obok ta sama liczba dla prostszego kierunku „różnica średnich” (bez odwracania kowariancji)
  oraz pole pod krzywą ROC.
- `cos_1000(t)`, `cos_500(t)`: cosinus kierunku odświeżanego (w surowych współrzędnych latentu) z kierunkiem z kroku
  1000 i z kierunkiem sprzed 500 kroków.
- `cka_koniec(t)`: liniowe CKA macierzy 180 sond × 768 w chwili `t` z tą samą macierzą na końcu biegu; CKA między
  ziarnami w ramieniu `bez` w krokach 1000, 2500 i 4500, obok cosinusa kierunków między ziarnami.

## Kryteria ustalone przed wynikiem

- **K1 (rozdzielenie odświeżane rośnie):** w ramieniu `bez` `d_odsw(4500) > d_odsw(1000)` w co najmniej 8 z 10 ziaren.
- **K2 (idzie w parze z mAP między ramionami):** średnia `d_odsw` z kroków 3500-4500 jest w ramionach `losowe` i
  `koopman` wyższa niż w `bez` w co najmniej 8 z 10 ziaren (osobno dla każdego ramienia), o ile ich mAP jest w tych
  krokach wyższe w co najmniej 8 z 10 ziaren. Ta sama tabela dla `sila_stala`.
- **K3 (korelacja wewnątrz chwili):** po odjęciu średniej z trzech ramion w każdej parze (ziarno, krok ≥ 1500 co 100)
  korelacja odchylenia mAP z odchyleniem `d_odsw` ma dolną granicę przedziału 95 procent (losowanie ziaren) powyżej
  zera. Ta sama liczba dla `sila_stala` i `d_staly`.
- **K4 (opisowo):** umiejętność prognozy zmiany na 500 kroków z krzywej innych ziaren, `1 − SSE/SSE „bez zmian”`, dla
  mAP, `sila_stala`, `d_odsw` i `cos_1000`; dla siły także osobno dla każdej sondy.
- **K5 (opisowo):** CKA między ziarnami wobec cosinusa kierunków między ziarnami.

Jeśli K2 i K3 są spełnione dla `d_odsw`, a nie dla `sila_stala`, to tożsamość ma sens jako cel, ale trzeba ją mierzyć
kierunkiem odświeżanym. Jeśli nie są spełnione dla żadnej z miar, to rozdzielenie narzędzia od tła w tej definicji nie
jest tym, czego brakuje detektorowi, i celem doboru powinno być coś innego (na przykład kierunek potrzebny głowie).

## Czego ten pomiar nie dowodzi

- Definicja tożsamości to wersja `stary` (narzędzie przeciw tłu i scenie). Według `../maly_vit_tozsamosc_tla/PLAN.md`
  mierzy ona w dużej części miejsce i scenę (`d'` 2,0 wobec tego samego miejsca bez narzędzia); sieć nieuczona też ją
  przechodzi. Definicji parowej (J128, 400 par z usuniętym narzędziem) te zapisy nie pozwalają policzyć.
- 192 sondy to 12 wycinków w 16 wersjach, poza rozkładem detektora; podział na 9 i 3 wycinki daje szumne `d'`.
- Trzy ramiona, jedna kampania, kroki do 4500. To korelacje, nie dowód przyczyny.
- Prognoza z krzywej innych ziaren wymaga wcześniejszej kampanii o tym samym protokole.
