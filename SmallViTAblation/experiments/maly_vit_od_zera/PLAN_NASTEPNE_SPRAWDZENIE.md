# Następne sprawdzenie: kiedy Koopman i Langevin zaczynają przewidywać tożsamość

Data: 2026-09-28. Status: **plan użytkownika do następnego sprawdzenia**. Nic nie uruchomiono. Przed zgłoszeniem
opisać szczegóły i czekać na zgodę.

## Słowa użytkownika (2026-09-28, sedno)

> na początku to sieć nic nie wie i ona się bardzo powoli uczy, więc na początku [Koopman/Langevin] nie będzie w stanie
> nic przewidywać […] przy pierwszych epokach […]. Jak zacznie się uczyć, no to po prostu zaczyna się zawsze
> przewidywanie i dopiero wtedy możemy obserwować po prostu to, co wychodzi z tożsamości.
>
> Trzeba to jakoś ugryźć dobrze, przy pomocy Koopmana, Langevin'a i zobaczyć jak to tam się zaczyna kreować na
> początku […] pierwsza, druga epoka, no to są takie epoki, w których Koopman i Langevin mają prawo się pomylić, więc
> te pomiary być może nie są zbyt dobre, natomiast w kolejnych pomiarach […] Koopman i Langevin powinny zyskiwać […]
> powinni wskazywać przebieg, w którą stronę latent będzie się uczył i co jest tożsamością w latencie dla narzędzia,
> pełną i rdzeniową […] sukcesywnie zwiększamy po prostu zbiór uczący, no i musimy to porównywać do losowego
> zwiększania różnymi klatkami, albo nawet […] dokładania kolejnych z rzędu klatek do uczenia.

## Co z tego wynika do zmierzenia

1. **Jakość prognozy w funkcji czasu uczenia.** Trzeba zmierzyć, od którego obrotu Koopman i Langevin zaczynają
   przewidywać lepiej niż założenie „bez zmian”.
   - Liczymy test wsteczny w każdym obrocie od kroku 0, także w rozgrzewce.
   - Oczekiwanie użytkownika: w pierwszych 1–2 epokach prognoza może się mylić, a potem stosunek do „braku zmiany”
     powinien spadać poniżej 1.
   - Pierwsze epoki raportujemy osobno i nie mieszamy ich z oceną metody.
2. **Narodziny tożsamości.** Obserwujemy, jak od początku uczenia powstaje pełny latent i rdzeń tożsamości
   narzędzia.
   - Śledzimy, kiedy pojawiają się pierwsze neurony, które zapalają się tylko na narzędziu.
   - Śledzimy, które warianty sond domykają się najpierw: obrót, tło, długość, jasność.
   - Sprawdzamy, czy Koopman i Langevin przewidują kierunek tego procesu, czyli które neurony urosną, a które
     zmaleją.
3. **Sterowanie dopiero po starcie przewidywania.** Klatki wybierane przez Koopmana i FastGAN liczą się dopiero
   wtedy, gdy prognoza zaczyna bić „brak zmiany”. Wcześniej można dokładać klatki tak samo jak w ramieniu losowym.
4. **Punkty odniesienia przy stopniowym powiększaniu zbioru uczącego:**
   - `losowe`: te same liczby losowych klatek z puli;
   - `kolejne z rzędu` (nowe ramię): kolejne klatki tych samych filmów, na przykład następne po klatkach bazy
     w czasie nagrania;
   - `bez`: nauka bez dodawania klatek.

## Na czym się oprzeć

Pętla `petla.py` już zapisuje latent sond co 10 kroków od kroku 0, więc punkty 1–2 da się częściowo policzyć bez
nowego GPU na istniejących `LATENT.npz`.

Dotyczy to:
- ziarna 42 lokalnie;
- ziaren 47–56 z Edenu (zadanie 1817935, `~/smallvit_runs/2026-09-28_maly_vit_od_zera/wyniki`), po ich pobraniu.

Nowe ramię `kolejne z rzędu` i próg „sterowanie dopiero po starcie przewidywania” wymagają zmiany kodu i nowej
kampanii, więc potrzebują zgody.

## Analiza wsteczna na zapisanych latentach (2026-09-28, przed liczeniem)

Użytkownik (2026-09-28) proponuje: pierwsze trzy epoki uczyć samą bazę 1500 klatek, dopiero potem zacząć dokładanie
(Koopman i losowe); wcześniej sprawdzić, jak w ViT rodzą się wektory tożsamości; jeśli lepszy jest inny moment
startu, zaproponować go z uzasadnieniem.

Przeliczenie jednostek: paczka 32, baza 1500 klatek, więc **1 epoka = 47 kroków**. Obecna rozgrzewka (3 obroty po
500 kroków = 1500 kroków) to już **32 epoki**. „Obrót” w pętli to ~10,7 epoki.

Skrypt `analiza_wsteczna.py`, bez GPU, na `LATENT.npz` ziaren 42 i 47–56 (zapis co 10 kroków, bloki po 5 = punkt
co 50 kroków). Dalszy ciąg po kroku 1500 bierzemy z ramienia `bez` (ta sama baza, bez dokładania); dla porównania
też z `losowe` i `koopman`.

1. **Narodziny tożsamości** co 50 kroków: wielkość pełnego latentu i rdzenia, „odległość do rdzenia” (najlepszy
   neuron wybiórczy: najmniejszy po 15 wariantach udział zapalonych źródeł; rdzeń wymaga ≥ 0,75), domknięcie
   każdego wariantu, stabilność zbioru (Jaccard pełnego latentu z chwili t z tym sprzed 250 kroków i z końcowym).
2. **Jakość prognozy w funkcji chwili startu** t (co 250 kroków od 250): wybór neuronów i standaryzacja tylko z danych
   do t (jak w pętli), dopasowanie Koopmana i Langevina do historii do t, prognoza na H = 1, 5, 10, 30 bloków
   (50, 250, 500, 1500 kroków) i porównanie z tym, co się naprawdę stało. Wzorce: „brak zmiany” i „trend” (ostatnie
   nachylenie z 5 bloków). Błąd w przestrzeni śledzonych neuronów.
3. **Kierunek**: czy prognoza trafia, które neurony tożsamości urosną, a które zmaleją (zgodność znaku zmiany
   średniej po sondach, dla neuronów o zmianie powyżej mediany; Spearman przewidzianej i prawdziwej zmiany).
   Wzorzec: „trend”.
4. **Trafność wykrywania** mAP@0,5 z ocen co 100 kroków (rozgrzewka + ramię `bez`) i jej udział w poziomie końcowym.

**Reguła wyboru chwili startu (ustalona przed liczeniem).** Najwcześniejsza chwila t (wielokrotność 250 kroków), dla
której Langevin albo Koopman przy H = 10 bloków (jeden obrót) ma medianę błędu po 11 ziarnach poniżej „braku zmiany”
i poniżej „trendu”, w co najmniej 8 z 11 ziaren poniżej „braku zmiany”, i to samo zachodzi w co najmniej 3 z 4
kolejnych chwil. Jeśli takiej chwili nie ma, prognoza nie nadaje się do sterowania na tej skali czasu, niezależnie od
momentu startu.

### Wynik analizy wstecznej (2026-09-28, 11 ziaren, `wyniki/ANALIZA_WSTECZNA.json`, `.png`)

Jednostka: 1 epoka bazy = 47 kroków. Dalszy ciąg po kroku 1500 z ramienia `bez` (ramiona `losowe` i `koopman` dają
ten sam obraz).

- **Trafność wykrywania** (mediana 11 ziaren, poziom końcowy po 96 epokach 0,436): 3 epoki (141 kroków) ≈ 0,04,
  czyli ~9 % poziomu końcowego; 11 epok 0,20 (47 %); 15 epok 0,28 (64 %); 21 epok 0,30 (69 %); 32 epoki 0,387 (89 %).
  Trzy epoki w dosłownym sensie to za wcześnie: sieć prawie nic jeszcze nie wykrywa. Obecna rozgrzewka (32 epoki)
  kończy się, gdy wykrywanie jest już blisko poziomu końcowego.
- **Tożsamość nie rodzi się z uczeniem, tylko kurczy.** Pełny latent jest największy na starcie (80–88 neuronów przy
  1–4 epokach, czyli w sieci prawie losowej) i maleje do 47 przy 32 epokach i 33 przy 96. „Odległość do rdzenia”
  (najlepszy neuron) najwyższa na starcie, 0,58, potem 0,33; próg rdzenia 0,75 nieosiągnięty nigdy (pojedynczy neuron
  rdzenia tylko w 1–2 epoce jednego ziarna). Miara wybiórczości na sondach łapie więc na starcie proste różnice
  pikseli (jasny metal kontra tkanka), a uczenie robi neurony bardziej zależne od wariantu, nie bardziej uniwersalne.
- **Zbiór neuronów tożsamości wymienia się cały czas.** Jaccard z chwilą sprzed 250 kroków (~5 epok) ≈ 0,45 od
  ~15 epoki do końca: co 5 epok wymienia się mniej więcej połowa neuronów. Jaccard ze zbiorem końcowym rośnie powoli:
  0,07 (start), 0,21 (32 ep), 0,31 (64 ep). Śledzimy więc obiekt, który sam się zmienia.
- **Prognoza nie zaczyna działać w żadnej chwili.** Koopman / brak zmiany ≈ 1,00 we wszystkich chwilach i horyzontach
  (model uczy się przewidywać „prawie brak zmiany”); Langevin 1,02–1,11 na horyzoncie 500 kroków i gorzej na 1500.
  Na krótkich horyzontach (50–250 kroków) pod koniec (75–91 epok) mediana spada do 0,97–0,98 w 8–9 z 11 ziaren —
  zysk 2–3 %, bez znaczenia praktycznego. Trend (dalsze nachylenie) jest 2–5 razy gorszy od braku zmiany.
- **Kierunek** (które neurony urosną, a które zmaleją): zgodność znaku 0,50–0,62 (rzut monetą = 0,50), Spearman
  0,0–0,2. Langevin minimalnie ponad losowość, ale bez trendu w czasie.
- **Reguła z planu: niespełniona.** Jedyne „trafienie” (Koopman, `bez`, krok 4000 = 85 epok) to ostatnia chwila, dla
  której nie ma już kolejnych punktów do potwierdzenia (artefakt reguły), a mediana 0,98 jest w granicy szumu.

**Wniosek dla propozycji użytkownika.** Pomysł „najpierw nauczyć, potem sterować” jest słuszny i był już zrealizowany:
dokładanie startowało po 32 epokach. Przesunięcie startu nic nie zmieni, bo prognoza nie działa w żadnym momencie
1–96 epok. Przeszkodą jest definicja tożsamości na pojedynczych neuronach: zbiór jest niestabilny i kurczy się, a rdzeń
nie istnieje. Najpierw trzeba znaleźć stabilny wektor tożsamości (np. kierunek w latencie oddzielający narzędzie od
tła, liczony na wszystkich 768 neuronach, zamiast listy neuronów), sprawdzić jego stabilność na tych samych latentach
(bez GPU), i dopiero gdy będzie stabilny i przewidywalny, wrócić do sterowania. Chwila startu z danych: nie wcześniej
niż ~20 epok (wykrywanie ~70 % poziomu końcowego); obecne 32 epoki są rozsądne.

**Sprawdzenie dodatkowe: wektor tożsamości na wszystkich 768 neuronach** (`kierunek_tozsamosci` w
`analiza_wsteczna.py`; średnia na narzędziu minus średnia negatywów, w jednostkach sd negatywów). Mediana 11 ziaren,
ramię `bez`:
- cosinus z chwilą sprzed 250 kroków: 0,72 (11 ep), 0,85 (16 ep), 0,89 (32 ep), 0,90 (64 ep) — **od ~16 epok wektor
  jest stabilny**, dużo stabilniejszy niż lista neuronów (Jaccard 0,45);
- cosinus z wektorem końcowym (96 ep): 0,20 (1 ep), 0,41 (11 ep), 0,59 (32 ep), 0,75 (64 ep), 0,83 (80 ep) — powolny,
  gładki dryf przez cały trening, czyli dokładnie taki ruch, który ma sens przewidywać;
- zgodność kierunków liczonych osobno dla 15 wariantów: ~0,65 i stała — wspólny kierunek istnieje tylko częściowo.
Ramiona `losowe` i `koopman` dają te same liczby (±0,05).

**Propozycja na następny krok (bez GPU, do zgody):** powtórzyć test prognozy na tym wektorze zamiast na liście
neuronów (Koopman/Langevin na trajektorii wektora 768 → PCA), z tą samą regułą. Jeśli prognoza wektora zacznie bić
„brak zmiany” od jakiejś epoki, to ta epoka jest uzasadnionym startem sterowania; z samej stabilności wektora
i wykrywania wynika dolna granica ~16–20 epok.
