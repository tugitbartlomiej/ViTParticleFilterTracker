> **Archiwum (2026-10-14).** Aktualna, zaktualizowana wersja zasad (razem z treścią `ZASADY_LASER_KOLOID.md`) jest w `app/ZASADY_DZIALANIA_UKLADU.md`. Od 2026-10-14 słowo „pułapka” oznacza pułapkę optyczną z laserów (pole i ognisko wiązek), a dawny „środek drgań” (średnia ±300 kroków) nazywa się „średnia skoków koloidu”. Ten plik zostaje bez zmian treści, bo odwołują się do niego skrypty filmów.

# Czy pułapka nadąża za koloidami? Pomiar na biegu s47 (2026-10-12)

Skrypt `pomiar_nadazania_pulapki.py` → `NADAZANIE_PULAPKI_S47.json`; ten sam log jest w JSON pod kluczem `log` i w pliku `pomiar_nadazania_pulapki.log`.
Bieg `assets/MalyViT_35000_od_zera_s47`, blok 6 (indeks 5). Kryteria werdyktu zapisano w skrypcie przed pełnym pomiarem.

**Dane i kontrola**
- Pytanie 1: 4 zdjęcia TRAIN (2400 tokenów), 747 checkpointów na siatce o stałym odstępie (co 10 kroków do kroku 1000, potem co 100), 746 przedziałów.
- Pytanie 2: 32 zdjęcia TRAIN (19 200 tokenów) w checkpointach końców epok 0–60. Klatki pochodzą z lokalnych zipów Cataract-LMM, z kontrolą SHA-256, i są wczytane jak w treningu. To pierwsze zdjęcia pomiaru ruch60, z filmów TR_0001–TR_0004.
- Przejście modelu odtwarza zapisany latent z R² ≥ 0,99993, sprawdzone na TRAIN 0 i 1 w krokach 0, 1000, 22 974 i 65 640.
- Wzór MLP zgadza się z wyjściem modelu z błędem względnym ≤ 5·10⁻⁷. Rozkład Δ_in + Δ_W odtwarza całą zmianę z błędem 5·10⁻¹⁷.
- Ustawienia liczenia: eps LN2 wzięty z modułu (1e-6), TF32 wyłączone, przejście w float32, rozkład w float64.

## Pytanie 1: czy pułapka idzie za ruchem koloidu?

Rozkład zmiany wyjścia MLP F_j między kolejnymi checkpointami:
- **Ruch koloidu:** Δ_in(t) = F(x(t+1); θ6(t)) − F(x(t); θ6(t)).
- **Zmiana pułapki:** Δ_W(t) = F(x(t+1); θ6(t+1)) − F(x(t+1); θ6(t)).

Oznaczenia w tabeli:
- C(τ) = ⟨cos(Δ_in(t), Δ_W(t+τ))⟩. C(+1) mierzy pułapkę po koloidzie, a C(−1) koloid po pułapce.
- a(1) to regresja Δ_W(t+1) ~ a·Δ_in(t) + b·Δ_W(t): pułapka po koloidzie, z kontrolą własnej przeszłości.
- c(1) to regresja Δ_in(t+1) ~ c·Δ_W(t) + e·Δ_in(t): odwrotny kierunek.
- Odchylenia sd liczone są względem zera z przesunięcia cyklicznego w czasie.

| faza | przedział | udział pułapki w energii zmiany | C(0) | C(+1) | C(−1) | zero C (sd) | autokor. Δ_in / Δ_W przy 1 | a(1) | c(1) |
|---|---|---|---|---|---|---|---|---|---|
| kroki 0–1000 | 10 kroków | 13 % | +0,149 | −0,034 | −0,071 | 0,010 | −0,25 / +0,06 | −0,023 (−5,0 sd) | −0,077 (−2,8 sd) |
| epoki 1–20 | 100 kroków | 23 % | +0,089 | −0,047 | −0,044 | 0,011 | −0,37 / −0,22 | −0,017 (−2,5 sd) | −0,027 (−1,7 sd) |
| epoki 20–40 | 100 kroków | 22 % | +0,093 | −0,030 | −0,048 | 0,008 | −0,34 / −0,18 | −0,006 (−0,9 sd) | −0,026 (−1,5 sd) |
| epoki 40–60 | 100 kroków | 26 % | +0,057 | −0,013 | −0,031 | 0,009 | −0,33 / −0,19 | +0,006 (+1,0 sd) | −0,029 (−1,5 sd) |

**Werdykt: niejednoznaczne według kryterium, w treści brak nadążania w jedną stronę.**

1. **W tym samym przedziale oba ruchy pchają wyjście bloku w tę samą stronę.** C(0) wynosi od +0,06 do +0,15, czyli 6–15 odchyleń zera. To wspólne zejście po gradiencie, a nie gonienie. Gdyby pułapka goniła koloid, żeby utrzymać wyjście, C(0) byłoby ujemne.
2. **Przy opóźnieniu 100 kroków korelacja jest ujemna w obie strony i podobnej wielkości.**
   - Po 1. epoce C(+1) mieści się między −0,013 a −0,047, a C(−1) między −0,031 a −0,048. To ten „efekt w obie strony”, przez który kryterium daje „niejednoznaczne”.
   - Wynika on z tego, że każda część drga osobno tam i z powrotem. Autokorelacja przy jednym przedziale wynosi od −0,33 do −0,37 dla koloidu i od −0,18 do −0,22 dla pułapki, a do tego dochodzi dodatnie C(0).
   - Po uwzględnieniu własnej przeszłości żaden kierunek nie przekracza 3 odchyleń: a(1) leży między −2,5 a +1,0 sd, c(1) między −1,7 a −1,5 sd.
   - Jedyny wynik poza 3 sd po 1. epoce to a(7) = −3,8 sd w epokach 40–60. Jest pojedynczy, a sąsiednie opóźnienia mają przeciwny znak (+2,1 i +2,3 sd). Przy 60 porównaniach to szum.
3. **Suma współczynników przez 10 przedziałów, czyli 1000 kroków.**
   - Pułapka odrabia 0–1 % zmiany pochodzącej od koloidu (suma a: −0,013, −0,000, −0,005).
   - Koloid odrabia 4–6 % zmiany pochodzącej od pułapki (suma c: −0,041, −0,040, −0,060), choć żaden pojedynczy współczynnik nie jest istotny.
   - Jeśli coś tu jest, to raczej odwrotnie niż w hipotezie.
4. **Rozgrzewka (kroki 0–1000, lr rośnie przez 300 kroków).** Widać słabą wzajemną kompensację. Pułapka odrabia 2,3 % ruchu koloidu w następnych 10 krokach (a(1) = −5,0 sd). Koloid odrabia około 8 % zmiany pułapki (c(1) = −2,8 sd), a w sumie przez 100 kroków 19 %.
5. **Prędkości |Δ| (logarytm, po odjęciu trendu z 21 przedziałów).**
   - Po 1. epoce nie ma zależności z opóźnieniem: |r| ≤ 0,08 przy zerze ±0,08–0,09. Tylko r(0) = 0,15 w epokach 1–20.
   - Na starcie r(0) = 0,35, r(+1) = 0,30, r(−1) = 0,16 przy zerze ±0,13, czyli zależność słaba.
6. **Ruch wspólny wszystkich tokenów.** Po przetasowaniu tokenów C(0) spada z 0,089, 0,093 i 0,057 do 0,054, 0,067 i 0,040. Zatem 60–70 % współruchu to wspólny „wiatr” wszystkich tokenów, a reszta jest własna tokenu. Ujemne C(±1) też w większości zostaje po tasowaniu.
7. **Kontrole.**
   - Rozkład symetryczny (Shapley, średnia obu kolejności) daje te same liczby: C(0) = 0,087, 0,096 i 0,062; a(1) = −0,017, −0,007 i +0,005.
   - Na latencie z (768 wymiarów, TRAIN 0) wychodzi ten sam wzór: C(0) ≈ +0,07, a C(±1) między −0,02 a −0,04.
   - Udział pułapki w energii zmiany z wynosi 2 % na starcie oraz 17 %, 16 % i 20 % w kolejnych fazach. Zgadza się to z pomiarem z Edenu (2 % w krokach 100–1000, 11–16 % w epokach 1–60).

## Pytanie 2: czy celowniki obracają się razem ze strumieniem?

**Metoda.**
- R_x to obrót Prokrustesa stanów x̂ wejścia bloku: LN bez γ i β, po odjęciu średniej po tokenach.
- Wybór x̂: czujnik widzi tylko x̂, bo a_c = s_c·x̂ + stała. Wspólne przesunięcie wszystkich tokenów wchłania próg b_c, a wzrost normy usuwa LN.
- R_x przykładam do czujników s_c = P₁(γ ⊙ w_c) i porównuję z trzema punktami odniesienia: najlepszym obrotem samych czujników R_s, zerem z przetasowaną zmianą oraz tym samym R_x z losowo obróconymi osiami (QR_xQᵀ).
- Liczę w podprzestrzeni wspólnej dla fazy, która niesie 95 % wariancji strumienia; 99 % służy jako kontrola.
- W pełnej 192D obrót strumienia jest nieokreślony w kierunkach bez wariancji. Próba techniczna przed pełnym pomiarem dała udziały rzędu −35, jak dla losowych osi.
- Kryteria werdyktu się nie zmieniły.

| faza (okna 1-epokowe, baza 95 %) | k | strumień: wspólny obrót (zero) | czujniki s: własny obrót (zero) | R_x przyłożony do s | zero: losowe osie | cos(log R_x, log R_s) | zero cos (niepasujące okna) | cos z R_x okna wcześniej |
|---|---|---|---|---|---|---|---|---|
| epoki 0–20 | 56 | 35 % (0,1 %) | 6,7 % (3,5 %) | −2,41 | −2,40 ± 0,06 | −0,033 | +0,005 ± 0,025 | +0,035 |
| epoki 20–40 | 58 | 33 % (0,1 %) | 6,1 % (3,5 %) | −1,78 | −1,66 ± 0,06 | −0,045 | +0,001 ± 0,025 | +0,015 |
| epoki 40–60 | 62 | 33 % (0,1 %) | 8,1 % (3,9 %) | −1,33 | −1,26 ± 0,04 | −0,026 | +0,001 ± 0,024 | +0,005 |

**Werdykt: obalone.**

1. **Strumień obraca się jako całość mocno.** W każdej epoce jeden obrót wyjaśnia 33–35 % zmiany stanów x̂ w podprzestrzeni 95 % (37–41 % w podprzestrzeni 99 %), a zero wynosi 0,1 %. Łączny kąt obrotu, czyli pierwiastek z sumy kwadratów kątów we wszystkich płaszczyznach, to około 1,4 rad na epokę po 20. epoce. W wariantach w pełnej 192D (okna 5-epokowe) wychodzi:
   - dla x̂ po odjęciu średniej 38–49 %;
   - dla x̂ bez odjęcia średniej 47–82 %;
   - dla surowego x̃ 9–44 %;
   - dla wyjścia bloku x⁺ prawie tyle samo, co dla x̂ po odjęciu średniej.
2. **Celowniki za nim nie idą.**
   - Obrót strumienia przyłożony do czujników działa tak źle jak obrót o tych samych kątach wokół losowych osi: późno −1,33 wobec −1,26 ± 0,04, a w oknach 5-epokowych −0,64 wobec −0,61 ± 0,02.
   - R_x wzięty z innego okna daje to samo (−1,30 ± 0,17).
   - Generatory obrotów nie są zgodne: cos −0,026 przy zerze +0,001 ± 0,024. Z obrotem strumienia z poprzedniego okna cos wynosi +0,005 późno i +0,035 w epokach 0–20.
   - Średnie tych cosinusów różnią się od zera o kilka błędów standardowych, ale ich kwadrat to poniżej 0,2 % wariancji, czyli praktycznie nic.
   - Wynik nie zależy od bazy: w bazie 99 % (k = 114–120) jest tak samo.
3. **To samo dla w_c (bez γ) i dla kierunków zapisu v_c.** Kontrolny wynik dla v_c późno: R_x −1,72 wobec −1,65 ± 0,06 dla losowych osi, cos −0,012.
4. **Skąd wcześniejszy wspólny obrót celowników (22–24 % wobec 9–10 % zera, PRZELICZENIE część O).**
   - Ten wynik się odtwarza: w pełnej 192D wychodzi 25 % w oknach 1-epokowych epok 40–60 i 23 % w 5-epokowych.
   - W podprzestrzeni niosącej 95 % wariancji strumienia wspólnego obrotu jest mało: 8,1 % wobec 3,9 % zera. W podprzestrzeni 99 % jest go więcej: 17,6 % wobec 6,6 %.
   - Wspólny obrót celowników zachodzi więc głównie w kierunkach, w których strumień prawie nie ma wariancji. Tam składowa czujnika daje prawie stałą, którą wchłania próg b_c.
   - Od 59 do 64 % normy czujników leży w podprzestrzeni 95 %. To nie jest nadążanie za strumieniem.

## Co z tego wynika dla obrazu „pułapka nadąża za koloidem”

- W skali 100 kroków pułapka i koloid nie gonią się nawzajem. W tym samym przedziale pchają wyjście w tę samą stronę, a potem każda część drga tam i z powrotem. Po odjęciu tego drgania nie zostaje reakcja z opóźnieniem w żadną stronę, poza słabą wzajemną kompensacją w pierwszych 1000 krokach.
- Strumień rezydualny co epokę obraca się jako całość (około jednej trzeciej zmiany), ale czujniki ostatniego bloku się za nim nie obracają. Ich własny wspólny obrót siedzi w kierunkach, których strumień prawie nie używa.
- Ograniczenia pomiaru:
  - rozdzielczość czasowa to odstęp checkpointów (10 kroków na starcie, potem 100), więc szybsza reakcja, w obrębie jednego przedziału, wchodzi do C(0), które jest dodatnie;
  - pytanie 1 liczyłem na 4 zdjęciach, pytanie 2 na 32 zdjęciach z 4 filmów;
  - w podprzestrzeni 95 % pomijam wszystko, co dzieje się w kierunkach o małej wariancji strumienia.
