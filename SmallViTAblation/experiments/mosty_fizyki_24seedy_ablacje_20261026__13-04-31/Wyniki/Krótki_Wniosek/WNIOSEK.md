# Krótki wniosek

Eksperyment `mosty_fizyki_24seedy_ablacje_20261026__13-04-31` (24 seedy × 12 checkpointów × 500 klatek TEST; 18 ablacji 6 × 3 seedy;
predykcje i progi zapisane przed pomiarem). Wynik predykcji: 8 z 18 PASS (P1, P2, P4, P6, P8, P10, P7b, A5), 10 FAIL.

**Co się replikuje (24/24 seedy).** Trening czyni pole MLP nienormalnym: ‖N‖/‖J‖ bloku 4 rośnie z 0,76 (wagi losowe) do 0,90–0,94,
najwięcej w blokach 3–5, mniej w 1 i 6; rośnie też bez weight decay (A5, 0,90–0,93). Uwaga ściska chmurę tokenów (stosunek odległości
0,95–0,98 w blokach 1–5) ruchem stycznym do sfery (0,97–0,99), MLP bloków 1 i 6 rozciąga, bloków 2–5 ściska. Separację klas narzędzie / tło
buduje MLP (Σ Δsep 0,69–1,35), nie uwaga (0,12–0,42). Trening przenosi ściskanie z uwagi (iloczyn 0,66 → 0,85) do MLP (1,00 → 0,83).

**Co pada.** Pole nie ściska bardziej wokół narzędzia (P3 11/24). Szczyt defektu komutatora jest w bloku 3, nie 4–5 (P9 10/24).
Separacja klas nie przewiduje IoU klatki: ρ ≈ 0,42 po wszystkich checkpointach razem było śladem trajektorii treningu (obie wielkości rosną),
per klatka przy końcowym checkpoincie ρ = −0,03 (P7a 0/24); między seedami ρ = 0,50 na 24 punktach to słaby dowód (P7b).

**Co mówią ablacje.** Podział pracy jest własnością wyuczonego rozwiązania, nie koniecznością. Zamrożony losowy MLP: AP 0,857–0,884
(base 0,864–0,888), separację przejmuje uwaga (A3 pada). Zamrożona uwaga: AP 0,786–0,861, uwaga staje się niemal identycznością
(iloczyn 0,92–0,97), MLP robi wszystko (A4 pada; była źle postawiona, bo kontrakcja zależy od stanu tokenów, nie tylko od wag).
Blok równoległy: AP 0,869–0,895, bez straty, separację dzielą uwaga i MLP po równo (A2 pada, teza C odrzucona). Losowe etykiety:
AP ≈ 0,005, a mimo to ‖N‖/‖J‖ rośnie do 0,92–0,96 (A6 pada): nienormalność jest skutkiem treningu gradientowego MLP, nie sygnaturą
nauczenia zadania.

**Do publikacji.** Nadaje się: (1) pre-rejestrowana replikacja nienormalnego ścinania w MLP i podziału pracy uwaga / MLP na 24 seedach,
(2) kompensacja funkcjonalna w ablacjach, która tę replikowaną strukturę relatywizuje, (3) negatywny wynik o separacji i IoU z rozbiorem
konfundatora. Nie nadaje się: lokalizacja mechanizmu w narzędziu ani w blokach 4–5, teza o kolejności uwaga → MLP, nienormalność jako
znak „zrozumienia” zadania. Brakuje drugiego zbioru lub architektury.
