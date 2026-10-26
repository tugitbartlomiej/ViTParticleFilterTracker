# Raport: mosty do fizyki, replikacja na 24 seedach i ablacje

Eksperyment `mosty_fizyki_24seedy_ablacje_20261026__13-04-31`. Kod pomiaru `Kod_eksperymentu/pomiar_mosty_eden.py`, treningu
`Kod_eksperymentu/trening_ablacje.py`, ocena `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`.
Predykcje i progi: `Opis/PREDYKCJE_PRZED_POMIAREM.md` (zapisane 2026-10-26 13:30, przed uruchomieniem). Progi nie były zmieniane.

## 1. Zakres i kompletność

- Część 1: 24 seedy kampanii `kompensacja_funkcjonalna_dgx2_20261017` (3047, 4047, …, 26047), 12 checkpointów (0 … 65 640 kroków),
  500 klatek TEST z 25 filmów spoza treningu (472 z ramką GT wchodzą do miar), 50 klatek diagnostycznych dla jakobianów.
  24/24 zadania SLURM COMPLETED (19–40 min), SHA256 wag zapisane per seed.
- Część 2: 6 wariantów × seedy 3047, 4047, 5047, trening od zera na A100 tym samym przepisem (60 epok, 65 640 kroków), potem ten sam
  pomiar. 18/18 COMPLETED (1 h 09 – 1 h 46). Wagi (135 MB na bieg) pozostały na Edenie; lokalnie pobrano pomiar, `loss.jsonl`,
  `WARIANT.json`, `COMPLETE.json`, `WAGI_SHA256.txt`.
- Jedna poprawka w kodzie oceny po pierwszym odczycie (indeks kolumn `miary_A`; opis w `Opis/OPIS_EKSPERYMENTU.md`), bez zmiany progów.
- Konwencja: bloki 1–6; mediany po tokenach i klatkach; test znaku po seedach.

## 2. Wynik predykcji

| nr | predykcja | wynik | liczby (24 seedy, k65640, chyba że inaczej) |
|---|---|---|---|
| P1 | trening instaluje nienormalność MLP | **PASS 24/24** | ‖N‖/‖J‖ bloku 4: k0 0,759–0,768 → k65640 0,899–0,939 |
| P2 | nienormalność nie rośnie monotonicznie z głębokością | **PASS 24/24** | blok 4 ≥ blok 6 w każdym seedzie (0,90–0,94 vs 0,87–0,89) |
| P3 | wokół narzędzia pole bardziej ściska | FAIL 11/24 | tr J narzędzia vs tła: znak losowy (mediany −1,42 vs −1,73) |
| P4 | uwaga ściska, MLP 1 rozciąga | **PASS 24/24** | półkroki uwagi < 1 w blokach 1–5; MLP 1: 1,06 |
| P5 | uwaga styczna we wszystkich blokach (> 0,9) | FAIL 14/24 | bloki 1–5: 0,97–0,99 we wszystkich seedach; blok 6: 0,80–0,90 w 10 seedach |
| P6 | separację robi MLP | **PASS 24/24** | Σ Δsep MLP 0,69–1,35 vs uwaga 0,12–0,42 |
| P7a | separacja przewiduje IoU w obrębie seedu (ρ > 0,5) | FAIL 0/24 | ρ 0,34–0,48, patrz § 5 |
| P7b | separacja przewiduje AP między seedami (ρ > 0,4) | PASS 1/1 | ρ = 0,50 (24 punkty), sep₁₂ 1,92–2,46, AP 0,827–0,902 |
| P8 | trening przenosi ściskanie z uwagi do MLP | **PASS 24/24** | iloczyn uwagi 0,66 → 0,85 (0,78–0,92); MLP 1,00 → 0,83 (0,75–0,91) |
| P9 | szczyt defektu komutatora w blokach 4–5 | FAIL 10/24 | argmax: blok 2 ×1, blok 3 ×13, blok 4 ×8, blok 5 ×2 |
| P10 | zero kontrolne przy k0 | **PASS 22/24** | 2 seedy: ρ(sep, IoU) przy k0 = −0,35 / −0,37 (reszta warunków spełniona) |
| A1 | base odtwarza P1, P4, P6, P9 | FAIL 1/3 | P1, P4, P6 3/3; P9 1/3 (szczyt w bloku 3, 5, 3) |
| A2 | parallel traci ≥ 0,02 AP | FAIL 0/3 | AP 0,895 / 0,869 / 0,873 vs base 0,888 / 0,885 / 0,864 |
| A3 | frozen_mlp: AP < 0,40, separacja z uwagi | FAIL 0/3 | AP 0,866 / 0,857 / 0,884; Σ Δsep uwaga +0,80 / +0,42 / +0,73, MLP −0,06 / +0,01 / −0,06 |
| A4 | frozen_attn: kontrakcja uwagi bez zmian | FAIL 0/3 | iloczyn uwagi 0,66 → 0,97 / 0,62 → 0,94 / 0,65 → 0,92 |
| A5 | no_wd: nienormalność > 0,85 | **PASS 3/3** | 0,922 / 0,927 / 0,900 |
| A6 | shuffled_labels: brak nienormalności i separacji | FAIL 0/3 | AP 0,005–0,009, ale ‖N‖/‖J‖ 0,925 / 0,955 / 0,938, Σ Δsep MLP +0,28 / +0,24 / +0,23 |

Przy progu 20/24 (p ≈ 0,0008 w teście znaku) wyniki 24/24 i 22/24 są jednoznaczne. Wyniki 10–14/24 oznaczają brak efektu systematycznego.

## 3. Teza A: nienormalne ścinanie w MLP (opisowo, mediany po 24 seedach)

| blok | ‖N‖/‖J‖ k0 | ‖N‖/‖J‖ k65640 | ‖Ω‖/‖S‖ k65640 | tr J k65640 | λ: Re>0 / Re<0 | λ zespolone | \|1+λ\|>1 |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 0,763 | 0,879 | 0,875 | +13,8 | 103 / 87 | 168 | 107 |
| 2 | 0,762 | 0,909 | 0,968 | −0,94 | 94 / 96 | 168 | 96 |
| 3 | 0,762 | 0,911 | 0,961 | −1,26 | 93 / 97 | 170 | 96 |
| 4 | 0,763 | 0,916 | 0,955 | −1,70 | 92 / 98 | 170 | 95 |
| 5 | 0,762 | 0,916 | 0,948 | −1,76 | 91 / 98 | 168 | 94 |
| 6 | 0,762 | 0,886 | 0,896 | −2,64 | 87 / 103 | 168 | 88,5 |

- Przy k0 nienormalność 0,76 to poziom macierzy losowej; przy k65640 0,88–0,92 we wszystkich blokach, najwyżej w 3–5.
- Przebieg w treningu (blok 4): 0,763 (k0), 0,772 (300), 0,829 (1000), 0,902 (3000), 0,923 (6000), 0,927 (10 000), potem lekki spadek
  do 0,916 (65 640). Nienormalność ustala się w pierwszych 5 epokach, razem z AP (0,54 przy 1000, 0,82 przy 6000).
- Każdy token jest siodłem w 192 wymiarach (ok. 92 wartości własnych z Re > 0 i 98 z Re < 0), jak na seedzie 47; ‖Ω‖/‖S‖ ≈ 0,95
  w blokach 2–5, czyli wir i ścinanie są porównywalne.
- Narzędzie vs tło w bloku 4: ‖N‖/‖J‖ 0,905 vs 0,917, ‖Ω‖/‖S‖ 0,934 vs 0,957, tr J −1,42 vs −1,73. Różnice małe, o losowym znaku
  między seedami (P3). Zmierzone na seedzie 47 „wokół narzędzia bardziej ściska” nie jest ogólne.
- MLP bloku 1 ma tr J = +13,8: silnie rozszerza (stosunek odległości 1,06; jego wyłączenie obniża IoU z 0,80 do 0,24, § 6). Blok 6
  najbardziej kurczy (tr J −2,64, 103 λ z Re < 0).

## 4. Teza B: uwaga ściska, MLP rozdziela, trening przenosi ściskanie

Stosunek median odległości par na sferze D_u po półkroku (k65640, mediany 24 seedów; k0 w nawiasie):

| blok | uwaga | MLP |
|---|---:|---:|
| 1 | 0,966 (0,908) | 1,058 (0,999) |
| 2 | 0,965 (0,918) | 0,949 (1,000) |
| 3 | 0,949 (0,938) | 0,944 (1,001) |
| 4 | 0,964 (0,933) | 0,898 (1,001) |
| 5 | 0,984 (0,945) | 0,906 (0,997) |
| 6 | 1,030 (0,954) | 1,102 (1,003) |

- Przy k0 ściska wyłącznie uwaga (0,91–0,95), MLP jest neutralny (1,00). Po treningu uwaga ściska słabiej (0,95–0,98), MLP bloków
  2–5 ściska (0,90–0,95), MLP 1 i 6 oraz uwaga 6 rozszerzają. Iloczyny: uwaga 0,66 → 0,85, MLP 1,00 → 0,83 (P8 24/24).
- Udział styczny ruchu: uwaga 0,993 / 0,983 / 0,972 / 0,972 / 0,976 / 0,914 (bloki 1–6), MLP 0,84 / 0,84 / 0,89 / 0,88 / 0,91 / 0,82.
  Uwaga porusza tokeny po sferze; blok 6 wyłamuje się (P5), tam zarówno uwaga, jak i MLP zmieniają normę przed głową.
- Parametr porządku r rośnie przez sieć 0,59 → 0,85 (stan 11), ranga efektywna PR_u spada 14,7 → 6,2: chmura się zagęszcza i
  spłaszcza (konsensus), a jednocześnie separacja klas rośnie 0,91 → 2,37 (stan 11); skoki po MLP bloków 2, 4 i 5
  (1,22 → 1,28, 1,65 → 2,14, 2,16 → 2,35), po uwadze bloku 4 nic (1,65 → 1,65). Blok 6 obniża separację (2,37 → 2,14).
- Σ Δsep: uwaga 0,24 (0,12–0,42), MLP 1,03 (0,69–1,35). P6 24/24.

## 5. Separacja a IoU: rozbiór konfundatora (uwaga po fakcie)

P7a liczyła Spearmana po parach (checkpoint, klatka), 12 × 472 punktów. ρ = 0,34–0,48 (mediana 0,42), poniżej progu 0,5. Rozbiór:

- W obrębie jednego seedu mediana sep₁₂ rośnie 1,13 → 2,33 i mediana IoU 0 → 0,80 monotonicznie z treningiem; ρ po wszystkich
  checkpointach mierzy tę wspólną trajektorię.
- Per klatka, tylko przy k65640: ρ = −0,16 … +0,11 (mediana −0,035, 24 seedy); w ablacjach −0,07 … +0,14; w shuffled_labels +0,24.
  Separacja klas nie przewiduje, która klatka zostanie dobrze zlokalizowana.
- sep₁₂ zależy od liczby tokenów narzędzia (ρ = −0,71: duże narzędzie, mała separacja na sferze), a IoU od niej rośnie (ρ = +0,36).
  Dodatnie ρ w shuffled_labels (model bez detekcji) pokazuje, że korelacje tego rzędu mogą być czysto geometryczne.
- P7b (ρ = 0,50 między seedami, 24 punkty) pozostaje jedynym dodatnim wynikiem, słabym przy tej liczbie punktów.

Wniosek: separacja klas w stanie końcowym jest miarą postępu treningu, nie jakości detekcji klatki. P7a odrzucona z powodu, nie progiem.

## 6. Teza C: kolejność uwaga → MLP

- Defekt komutatora ‖S−P‖/‖S−X‖ (mediany po 24 seedach, bloki 1–6): 0,217 / 0,277 / 0,359 / 0,291 / 0,233 / 0,252. Szczyt w bloku 3
  (13 seedów), nie 4–5 (10 seedów). ‖S−R‖/‖S−X‖: 0,35–0,43, płasko.
- Skutek dla wyjścia (IoU ramki szczytu po zamianie jednego bloku; mediana IoU pełnej sieci 0,801): odwrócona kolejność R 0,76–0,80,
  równolegle P 0,78–0,80 (prawie bez straty), tylko uwaga A: blok 1 **0,24**, bloki 2–6 0,75–0,80; tylko MLP M: 0,73–0,79.
  Kolejność w żadnym bloku nie decyduje; decyduje MLP bloku 1.
- Ablacja parallel: AP 0,869–0,895 wobec base 0,864–0,888; separację budują uwaga (+0,44 … +0,67) i MLP (+0,27 … +0,50) po równo,
  nienormalność bloku 4 0,92–0,93 jak w base. Teza C odrzucona: defekt komutatora jest cechą wyuczoną przy sekwencyjnym bloku, bez
  znaczenia dla zadania (zgodnie z uwagą do A2 w predykcjach).

## 7. Ablacje: kompensacja funkcjonalna

| wariant | AP@0,5 (s3047 / s4047 / s5047) | ‖N‖/‖J‖ bloku 4 | iloczyn uwagi k0 → kk | iloczyn MLP k0 → kk | Σ Δsep uwaga / MLP | uwaga styczna, blok 6 |
|---|---|---|---|---|---|---|
| base | 0,888 / 0,885 / 0,864 | 0,93 / 0,93 / 0,90 | 0,66→0,81 / 0,62→0,89 / 0,65→0,84 | 1,01→0,86 / 1,02→0,77 / 0,99→0,90 | +0,25/+1,20, +0,44/+0,91, +0,10/+0,91 | 0,85–0,89 |
| parallel | 0,895 / 0,869 / 0,873 | 0,93 / 0,92 / 0,92 | →0,76 / →0,82 / →0,79 | →1,01 / →0,94 / →0,95 | +0,44/+0,50, +0,67/+0,27, +0,58/+0,44 | 0,92–0,97 |
| frozen_mlp | 0,866 / 0,857 / 0,884 | 0,76 (bez zmian) | →0,78 / →0,86 / →0,78 | →1,02 / →1,00 / →1,03 | +0,80/−0,06, +0,42/+0,01, +0,73/−0,06 | 0,95–0,96 |
| frozen_attn | 0,861 / 0,861 / 0,786 | 0,95 / 0,97 / 0,95 | →0,97 / →0,94 / →0,92 | →0,73 / →0,87 / →0,85 | +0,07/+0,85, +0,03/+0,65, +0,05/+0,52 | 0,98–1,00 |
| no_wd | 0,886 / 0,873 / 0,865 | 0,92 / 0,93 / 0,90 | →0,71 / →0,77 / →0,70 | →0,89 / →0,88 / →0,96 | +0,24/+1,00, +0,27/+0,98, +0,15/+0,96 | 0,87–0,93 |
| shuffled_labels | 0,005 / 0,005 / 0,009 | 0,92 / 0,96 / 0,94 | →1,04 / →1,03 / →1,02 | →0,75 / →0,73 / →0,78 | −0,02/+0,28, −0,03/+0,24, −0,03/+0,23 | 0,94–0,98 |

- **frozen_mlp.** Losowy, zamrożony MLP (i norm2) we wszystkich blokach kosztuje ≤ 0,03 AP. Separację klas w całości przejmuje uwaga
  (Σ Δsep 0,42–0,80 wobec 0,10–0,44 w base), MLP nie rozdziela (−0,06 … +0,01). Nienormalność zostaje na poziomie losowym (0,76), więc
  nienormalne ścinanie w MLP nie jest potrzebne do detekcji.
- **frozen_attn.** Zamrożona uwaga (i norm1) kosztuje 0,03–0,08 AP. Iloczyn kontrakcji uwagi dochodzi do 0,92–0,97: sieć przestawia
  rozkład wejść tak, że zamrożona uwaga prawie nic nie robi (ruch w 100 % styczny, Σ Δsep 0,03–0,07), a MLP ściska mocniej (0,73–0,87)
  i jest bardziej nienormalny (0,95–0,97). Predykcja A4 zakładała, że kontrakcja to własność wag; jest własnością pary (wagi, stan).
- **parallel.** Bez straty; obie podsieci dzielą separację. Patrz § 6.
- **no_wd.** Wszystkie miary jak w base. Weight decay nie jest źródłem nienormalności ani podziału pracy (A5 PASS).
- **shuffled_labels.** Sieć zapamiętuje losowe przypisania (AP ≈ 0,005), a mimo to nienormalność bloku 4 rośnie do 0,92–0,96, MLP
  ściska (0,73–0,78), uwaga przestaje ściskać (1,02–1,04), separacja prawdziwych klas rośnie o 0,23–0,28 po MLP. Nienormalność i
  „MLP ściska” są więc ogólnym skutkiem treningu gradientowego tego bloku, nie sygnaturą nauczenia zadania. Zadanie od szumu odróżnia
  wielkość separacji (1,0 vs 0,25) i to, że w szumie uwaga nie ściska.

## 8. Co z tego wynika dla tez

- **A (nienormalne ścinanie):** replikuje się jako opis (P1, P2, A5), ale pada jako wyjaśnienie: nie zależy od zadania (A6), nie jest
  potrzebne (frozen_mlp), nie lokalizuje się w narzędziu (P3).
- **B (podział pracy i przeniesienie ściskania):** replikuje się w pełni opisowo (P4, P6, P8 24/24, P5 w blokach 1–5), ale ablacje
  pokazują kompensację: każda z podsieci może przejąć separację. Związku z IoU per klatka nie ma (P7a, § 5).
- **C (kolejność):** odrzucona (P9, A2, § 6).
- **D (rzut 3D nie klasyfikuje punktów stałych):** nie testowana tu; pozostaje z pomiaru 24.10.

Narracja do artykułu: pre-rejestrowana replikacja geometrii wyuczonego rozwiązania na 24 seedach plus ablacje, które pokazują, że ta
geometria jest jednym z wielu równoważnych rozwiązań (kompensacja funkcjonalna). Negatywne wyniki (P3, P7a, P9, A2–A4, A6) są
częścią wyniku i powinny być raportowane z progami sprzed pomiaru.

## 9. Ograniczenia

- Jeden zbiór (cataract, cystotom), jedna architektura (MałyViT 6 × 192); brak drugiego układu.
- Ablacje po 3 seedy: wystarczają do kierunku, nie do przedziałów ufności. Trening ablacji na A100 nie jest bitowo zgodny z H100 kampanii;
  base w części 2 jest własną kontrolą (A1: P1, P4, P6 odtworzone).
- Jakobiany dla ≤ ~80 tokenów na klatkę na 50 klatkach; miary B i C na 472 klatkach; 12 checkpointów (rzadko po 10. epoce).
- Rozbiór konfundatora w § 5 i interpretacja A4 są uwagami po fakcie, oznaczonymi jako takie.
- Wykresy nie były robione (folder `Wyniki/Wykresy` pusty).
