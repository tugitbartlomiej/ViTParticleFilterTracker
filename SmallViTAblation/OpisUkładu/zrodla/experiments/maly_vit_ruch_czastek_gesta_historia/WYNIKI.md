# Wynik: ruch tokenów narzędzia w gęstej historii uczenia

Data: 2026-10-01. Eden, zadanie 1818732 (A100, `dgx-4`), 15 min 25 s, w tym uczenie 543 s. Próba kodu: 1818729.
Plan: [../maly_vit_ruch_czastek_maska_1px/PLAN.md](../maly_vit_ruch_czastek_maska_1px/PLAN.md). Różnice względem planu: [README.md](README.md).

## Werdykt

Po pierwszych ~100 krokach tokeny narzędzia ruszają się **mocniej** niż odległe tło, ale **nie w swoistym kierunku**.
Kierunki SVD dopasowane na filmach uczących prawie nie wygrywają z losowymi kierunkami tej samej liczby i nie
przewyższają samej normy ruchu. Według kryterium planu ruch jest więc nieswoisty co do kierunku.

Wspólny kierunek ruchu narzędzia istnieje tylko na początku uczenia (kroki ~20–100) i potem znika.
Wczorajszy sygnał z kroków 60 i 120 (AUC 0,67) jest prawdziwy i powtarza się w nowym biegu (0,68), ale dotyczy
tylko tej wczesnej fazy.

## Aparat

- Uczenie od zera, ziarno 47, przepis wspólnej rozgrzewki pętli małego ViT, 1000 aktualizacji.
  mAP@0,5 próbki val: 0,025 (krok 100), 0,097 (300), 0,24 (500), 0,35 (900), 0,31 (1000).
- Tokeny 12 pełnych klatek 480 × 320 po każdej aktualizacji: `LATENT_GESTY.npy` (1001, 12, 600, 768) float32,
  22,1 GB, tryb eval, fp32 bez TF32. Powtórzony przebieg w kroku 0 daje błąd 0,0.
- Podział po filmach jak w pomiarze lokalnym: 8 filmów do dopasowania, 4 odłożone (obrazy 2, 5, 6, 10).
- Względna zmiana wszystkich tokenów w jednym kroku (mediana): 0,023 w krokach 1–10, 0,047 w 10–100,
  0,072 w 100–300, 0,066 w 300–1000. Ruch nie wygasa do kroku 1000.

## Liczby (ROC AUC narzędzie/tło na odłożonych filmach, średnia po przedziałach i obrazach)

0,5 oznacza brak rozdzielenia. Maska od jednego piksela.

| Wynik | co 1 krok (1001 stanów) | co 10 kroków (101) | kroki 60 i 120 | lokalnie 60 i 120 (2026-09-30) |
|---|---|---|---|---|
| Norma całego ruchu | 0,711 | 0,685 | 0,505 | 0,506 |
| Norma po odjęciu dryfu tła | **0,761** | **0,740** | 0,594 | 0,567 |
| Średni kierunek (ze znakiem) | 0,519 | 0,522 | **0,684** | **0,669** |
| 4 kierunki SVD / losowe | 0,705 / 0,649 | 0,667 / 0,625 | 0,616 / 0,561 | 0,567 / 0,539 |
| 8 kierunków SVD / losowe | 0,708 / 0,686 | 0,672 / 0,661 | 0,620 / 0,607 | 0,607 / 0,561 |
| 32 kierunki SVD / losowe | 0,730 / 0,733 | 0,700 / 0,705 | 0,625 / 0,621 | 0,606 / 0,573 |
| Osi SVD do 90 % energii ruchu | 192 | 202 | 17 | 16 |

Kolumna „kroki 60 i 120” to nowy bieg ograniczony do tych dwóch stanów. Zgadza się z pomiarem lokalnym, choć to
inny przebieg na innym GPU: wspólny kierunek 0,68 wobec 0,67, norma 0,51 wobec 0,51.

## Fazy uczenia (co 1 krok, maska 1 px, średnia AUC przedziałów w fazie)

| Kroki | Norma po odjęciu dryfu | Średni kierunek | 8 SVD | 8 losowych |
|---|---|---|---|---|
| 0–20 | 0,46 | 0,59 | 0,47 | 0,46 |
| 20–100 | 0,62 | 0,62 | 0,63 | 0,58 |
| 100–300 | 0,78 | 0,51 | 0,73 | 0,71 |
| 300–1000 | 0,78 | 0,51 | 0,72 | 0,70 |

Wykres: [wyniki/pelny/analiza_co1/AUC_W_CZASIE.png](wyniki/pelny/analiza_co1/AUC_W_CZASIE.png)
(fazy w `FAZY.json` obok). Średni kierunek ma szczyt ~0,75 około kroku 30–40, a po kroku 100 waha się wokół 0,5.
Po obrazach testowych (cały bieg, co 1 krok): surowa norma ruchu rozdziela narzędzie od tła w obrazach 5, 6 i 10
(0,76–0,81), a w obrazie 2 nie (0,52; po odjęciu dryfu 0,64). Obraz 2 ma największą maskę, 31 tokenów.

Na [wyniki/pelny/analiza_co1/RUCH_TOKENOW.png](wyniki/pelny/analiza_co1/RUCH_TOKENOW.png) osobne trajektorie tokenów
obrazu 2 w płaszczyźnie dwóch pierwszych osi błądzą tam i z powrotem, bez wyraźnego kierunku. Panel „AUC w czasie”
tego rysunku ma sztuczne spadki na obu brzegach (wygładzanie z dopełnieniem zerami); poprawione w kodzie
i w `AUC_W_CZASIE.png`.

## Czego wynik nie dowodzi

- „Mocniej się rusza” nie znaczy „niesie tożsamość narzędzia”. Tokeny narzędzia leżą na krawędziach i jasnych
  odbiciach; większy ruch może wynikać z samej treści obrazu, a nie z nauki rozpoznawania cystotomu.
- Średni kierunek dopasowano na wszystkich 1000 przedziałach naraz. Kierunek wspólny tylko dla części czasu
  (np. zmieniający się powoli) ta metoda rozmywa. Plan nie przewidywał kierunków zależnych od czasu; to byłby nowy pomiar.
- Jeden bieg, jedno ziarno, 12 obrazów z 12 filmów, 4 obrazy testowe. Kolejne przedziały i tokeny jednego obrazu
  nie są niezależnymi próbami. Porównanie z jedną losową bazą to kontrola orientacyjna, nie test istotności.
- Ruch w trybie eval zawiera zmianę statystyk biegnących BatchNorm łodygi (README.md).
- Prognozy Koopmana ani Langevina nie liczono. Gęsta historia do tego już jest: 1001 stanów co 1 krok.

## Gdzie co leży

Eden: `~/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/` (23 GB): tokeny,
wagi co 10 kroków (101 plików), stan z AdamW co 100 kroków, telemetria sond, trzy analizy.
Kwota projektu po biegu: 175,2 GB z 200 GB.
Lokalnie: `wyniki/pelny/analiza_*`, `wyniki/pelny/dane/` (metadane, oceny, ruch kroków, telemetria sond 54 MB), `wyniki/logi/`.
