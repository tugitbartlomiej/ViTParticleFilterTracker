# Wyniki: pętla małego ViT ze Stable Diffusion i równaniem ruchu latentu

## Kampania wersji 4.3 na Edenie (2026-10-01, zadanie 1818736, 10 ziaren 47-56)

Metoda użytkownika krok po kroku: `../../docs/26_PIPELINE_TOZSAMOSC_PROGNOZA_SD.md`. 14 obrotów × 500 kroków po
rozgrzewce 21 epok (z wersji 2), 134 klatki na obrót w każdym ramieniu (równy budżet), listy DINOv2 180 (wydłużane, gdy się
wyczerpią). Analiza `analiza43.py`, wyniki `wyniki_eden43/WYNIKI_EDEN43.json` i `WYNIKI_EDEN43.png`; kopie OBROT.json na
Edenie w `~/smallvit_runs/2026-10-01_petla43_sd_prowadzenie/wyniki/petla43b_s*`.

| ramię | pole pod krzywą mAP@0,5 | mAP@0,5 na końcu | najlepsze |
|---|---|---|---|
| metoda (Langevin z jądrem + SD prowadzone) | **0,4883** | **0,566** | 0,597 |
| metoda_kl3 (Koopman/Langevin w. 3 + SD prowadzone) | 0,4874 | 0,557 | 0,590 |
| sd_bez_prognozy (SD prowadzone do obecnego latentu) | 0,4871 | 0,554 | 0,588 |
| losowe | 0,4720 | 0,544 | 0,572 |
| kolejne (klatki po kolei z filmów) | 0,4646 | 0,519 | 0,560 |
| bez dokładania | 0,4219 | 0,451 | 0,497 |

Porównania sparowane po ziarnach (pole pod krzywą = szybkość uczenia):

| kontrast | różnica | wygrane | Wilcoxon p |
|---|---|---|---|
| metoda − losowe | +0,0163 | 10/10 | 0,002 |
| metoda − kolejne | +0,0237 | 10/10 | 0,002 |
| metoda − sd_bez_prognozy | +0,0012 | 5/10 | 1,0 |
| metoda − metoda_kl3 | +0,0008 | 6/10 | 0,77 |
| sd_bez_prognozy − losowe | +0,0150 | 9/10 | 0,006 |
| metoda − losowe, mAP na końcu | +0,022 | 7/10 | 0,13 |
| metoda − kolejne, mAP na końcu | +0,048 | 10/10 | 0,002 |

Kryteria ustalone przed wynikiem: **K1 spełnione** (metoda > losowe w 10/10, p = 0,002), **K2 spełnione** (metoda > kolejne
w 10/10), **K3 niespełnione** (metoda ≈ sd_bez_prognozy, 5/10).

Pomiary pomocnicze (mediana po wszystkich obrotach i ziarnach):
- trafność prognozy siły tożsamości na 500 kroków: Langevin z jądrem pamięci 28,7 %, Koopman/Langevin wersji 3 16,2 %;
- pokrycie przewidzianego latentu przez obrazy SD: tożsamość 88,5 %, cały latent 88,8 %; obu ≥ 90 % w 24 % obrazów
  (w teście lokalnym na 3 kotwicach było 93,5 i 91,9 %).

### Werdykt

1. Pipeline (tożsamość → kotwice → SD prowadzone latentem → DINOv2 → podobne prawdziwe klatki) uczy szybciej niż ta sama
   liczba klatek losowych i kolejnych: we wszystkich 10 ziarnach, p = 0,002. Na końcu uczenia przewaga nad losowymi
   (+0,022 mAP) nie jest istotna (7/10), nad kolejnymi jest (+0,048, 10/10).
2. Zysk daje część „SD + wyszukanie”, nie prognoza: ramię z SD prowadzonym do obecnego latentu (Δs = 0) uczy się tak samo
   jak metoda (5/10) i samo bije losowe klatki (9/10, p = 0,006).
3. Nowe równanie Langevina z jądrem pamięci przewiduje siłę tożsamości na 500 kroków trafniej niż Koopman/Langevin wersji 3
   (28,7 wobec 16,2 %), ale lepsza prognoza nie zmienia doboru klatek (metoda − metoda_kl3: +0,0008, 6/10).
4. Ten sam obraz co w wersji 3 z FastGAN-em (prognoza 0,4432, bez prognozy 0,4428), teraz na 10 ziarnach i z kontrolami
   losową i kolejną.

### Czego to NIE dowodzi

- Jeden mały ViT, jedna pula klatek (Cataract-LMM, cystotom); walidacja, nie test.
- 95 % pewności wykrywania nie osiągnięto: najlepsze mAP@0,5 około 0,60 po 7000 krokach.
- SD prowadzone pokrywa przewidziany latent w medianie w 89 %, nie we wszystkich obrazach ≥ 90 %.
- Brak ramienia „wyszukanie z prawdziwej klatki kotwicy bez SD”, więc nie rozdzielono wkładu SD od samego wyszukiwania
  podobnych klatek.

---

# (historia) Wyniki wersji 4.1-4.2


Plan i różnice wobec wersji 3: `PLAN.md`. Wszystko poniżej to sprawdzenie kodu i diagnoza, nie wynik pętli.

## Diagnoza prognozy wieloetapowej wersji 4.1 (2026-09-30, `diagnoza_prognozy.py`, `wyniki/diagnoza_s47/DIAGNOZA.json`)

- Wektor tożsamości wyznaczony dobrze: AUC 0,98 na odłożonych sondach (narzędzie kontra to samo miejsce bez narzędzia).
- Jeden krok równania trafny; prognoza na 100-500 kroków przy stałej średniej sile 1,5-50 razy gorsza od „braku zmiany”.
- Przyczyna: średnia siła paczek (około 0,0007 na krok) jest około 30 razy mniejsza od losowego rozrzutu sił kolejnych
  paczek, więc siła tożsamości na długim odcinku zachowuje się jak błądzenie losowe.
- Wniosek przyjęty przez użytkownika: „zróbmy przewidywanie na jeden krok do przodu” (wersja 4.2).

## Próba kodu wersji 4.2 (2026-09-30, `wyniki/proba_krok_s47`, log `wyniki/proba_krok_s47.log`)

Skala próby: rozgrzewka 2 × 60 kroków, 2 obroty × 40 kroków, 24 klatki na obrót, 2 kotwice × 4 wersje SD
(migawka `ema_015000`), 300 sekund na RTX 3070. Sprawdza tylko działanie kodu; mAP po 40 krokach nic nie mówi.

| ramię | równanie 1 krok, siła tożsamości (obrót 3 / 4) | sam pęd | przewidziane pchnięcie dodanych klatek (3 / 4) |
|---|---|---|---|
| `sd_prognoza` | 97,9 / 97,5 % | 48 / 30 % | +0,106 / +0,155 |
| `sd_bez_prognozy` | 97,1 / 98,8 % | 38 / 32 % | +0,074 / −0,097 |
| `losowe` | 98,9 / 99,1 % | 50 / 60 % | +0,022 / +0,067 |
| `kolejne` | 97,0 / 98,6 % | 30 / 49 % | +0,056 / −0,035 |
| `bez` | 99,2 / 98,0 % | 73 / 50 % | - |

- Równanie Δz(t+1) = S_pęd + S_paczka na prawdziwych krokach uczenia w pętli: 97-99 % dla siły tożsamości i 99 % dla
  całego latentu (10 kroków na obrót, wczesne uczenie). Na nagraniach z późniejszego uczenia było 95 %.
- Wybór wersji SD równaniem daje zestawy, które według równania mocniej podnoszą tożsamość niż losowe i kolejne klatki.
  To przewidywanie jednego kroku, nie zmierzony skutek całego obrotu.
- W próbie częściej wygrywały wersje z tym samym układem narzędzia (R2) niż z przestawionym (R1).

## Czego próba NIE dowodzi

- Nic o mAP: 40 kroków na obrót i jedno ziarno.
- Nic o tym, czy większe przewidziane pchnięcie jednego kroku przekłada się na siłę tożsamości po 500 krokach.
- Umiejętność 97-99 % zmierzona na wczesnym uczeniu (kroki około 120-200); pełne ziarno sprawdzi ją po rozgrzewce 1000 kroków.
