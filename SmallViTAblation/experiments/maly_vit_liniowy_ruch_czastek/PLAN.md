# Liniowy opis ruchu każdej cząstki w uczeniu małego ViT

Data: 2026-10-07. Prośba użytkownika: „Musimy zrozumieć, z czego wynika ruch cząsteczki każdej jednej, i potem
zamodelować jej ruch w sposób liniowy: Koopmanem, Langevinem, obserwatorem stanu, nieważne jak. Musimy mieć opis
liniowy i ruch liniowy cząstki, w jakiejś przestrzeni.” Zgoda na trening na Edenie i pobieranie wyników.

Cząstka = token (patch 16 x 16) klatki próbnej; jej położenie = latent z_i(t) w R^768 (MLP ostatniego bloku po
GELU, tryb eval), t = krok uczenia.

## Co już wiadomo (przegląd z 2026-10-07)

- `docs/23`: na średniej po tokenach narzędzia 180 sond jeden krok to dz_(t+1) = a dz_t + b S_t, gdzie S_t to
  przesunięcie latentu od części aktualizacji AdamW z gradientu bieżącej paczki. Wyjaśnia to 85 % ruchu
  (pęd i paczka osobno: 94,9 %). Tego nigdy nie sprawdzono na pojedynczych tokenach, bo gęsta historia
  tokenów nie ma sił paczki.
- Prognoza z samej historii utyka na 20-35 % umiejętności, niezależnie od modelu (Koopman, GLE, EDMD, Kalman,
  OU). Ruch to drgania wokół wolno płynącego środka, a pchnięcie paczki jest w 85-89 % nową informacją.

## Etap 1 (lokalnie, gotowa gęsta historia 1001 kroków; wyniki w `wyniki/`)

Skrypty w `kod/`, uruchamiane z katalogu `app` aplikacji:

1. `mody_czasowe.py`: wspólne mody czasowe wszystkich cząstek klatki, x_t = m + sum_k a_k(t) C_k, i liniowy układ
   a(t+1) = A a(t) + b lr(t) + c.
2. `struktura_szumu.py`: korelacja kolejnych przyrostów, MSD(h), autokorelacja reszty od dryfu.
3. `oscylator.py`: dyskretny Langevin z bezwładnością dla każdej cząstki osobno,
   v_(t+1) = beta_i v_t − kappa_i (z_t − m_t), z_(t+1) = z_t + v_(t+1), m_t = średnia ostatnich 50 kroków
   (tylko przeszłość); porównanie z persystencją, stałą prędkością, samym pędem i wspólnym Koopmanem PCA-32.

## Etap 2 (Eden, zadania 1823538 próba i 1823539 pełny)

`kod_eden/bieg_sily.py`: ten sam przepis uczenia co gęsta historia (ziarno 47, 1000 kroków). W każdym kroku, dla
3 klatek (2, 3, 6) i wszystkich 600 tokenów, przesunięcie cząstki od każdej przyczyny osobno (podstawienie wag):
nowa paczka S_nowa, pęd AdamW S_ped, zanik wag S_zan, statystyki BatchNorm S_buf. Co 10 kroków to samo dla
13 grup wag. Około 17 GB float16.

Model liniowy do sprawdzenia na każdej cząstce (stan x_i = (z_i, v_i), wejścia u = siły):

    v_i(t+1) = beta_i v_i(t) − kappa_i (z_i(t) − m_i(t)) + b_i^T [S_nowa, S_buf, S_zan](t) + e_i(t)
    z_i(t+1) = z_i(t) + v_i(t+1)

Jest to liniowy układ stanowy z wejściem, czyli dyskretne równanie Langevina z bezwładnością, sprężyną i
wymuszeniem. Rozstrzygnie dwie rzeczy:
- **Wyjaśnienie** (wolno używać sił bieżącego kroku, docs/23): jaki procent ruchu każdej cząstki opisuje model
  liniowy. Rozkład na składniki pokaże, z czego wynika ruch (pęd, paczka, BatchNorm, zanik, grupy wag).
- **Prognoza** (bez przyszłych paczek): to samo bez wejścia, czyli oscylator. To jest górna granica przewidywania
  z samego stanu cząstki.

Co by obaliło liniowość: duża reszta liniowości (dz − suma sił) albo słaby model z wejściem na tokenach, gdy na
średniej działał (85 %).

## Etap 3 (po pobraniu)

- Dopasowanie modelu na kroki 300-700, test 700-1000, osobno dla każdego tokenu, mediana po narzędziu i tle.
- Wspólne parametry (beta, kappa, b) dla wszystkich cząstek i porównanie z parametrami osobnymi dla każdej.
- Widmo układu (tłumienie i okres drgań) i jego zgodność z beta1 = 0,9 Adama.
- Pokazanie w aplikacji: zakładka „Uczenie ViT”, strzałki sił na cząstce, model liniowy obok rzeczywistego ruchu.

## Etap 4: ulepszenie uczenia tym, co wiemy o ruchu cząstek (prośba użytkownika 2026-10-07, do zaplanowania)

„Potem trzeba będzie znaleźć sposób na ulepszenie uczenia tym, co mamy, sieci neuronowej.” Kandydaci wynikający
wprost z modelu liniowego (każdy wymaga osobnego planu i zgody przed treningiem):
1. **Wygaszenie drgań.** Ruch cząstki to tłumiony oscylator wokół wolnego środka m_t. Środek jest lepszym stanem
   niż bieżący punkt drgań: średnia krocząca wag (EMA, SWA) albo dobór beta1 / lr tak, by tłumienie było krytyczne
   (z beta i kappa zmierzonych na cząstkach). Sprawdzian: mAP sieci z wag uśrednionych wobec bieżących.
2. **Siła BatchNorm.** Na początku uczenia zmiana statystyk BatchNorm łodygi (S_buf) przesuwa cząstki bardziej niż
   gradient paczki (próba: |S_buf| 3,6 wobec |S_nowa| 1,7 w kroku 0). Kontrola: łodyga z GroupNorm albo zamrożone
   statystyki po rozgrzewce.
3. **Paczki z celem.** Siła S_nowa pokazuje, które paczki pchają cząstki narzędzia w stronę lepszej detekcji.
   Dobór klatek do paczki według przewidywanej siły na cząstki narzędzia (pętla doboru danych projektu).
4. **Sterowanie lr.** Z modelu stanowego z wejściem (lr jako sterowanie) dobór harmonogramu lr, który prowadzi
   cząstki do środka szybciej i z mniejszymi drganiami.
