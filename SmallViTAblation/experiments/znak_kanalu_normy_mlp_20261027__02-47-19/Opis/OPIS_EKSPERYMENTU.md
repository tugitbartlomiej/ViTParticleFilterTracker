# Znak kanału normy: czy MLP przy pchnięciu tokenu w stronę odczytu zwiększa czy zmniejsza jego normę

**Data:** 2026-10-27, 02:47 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach trzech układów (bez treningu).
**Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane 02:55, przed kodem, testem lokalnym i jakimkolwiek zadaniem na Edenie).

## Cel

Rodzic `odbiorca_zapisu_mlp_uwaga_20261027__01-55-25` pokazał, że kierunek zapisu u₁ wzmacnianego modu MLP nie ma wyróżnionego
odbiorcy (uwaga, MLP i głowa czytają go 1,3–4,5× ponad losowy), ale w MałymViT 20–26 % jego energii (detekcja; CIFAR 8 %; DeiT 0,7 %
= losowe) leży w jądrze LayerNorm odbiorcy, span{1̂, ê}. Rodzic mierzył obie składowe łącznie. Są funkcjonalnie różne:

- **średnia** 1̂ = 1/√C: każdy dalszy odbiorca jest za LayerNorm (także głowa), więc dodanie stałej do tokenu nie zmienia niczego —
  kanał „kosza”;
- **radialny** ê (kierunek scentrowanego tokenu): zmienia skalę ρ = ‖P x‖ tokenu; LN ją usuwa, ale każda późniejsza aktualizacja obraca
  token o kąt ∝ ‖Δ‖/ρ, więc ρ działa jak bezwładność tokenu dla całej reszty sieci.

Eksperyment rozstrzyga: (1) radialny czy średnia; (2) znak zmiany normy, gdy token jest pchnięty w stronę odczytu (więcej „narzędzia”
albo poprawnej klasy), osobno narzędzie i tło; (3) czy zmiana normy przeciwdziała skutkowi pchnięcia dla wyjścia (stabilizator, ujemne
sprzężenie; odpowiednik stabilizacji hamowaniem i normalizacji dzielącej w korze) czy go utrwala (zatrzask).

## Hipotezy

- **Stabilizator:** pchnięcie w stronę odczytu zwiększa normę tokenu, a ta zmiana osłabia skutek pchnięcia dla wyjścia (F_rad < 0).
- **Zatrzask:** zmiana normy wzmacnia skutek pchnięcia (F_rad > 0).
- **Kosz:** zapis w jądrze LN to głównie średnia (E_mean ≥ E_rad); kanał normy nieistotny.
- Predykcje R1–R2, Z1–Z2, S1–S2, T1, kontrole K1–K3 z progami i regułą rozstrzygania: `PREDYKCJE_PRZED_POMIAREM.md`.

## Rodzice i źródła

- Rodzic: `experiments/odbiorca_zapisu_mlp_uwaga_20261027__01-55-25` (wynik O8 jako motywacja i kontrola K1; `odbiorca_eden.py`
  importowany przez `importlib` bez zmian: ręczna uwaga i blok w float64, reszta sieci, zgodność z timm).
- Dziadek `balanced_amplification_schur_odczyt_20261027__00-31-41` (definicja J i pary osobliwej; BA5b), pradziadek
  `wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11` (klasa `Uklad`, dane, tokeny; odporność wyjścia).
- Szczegóły, ścieżki i SHA256: `Kod_eksperymentu/pochodzenie.json`; SHA256 wag w `Wyniki/Dane/pomiar/<bieg>/WAGI_SHA256.txt`.

## Model, checkpointy, dane

Jak w rodzicu: MałyViT detekcja, seedy 3047 / 4047 / 5047 kampanii kompensacji (kroki 0, 3000, 65 640; 50 klatek diagnostycznych);
MałyViT CIFAR s47 i DeiT-Tiny CIFAR s47 (kroki 0, 3000, 93 780; 50 obrazów diagnostycznych, 5 na klasę). Tokeny: reguła rodzica
(detekcja: wszystkie narzędzia + 20 tła na klatkę; CIFAR: 16 tokenów obrazu), te same co w rodzicu.

## Procedura i miary

`Kod_eksperymentu/znak_normy.slurm` (tablica 5 zadań, 1 × A100 + 5 CPU) → `znak_normy_eden.py`. Dla bloków b = 0…NB−2, stanu x_a
(wejście MLP) i x_c = x_a + MLP(x_a):

- **L** (wszystkie wybrane tokeny): E_rad i E_mean dla u₁, v₁ i losowego kierunku; opisowo cos rzeczywistej aktualizacji MLP z kierunkiem
  radialnym, wzrost normy przez MLP i przez uwagę, profil normy po blokach.
- **S** (8 tokenów na klatkę; detekcja: do 4 narzędzia, reszta tło): odczyt y (detekcja: logit ciepła tokenu; CIFAR: margines klasy),
  gradienty r_a, r_c (autograd); pchnięcia d ∈ {v₁, r̂_a, losowy} skierowane tak, by zwiększały y; znak radialnej odpowiedzi MLP c_rad,
  elastyczność normy η; liniowy rozkład zmiany y na tor tożsamości, radialny, średniej i prostopadły (suma = 1, kontrola kodu).
- **B** (te same tokeny, δ ∈ {0,03; 0,1} × mediana normy tokenu): pchnięcie jednego tokenu i kontrfakty stanu po bloku b (pełny,
  bez składowej radialnej odpowiedzi MLP, bez średniej, bez prostopadłej, bez całej odpowiedzi MLP), każdy przez resztę sieci.

Wyjście: `Wyniki/Dane/pomiar/<bieg>/{wyniki.npz, podsumowanie.json, WAGI_SHA256.txt, COMPLETE.json}`.
**Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`.

## Ograniczenia

- Pchnięcie jednego tokenu na kopię; skutek przez innych tokenów (uwaga) jest w Δy, ale nie rozdzielony.
- Znak liniowej odpowiedzi jest nieparzysty w d; orientację ustala znak ⟨d, r_a⟩, więc dla kierunków prawie prostopadłych do odczytu
  (losowy) orientacja jest szumem; stąd kierunek losowy służy za odniesienie zerowe.
- Odczyt y to jedno wyjście tokenu (detekcja: ciepło, bez ramki); CIFAR: margines całego obrazu.
- 5 biegów, 3 checkpointy na bieg; 8 tokenów na klatkę do odczytu.

## Odtwarzanie

```
python experiments/eden.py init znak_kanalu_normy_mlp_20261027__02-47-19
python experiments/eden.py push znak_kanalu_normy_mlp_20261027__02-47-19
python experiments/eden.py submit znak_kanalu_normy_mlp_20261027__02-47-19 --script Kod_eksperymentu/znak_normy.slurm --sbatch-arg=--nodelist=dgx-X
python experiments/eden.py pull znak_kanalu_normy_mlp_20261027__02-47-19
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próba dymna: `--sbatch-arg=--array=0,4 --script-arg=proba` (2 klatki, 2 checkpointy, wynik w `Wyniki/Dane/proby_dymne/`).
Test lokalny logiki na losowych wagach (CPU/GPU, poza repozytorium) sprawdzał: zgodność bloku ręcznego z timm, domknięcie rozkładu
liniowego, zerowy kanał średniej, zgodność liniowej i nieliniowej odpowiedzi.

## Stan

- 2026-10-27 02:47–02:54: folder, predykcje (02:49), wpis w indeksie, kod (02:51), test lokalny na losowych wagach (MałyViT detekcja i DeiT CIFAR,
  2 klatki): zgodność ręczna 5·10⁻¹⁶, domknięcie rozkładu liniowego 4·10⁻¹³, |⟨r_c, 1̂⟩|/‖r_c‖ 6·10⁻¹⁷, kanał średniej nieliniowo 2·10⁻¹⁵,
  liniowa vs nieliniowa zmiana y: 0,4–3 % różnicy. Skrypt oceny napisany 02:55, w czasie próby dymnej, przed odczytem jakichkolwiek
  wyników; sprawdzony na wynikach testu lokalnego (liczby bez znaczenia).
- 02:54–02:58: próba dymna 1933189_0 i _4 na dgx-3 (4 min): kontrole jak lokalnie (domknięcie 6·10⁻¹⁴–1·10⁻¹³, kanał średniej 1·10⁻¹⁴).
- 03:35: pełny bieg 1933192_0…_4 (5 CPU, A100): _0, _1 na dgx-4; _2, _3, _4 przestawione z kolejki dgx-4 na dgx-3
  (`scontrol update ReqNodeList`, ten sam numer zadania).
- 03:35–03:43: pełny bieg 1933192_0…_4 COMPLETED 0:0 (detekcja 5–6 min, MałyViT CIFAR 4,6 min, DeiT 8 min). 04:40: pobranie przez
  `eden.py pull` (pierwsza próba: zerwane SSH, druga OK, SHA256 zweryfikowane), ocena `Wyniki/Dane/OCENA_PREDYKCJI.txt` (progi bez zmian;
  koniec treningu, mediana po blokach z median po tokenach; kolejność biegów: detekcja s3047 / s4047 / s5047; MałyViT CIFAR; DeiT):
  **R1 PASS silnie 5/5** — jądro LN niesie **norma**, nie średnia: E_rad(u₁) 0,197 / 0,253 / 0,240; 0,082; 0,0072, E_mean(u₁)
  0,0001–0,001 (poniżej losowego 0,0025). **R2 PASS słabo 4/5** — udział radialny wyuczony: k0 0,001–0,005 → koniec jak wyżej; DeiT 0,007
  (3× losowe, nie spełnia).
  **Z1 FAIL 3/5** — znak spójny tylko w detekcji: c_rad(v₁) −0,23 / −0,25 / −0,24 (66–71 % tokenów ujemnych; losowy kierunek −0,02);
  MałyViT CIFAR −0,05 (52 %), DeiT +0,003. **Z2 FAIL 1/5** — znak **odwrotny** do predykcji: pchnięcie w stronę odczytu **zmniejsza** normę
  tokenu, c_rad(r̂) −0,06 / −0,09 / −0,18 (detekcja; 28–34 % tokenów dodatnich), MałyViT CIFAR −0,02, DeiT +0,002.
  **S1 FAIL 0/5, S2 FAIL 0/5** — kanał normy **nie wpływa istotnie** na odczyt: F_rad(r̂, δ = 0,03) +0,0005 / +0,0005 / +0,0014; 1·10⁻⁵;
  3·10⁻⁶ (δ = 0,1: +0,003…+0,015), wobec F_perp 0,18–0,23 (detekcja), 0,12 (MałyViT CIFAR), 0,011 (DeiT); udział normy w skutku MLP
  0,2–0,8 %. **T1 PASS silnie 3/3** — |η| na narzędziu 0,05–0,11, na tle 0,27–0,30.
  **K1 PASS 5/5** (E_rad + E_mean = 0,201 / 0,253 / 0,240; 0,082; 0,0074 wobec Enull rodzica 0,199 / 0,259 / 0,239; 0,076; 0,0074),
  **K2 PASS 5/5** (|⟨r_c, 1̂⟩|/‖r_c‖ ≤ 1·10⁻¹⁶, kanał średniej nieliniowo ≤ 2·10⁻¹¹), **K3 PASS 5/5** (znak liniowy = nieliniowy).
- Opisowo (koniec, detekcja): rzeczywista aktualizacja MLP **zwiększa** normę tokenu o 16–21 % na blok (mediana; narzędzie 9–13 %, tło
  20–29 %; uwaga 6–7 %), cos(m₀, ê) +0,28–0,41 — MLP „pompuje” token na zewnątrz; blok 0 +72–89 % (MałyViT CIFAR +52 %, DeiT +1750 %,
  tworzenie dużej normy w pierwszym bloku). Odpowiedź na pchnięcie w stronę odczytu ma znak przeciwny: MLP pompuje token **mniej**, gdy
  ten staje się bardziej „narzędziem” (tło: c_rad(v₁) −0,22…−0,51). Rozkład liniowy skutku pchnięcia r̂ dla odczytu: tożsamość 0,77–0,80,
  tor prostopadły MLP 0,19–0,24, norma ≤ 0,001, średnia 0 (detekcja); MałyViT CIFAR 0,90 / 0,10; DeiT 0,99 / 0,014.
- Interpretacja (reguła z góry: **R1 PASS, S2 FAIL → „kanał nieistotny”**): piąta część zapisu wzmacnianego modu w MałymViT to zmiana normy
  tokenu (nie średniej), wyuczona i w detekcji o spójnym znaku (pchnięcie ku narzędziu → mniejsza norma), ale dla odczytu tego tokenu jest
  bez znaczenia (< 1 % skutku). Całe działanie MLP na odczyt idzie torem prostopadłym. Hipoteza „stabilizator przez normalizację dzielącą”
  (S1) i kierunek Z2 odrzucone; wcześniejsze zdanie „MLP reguluje bezwładność tokenu” jest **nieprawdziwe w sensie funkcjonalnym** dla
  własnego odczytu tokenu. Niezmierzone: wpływ zmiany normy na odczyty **innych** tokenów (Δy liczone tylko dla pchniętego tokenu, choć
  przez całą sieć z uwagą) i na odporność na późniejsze zaburzenia.
