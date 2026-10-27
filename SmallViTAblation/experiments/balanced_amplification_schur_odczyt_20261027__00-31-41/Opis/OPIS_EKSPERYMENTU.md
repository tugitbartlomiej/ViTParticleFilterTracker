# Balanced amplification w MLP transformera: forma Schura, para osobliwa, odczyt, reflektory

**Data:** 2026-10-27, 00:31 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach trzech układów (bez treningu).
**Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane przed uruchomieniem).

## Cel

Rodzice wykazały, że trening czyni pole MLP nienormalnym (0,76 → 0,92; 24 seedy, CIFAR, DeiT) i że ta nienormalność ma skutek
dynamiczny (stała Kreissa rośnie, najgorszy kierunek wzmacniany 2–4,5×, zgodnie z σ_max co do 1 %). Literatura (Undermind 26.10, zapis
w rodzicu `wzmocnienie_przejsciowe_pseudospektrum/Wyniki/Dane/literatura_undermind.md`) wskazuje dwa odpowiedniki: **balanced
amplification** w korze (Murphy & Miller 2009: ukryty łańcuch feedforward w formie Schura macierzy połączeń E/I, stabilne widmo,
selektywne wzmocnienie wzorów czytanych dalej; Hennequin 2012, 2014) i **nienormalny wzmacniacz fotoniczny** (Makris 2014: ośrodek
stratny wzmacnia sygnał w kształcie prawego wektora osobliwego; Midya 2022: wzmocnienie kierunkowe, ω > 0, stała Kreissa). Ten eksperyment
sprawdza, czy jakobian MLP ma tę strukturę, i skąd nienormalność pochodzi (wagi czy bramkowanie GELU), oraz jak zależy od liczby
„żywych reflektorów” (neuronów w zakresie liniowym GELU), co wiąże pomiar z obrazem laserów aplikacji „Lasery i koloidy”.

## Rodzice i źródła

- `experiments/wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11`: klasa `Uklad` (modele, dane, tokeny), wynik W/F jako motywacja;
  kod importowany bez zmian.
- `experiments/mosty_fizyki_24seedy_ablacje_20261026__13-04-31` i `kompensacja_lodyga_i_cifar_20261026__19-01-59`: jakobiany, modele,
  dane (import przez rodzica). Checkpointy: kampania kompensacji seedy 3047/4047/5047 (k0, 3000, 65 640); biegi CIFAR
  `maly_s47_obciecie`, `s47_obciecie` (k0, 3000, 93 780).
- Teoria: Murphy & Miller, Neuron 2009; Hennequin, Vogels, Gerstner, PRE 2012 i Neuron 2014; Makris, Ge, Türeci, PRX 2014; Midya, PRA 2022;
  Trefethen & Embree 2005; Henrici 1962.

## Procedura

`Kod_eksperymentu/balanced.slurm` (tablica 5 zadań, 1 × A100 + 8 CPU): `balanced_eden.py --uklad {detekcja, cifar_maly, cifar_deit}`.
Klatki jak w rodzicu (detekcja 50 diagnostycznych, tokeny: narzędzia + 20 tła; CIFAR 50 obrazów, 16 tokenów). Dla każdego bloku i tokenu
J jak w rodzicach, A = I + J.

- **S (pula procesów):** forma Schura bez scipy (eig → QR wektorów własnych → T = RΛR⁻¹), N = T − diag T; c5 = udział 5 największych
  σ(N)² w ‖N‖², PR_N = ranga efektywna N; para osobliwa A: σ₁, u₁, v₁; |⟨v₁, u₁⟩|; zawór ‖A u₁‖/σ₁; ρ, G₁ = σ₁/ρ, ‖N‖/‖J‖.
- **O (GPU, autograd, 8 tokenów na klatkę):** kierunek odczytu r = gradient wyjścia tokenu po jego stanie po bloku b przez resztę sieci
  (detekcja: logit ciepła tego tokenu; CIFAR: margines logitów; DeiT: przez uwagę do CLS); cos²(u₁, r), cos²(v₁, r), cos²(losowy, r).
- **R (GPU):** per token liczba żywych reflektorów (GELU′ ∈ (0,1; 0,9)), średnia GELU′, liczby nasyconych; per blok: ‖N‖/‖J‖ macierzy
  W₂W₁ i W₂·diag(ḡ)·W₁ (ḡ = średnia GELU′ po tokenach i klatkach), rozkład |cos(w₁ᵢ, w₂ᵢ)| po 768 neuronach (średnia, ważona ḡᵢ, kwantyl 0,9).

Wyjście: `Wyniki/Dane/pomiar/<bieg>/{wyniki.npz, podsumowanie.json, WAGI_SHA256.txt, COMPLETE.json}`.
**Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` → `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`.

## Ograniczenia

- Forma Schura przez rozkład własny zakłada diagonalizowalność J (prawie zawsze spełnione dla macierzy z danych; T jest górnotrójkątne
  z dokładnością numeryczną R⁻¹).
- Odczyt r dla 8 z ~37 (detekcja) lub 16 (CIFAR) tokenów na klatkę; dla DeiT kierunek pośredni (przez uwagę do CLS).
- Analogia do kory dotyczy struktury matematycznej (łańcuch feedforward w formie Schura, zawór, wyrównanie z odczytem), nie danych
  neuronalnych; „to samo zjawisko w mózgu” nie jest tu testowane.
- 5 biegów, 3 checkpointy na bieg.

## Odtwarzanie

```
python experiments/eden.py init balanced_amplification_schur_odczyt_20261027__00-31-41
python experiments/eden.py push balanced_amplification_schur_odczyt_20261027__00-31-41
python experiments/eden.py submit balanced_amplification_schur_odczyt_20261027__00-31-41 --script Kod_eksperymentu/balanced.slurm
python experiments/eden.py watch balanced_amplification_schur_odczyt_20261027__00-31-41 --pull-on-complete
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próba dymna: `--sbatch-arg=--array=0,4 --script-arg=proba`.

## Stan

Zobacz `Wyniki/Dane/Eden/STATUS_EDEN.txt` i koniec tego pliku.

- 2026-10-27 00:40–00:46: próby dymne 1933141 (padły: usunięta linia definiująca EXTRA w SLURM, naprawione) i 1933144_0/_4 (OK, 2 klatki,
  2 checkpointy, Schur 384 macierzy w 2 s na 8 CPU). Pełne zadania 00:47–00:53: 1933146_0…_2 (detekcja, dgx-3, 8 CPU, 24 GB),
  1933147_3/_4 (CIFAR, hopper H100, 8 CPU); wszystkie COMPLETED w 3–5 min. Wyniki pobrane.
- 2026-10-27 00:56: ocena końcowa (`Wyniki/Dane/OCENA_PREDYKCJI.txt`, progi bez zmian; wartości dla bloku środkowego, k0 → koniec):
  **BA1 PASS 5/5**: część nilpotentna N skupia się w kilku modach: c5 0,15 → 0,63–0,74, ranga efektywna PR_N 60–63 → 3,3–8,3
  (macierz Ginibre'a: rozmyta, 62). To jest ukryty łańcuch feedforward o niskiej randze w sensie Murphy–Miller / Goldman.
  **BA2b PASS 5/5**: zawór ‖J u₁‖/σ₁ 0,39–0,45 → 0,18–0,36: to, co pole pisze, jest po treningu czytane przez pole 2× słabiej niż przy
  wagach losowych (kierunkowość rośnie). BA2a 4/5: |⟨v₁,u₁⟩| 0,03–0,08 → 0,13–0,18 (DeiT 0,32 > próg); uwaga po fakcie: przy k0
  rozdzielność jest trywialna (losowe kierunki), trening lekko ją zmniejsza, więc BA2a nie dyskryminuje; dyskryminuje BA2b.
  **BA3a FAIL 2/5, BA3b FAIL 0/5** — wynik odwrotny do predykcji w detekcji: z gradientem odczytu (logit ciepła) wyrównany jest kierunek
  **wejściowy** v₁ (cos² 0,06–0,11 w bloku 4, 30–50× losowe 0,002), a nie wyjściowy u₁ (0,003–0,008, 1–4× losowe). Wzmacniany mod MLP
  **czyta** współrzędną, którą czyta głowa, i pisze w kierunkach do niej prostopadłych. W CIFAR oba małe (0,01, 4–5× losowe), u₁ ≳ v₁.
  Hipoteza po fakcie: w detekcji u₁ może być wyrównane z odczytem ramki (l, t, r, b), który tu nie był mierzony (r = tylko logit ciepła).
  **BA4a FAIL 4/5** (‖N‖/‖J‖ samych wag W₂W₁ 0,73 → 0,85–0,90; MałyViT CIFAR 0,845 tuż pod progiem), **BA4b FAIL 3/5** (W₂·diag(ḡ)·W₁
  zgodne z J w detekcji co do 0,02, w CIFAR różnica 0,05–0,06). Nienormalność siedzi głównie w wagach (0,73 → 0,85–0,90), bramkowanie
  dodaje resztę (J 0,92–0,96).
  **BA5a PASS 5/5**: korelacja (żywe reflektory, G₁) po tokenach 0,41–0,84 na końcu (k0: −0,38…+0,11). **BA5b PASS 3/3**: tokeny narzędzia
  mają mniej żywych reflektorów (73–103 vs 127–162) i średnią GELU′ bliską zeru (0,01–0,04 vs 0,11–0,18): **na narzędziu MLP jest prawie
  wyłączony**, stąd mniejsze σ₁ (1,1–1,9 vs 1,8–4,5) i mniejsze wzmocnienie — wyjaśnienie obserwacji z rodzica.
  **BA6 FAIL 2/5**: |cos(w₁ᵢ, w₂ᵢ)| ważone 0,058 (losowe) → 0,10–0,24 (2–4× losowe; kwantyl 0,9: 0,25–0,44, ostatni blok DeiT 0,60).
  Reflektory częściowo **stają się wzajemne** (piszą w kierunku, z którego czytają: składowa normalna, samowzmacniająca, zgodna
  z wcześniej znalezionymi parami przeciwstawnymi), a mimo to nienormalność całości rośnie. Oba zjawiska współistnieją: mniejszość
  reflektorów wzajemnych, większość tworzy łańcuch feedforward.
- Opisowo: liczba żywych reflektorów spada w treningu z ~510 (k0, połowa z 768… DeiT 733) do 90–200 (detekcja), 30–140 (MałyViT CIFAR:
  bloki 2–5 prawie całkiem nasycone, GELU′ ≈ 0), 120–680 (DeiT: bloki środkowe pozostają żywe, pierwszy i ostatni rzadkie). To jest
  „lazy neuron” (Li i in., ICLR 2023) widziane przez pochodną GELU; sprzęga się z wzmocnieniem przez BA5a.
- Interpretacja: struktura balanced amplification jest **potwierdzona w części algebraicznej** (BA1, BA2b: łańcuch Schura o niskiej randze,
  zawór kierunkowy) i **odwrócona w części funkcjonalnej** (BA3: mod wzmacniany czyta kierunek odczytu, nie pisze w nim). W języku kory:
  MLP wzmacnia różnicowo to, co reszta sieci czyta, i przekierowuje to na nowe współrzędne; w języku fotoniki: optymalne wzbudzenie
  wzmacniacza leży w kierunku odczytu. Nienormalność siedzi w wagach; bramkowanie (liczba żywych reflektorów) steruje wielkością
  wzmocnienia per token i wyłącza MLP na tokenach narzędzia.
- 2026-10-27 01:21–01:50: **wersja 2** (BA3c / BA3c′ / BA3d dopisane do predykcji przed pomiarem; kod `balanced_eden.py` rozszerzony
  o energię u₁/v₁ w podprzestrzeni odczytu głowy (QR gradientów wszystkich wyjść: detekcja 5, w tym 4 ramki; CIFAR 10 logitów) oraz
  o wyrównanie między blokami cos²(u₁ᵇ, v₁ᵇ⁺¹), cos²(v₁ᵇ, v₁ᵇ⁺¹), cos²(u₁ᵇ, u₁ᵇ⁺¹) dla tego samego tokenu). Wyniki v1 przeniesione do
  `Wyniki/Dane/pomiar_v1/`. Zadania 1933157_0…_2 (detekcja, dgx-3, 8 CPU; 6–8 min) i 1933161_3/_4 (CIFAR, dgx-1, 6 CPU, przekierowane
  z hoppera; 5–10 min). BA1–BA6 odtworzone co do wartości (ten sam kod). Ocena uzupełnienia (`OCENA_PREDYKCJI.txt`):
  **BA3c FAIL 0/3** — w detekcji wyjście u₁ **nie** leży w podprzestrzeni odczytu ramki: E(u₁ | ramka) 0,021–0,039 w bloku 4
  (losowe 4/192 = 0,021), E(u₁ | 5 wyjść) 0,030–0,046 (losowe 0,022–0,023); tylko w ostatnim bloku u₁ wchodzi w podprzestrzeń głowy (0,10–0,14, 4–6×).
  Wejście v₁ natomiast leży w niej wyraźnie: E(v₁ | 5 wyjść) 0,14–0,17 w bloku 4, 0,21–0,26 w bloku 6 (6–10× losowe), w tym ramka
  0,05–0,12. Wzmacniany mod czyta całą współrzędną odczytu (ciepło i ramkę), nie tylko logit ciepła.
  **BA3c′ PASS 2/2** — w CIFAR wyjście u₁ leży w podprzestrzeni logitów: E(u₁ | 10 logitów) 0,17–0,20 w bloku środkowym (losowe 0,048–0,050,
  3,5–4×), v₁ podobnie (0,14–0,21); w ostatnim bloku MałegoViT E(u₁ | logity) = 0,82 — MLP pisze prawie wyłącznie do logitów. W DeiT
  0,16–0,24 we wszystkich blokach.
  **BA3d FAIL 3/5** — cos²(u₁ᵇ, v₁ᵇ⁺¹) 0,008–0,021 (1,6–4× losowe 0,0052; detekcja s3047 i s5047 pod progiem 3/192): następny blok
  **nie czyta tego, co ten napisał**. Zamiast tego, nieprzewidziane: kolejne bloki **czytają ten sam kierunek** — cos²(v₁ᵇ, v₁ᵇ⁺¹)
  0,09–0,21 (17–40× losowe; k0: 0,003) — i częściowo **piszą w tym samym kierunku** — cos²(u₁ᵇ, u₁ᵇ⁺¹) 0,04–0,09 (8–18× losowe).
- Interpretacja v2 (odpowiedź na pytanie „dokąd pisze wzmacniany mod”): zgodnie z zapisaną z góry regułą, w detekcji **oba nie zaszły**,
  więc odwrócenie BA3 jest realne — mod wzmacniany przez MLP czyta współrzędną odczytu (tę samą, którą czyta głowa i którą czytają MLP
  wszystkich bloków) i pisze w kierunkach prostopadłych, których nie czyta ani głowa (poza ostatnim blokiem), ani pole MLP następnego
  bloku; kierunki zapisu są częściowo wspólne dla bloków. Struktura jest więc **równoległa, nie łańcuchowa**: nie „czytaj tu → pisz tam →
  czytaj tam” (feedforward Murphy–Miller między blokami), lecz „wszystkie bloki czytają jedną współrzędną (czy to narzędzie) i piszą
  do wspólnej podprzestrzeni, której odbiorca nie został zmierzony”. Kandydat na odbiorcę: wejście **uwagi** kolejnych bloków (q/k/v),
  nieobjęte tym pomiarem — otwarte, do zmierzenia (projekcja u₁ na wiodące prawe kierunki osobliwe W_qkv·J_LN następnego bloku).
  W klasyfikacji (CIFAR) jest inaczej: u₁ i v₁ leżą w podprzestrzeni logitów, a następny blok czyta zapis (BA3d 2/2) — tam MLP
  pisze wprost do odczytu i łańcuch między blokami istnieje słabo. Różnica wiąże się z odczytem: w detekcji odczyt jest per token
  (ciepło, ramka), w CIFAR globalny (średnia / CLS), więc każdy token może pisać do logitów. Łączny obraz dla artykułu: łańcuch Schura
  o niskiej randze i zawór (BA1, BA2b) są uniwersalne; **kierunek** wyrównania z zadaniem zależy od typu odczytu.
- 2026-10-27 01:55: `eden.py push --update-code` (opis, kod oceny) i `pull` — komplet plików pobrany, SHA256 zweryfikowane
  (`Wyniki/Dane/Eden/STATUS_EDEN.txt`). Stan mechanizmu pozostaje `BLAD_LICZENIA`, bo wśród zarejestrowanych zadań są padnięta próba
  dymna 1933141 (błąd EXTRA, naprawiony) i 1933158 anulowane na polecenie użytkownika (przekierowanie z hoppera na dgx-1); narzędzie nie
  ma opcji oznaczenia ich jako zastąpionych. Wszystkie 5 zadań właściwych (1933157_0…_2, 1933161_3/_4) COMPLETED 0:0 z `COMPLETE.json`.
- 2026-10-27 04:55 (poprawka po fakcie, przy pisaniu artykułu): zakresy BA3c / BA3c′ w punkcie wersji 2 poprawione według
  `Wyniki/Dane/OCENA_PREDYKCJI.json` (wcześniej odczytane z wydruku per blok: ramka 0,017–0,042 → 0,021–0,039; 5 wyjść 0,03–0,07 → 0,030–0,046;
  logity CIFAR 0,20–0,21 → 0,17–0,20; „blok 5” → „blok 6”). Werdykty PASS/FAIL bez zmian.
