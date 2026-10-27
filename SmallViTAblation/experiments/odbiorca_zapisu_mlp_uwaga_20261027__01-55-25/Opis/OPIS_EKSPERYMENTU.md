# Odbiorca zapisu wzmacnianego modu MLP: uwaga (q/k/v), MLP następnego bloku, głowa czy nikt

**Data:** 2026-10-27, 01:55 (Europe/Warsaw). **Typ:** pomiar na istniejących checkpointach trzech układów (bez treningu).
**Predykcje:** `PREDYKCJE_PRZED_POMIAREM.md` (zapisane 02:05, przed testem lokalnym i przed jakimkolwiek zadaniem na Edenie).

## Cel

Rodzic `balanced_amplification_schur_odczyt_20261027__00-31-41` zmierzył jakobian pola MLP J = W₂·diag(GELU′)·W₁·J_LN2 per token
i jego parę osobliwą: u₁ (kierunek, w którym MLP **pisze**) i v₁ (kierunek, który MLP **czyta**). Wynik wersji 2: v₁ jest wyrównane
z gradientem odczytu głowy i wspólne dla kolejnych bloków (cos²(v₁ᵇ, v₁ᵇ⁺¹) 0,09–0,21 przy losowym 0,0052), natomiast u₁ nie leży
w podprzestrzeni odczytu głowy (detekcja: energia 0,02–0,04 przy losowej 0,021) i nie jest czytane przez MLP następnego bloku
(cos²(u₁ᵇ, v₁ᵇ⁺¹) 0,008–0,02). Kierunki zapisu są częściowo wspólne (cos²(u₁ᵇ, u₁ᵇ⁺¹) 0,04–0,09). Rodzic nazwał to strukturą
„równoległą, nie łańcuchową” i wskazał uwagę jako niezmierzonego kandydata na odbiorcę. Ten eksperyment pyta: **kto czyta to, co pisze
wzmacniany mod MLP?**

## Hipotezy

- **H1 (główna):** wejście uwagi kolejnych bloków (q/k/v po LN1), w szczególności tor routingu (q, k), nie tor treści (v).
- **H2:** uwaga czyta u₁ przez v (treść), nie przez q/k.
- **H3:** nikt nie czyta: u₁ rozprasza się w kierunkach o małej wrażliwości reszty sieci (zapis „do kosza”), np. w jądrze LayerNorm.
- Alternatywa sprawdzana tą samą miarą: MLP bloku b+1 czyta u₁ pełną macierzą W₁·J_LN2, choć nie jego jednym kierunkiem v₁ᵇ⁺¹.

Predykcje O1–O8 i kontrola K1 z progami liczbowymi i regułą rozstrzygania: `PREDYKCJE_PRZED_POMIAREM.md`.

## Rodzice i źródła

- Rodzic: `experiments/balanced_amplification_schur_odczyt_20261027__00-31-41`: motywacja (BA3c, BA3d), definicja J, pary osobliwej
  i odczytu głowy; `balanced_eden.py` importowany przez `importlib` (bez kopiowania).
- Dziadkowie: `wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11` (klasa `Uklad`, import przez `importlib`, pod nazwą modułu
  `wzmocnienie_eden`, aby rodzic użył tego samego obiektu); `mosty_fizyki_24seedy_ablacje_20261026__13-04-31` (`pomiar_mosty_eden.py`:
  jakobiany, klatki, maski, model) i `kompensacja_lodyga_i_cifar_20261026__19-01-59` (`pomiar_mosty_cifar.py`, `model_deit_cifar.py`),
  oba przez kopie w folderze `wzmocnienie_przejsciowe` (SHA256 kopii w rodzicu i dziadku identyczne lokalnie i na Edenie, sprawdzone 02:00).
- Powiązania: `Kod_eksperymentu/pochodzenie.json`.

## Model, checkpointy, dane

Jak w rodzicu (bez zmian):
- detekcja MałyViT (192 wymiary, 6 bloków, 3 głowy, gęsta głowa 5 wyjść na token), kampania kompensacji, seedy 3047 / 4047 / 5047:
  `/mnt/evafs/faculty/home/bpiotrowski/smallvit_runs/SmallViTAblation/outputs/kompensacja_funkcjonalna_dgx2_20261017/replications/s<seed>/weights/k<krok>.pt`,
  kroki 0, 3000, 65 640; dane: `.../kompensacja_funkcjonalna_dgx2_20261017/input`, 50 klatek diagnostycznych (rola evaluation),
  tokeny: wszystkie narzędzia + 20 tła na klatkę;
- MałyViT CIFAR-10 s47 (6 bloków, 64 tokeny) i DeiT-Tiny CIFAR-10 s47 (12 bloków, 64 tokeny + CLS):
  `/home2/faculty/bpiotrowski/smallvit_runs/2026-10-13_deit_cifar10/wyniki/{maly_s47_obciecie, s47_obciecie}/wagi/k<krok>.pt`,
  kroki 0, 3000, 93 780; dane `.../dane/cifar10_uint8.npz`, 50 obrazów diagnostycznych (5 na klasę), 16 tokenów obrazu.
- SHA256 każdego użytego checkpointu: `Wyniki/Dane/pomiar/<bieg>/WAGI_SHA256.txt` (liczone przez zadanie). Wznowienie treningu nie
  dotyczy (tylko wagi, tryb eval).

## Procedura

`Kod_eksperymentu/odbiorca.slurm` (tablica 5 zadań, 1 × A100 na dgx-1…dgx-4, partycja short, 6 CPU, 24 GB):
`odbiorca_eden.py --uklad {detekcja, cifar_maly, cifar_deit}`. Cała sieć w float64, uwaga liczona ręcznie z wag timm (bez SDPA),
zgodność z blokiem timm sprawdzana przy każdym checkpoincie (względna różnica < 1e-9, zapis `zgodnosc_reczna`). Para osobliwa J przez
eigh JᵀJ (v₁ = wiodący wektor własny, u₁ = J v₁ / σ₁; równoważne SVD rodzica, sprawdzone lokalnie: cos² = 1). Tokeny: reguła rodzica,
generator ziarnem 2026 od nowa dla każdego checkpointu (te same tokeny we wszystkich checkpointach), kierunki losowe z ziarna 2027.
Dla każdego bloku b = 0…NB−2 i tokenu kierunki d ∈ {u₁ᵇ, v₁ᵇ, losowy, v₁ᵇ⁺¹}:

- **A (liniowo, wszystkie tokeny):** dla odbiorcy c = b+1 i c = b+2: R_q, R_k, R_v = W_q/k/v·J_LN1(x_c) (W_qkv = `attn.qkv.weight`
  576 × 192 podzielone po 192 wiersze), R_att = [R_q; R_k; R_v], R_m = W₁·J_LN2(x_c po uwadze) (768 × 192), opisowo
  R_mg = diag(GELU′)·R_m. Względny zysk g_X(d) = ‖R_X d‖² / mediana ‖R_X r‖² po 64 losowych r; sufit gmax = σ₁(R_X)² / ta mediana;
  energia E_X(d) w 16 wiodących prawych kierunkach osobliwych R_X (losowe 0,083); energia w jądrze J_LN1 (span{1, x̂}, losowe 2/192);
  kontrola zgodności z rodzicem: cos²(u₁ᵇ, v₁ᵇ⁺¹), cos²(v₁ᵇ, v₁ᵇ⁺¹), cos²(u₁ᵇ, u₁ᵇ⁺¹).
- **O (odczyt głowy, 8 tokenów na klatkę):** gradienty wszystkich wyjść tokenu (detekcja 5, CIFAR 10 logitów) po stanie tokenu po bloku b;
  energia d w ich rozpięciu i zysk g_glowa (jak rodzic).
- **B (nieliniowo, 8 tokenów):** δ·s·d dodane do stanu tokenu tuż po MLP bloku b (δ ∈ {0,03; 0,1}, s = mediana normy tokenu w klatce),
  jedna kopia sieci na token i kierunek; przez blok b+1 i resztę sieci: zmiana wyjścia podwarstwy uwagi b+1 dla tego tokenu i dla
  pozostałych, zmiana mapy uwagi (wiersz, kolumna, wiersze innych tokenów), zmiana wyjścia uwagi, gdy zaburzenie wchodzi tylko przez q,
  tylko przez k albo tylko przez v, zmiana wyjścia MLP b+1 (całkowita i tylko przez rezydual), wyjście bloku b+1 i wyjście sieci
  (detekcja: mapa 600 × 5, w tym kanał ciepła i wyjścia własnego tokenu; CIFAR: logity) — dla d ∈ {u₁ᵇ, losowy, v₁ᵇ⁺¹, v₁ᵇ}.
- **C (wrażliwość, 8 tokenów):** ‖J_reszty d‖ przez `torch.func.jvp` od stanu po bloku b do wyjścia sieci, d ∈ {u₁ᵇ, v₁ᵇ, v₁ᵇ⁺¹,
  4 × losowy}.

Wyjście: `Wyniki/Dane/pomiar/<bieg>/{wyniki.npz, podsumowanie.json, WAGI_SHA256.txt, COMPLETE.json}`; próby dymne
`Wyniki/Dane/proby_dymne/`; logi `Wyniki/Dane/logi/`. **Ocena:** `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` →
`Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}` (napisana przed odczytem wyników; sprawdzona na wynikach testu z losowymi wagami: g ≈ 1,
E ≈ 0,08 dla wszystkich kierunków, jak należy).

## Miary i statystyka

M[·] = mediana po blokach b = 0…NB−2 z median po tokenach i klatkach (c = b+1, koniec), stosunki nieliniowe: w bloku stosunek median,
potem mediana po blokach. Próg: ≥ 4/5 biegów (5/5 „silnie”, p = 0,031 test znaku; 4/5 „słabo”), K1 5/5.

## Ograniczenia

- Pomiar liniowy (A) mierzy wrażliwość wejścia q/k/v na zaburzenie, nie to, czy zmiana q/k/v zmienia coś istotnego dalej; to sprawdzają B i C.
- u₁ to kierunek maksymalnego wzmocnienia dla danego tokenu; rzeczywisty zapis MLP jest sumą wielu kierunków (σ₂, σ₃…), tu nieobjętych.
- B i C na 8 tokenach na klatkę; jeden znak u₁ (+); δ = 0,1 może wychodzić poza zakres liniowy.
- g_X jest względne do losowego kierunku w tym samym tokenie; nie mówi, ile informacji przenosi tor, tylko jak mocno jest czytany.
- Kierunki losowe w 192 wymiarach; odniesienia 16/192 i 2/192 są wartościami oczekiwanymi.
- 5 biegów, 3 checkpointy, jeden seed na układ CIFAR.

## Odtwarzanie

```
python experiments/eden.py init odbiorca_zapisu_mlp_uwaga_20261027__01-55-25
python experiments/eden.py push odbiorca_zapisu_mlp_uwaga_20261027__01-55-25
python experiments/eden.py submit odbiorca_zapisu_mlp_uwaga_20261027__01-55-25 --script Kod_eksperymentu/odbiorca.slurm --sbatch-arg=--nodelist=dgx-3
python experiments/eden.py status odbiorca_zapisu_mlp_uwaga_20261027__01-55-25
python experiments/eden.py pull odbiorca_zapisu_mlp_uwaga_20261027__01-55-25
python Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

Próba dymna: `--sbatch-arg=--array=0,4 --script-arg=proba`. Kod wymaga obecności folderów rodzica i dziadka w tym samym `experiments/`.

## Stan

Zobacz `Wyniki/Dane/Eden/STATUS_EDEN.txt` i koniec tego pliku.

- 2026-10-27 01:55–02:05: folder (nazwa nadana przez zlecającego), `eden.py init`, wpis w indeksie (z odnośnikiem w wierszu rodzica),
  predykcje przed pomiarem.
- 02:05–02:15: kod `odbiorca_eden.py`, test lokalny na CPU z losowymi wagami (2 klatki, 2 „checkpointy”, wszystkie trzy układy; pliki
  próbne w katalogu tymczasowym, usunięte): blok ręczny = timm co do 1e-15, R_q i R_m zgodne z jakobianem z autogradu co do 1e-16,
  J_LN·1 = 0, JVP zgodne z różnicą skończoną co do 4e-7, zaburzenie tylko przez q nie zmienia innych tokenów (0,0), para osobliwa
  z eigh JᵀJ zgodna z SVD numpy (cos² = 1); ocena na wynikach testu: g ≈ 1, E ≈ 0,08 (losowe wagi, jak należy). Poprawka przed
  wysłaniem: SVD float64 na GPU (3 s na 37 macierzy) zastąpione eigh JᵀJ; pierwsze uruchomienie testu padło na podwójnym wywołaniu głowy
  w części O (naprawione).
- 02:15–02:21: `push`, próba dymna 1933168_0 (detekcja s3047) i 1933168_4 (DeiT) na dgx-3 (A100 40 GB, 6 CPU): obie zakończone
  02:21, zgodność ręczna 2e-16…1e-15, 2 klatki: detekcja 8–37 s na checkpoint, DeiT 21–46 s.
- 02:22: pełny bieg 1933172_0…_4 na dgx-3 (5 CPU na zadanie zamiast 6: na dgx-3 wolne było 26 rdzeni, dgx-1/2/4 bez wolnego CPU;
  kod jest ograniczony przez GPU).
- 02:22–02:48: zadania 1933172_0…_4 na dgx-3 COMPLETED 0:0: detekcja 13 min 14–18 s, MałyViT CIFAR 10 min 13 s, DeiT 25 min 40 s
  (A ≈ 165 s, O ≈ 30 s, B ≈ 11 s, C ≈ 21 s na checkpoint detekcji; DeiT: A ≈ 225 s, O ≈ 190 s). Zgodność ręczna 2e-16…1e-15 we
  wszystkich checkpointach. 02:49: `push --update-code` (opis), `pull`, SHA256 pobrania zweryfikowane przez mechanizm. SHA256 wszystkich
  15 checkpointów identyczne z `WAGI_SHA256.txt` rodzica.
- 02:50: ocena (`Wyniki/Dane/OCENA_PREDYKCJI.{txt,json}`, progi bez zmian; M[·] = mediana po blokach b < NB−1 z median po tokenach,
  koniec; w nawiasach k0 → koniec, odniesienia: losowy kierunek g ≈ 1,0, energia top-16 0,083, jądro LN 0,007):
  **O1 FAIL 2/5**: g_qk(u₁) = 1,0 → 4,0 / 2,5 / 2,9 (detekcja s3047 / s4047 / s5047), 2,3 (MałyViT CIFAR), 3,6 (DeiT); losowy 1,0;
  v₁ᵇ⁺¹ 3,5–4,4. Uwaga czyta zapis wyraźnie ponad losowy i ponad k0, ale próg 3× przekracza tylko s3047 i DeiT.
  **O2 PASS słabo 4/5**: g_qk(u₁) > g_v(u₁) (detekcja 2,5–4,0 vs 2,1–2,4; MałyViT 2,28 vs 2,08); w DeiT odwrotnie (3,6 vs 4,5).
  W detekcji przewaga pochodzi z **k** (g_k 2,5–4,0), nie z q (1,3–1,7).
  **O3 PASS słabo 4/5**: E_att(u₁) 0,24–0,27 vs E_m(u₁) 0,16–0,21 (detekcja), DeiT 0,40 vs 0,32; MałyViT CIFAR 0,201 vs 0,205 (remis).
  **O4 FAIL 1/5**: Δ wyjścia uwagi innych tokenów przy δ = 0,03 wzdłuż u₁ / wzdłuż losowego = 1,0 → 1,7–1,8 (detekcja), 1,5 (MałyViT),
  2,26 (DeiT, jedyny ≥ 2); v₁ᵇ⁺¹ daje 1,4–2,1, czyli tyle samo. δ = 0,1: 1,5–2,3.
  **O5 PASS silnie 5/5**: wrażliwość reszty sieci (JVP) wzdłuż u₁ / losowy = 1,0–1,2 → 1,30–1,49 (detekcja), 1,99 (MałyViT), 2,15 (DeiT);
  v₁ᵇ 1,8–3,3, v₁ᵇ⁺¹ 1,9–4,9. Zapis nie trafia „do kosza” (H3 w mocnej wersji odrzucona), ale w detekcji wyjście jest wzdłuż u₁
  2–3× mniej czułe niż wzdłuż kierunków odczytu (v₁ᵇ 3,0–3,3, v₁ᵇ⁺¹ 4,3–4,9); w CIFAR porównywalnie (DeiT: u₁ 2,15, v₁ᵇ 1,83, v₁ᵇ⁺¹ 1,90).
  **O6 FAIL 1/5**: dwa bloki dalej g_qk(u₁) 1,7–1,9 (detekcja), 2,25 (MałyViT), 3,5 (DeiT).
  **O7 FAIL 2/5**: u₁ nie przesuwa nieliniowej równowagi uwagi ku routingowi względem losowego (detekcja q/k : v 7–15 dla u₁ vs 10–14
  dla losowego; DeiT 2,4 vs 2,8).
  **O8 PASS słabo 4/5**: energia u₁ w jądrze J_LN1 (span{1, x̂}) 0,006–0,014 → **0,20–0,26 w detekcji** (przy wczesnym 3000: 0,34–0,37),
  0,077 w MałymViT CIFAR, 0,007 w DeiT (= losowe); losowy 0,007, v₁ 0,001. W detekcji około jednej czwartej wzmacnianego zapisu to
  kierunek radialny (wzdłuż samego tokenu): zmienia normę tokenu, a nie to, co widzi LayerNorm któregokolwiek odbiorcy.
  **K1 PASS 5/5**: E_m(v₁ᵇ⁺¹) 0,75–0,83 (miara czuła; losowe 0,08).
- Opisowo (koniec, c = b+1): odczyt v₁ᵇ⁺¹ przez MLP b+1 g_m = 9–16 (bramkowany 20–37), a u₁ przez MLP b+1 g_m = 1,65–2,5 (DeiT 3,4).
  Głowa: E_glowa(u₁) 0,035–0,043 vs losowy 0,023 i v₁ᵇ⁺¹ 0,25–0,28 (detekcja); CIFAR 0,17–0,18 vs 0,05 i 0,15–0,31. Nieliniowo udział
  uwagi w zmianie w bloku b+1 dla u₁ (0,21–0,34; DeiT 0,56) jest taki jak dla losowego (0,25–0,39; DeiT 0,58). Kolumna mapy uwagi
  (jak inni patrzą na ten token) zmienia się wzdłuż u₁ 1,7–2,0× bardziej niż losowo, wiersz 1,4–1,55×. Kontrola zgodności z rodzicem
  (koniec): cos²(u₁ᵇ, v₁ᵇ⁺¹) 0,008–0,029, cos²(v₁ᵇ, v₁ᵇ⁺¹) 0,07–0,22, cos²(u₁ᵇ, u₁ᵇ⁺¹) 0,037–0,097 — ten sam rząd co w rodzicu.
  Przy k0 wszystkie zyski g ≈ 0,92–1,05 i energie ≈ 0,07–0,09 (losowe), więc całe odchylenie od losowego powstaje w treningu
  (największe przy wczesnym 3000 w DeiT: g_qk(u₁) 5,1).
- Uwaga po fakcie (nieprzewidziane, nieobjęte progami): w detekcji projekcja **k** jest jedynym odbiorcą, który czyta u₁ mocniej niż
  wspólny kierunek odczytu: g_k(u₁) 2,5–4,0 > g_k(v₁ᵇ⁺¹) 1,9–2,3 we wszystkich 3 seedach (dla q, v i MLP jest odwrotnie). Słabo zgodne
  z tym: stosunek „inne tokeny / własny” w zmianie uwagi dla u₁ 0,09–0,23 vs losowy 0,08–0,13. Zapis może więc lekko zmieniać to, jak
  inne tokeny patrzą na dany token (klucz), ale efekt nieliniowy (O4) nie osiąga progu 2×.
- Rozstrzygnięcie według reguły zapisanej z góry: **H1 nie jest potwierdzona** (O1 i O4 FAIL przy O2, O3 PASS słabo). **H2** zachodzi
  tylko w DeiT (g_v(u₁) 4,5 > 3 × losowe i > g_qk 3,6). **H3 odrzucona w mocnej wersji** (O5 PASS 5/5); słaba wersja ma w detekcji
  realny mechanizm: około 1/4 zapisu u₁ jest radialna (O8). Alternatywa „MLP b+1 czyta” odrzucona (g_m(u₁) < 3 poza DeiT; O3 4/5).
  Wynik nie mieści się w żadnej z zapisanych kategorii: **zapis wzmacnianego modu nie ma wyróżnionego odbiorcy**. Jest czytany
  rozproszenie i słabo przez wszystkich odbiorców naraz (q, k, v, MLP b+1, głowa: 1,3–4,5× losowe, w top-16 energii 0,16–0,40 przy
  0,08), z niewielką przewagą uwagi nad MLP i, w detekcji, toru klucza; wpływa na wyjście sieci 1,3–2,2× silniej niż losowy kierunek,
  w detekcji lecz 2–3× słabiej niż kierunki odczytu v₁ᵇ i v₁ᵇ⁺¹; w detekcji około 1/4 jego energii idzie w normę tokenu.
- Znaczenie dla obrazu „struktura równoległa, nie łańcuchowa” z rodzica: obraz się **utrzymuje i doprecyzowuje**. Brak łańcucha
  MLP → MLP (rodzic) i brak zastępczego łańcucha MLP → routing uwagi (tu). Wspólną „walutą” bloków jest kierunek **odczytu** v₁ (czytany
  mocno przez MLP b+1: g_m 9–16, przez głowę i przez uwagę na poziomie 2–4,6× losowe), a nie kierunek zapisu: wzmacniany mod MLP
  czyta tę wspólną współrzędną i rozpisuje wynik szeroko — częściowo w normę tokenu (detekcja), częściowo w kierunki widziane przez
  wszystkich odbiorców po trochu. Dla detekcji (odczyt per token) zapis jest najmniej celowany (u₁ prawie poza głową, 1/4 radialnie);
  dla CIFAR (odczyt globalny) u₁ jest bardziej czytany przez głowę i uwagę (DeiT: uwaga, w tym v, 3,6–4,5×), zgodnie z różnicą
  typu odczytu opisaną w rodzicu. Pytanie otwarte: czy rozproszony zapis to kanał wzmocnienia (gain) przez normę i LayerNorm, co
  można sprawdzić zaburzeniem samej składowej radialnej u₁ w porównaniu ze składową styczną.
