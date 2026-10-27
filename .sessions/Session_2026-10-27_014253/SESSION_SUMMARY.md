# Session Summary: Mosty do fizyki MałegoViT — ocena, kompensacja, wzmocnienie przejściowe, balanced amplification

## Metadata
- **Date:** 2026-10-26 → 2026-10-27
- **Time:** ok. 13:00 → 01:42 (Europe/Warsaw)
- **Status:** In Progress (zadanie Eden 1933161_4 w toku; trzy nocne eksperymenty niezacommitowane)
- **Type:** Mixed (Analysis + SSH Eden + Literature)
- **Transkrypt:** `W:\Chaty\Claude\CodeCLI\config\projects\W--Studia-PhD-projekt-VIT-ViTParticleFilterTracker\7a3f5b79-c883-4cdf-9559-5f8815a97330.jsonl`

## Objective
Domknąć eksperyment `mosty_fizyki_24seedy_ablacje` (24 seedy, 18 ablacji) i odpowiedzieć na pytania: co można stwierdzić, czego
brakuje do artykułu, co jeszcze policzyć. Następnie sprawdzić, czy nienormalność pola MLP ma skutek dynamiczny (wzmocnienie
przejściowe) i strukturę balanced amplification znaną z kory (Murphy–Miller) i fotoniki nie-hermitowskiej (Makris, Midya).
Ocenić gotowość publikacyjną (TMLR / ICML / NeurIPS).

## Context
Wcześniejsze sesje pokazały na pojedynczych seedach, że trening czyni jakobian podwarstwy MLP MałegoViT nienormalnym (odchylenie
Henriciego) i że istnieją „mosty” do fizyki (ścinanie, kontrakcja, defekt komutatora). Przed sesją uruchomiono na Edenie
replikację na 24 seedach i ablacje (frozen_mlp, frozen_attn, parallel, bez wd, losowe etykiety). Aplikacja „Lasery i koloidy”
(LaserLatentSpaceApp) wizualizuje te zjawiska jako lasery/reflektory.

## Actions Taken
1. Pobranie wyników 24 seedów i 18 ablacji z Edenu, poprawka oceny (`miary_A` przesunięte o 5 etykiet → `[5:]`), ocena predykcji.
2. „zapisuj wnioski!”: `WNIOSEK.md`, `RAPORT.md` (9 sekcji: korelacja częściowa 0,25, kto kompensuje, skala ‖J‖), OPIS, indeks;
   commit 50ceb2e1 i push.
3. Eksperyment pochodny `kompensacja_lodyga_i_cifar_20261026__19-01-59`: kontrola `frozen_both` (uwaga + MLP zamrożone, uczy się
   tylko łodyga i głowa) oraz pomiar mostów na CIFAR-10 (DeiT-Tiny 12 bloków, MałyViT-CIFAR); predykcje K1–K5, C1–C9b przed
   pomiarem; ocena przez import rodzica (`importlib.util.spec_from_file_location`); commit efd2b6a8 i push.
4. Eksperyment `wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11`: G_k = ‖Aᵏ‖/ρᵏ, stała Kreissa, abscysa numeryczna vs
   max Re λ, złożenie bloków, zaburzenia losowe vs najgorszy kierunek (prawy wektor osobliwy), skutek na IoU/margines;
   predykcje W1–W6, F1–F5. Błąd w części F (stan 2b zamiast 2b+1) → poprawka, tryb `--bez_W` / `tylkoF`, powtórka części F.
5. Undermind deep search (2×): nienormalność/ścinanie w fizyce, chemii, korze, fotonice; audyt nowości. Notatki:
   `wzmocnienie_przejsciowe.../Wyniki/Dane/literatura_undermind.md`, `balanced_amplification.../Wyniki/Dane/audyt_nowosci_undermind.md`.
6. Odniesienie prac fotonicznych (Makris 2014, Midya 2022) do układu laserów aplikacji: wzmacniacz nienormalny, kierunek
   optymalnego wzbudzenia = prawy wektor osobliwy, zawór.
7. Eksperyment `balanced_amplification_schur_odczyt_20261027__00-31-41`: forma Schura bez scipy (eig → QR → T = RΛR⁻¹),
   c5, PR_N, para osobliwa J (u₁ pisz, v₁ czytaj), zawór, odczyt r przez autograd (gradient wyjścia głowy po stanie tokenu),
   cos²(u₁,r), cos²(v₁,r); W₂W₁ vs W₂·diag(ḡ)·W₁; żywe reflektory (GELU′ ∈ (0,1; 0,9)); |cos(w₁ᵢ,w₂ᵢ)|. Predykcje BA1–BA6.
   Wersja 2 (po „Licz”): BA3c (energia u₁ w podprzestrzeni odczytu ramki / 10 logitów), BA3d (cos²(u₁ᵇ, v₁ᵇ⁺¹));
   pierwsze wyniki przeniesione do `Wyniki/Dane/pomiar_v1/`; zadania 1933157 (dgx-3) i 1933161 (dgx-1, przekierowane z hoppera
   na polecenie użytkownika).
8. Rozmowy o publikacji: TMLR ~75 %, ICML 2027 ~45 % (brak skali: duże ViT z timm), NeurIPS ~25–30 %; brakuje tylko tekstu
   i wykresów. Czym jest warsztat; dlaczego jedno zdanie to za mało na artykuł; czy to dowód, że pamięć transformera jest w MLP (nie).

## Results

### Key Findings
- **Teza A/B** (nienormalność pola MLP 0,76 → 0,92 rośnie w treningu) potwierdzona na 24 seedach; **teza C** odrzucona;
  konfundator P7 wyjaśniony (trajektoria treningu, rozmiar narzędzia).
- **Kompensacja funkcjonalna** prawdziwa: frozen_mlp / parallel bez straty AP; frozen_both AP 0,46–0,62.
- **CIFAR**: nienormalność replikuje się w DeiT i MałymViT (0,75 → 0,92), także przy losowych etykietach (kształt zostaje, skala
  zanika); blok 1 krytyczny we wszystkich układach; podział pracy zależy od odczytu (CLS).
- **Wzmocnienie przejściowe**: Kreiss i luka abscysy rosną 5/5; najgorszy kierunek wzmacniany 2,2–4,5× w detekcji (σ_max zgodne
  co do 1 %), 1,2 w DeiT; wyjście odporne (ΔIoU ≤ 0,013); tokeny tła wzmacniane silniej niż narzędzia. F1 źle skalibrowana (FAIL,
  uwaga po fakcie). W5 FAIL: najgorsze kierunki kolejnych bloków nie są zgodne (kontrast z Aubry 2024).
- **Balanced amplification (v1)**: BA1 5/5 (c5 0,15 → 0,63–0,74; PR_N 62 → 3–8), BA2b 5/5 (zawór 0,43 → 0,18–0,36),
  BA2a 4/5, **BA3a 2/5, BA3b 0/5 — odwrócone**: z odczytem logitu ciepła wyrównane jest wejście v₁ (cos² 0,06–0,11, 30–50× losowe),
  nie wyjście u₁. BA4a 4/5, BA4b 3/5: nienormalność siedzi w wagach W₂W₁ (0,73 → 0,85–0,90). BA5a 5/5 (korelacja żywe–G₁
  0,41–0,84), BA5b 3/3 (na narzędziu MLP prawie wyłączony: GELU′ 0,01–0,04). BA6 2/5: reflektory częściowo wzajemne
  (|cos| 0,058 → 0,10–0,24).
- **Audyt nowości**: nikt nie mierzył jakobianu podwarstwy MLP per token w funkcji checkpointu z kontrolami; najbliżsi Li26ad
  (uwaga, LLM), Fer26d (cały blok, LLM, monotoniczny gradient), Aub24; ryzyko czasowe → argument za szybkim arXiv.

### Issues Encountered
- Brak tożsamości git po restarcie → commit z `-c user.name/-c user.email` autora HEAD.
- Kolizja nazw modułów `ocena_predykcji` (dziecko importuje rodzica) → `importlib.util.spec_from_file_location`.
- Heredoc z apostrofem / wiele heredoków w Bash tool padają → Write tool lub plik w scratchpadzie.
- `eden.py submit` odmawia przy aktywnych zadaniach → `ssh sbatch --parsable --output=/dev/null` + `register-job`;
  zbłąkane `slurm-*.out` blokują register-job → usuwać.
- Zadania pending (brak CPU/pamięci na dgx; ReqNodeNotAvail) → scancel i resubmit (sr-3 CPU-only, dgx-1 zamiast hopper).
- Błąd stanu w części F wzmocnienia (2b zamiast 2b+1) → poprawka i powtórka tylko F.
- BA2a: para osobliwa na A ≈ I daje cos ≈ 0,96 niezależnie od struktury → definicja przeniesiona na J przed pomiarem.
- SLURM balanced: usunięta linia `EXTRA=""` → „unbound variable”, przywrócona.

## Files Generated/Modified
Wszystko w `SmallViTAblation/experiments/`:
- `mosty_fizyki_24seedy_ablacje_20261026__13-04-31/`: `Opis/OPIS_EKSPERYMENTU.md`, `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`,
  `Wyniki/Krótki_Wniosek/WNIOSEK.md`, `Wyniki/Raport/RAPORT.md` (commit 50ceb2e1).
- `kompensacja_lodyga_i_cifar_20261026__19-01-59/`: `Kod_eksperymentu/{trening_ablacje.py, pomiar_mosty_eden.py, pomiar_mosty_cifar.py,
  ablacje_lodyga.slurm, pomiar_cifar.slurm, pochodzenie.json}`, `Opis/{PREDYKCJE_PRZED_POMIAREM.md, OPIS_EKSPERYMENTU.md}`,
  `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py` (commit efd2b6a8; późniejsze uzupełnienia Stan niezacommitowane).
- `wzmocnienie_przejsciowe_pseudospektrum_20261026__21-12-11/`: `Kod_eksperymentu/{wzmocnienie_eden.py, wzmocnienie.slurm}`,
  `Opis/*`, `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`, `Wyniki/Dane/{literatura_undermind.md, OCENA_PREDYKCJI.txt}` (niezacommitowane).
- `balanced_amplification_schur_odczyt_20261027__00-31-41/`: `Kod_eksperymentu/{balanced_eden.py, balanced.slurm, pochodzenie.json}`,
  `Opis/*`, `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`, `Wyniki/Dane/audyt_nowosci_undermind.md` (niezacommitowane).
- `experiments_index.md` (wiersze czterech eksperymentów; plik CRLF, edycja bajtowa przez Python).

## Commands Used
```
python experiments/eden.py push <eksp> [--update-code]
python experiments/eden.py register-job <eksp> --job-id <id>
python experiments/eden.py pull <eksp>
ssh eden "cd $B && sbatch --parsable --output=/dev/null --array=... Kod_eksperymentu/<skrypt>.slurm"
ssh eden "cd $B && tar cf - Wyniki/Dane/pomiar Wyniki/Dane/logi" | tar xf -
PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
git -c user.name=... -c user.email=... commit -m "..."
```

## Next Steps
- [x] 01:50: wyniki v2 pobrane, ocena: BA3c FAIL 0/3, BA3c′ PASS 2/2, BA3d FAIL 3/5; nieprzewidziane: kolejne bloki czytają ten sam
      kierunek (cos²(v₁ᵇ, v₁ᵇ⁺¹) 0,09–0,21, 17–40× losowe) — struktura równoległa, nie łańcuchowa; dopisane do OPIS (Stan) i indeksu.
- [ ] Pomiar odbiorcy zapisu u₁: projekcja na wiodące kierunki osobliwe wejścia uwagi (W_qkv·J_LN) następnego bloku.
- [ ] Commit i push trzech nocnych eksperymentów (po zgodzie użytkownika).
- [ ] Zapytać o wykresy / raport / krótki wniosek dla nowych eksperymentów.
- [ ] Rozważyć pomiar na dużych ViT z timm (skala, pod ICML) i szybki preprint arXiv (ryzyko czasowe z audytu nowości).
