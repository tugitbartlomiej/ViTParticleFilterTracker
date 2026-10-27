# Session Summary: Odbiorca zapisu MLP, kanał normy i rodzina efektów balanced amplification

## Metadata
- **Date:** 2026-10-27
- **Time:** 01:42 → 03:57 (Europe/Warsaw)
- **Status:** In Progress (pełny bieg 1933192_0…_4 znaku normy liczy się: _0, _1 dgx-4; _2…_4 dgx-3; start 03:35)
- **Type:** Mixed (Analysis + SSH Eden + Literature)
- **Poprzednia sesja:** `.sessions/Session_2026-10-27_014253`
- **Transkrypt:** `W:\Chaty\Claude\CodeCLI\config\projects\W--Studia-PhD-projekt-VIT-ViTParticleFilterTracker\7a3f5b79-c883-4cdf-9559-5f8815a97330.jsonl`

## Objective
Dokończyć balanced amplification v2; znaleźć zjawiska pokrewne „pobudzeniu stabilizowanemu hamowaniem / selektywnemu, szybkiemu
wzmocnieniu wzorów” w innych dziedzinach; zmierzyć, kto odbiera zapis wzmacnianego modu MLP (na polecenie: „zmierz koniecznie teraz,
agentem w tle opus 5.5”); zmierzyć znak kanału normy (na polecenie, na Edenie, na dgx-ach); przygotować start pisania artykułu pod TMLR.

## Context
Wcześniej ustalono: trening czyni jakobian pola MLP nienormalnym (24 seedy, CIFAR, DeiT), ze skutkiem dynamicznym (wzmocnienie przejściowe
2–4,5×) i strukturą Schura niskiej rangi z zaworem (BA1, BA2b). Otwarte było pytanie, dokąd pisze wzmacniany mod.

## Actions Taken
1. Pobranie balanced v2, ocena: BA3c FAIL 0/3, BA3c′ PASS 2/2, BA3d FAIL 3/5; dopisanie do OPIS i indeksu; `push --update-code` + `pull`
   (SHA256 zweryfikowane). Status mechanizmu „BLAD_LICZENIA” z powodu padniętej próby 1933141 i anulowanego 1933158 — opisane w OPIS.
2. Zapis sesji `Session_2026-10-27_014253`; podanie ścieżki transkryptu.
3. Literatura: Undermind `/Balanced amplification family of effects across cortex fluids ecology photonics and learned networks`
   (134 prace); workflow deep-research zatrzymany na polecenie („dobra starczy”); scite niedostępny (limit 250 wywołań, reset 13.11).
   Notatka `balanced_amplification.../Wyniki/Dane/literatura_rodzina_efektow_undermind.md`.
4. Wyjaśnienia: analogia sfery-szkła-kolimatora w aplikacji laserów (oś odczytu = oś kolimatora, załamanie = nienormalność, sfera =
   LayerNorm; braki: wzmocnienie, tryb zbiorczy); „co odkryliśmy” prostymi słowami; cała teoria krok po kroku.
5. Agent Opus 5.5: eksperyment `odbiorca_zapisu_mlp_uwaga_20261027__01-55-25` (predykcje O1–O8, K1; liniowe zyski odczytu q/k/v/MLP/głowy,
   zaburzenia nieliniowe przez blok b+1, JVP reszty sieci); próba 1933168, pełny bieg 1933172 na dgx-3; polecenie użytkownika „wszystko
   na Edenie, na dgx-ach” przekazane agentowi.
6. Wstępna ocena 4/5 biegów odczytana w scratchpadzie (bez ingerencji w pobieranie agenta), potem pełny raport agenta 5/5.
7. Eksperyment `znak_kanalu_normy_mlp_20261027__02-47-19`: predykcje R1–R2, Z1–Z2, S1–S2, T1, K1–K3 (02:49), kod `znak_normy_eden.py`
   (import `odbiorca_eden.py` rodzica), test lokalny na losowych wagach (domknięcie 4·10⁻¹³, kanał średniej 2·10⁻¹⁵), SLURM, ocena (02:55),
   `pochodzenie.json`, OPIS; `eden.py init/push/submit`; próba dymna 1933189 (dgx-3, 4 min, OK); pełny bieg 1933192 (dgx-4 + dgx-3 przez
   `scontrol update ReqNodeList` dla zadań tablicy pojedynczo).
8. Zaplanowany start pisania artykułu TMLR (CronCreate, jednorazowo 04:38, tylko w tej sesji).

## Results

### Key Findings
- **Dokąd pisze wzmacniany mod:** nie do głowy, nie do v₁ następnego MLP, nie do konkretnego toru uwagi; zapis czytają wszyscy słabo
  (uwaga q/k 2,3–4,0×, v 2,1–4,5×, MLP b+1 1,7–2,5×, głowa 1,8–2,4×; najczulsze kierunki odbiorców 16–25×). Nieliniowo zmienia uwagę innych
  tokenów 1,5–1,8× losowe (DeiT 2,3×). Wpływ na wyjście 1,3–2,2× losowe, 2–3× słabszy niż kierunku odczytu.
- **Wspólna szyna:** kierunek czytania v₁ jest czytany silnie przez wszystkich (MLP b+1 9–16×, uwaga 3,5–4,4×, głowa 21×) — MLP-y to
  równoległe wzmacniacze podpięte do jednej współrzędnej zadania.
- **Jądro LN:** 20–26 % energii u₁ w detekcji (k0 0,8 %; krok 3000 34–37 %), CIFAR MałyViT 8 %, DeiT 0,7 %. Rozdział na normę i średnią
  — pomiar w toku (rodzic łączył obie; wcześniejsza interpretacja „bezwładność tokenu” jest hipotezą).
- **Literatura:** wspólny obiekt rodziny — kierunkowo selektywne wzmocnienie skończonoczasowe przy stabilnym widmie; ISN i efekt
  paradoksalny to inne diagnostyki niż łańcuch Schura; w artykule: „nienormalne wzmocnienie przejściowe o strukturze balanced amplification”.

### Issues Encountered
- Scite: wyczerpany limit miesięczny MCP.
- `scontrol update` na całej tablicy: „Job resizing not supported for job arrays” → aktualizacja każdego zadania `_2`, `_3`, `_4` osobno.
- Test lokalny: tensor na CPU przy sieci na GPU → `.to(dev)`; `-I` ignoruje `PYTHONIOENCODING` → `-X utf8`.
- Godziny w predykcjach, ocenie i OPIS wpisane szacunkowo → poprawione według czasów modyfikacji plików i `sacct`.
- Heredoc w Bash z tekstem zawierającym apostrof/znaki specjalne padał → zapisy przez narzędzie Write.

## Files Generated/Modified
W `SmallViTAblation/experiments/`:
- `balanced_amplification_schur_odczyt_20261027__00-31-41/Opis/OPIS_EKSPERYMENTU.md` (Stan v2), `Wyniki/Dane/literatura_rodzina_efektow_undermind.md`,
  `Wyniki/Dane/pomiar/*`, `Wyniki/Dane/OCENA_PREDYKCJI.{json,txt}`, `Wyniki/Dane/Eden/*`.
- `odbiorca_zapisu_mlp_uwaga_20261027__01-55-25/` (cały, przez agenta): kod, SLURM, pochodzenie, OPIS, predykcje, ocena, wyniki 5 biegów.
- `znak_kanalu_normy_mlp_20261027__02-47-19/`: `Kod_eksperymentu/{znak_normy_eden.py, znak_normy.slurm, pochodzenie.json}`,
  `Opis/{PREDYKCJE_PRZED_POMIAREM.md, OPIS_EKSPERYMENTU.md}`, `Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py`, `Wyniki/Dane/Eden/*`.
- `experiments_index.md`: wynik v2 balanced, wiersz odbiorcy (agent), wiersz znaku normy.
- `.sessions/Session_2026-10-27_014253/`, `.sessions/Session_2026-10-27_035705/`.
Nic nie zacommitowano.

## Commands Used
```
python experiments/eden.py init|push [--update-code]|submit|pull <eksp>
python experiments/eden.py submit <eksp> --script Kod_eksperymentu/znak_normy.slurm --sbatch-arg=--nodelist=dgx-4 --sbatch-arg=--cpus-per-task=5
ssh eden 'scontrol update JobId=1933192_2 ReqNodeList=dgx-3'
ssh eden "sinfo -N -p short -o '%N %C %e' | grep dgx"
W:/Python/3.11/python.exe -I -X utf8 test_znak_normy.py        # test lokalny w scratchpadzie
PYTHONIOENCODING=utf-8 W:/Python/3.11/python.exe Wyniki/Code_Interpretujacy_Wyniki/ocena_predykcji.py
```

## Next Steps
- [ ] Po zakończeniu 1933192: `eden.py pull`, ocena R1–R2, Z1–Z2, S1–S2, T1, K1–K3, dopisanie do OPIS (Stan) i indeksu; dopisać odnośnik do
      dziecka we wpisie rodzica `odbiorca_zapisu_mlp_uwaga` w indeksie.
- [ ] 04:38: start artykułu pod TMLR (LaTeX, szablon TMLR; liczby z plików OCENA_PREDYKCJI i OPIS sześciu eksperymentów).
- [ ] Commit i push nocnych eksperymentów (po zgodzie użytkownika).
- [ ] Zapytać o wykresy, raport i krótki wniosek dla nowych eksperymentów.
