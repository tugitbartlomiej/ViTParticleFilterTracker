# Kontrakt analizy temporalnej, wersja 1

Odbiorca: `vit_k1000_wzorce_pydmd_deeptime_20261009`. Ownership odbiorcy nie obejmuje replay ani wizualizacji Hubble'a. Status: gotowy kontrakt odbiorcy; ścieżka manifestu i ID joba producenta wymagają powiązania przez main.

## Dwie odrębne populacje

1. Replay 900 000 cząsteczek co 5 kroków w stałej PCA3 służy wizualizacji. **Nie jest wejściem do modelowania 768D.**
2. Bogaty panel: pełne 768D, prawdziwe kolejne kroki 0..500 (krok 1, także gdy projekcja chmury jest co 5). Wymagane target, naturalny top1 i konkurenci oraz role kontrolne. Wszystkie g/m/v dla każdej zapisanej trajektorii, przed każdym krokiem. Panel i filmy stanowią jawny zakres wnioskowania. Analiza może odczytać też stary schemat control_* jako osobny smoke bez targetów.

## Minimalny plik NPZ per obraz

Nazwy kompatybilne ze starym oracle, aby producent mógł zachować writer:

| Pole | Kształt | Znaczenie |
|---|---|---|
| control_z | (T+1,P,768), float32 | Stan po t rzeczywistych aktualizacjach |
| control_gradient_before_step | (T,P,768), float32 | g_t przed aktualizacją z_t→z_t+1 |
| control_adam_m1_before_step | (T,P,768), float32 | m_t przed włączeniem g_t |
| control_adam_m2_before_step | (T,P,768), float32 | v_t przed włączeniem g_t, nieujemne |
| control_tokens | (P,), int | Stałe identyfikatory tokenów; bez zmiany tożsamości w czasie |
| telemetry_step | (T+1,), int | Dokładnie 0..T, brak padding/interpolacji |
| telemetry_phase_min_norm | (T+1,), 0/1 | Czy po obejrzeniu stanu t sukces jest znany; określa cel przejścia t→t+1 |
| telemetry_learning_rate | (T+1,), float | lr[0]=0, lr[t+1] użyty w przejściu t→t+1 |
| telemetry_top_token | (T+1,), int | Faktyczny top1 po każdym kroku |
| metadata_json_utf8 | uint8 bytes | JSON poniżej |

Metadane: index/image_id, frame_id (z filmem), film_id opcjonalnie jawne, status success/budget_exhausted, first_success_step (null gdy brak), target.token_index, after_step, checkpoint_sha256, manifest_sha256. Pożądane: role każdego tokenu (lista wieloetykietowa), naturalny top1 i lista konkurentów, clamp_floor=-0.16997121, Adam betas=[0.9,0.999], eps=1e-8, autentyczne metryki replay wobec historycznego oracle.

Przy zbliżonym writerze odbiorca akceptuje manifest mapujący nazwy pól. Nie wolno podać PCA3 pod nazwą control_z. GT może służyć do etykiet roli i oceny, nie do deklaracji deploy bez GT.

## Publikacja gotowości przez producenta

Preferowany `TEMPORAL_PANEL_MANIFEST.json`:

```json
{
  "schema_version": 1,
  "status": "complete",
  "step_stride": 1,
  "dimensions": 768,
  "files": [
    {"path": "panel/000000.npz", "sha256": "...", "image_index": 0}
  ],
  "producer_job_id": "...",
  "source_checkpoint_sha256": "...",
  "validation_path": "REPLAY_VALIDATION.json"
}
```

Ścieżki względne wobec katalogu manifestu lub absolutne Edena. Manifest publikowany atomowo po ukończeniu plików i walidacji. Dopuszczalny istniejący manifest producenta po jawnym adapterze; nie wymaga przebudowy replay. Odbiorca odmawia biegu dla niekompletnego manifestu, rozbieżności hashy, kroku !=1, wymiaru !=768 i brakujących g/m/v.

## Analiza i split

Split deterministyczny po całych filmach: train/validation/test. Identyfikatory i hashe zapisane przed fit. Skalowanie, PCA analityczna (domyślnie 16D), VAMP i reżimy fit tylko train; validation służy do oceny kontrolnej, test nie zmienia fit. Stan modelowany jako z_t−z0; błędy raportowane także po powrocie do pełnych 768D, z zamrożonym residuum stanu startowego. Baza, normy i skale wspólne dla wszystkich obrazów.

Osobne modele fit/min_norm; rozbicia oceny success/failure, target/other, natural-top1/competitor/other oraz film. Przejścia zmiany fazy i okna 1/5/20 kroków oznaczone; główne wyniki wewnątrz faz, bez udawanej synchronii różnych obrazów.

Rzeczywiste `pydmd.DMDc.fit` ma wejście stanów i sterowania. Aby podać wiele trajektorii bez fałszywych przejść, jawne pary X,Y są pakowane jako [X,Y], lag=N_pairs. Lag jest technicznym adapterem API, fizyczny krok nadal 1. Operator jest używany przez basis/operator/B; nie używa się zegara ani reconstructed_data, które interpretowałyby techniczny lag jako czas.

Learned-B: skalowane train-only bloki gradientu/m1/m2, lr i zegar; raport rangi i uwarunkowania [X;U]. Known-B: wejście to analityczny krok Adama z g/m/v/lr/clamp, wyliczony przed krokiem, B jest znaną transformacją do bazy. To kontrola teacherowa, nie nowy wzorzec i nie deployable corrector. Wszystkie rollouty z prawdziwym przyszłym sterowaniem i fazą oznaczamy `teacher_forced_oracle_inputs`.

deeptime DMD operuje na jawnych parach; deeptime VAMP tworzy cechy temporalne fit na train, następnie grupowanie i zajętość/transition counts na osobnych trajektoriach. Persistence, stały przyrost i fazowy skurcz są obowiązkowymi baseline.

## Budżet

Brak pełnego TB zapisu. Odczyt panelu strumieniowo po obrazie; zapis tylko redukcji, statystyk, modeli i raportów. Walidacja kształtów przed alokacją. Job CPU, prywatny venv w root nowej rodziny, bez modyfikacji środowisk innych agentów. Produkcja dependency=afterok:ID_replay; brak automatycznych powtórzeń nieudanego producenta i brak cancel cudzych jobów.

## Zgodność z producentem i rozszerzenie na całą populację

2026-10-09: odczytano rzeczywiste `uruchom_replay.py`, `replay_core.py` i `scal_replay.py` rodziny Hubble. Adapter odbiorcy przyjmuje `control_token_role_mask` (bity 1=target, 2=natural top1, 4=konkurent), `metadata_json_utf8.analysis_split`, `video_disjoint_split`, ścieżki manifestu względem jego katalogu, oraz `phase_before_update`. Nowy panel ma 64 obrazy ×16 tokenów =1024 trajektorie; to **podzbiór**, nie 1500×600. Dotychczasowy smoke ma 16×8=128. Projekcja wizualizacyjna PCA3 nie jest czytana przez fit.

Preferowane rozszerzenie populacyjne: zamrożona, wspólna analityczna PCA32 dopasowana wyłącznie na surowych 768D stanach/przyrostach filmów train. Rozdzielić PCA pozycji i PCA przyrostów, jeśli jedna nie zachowuje energii ruchu; wybór wyłącznie train/validation. Manifest musi identyfikować filmy fit, rodzaj centrowania, skale, kroki próbki, hash bazy i wariancję zachowaną osobno dla pozycji oraz ruchu. Jedna stała baza dla wszystkich obrazów i kroków. PCA3 wizualizacji nie jest tą bazą.

Pełna populacja przy 101 zapisach (0,5,...,500), 32 współrzędnych, float32: 1500×101×600×32×4=11 635 200 000 B =11.6352 GB (10.8361 GiB), przed kompresją i dodatkowymi statystykami. Docelowy tensor per obraz `(101,600,32)`, indeksy tokenów, rzeczywiste `steps`, hash bazy, metadane filmu/fazy/roli i kontrola integralności. Do pełnych błędów 768D potrzebne równolegle normy ortogonalnego residuum i jego zmiany; sam błąd PCA32 nie jest błędem 768D.

Baza musi istnieć **przed** emisją projekcji wszystkich filmów. Jeśli producent robi pojedynczy replay, dozwolone są: wcześniej dopasowana train-only baza z panelu (jawnie panelowa), dwa przebiegi train/statystyki potem projekcja, lub statystyki strumieniowe bez zapisu pozycji. Online PCA zmieniająca osie między krokami bez transformacji wcześniejszych stanów narusza wspólny układ i nie może zasilać DMD.

Tańsze rozszerzenie bez pozycji PCA: dla każdego tokenu co 5 kroków zapisać pełne768 normy przemieszczenia od z0, sumę długości 5 rzeczywistych kroków, normę ich wypadkowej, sumę kwadratów, cos/dot kolejnych kroków i licznik aktywnych par, fazę/zdarzenia i role. To umożliwia populacyjny audyt skali, krzywizny i spójności, **nie** identyfikację wspólnych kierunków ani operatora. Dla kierunków potrzebne średnie wektory/Gramy w wspólnej 768D bazie, stratyfikowane train/val/test, fazą i rolą; agregaty nie są trajektoriami.

Streaming normal equations/Gramy dla DMD należy akumulować z prawdziwych par `(z_t,z_t+1)` lub `(z_t,z_t+5)` tego samego tokenu. Osobne sumy X'X, X'Y, Y'Y, U'U, X'U, U'Y, liczności, średnie i normalizacja train-only; nigdy kolejnych obrazów. Zachować pary walidacyjne i dane do błędu wielokrokowego, którego same globalne momenty nie identyfikują. DMDc całej populacji wymaga także właściwych **populacyjnych kontroli** i ich wyrównania; panelowe g/m/v nie zastąpią kontroli pozostałych tokenów. Wariant bez tych kontroli należy nazwać autonomiczną analizą statystyczną ruchu.

Próbkowanie co5 daje fizyczny lag5. Prognoza 5/20 kroków jest wtedy 1/4 krokiem operatora; prognozy rzeczywistego pojedynczego kroku z samych zapisów co5 nie da się zweryfikować. Zmiany fazy wewnątrz interwału muszą mieć znacznik i osobny model/wykluczenie. Ruch pomiędzy zapisami, kierunek każdego kroku i pamięć Adama pozostają nieodtwarzalne bez dodatkowych deskryptorów/kontroli. Pełnopopulacyjne rozszerzenie nie blokuje smoke bibliotek na 128 trajektoriach; wyniki obu zakresów muszą mieć oddzielne nazwy i mianowniki.

### Potwierdzony adapter populacyjny po aktualizacji producenta

Odczyt rzeczywistego nowego kodu Hubble potwierdza `analysis_positions_pca32` `(101,600,32)`, `analysis_state_residual_norm` i `analysis_increment_residual_norm`, obok pełnych768 `interval_increment_norm`, `interval_previous_increment_dot/cosine`, `interval_previous_increment_valid`, fazy i `token_role_mask`. Manifesty po scaleniu: `REPLAY_INDEX.jsonl`, `REPLAY_VALIDATION.json`, `ANALYTICAL_PCA32_MANIFEST.json`; latter wskazuje `basis_path`, `basis_sha256`, `container_sha256`, `fit_image_indices` oraz `per_image_key`.

Consumer `analiza_populacji.py` używa tych nazw. Kontroluje hash kontenera i każdego pliku, wyklucza mieszane interwały, zachowuje split filmów panelu. Dla filmów nieobecnych w panelu ustala przed fit deterministyczny hash split 60/20/20; lista utrwalona w wynikach. Dopasowuje VAMP i kmeans tylko na maksymalnie160 obrazach train wybranych równomiernie po indeksie, do24 interwałów i 24 przestrzennych tokenów plus role. Potem przypisuje **wszystkie** 1500×600×100 interwałów w dwóch fazach, zapisując uint8 etykiety, zajętości/przejścia i osobne pełne768 normy/cosinusy. Etykieta255 oznacza interwał wykluczony, nie reżim.

To drugi, rzeczywiście populacyjny zakres analizy opisowej. Jego cechy VAMP to przyrosty PCA32 między prawdziwymi stanami co5; nie wejścia do DMDc i nie predictor poprawy detekcji. Średni kierunek jest w32D, natomiast norma/kąt i udział energii są sprawdzane z pełnych768 deskryptorów. Panelowy DMDc nadal ma swój zakres1024 trajektorii. Nie przenosi się parametrów pomiędzy różnymi bazami16/32 i lagami1/5.
