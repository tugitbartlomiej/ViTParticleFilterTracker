# Uzupełnienie: kontrolowany DMDc obejmujący całą populację

Status: zgłoszony **1825187**, dependency **afterok1825166**,2026-10-09 02:10:31UTC. Wyników produkcyjnych jeszcze nie ma. Nie zmieniono istniejących konsumentów1825171/1825172 ani ich zamrożonego kodu. [Marker, komenda i hashe](SUPPLEMENT_DMDC_SUBMITTED.json), [testy](TESTS_READY_POPULATION_DMDC_SUPPLEMENT.txt), [plan](../../experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/supplement_population_dmdc/PLAN.md).

Matematycznie to uzasadniony **test predykcyjnego modelu z niepełnymi wejściami**, nie pełna identyfikacja sterowanej dynamiki Adama. U zawiera normy gradientu i momentów znane na początku interwału, LR pierwszej aktualizacji, clock i bias. Dwa różne wektory gradientu mogą mieć tę samą normę i prowadzić do przeciwnego ruchu. Dlatego nawet pełna ranga designu nie dowodzi pełnego opisu siły lub związku interwencyjnego. Stan PCA32 może częściowo dostarczać informacji o kontekście i kierunku; to hipoteza sprawdzana na odłożonych filmach.

Jedna wspólna A32×32 i B32×6 jest dopasowana rzeczywistym PyDMD.DMDc.fit tylko na próbie train. X=z(t),Y=z(t+5),w jednej zamrożonej bazie producenta; skale wyłącznie train. Role, fazy i status nie są cechami modelu. Role służą jedynie stratyfikacji samplera/raportu. Hiperparametry ustalono przed tym biegiem; starego niekorzystnego testu learnedB nie użyto do ich strojenia.

Po fit analizowane są wszystkie90mln przejść co5 oraz97 okien h20 każdego z900000 tokenów. H5 używa wyłącznie startU. H20 ma osobne warianty: wszystkie składowe U zamrożone w początku oraz prawdziwe futureU opisane jako oracle. Nawet startU pochodzi z oracle loss zależnego od GT: dostępność przed przejściem w zarejestrowanej optymalizacji nie jest dostępnością podczas deploy bez GT.

Porównania: autonomiczne deeptime DMD dopasowane na tych samych stanach i parach oraz persistence. Raport rozbija wyniki na filmy/splity, fazy(z przejściami osobno), role i sukcesy/porażki. Zapisuje OOD każdego wejścia, rank/condition, promień spektralny, A/B, retained energy, kierunki/normy prognoz. Pełny błąd768 co5 uwzględnia znany residual increment; h20 ma tylko dolną i górną granicę błędu768. Nie deklaruje poprawy detekcji/mAP ani gotowego correctora.

Gate7testów sprawdza: brak future mean/last/delta w U, LR(t+1), prawdziwe PyDMD odzyskujące znany układ, brak zmian skali po odczycie testu, sampler train-only, rozdzielenie frozen/futureU, oddzielne okna mieszane, brak utraty residualnej energii768 i pełną liczbę tokenów/przejść w ocenie.

Folder dla collectora: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/results/population900k_dmdc_supplement/`.

Snapshot: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009/code_frozen_population_dmdc_supplement/`.

SHA256 głównego skryptu: `93b388612055ad0a71f28b23ef465ec3b4b7a9bbf00fe7643f43e59095a767c2`. Biblioteki pozostają w prywatnym venv: PyDMD2025.8.1, deeptime0.4.5. Model wykonawcy:gpt-6-astra,effort:xhigh.
