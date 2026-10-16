# Uzupełnienie populacyjne: PyDMD DMDc z amplitudami sterowania

Wyraźne zlecenie użytkownika po uruchomieniu konsumentów1825171/1825172. Ich `code_frozen_production` pozostaje bez zmian. Nowy pojedynczy job ma dependency afterok1825166 i oddzielny snapshot/wyniki.

X=stan w stałej analitycznej PCA32 w t; Y=stan w t+5. U: zapisane przed aktualizacją normy gradientu/m1/m2, LR z telemetry[t+1], clock=t/500 oraz bias. Bez mean/last gradientu, delta, przyszłego stanu, fazy/GT/roli jako wejścia. Startowy gradient pochodzi jednak z oracle GT-dependent celu: causal w osi optymalizacji nie oznacza deploy bez GT. Normy są skalarnymi proxy amplitudy, nie pełnym kierunkowym sterowaniem. LearnedB ma32×6 i jest jeden dla wszystkich obrazów/tokenów/faz; nie ma uzasadnionego znanego B dla tych norm.

Preregistered fit: do160 obrazów train równomiernie po indeksie, do12 interwałów każdej czystej fazy,24 równomierne tokeny przestrzenne plus oznaczone role. Role służą jedynie stratyfikacji samplera/raportu. Wszystkie filmy fit PCA producenta muszą pozostać train. Skale tylko train. PyDMD32, rank omega według istniejącego progu1e-8; bez strojenia testem. Jedna wspólna A/B; osobne raporty faz−1mixed/0fit/1min_norm, masek ról i sukcesów/porażek.

Inference strumieniowo po obrazie, wszystkie1500×600×100 przejść co5=90mln. H20:97 okien na token, wariant frozen-start U zamraża wszystkie składowe łącznie z LR/czasem; osobny futureU_oracle używa rzeczywistych przyszłych amplitud i jest teacher-forced. DMD autonomiczne na identycznych stanach i parach oraz persistence stanowią baseline. Porównanie nie miesza z wcześniejszym DMD prędkości.

Pełny błąd768 h5 to błąd projekcji plus kwadrat zapisanego residualnego przyrostu; zachowujemy residuum stanu początkowego prognozy. H20 ma jedynie uczciwe triangle bounds, nigdy pozorny dokładny768D błąd. Retained energy liczona z prawdziwych norm768. Zapis OOD według pięciu wejść, rangi, condition, promienia spektralnego, A/B, hashy i wersji. Brak mAP/correctora. Żadnej obietnicy poprawy.

Jedyny nowy submit po testach i immutable snapshot: `sbatch --dependency=afterok:1825166 --kill-on-invalid-dep=yes population_dmdc.slurm`. Bez listy export. Root wyników: `results/population900k_dmdc_supplement/`; kolektor może pobrać ten trzeci folder niezależnie od dwóch istniejących.
