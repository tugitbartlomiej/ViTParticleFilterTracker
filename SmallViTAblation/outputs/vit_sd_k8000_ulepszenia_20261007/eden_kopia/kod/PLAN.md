# Ulepszenia SD K8000 — 2026-10-07
Status: implementacja i weryfikacja; uruchomienie po próbie kontrolnej.
Użytkownik: „No rób to” w odpowiedzi na propozycje poprawy danych, walidacji, optymalizacji i obszaru narzędzia.

## Ograniczenia
- Pełny latent 600×768 i dokładnie teacher K8000 SHA256 0844a20aabaa2ed340cc68f75ec8a23094382285e0d7eeb01930153bdea1490b.
- Start wszystkich wariantów z własnego best_sd.pt, krok7500, SHA256 6df4710b6c84eb7ea3bd7b3781347172a7a7f787772539ecdc11b53c209777b7. Własny VAE zamrożony. Pochodzenie wag: wcześniejsze uczenie od zera.
- Nowy optymalizator w każdym wariancie: best nie zawiera starego stanu optymalizatora. Nie przedstawiać tego jako dokładnego resume kroku7500.
- Nie zmieniać poprzedniego zakończonego eksperymentu. Filmy, obrazy i latenty tylko RAM; kod, logi, metryki i checkpointy mogą być zapisane.
- Wszystkie przebiegi w nowej rodzinie; blokady zapobiegające duplikatom; STOP zachowany, STOP_TIME pozwala wznowić wyłącznie ten nowy eksperyment.
- Zachować podział całymi filmami i nie uznawać braku etykiety za negatywny przykład.

## Etapy
1. Testy CPU i smoke na Edenie w odrębnym katalogu.
2. Diagnoza DDIM 50/100/200 × leading/trailing, ten sam model, te same obrazy i seedy; wybór na walidacji, matched TRAIN tylko diagnostycznie. Najszybszy wariant w2% najmniejszego błędu, pełny zapis reguły.
3. Cztery porównania po maksymalnie4000 NOWYCH aktualizacji każde, wspólny start i ewaluacja:
   A_control: dotychczasowy sampler sekwencyjny stride600, LR5e-5.
   B_mixed: losowa klatka w każdym oknie600, mieszanie filmów, reszta jakA.
   C_optim: danejakB, LR1e-5 maleje do2e-6, EMA0.999.
   D_roi: jakC i dodatkowy błąd pełnych768kanałów tokenów zachodzących na prawdziwą ramkę narzędzia.
   CvsB testuje pakiet optymalizacji, nie osobno wpływEMA iLR. DvsC izoluje dodanyROIloss.
4. Stała walidacja: odłożone25filmów, dwie klatki33%/67%, dwa seedy. TRAINdiagnostyka3filmy z tymi samymi pozycjami/seedami. Raportować pełny błąd i błądROI, cosine, wygląd i kontrolęwarunkowania. IoU generacji vsramkaźródła to diagnostyka układu, nie groundtruthgeneracji.
5. Zachować w każdym wariancie najlepszy checkpoint oraz latest ze stanem optymalizatora/EMA/RNG. Earlystop po>=2000 i>=90%filmówTRAIN: jakość lub3ocenybrakupoprawy; silnedegradacjepowtarzane takżeprzerwać. STOPmanualnykończycałypipelinebezautoresume.
6. Po wszystkich etapach raportporównawczy i lokalny test najlepszego wyniku. Uczciwie zgłosić także brak poprawy. Nie kontynuować dłuższego treningu bez wyniku tego porównania.

## Weryfikacja
Testy doboru klatek/shardingu/przeplotu; ROI maska/strata; LR/EMA/checkpointresume; agregacja filmowa i wybórsamplera; smoke wszystkich4wariantów; świeżewczytaniebest; walidacjahashybazy/teachera.
Diagnostyka i próbki nie zapisują obrazów/latentów. Po ukończeniu zabezpieczyć najlepszy model i usunąć wyłącznie własne zbędne duże zapisy, zgodnie z wcześniejszą zgodą użytkownika.

