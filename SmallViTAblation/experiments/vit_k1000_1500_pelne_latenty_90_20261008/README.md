# Pełne latenty K1000 dla bazy 1500

Eksperyment klasy badawczej wykonuje per-image oracle na dokładnych 1500
klatkach bazowych K1000. Wagi pozostają zamrożone; zmieniany jest wyłącznie
pełny latent `600×768` po GELU ostatniego MLP.

Manifest zachowuje dokładnie te same obrazy i adnotacje. Dla 203 klatek
naprawia historyczny błąd skalowania ramki po resize i przechowuje dawną
ramkę kampanii jako osobne pole audytowe.

Stan przed zgłoszeniem: kompletny pipeline i skrypty SLURM są przygotowane
lokalnie. Sprzątanie Nasha oraz niezależna ponowna weryfikacja miejsca,
środowisk i wszystkich wejść zakończyły się poprawnie. Użytkownik zwolnił
blokadę wdrożenia; smoke i produkcja wymagają nadal własnych preflightów.

Wyniki trafiają wyłącznie do
`outputs/vit_k1000_1500_pelne_latenty_90_20261008/`.

Etapy są rozdzielone: oracle na 8 GPU, scalanie, analiza wspólnego ruchu,
reżimy z holdoutem po filmach, opisowe DMD/DMDc, pakiety wykonywalne A/B,
fresh-load ewaluacja oraz finalizacja. Zależności SLURM uruchamiają ciężkie
analizy dopiero po kompletnym oracle.
