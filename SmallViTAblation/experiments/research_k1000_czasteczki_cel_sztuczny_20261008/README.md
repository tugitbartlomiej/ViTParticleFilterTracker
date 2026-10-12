# Krytyczny research K1000: cząsteczki i sztuczny cel

Data: 2026-10-08. Research/spike jawnie autoryzowany przez użytkownika.

Cel: ocenić przesłanki oracle edycji 600×768 po GELU ostatniego MLP zamrożonego K1000, transferu korekty bez GT, inwersji SD i wyboru danych według rzeczywistego wpływu aktualizacji AdamW. Wyniki mają rozdzielać literaturę, dowody z repozytorium i własne wnioskowanie.

Zakres zapisu: wyłącznie ten katalog i odpowiadający mu katalog `outputs/research_k1000_czasteczki_cel_sztuczny_20261008`. Bez zmian innych eksperymentów, treningów GPU, commitów i push.

`audit_cpu.py` to jednorazowy probe badawczy, nie kod produktu. Wczytuje istniejący checkpoint wyłącznie na CPU, mierzy rangę fc2, sprawdza projekcję i nullspace na syntetycznych wektorach oraz zachowuje hashe i małe migawki odczytanych dowodów. Nie uruchamia treningu ani forwardów obrazów. Odtworzenie: `F:/Python/python.exe experiments/research_k1000_czasteczki_cel_sztuczny_20261008/audit_cpu.py`.

Główny raport, tabela claims/evidence/failtest, bibliografia i pełne teksty znajdują się w odpowiadającym katalogu outputs. Proponowane testy w raporcie są propozycjami; nie zostały uruchomione w tym researchu.
