# Dziennik źródeł i granic weryfikacji

Data: 2026-10-16. Zakres: mosty z optyki, Koopmana, Moriego–Zwanziga i opisu wirów do SmallViT. Badania były prowadzone równolegle w trzech wątkach; główny agent integrował wyniki i obsługiwał lokalny RAG. Nie wszystkie trafienia wyszukiwarek były czytane w pełnym tekście.

## Nawigacja lokalna

1. Najpierw odczyt i selektywne wyszukiwanie w `F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/index.md`. Pełny indeks był zbyt duży na jeden wynik narzędzia, dlatego dalsza nawigacja używała `rg` po wskazanych pojęciach.
2. Potem konkretne `concepts`, `sources`, MathIndex i wybrane tezy. Ścieżki oraz zakresy odczytu zapisano w załącznikach tematycznych.
3. Nie wykonano masowego skanowania `raw/papers`, ingestu, budowy ani aktualizacji indeksu. Nie modyfikowano Grant.
4. Odnalezione skille: `F:/Studia/MCHTR/Grant/Projekt/.agents/skills/wiki-rag/SKILL.md`, `.../szukaj-wiedzy/SKILL.md`. Odczytano też starszą komendę wiki-rag; jej ścieżka do starego indeksu nie została użyta. Kierowano się nowszym skillem v2.
5. Lokalny serwer `127.0.0.1:8765` nie odpowiadał. Uruchomiono bezpośredni klient w istniejącym venv, z modelami ograniczonymi do lokalnego cache. Kod klienta znajduje się w `experiments/mosty_fizyka_tozsamosc_20261016/przeszukaj_rag.py`.

## RAG: stan wykonania

Zweryfikowano właściwy korpus: `qdrant_korpus_v2`, **2687 dokumentów, 91489 chunków**, late chunking, BGE-M3. Zimny start embedded Qdrant był powolny; biblioteka zgłosiła ograniczenie wydajności dla dużych kolekcji. Nie przeprowadzano migracji ani naprawy bazy.

Ostatecznie klient został zatrzymany po przedłużonym oczekiwaniu bez pierwszej odpowiedzi. Stan: `aborted_no_search_response`, **0 ukończonych z 6 planowanych zapytań**. Szczegóły, potwierdzona tożsamość korpusu i czasy: `RAG_WYNIKI.json`. Nie ma trafień ani score RAG do interpretowania. Skorzystano z wyszukiwania tekstowego i bezpośredniego odczytu źródeł; nie naprawiano środowiska wyszukiwarki. Liczba i rozmiar korpusu nie potwierdzają aktualności każdego chunku; źródła mogły być później zmieniane.

## Wyszukiwanie zewnętrzne

- SciSpace: pytanie o aktywną odpowiedź pułapek, niekonserwatywność i zastosowania w reprezentacjach; 10 kandydatów. Większość była stricte optyczna. Nie wynika z tego brak wcześniejszych zastosowań w ML.
- SciSpace: pytanie o przyczynowe interwencje w tożsamość, z kontrolą pozy/tła; 10 trafień, w tym duplikaty. Odczytano pierwotny tekst Huang–Chang o counting ViT oraz tekst LEACE znaleziony dodatkowym wyszukiwaniem.
- Scite: weryfikacja DOI Roichman 2008 i Venturi–Li 2023. Zwrócono oba rekordy; cytowane fragmenty były dostępne dla pierwszego. `editorialNotices` i `fulltextExcerpts` nie były obecne w odpowiedzi. **Nie twierdzimy, że sprawdzono brak korekt/retrakcji.** Tekst MZ czytano bezpośrednio u wydawcy. Nie kupowano dostępu.
- Paper-search i strony autorów/wydawców: zakresy odczytu FORMA, VAMP, LAVD, FTLE, MZ i pozostałych prac podano w notatkach agentów. Nie oparto żadnego twierdzenia o ViT wyłącznie na abstrakcie optycznym.
- Nie uruchamiano nowego zadania deep-search w Undermind w tej części pracy. Nie przedstawiamy wcześniejszego researchu Undermind jako świeżego przeszukania tej bazy Grant.

Metadane wyszukiwania: `ZAPYTANIA_ZEWNETRZNE.json`, `ZAPYTANIE_TOZSAMOSC.json`. Daty i etykiety indeksów akademickich mogą być niedokładne; wnioski opierają się na wskazanych pierwotnych wersjach.

## Wykryte rozbieżności

- Okubo–Weiss: różne konwencje znaku w Wiki i MathIndex, a w jednym opisie znak nie odpowiada wzorowi. W raporcie użyto jawnych definicji i analitycznych kontrprzykładów.
- Izotropowa kowariancja w równaniu Lyapunova nie identyfikuje części antysymetrycznej. Nie dowodzi jej powstania.
- SVD surowej kowariancji C01 nie zastępuje wybielonego operatora VAMP. Singular value bliska jeden nie jest dowodem stałej semantycznej cechy.
- Część konceptów FTLE/LCS/obserwowalności to stuby. Weryfikowano publikacje, a nie ich automatyczne etykiety pewności.
- W zapisach MZ trzeba zdefiniować projektor i miejsce operatora Q; użyto bezpiecznego przykładu dokładnej eliminacji liniowego stanu ukrytego.

Nie naprawiano plików źródłowych. Raport obejmuje jedynie znalezione rozbieżności, nie jest audytem całej bazy.

## Niezależny przegląd opracowania

Dwaj agenci sprawdzili root RAPORT i PLAN. Doprecyzowano miejsce Jacobianu C_out, rangi i dodatnią półokreśloność Gramianu, warunek pominięcia gauge oraz pełnowymiarowy fallback. Dodano zamrożoną regułę fit-prefiks/score-sufiks dla modelu pamięci w nowym seedzie. Nie weryfikowano wyników eksperymentalnych, bo ta sesja ich nie wytwarza.

## References — dodatkowe źródła głównego raportu

- Roichman, Y., Sun, B., Stolarski, A., & Grier, D. G. (2008). *Influence of nonconservative optical forces on the dynamics of optically trapped colloidal spheres: The fountain of probability.* Physical Review Letters, 101, 128301. https://doi.org/10.1103/PhysRevLett.101.128301
- Venturi, D., & Li, X. (2023). *The Mori–Zwanzig formulation of deep learning.* Research in the Mathematical Sciences, 10, 23. https://doi.org/10.1007/s40687-023-00390-2
- Huang, L., & Chang, Y. (2025). *Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs.* arXiv:2510.09794v1. https://arxiv.org/html/2510.09794v1
- Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., & Biderman, S. (2023). *LEACE: Perfect linear concept erasure in closed form.* Advances in Neural Information Processing Systems, 36. https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html
- Ravfogel, S., Twiton, M., Goldberg, Y., & Cotterell, R. D. (2022). *Linear Adversarial Concept Erasure.* Proceedings of Machine Learning Research, 162, 18400–18421. https://proceedings.mlr.press/v162/ravfogel22a.html — rozważany dodatkowy baseline, nie wymagany równolegle z każdą wersją LEACE.

Dalsze zweryfikowane źródła i zakresy lektury znajdują się w trzech załącznikach tematycznych.
