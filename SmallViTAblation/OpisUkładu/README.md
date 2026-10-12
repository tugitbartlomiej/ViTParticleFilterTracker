# Opis układu i dostęp do istniejących danych

Zacznij od [opisu układu](01_OPIS_UKLADU.md) albo [przewodnika po surowych danych](02_DANE_I_TRAJEKTORIE.md).

Najważniejsze skróty Windows znajdują się w folderze **linki**:

- **DANE_60_EPOK_768D_KONTROLE.lnk** — pełne aktywacje 768D ośmiu zdjęć w 75 wybranych checkpointach modelu 60-epokowego;
- **WAGI_60_EPOK_806_CHECKPOINTOW.lnk** — wszystkie checkpointy tego modelu, uczonego na 35 000 zdjęć;
- **DANE_GESTE_21_EPOK_768D_I_PACZKI.lnk** — osobny bieg 21 epok, pełne 768D ośmiu zdjęć i telemetria paczek;
- **K1000_LATENT_GESTY_768D.lnk** — oryginalna tablica 1001 × 12 × 600 × 768 FP32, około 22 GB;
- **ZBIORCZE_ARCHIWUM_RUCHU.lnk** — istniejące archiwum danych ruchu w assets.

Dwuklik otwiera cel. Prawy przycisk → Właściwości pokazuje jego ścieżkę. Dla skrótu pliku „Otwórz lokalizację pliku” pozwala przejść do źródła i skopiować je w Eksploratorze. Duże dane pozostają w dotychczasowych lokalizacjach. Skróty nie zawierają ich bajtów.

[Katalog eksperymentów](03_KATALOG_EKSPERYMENTOW.md) obejmuje 212 folderów rodzin w experiments i outputs, 243 rozpoznane raporty Markdown. Nie oznacza to 212 niezależnych treningów: eksperyment i jego wyniki często mają osobne foldery. [Scalone raporty](04_RAPORTY_SCALONE.md) zawierają treść wszystkich tych raportów. Ich osobne kopie są w **zrodla**, a metadane głównych trajektorii w **metadane**. W sumie skopiowano 305 małych plików i przygotowano 315 skrótów.

Priorytetem są istniejące dane. Po doprecyzowaniu polecenia zatrzymano odtwarzanie aktywacji; niepełny plik utworzony przez tę operację usunięto. Historyczne wyniki pozostały na miejscu. Pakiet nie zawiera nowego treningu ani nowego kompletnego eksportu 806 stanów 768D.

Istniejące macierze Grama, PCA3/PCA32 i RP128 nie są pełnymi współrzędnymi cząsteczek. Dla modelu 60 epok zachowane pełne aktywacje wszystkich 806 checkpointów nie zostały odnalezione; pełne 768D są zachowane w plikach kontrolnych wybranych stanów, w tym końców wszystkich 60 epok.
