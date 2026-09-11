# Analiza ponownej publikacji — 2026-09-11

Ten folder został przeniesiony przez autora do `SmallViTAblation/docs/Analiza_ponownej_publikacji/`. Kolekcja PDF/OCR i listy pobierania pozostają w `F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja`.

1. [24 — Szybkie ablacje i optymalizacja Edena](24_SZYBKIE_ABLACJE_I_OPTYMALIZACJA_EDEN.md): najpierw YOLOv8n, następnie kontrolowany transfer; modele, budżety, 9/18/24 biegi i pomiar kosztu.
2. [23 — Połączenie z grantem](23_POLACZENIE_Z_GRANTEM_DYNAMIKA_LATENTU.md): dynamika latentu jako selektor, istniejące artefakty grantu, granice teorii i plan falsyfikacji.
3. [25 — Diagnostyka latentu i aktywna kuracja](25_DIAGNOSTYKA_LATENTU_I_AKTYWNA_KURACJA.md): osobny kierunek badawczy — wybór między dodatkowymi danymi, historią i obserwacją; tanie testy falsyfikacyjne oraz plan AL. Nie zwiększa automatycznie zakresu treningów dokumentu 24.
4. Główny wcześniejszy raport, kanoniczna wersja w Git: [dokument 22](../22_AUDYT_LITERATURY_I_PLAN_PUBLIKACJI.md). Przeniesiona kopia lokalna ma nazwę `UWAGI_DO_PONOWNEJ_PUBLIKACJI.md`; dalsze zmiany wcześniejszego audytu należy utrzymywać w dokumencie 22.
5. [26 — Swift Sampling: pełna analiza krytyczna](26_SWIFT_SAMPLING_ANALIZA_KRYTYCZNA.md): dostarczony OCR/PDF v1 i kod autorów, różnice implementacyjne, rachunek predyktora i kosztu, konsekwencje dla CATARACT-LMM oraz osobny plan porównania z diagnostyką historii. Ujawnia też różnicę przypisania score do klatki w naszym prototypie. To analiza, bez zmian kodu i bez uruchamiania testów/treningów.
6. [27 — Protokół szybkich badań](27_PROTOKOL_SZYBKICH_BADAN_ACTIVE_CURATION_GLE.md): hipotezy, małe modele, sparowane seedy, aktywna pętla, metryki oraz granice interpretacji GLE/Mori–Zwanziga.
7. [28 — Plan implementacji](28_PLAN_IMPLEMENTACJI_ACTIVE_CURATION_GLE.md): kontrakty modułów, profile kosztu, generator manifestu i polecenia lokalne/Eden.
8. [29 — Uruchomienie lokalne i Eden](29_URUCHOMIENIE_LOKALNE_I_EDEN.md): konkretna kolejność smoke, pilot, aktywnych rund, kolejki SLURM i sparowanego bootstrapu operacji.

To plan badania i ocena literatury. Nie wykonano OCR ani nowych treningów. Zarejestrowano 123 pliki OCR; dla 30 przeczytano abstrakty lub wybrane fragmenty, dla pozostałych wykonano screening tytułów. Rejestr podaje zakres i hashe; różne wersje tego samego artykułu nie są niezależnymi dowodami.

Undermind zakończył przegląd z 190 wynikami; pierwsze 50 zapisano z opisem wyszukiwania. Trzy PDF-y odczytano jego narzędziem pełnotekstowym. Surowe odpowiedzi w plikach undermind_*.json są materiałem pomocniczym; wnioski naukowe i ograniczenia zawiera raport.

[Wcześniejsza lista do pobieracza](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/audyt-publikacyjny-kuracji-wideo-rt-detr-i-projekt-eksperymentow-2026-09-11.txt) zawiera 25 adresów zapisanych podczas audytu 22. Nowe źródła do dokumentów 23–24: [lista tematyczna](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/grant-dynamika-latentu-i-szybkie-ablacje-detektorow-2026-09-11.txt). Część adresów pominięto jako obecne w innych listach kolekcji; pełny zestaw wejściowy jest w [rejestrze linków](evidence_grant_2026-09-11/linki_wejsciowe.txt).

Nowa runda 23–24 wykorzystała dodatkowo wcześniejszy przegląd Koopmana (139 rekordów; odczyt opisu i pierwszych 18), dwie lektury PDF przez Undermind oraz źródła pierwotne i wybrane artefakty grantu. Nie wykonano OCR, nowych treningów ani pobierania PDF-ów na dysk. [Dowody i zakres odczytu](evidence_grant_2026-09-11/zrodla_lokalne.json).

Osobna runda 25: nowy przegląd Undermind zakończony 161 rekordami; odczyt pierwszych 15 i trzech PDF-ów dotyczących EnKode, projekcji regresyjnej MZ i LRTD. [Zakres i uwagi krytyczne](evidence_active_learning_2026-09-11/zrodla.json). [Lista do pobieracza](F:/Studia/Articles/Zbiory/2026-09-11_RT-DETR_selekcja_ewaluacja/linki/diagnostyka-latentu-active-learning-i-uzupelnianie-stanu-2026-09-11.txt): 11 nowych adresów, jeden pominięty duplikat. To propozycja kolejnego badania; nie uruchomiono implementacji ani treningów.
