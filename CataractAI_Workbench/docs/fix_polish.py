#!/usr/bin/env python3
"""Fix Polish diacritical marks in LaTeX files."""
import re
import sys

# Whole-word replacements: (ascii_form, correct_polish_form)
# Sorted longest-first within each group to avoid partial matches
WORD_REPLACEMENTS = [
    # === częstotliwość family ===
    ('czestotliwosciowych', 'częstotliwościowych'),
    ('czestotliwosciowego', 'częstotliwościowego'),
    ('czestotliwosciowej', 'częstotliwościowej'),
    ('czestotliwosciowym', 'częstotliwościowym'),
    ('Czestotliwosciowa', 'Częstotliwościowa'),
    ('czestotliwosciowy', 'częstotliwościowy'),
    ('czestotliwosciowe', 'częstotliwościowe'),
    ('czestotliwosciowo', 'częstotliwościowo'),
    ('czestotliwosciowa', 'częstotliwościowa'),
    ('czestotliwosciami', 'częstotliwościami'),
    ('czestotliwosci', 'częstotliwości'),
    ('czestotliwosc', 'częstotliwość'),

    # === obciążenie ===
    ('Obciazenie', 'Obciążenie'),
    ('obciazeniem', 'obciążeniem'),
    ('obciazeniu', 'obciążeniu'),
    ('obciazenia', 'obciążenia'),
    ('obciazenie', 'obciążenie'),

    # === zbiór ===
    ('zbiorow', 'zbiorów'),

    # === narzędzie ===
    ('narzedzia', 'narzędzia'),
    ('narzedzi', 'narzędzi'),

    # === sieć (standalone) ===
    # handled via regex below

    # === włączenie ===
    ('wlaczenia', 'włączenia'),
    ('wlaczenie', 'włączenie'),
    ('wlaczeniu', 'włączeniu'),
    ('wlaczaja', 'włączają'),

    # === zbieżny ===
    ('zbieznych', 'zbieżnych'),
    ('zbieznosci', 'zbieżności'),

    # === współczesny ===
    ('wspolczesnych', 'współczesnych'),
    ('wspolczesne', 'współczesne'),
    ('wspolczesny', 'współczesny'),

    # === współczynnik ===
    ('wspolczynnikiem', 'współczynnikiem'),
    ('wspolczynnik', 'współczynnik'),

    # === wpływ ===
    ('wplywa', 'wpływa'),
    ('wplyw', 'wpływ'),

    # === Bartłomiej ===
    ('Bartlomiej', 'Bartłomiej'),

    # === głęboki ===
    ('glebokch', 'głębokich'),   # typo fix
    ('glebokiego', 'głębokiego'),
    ('glebokiej', 'głębokiej'),
    ('glebokich', 'głębokich'),
    ('glebokim', 'głębokim'),
    ('glebokie', 'głębokie'),
    ('gleboki', 'głęboki'),

    # === wykazywać ===
    ('wykazuja', 'wykazują'),

    # === również ===
    ('rowniez', 'również'),

    # === jądro ===
    ('Jadro', 'Jądro'),
    ('jadro', 'jądro'),
    ('jadra', 'jądra'),

    # === nieskończony ===
    ('nieskonczonej', 'nieskończonej'),
    ('nieskonczone', 'nieskończone'),
    ('nieskonczona', 'nieskończona'),

    # === stały ===
    ('stale', 'stałe'),  # "pozostaje stałe"

    # === błąd ===
    ('bledu', 'błędu'),

    # === wartość ===
    ('wartoscia', 'wartością'),
    ('wartosci', 'wartości'),

    # === własny ===
    ('wlasnej', 'własnej'),
    ('wlasnych', 'własnych'),
    ('wlasne', 'własne'),
    ('wlasna', 'własną'),

    # === zanikać ===
    ('zanikaja', 'zanikają'),

    # === powiązany ===
    ('powiazanych', 'powiązanych'),
    ('powiazana', 'powiązaną'),
    ('powiazany', 'powiązany'),

    # === składowe ===
    ('skladowych', 'składowych'),
    ('skladowe', 'składowe'),
    ('skladowa', 'składowa'),

    # === wykładniczo ===
    ('wykladniczo', 'wykładniczo'),

    # === wyższy ===
    ('wyzszych', 'wyższych'),
    ('wyzszym', 'wyższym'),
    ('wyzsza', 'wyższą'),
    ('wyzsze', 'wyższe'),
    ('wyzszej', 'wyższej'),

    # === zarówno ===
    ('zarowno', 'zarówno'),

    # === próbować ===
    ('probowalyc', 'próbowały'),  # typo fix
    ('probowaly', 'próbowały'),

    # === próbka ===
    ('probkami', 'próbkami'),
    ('probek', 'próbek'),
    ('probki', 'próbki'),
    ('probke', 'próbkę'),
    ('probka', 'próbka'),

    # === złagodzić ===
    ('zlagodzic', 'złagodzić'),

    # === tłumić (typo: "tluimi" → "tłumi") ===
    ('tluimi', 'tłumi'),

    # === działać ===
    ('dzialajac', 'działając'),
    ('dzialaja', 'działają'),
    ('dziala', 'działa'),

    # === płytki ===
    ('plytkich', 'płytkich'),

    # === złożenie ===
    ('zlozenia', 'złożenia'),
    ('zlozone', 'złożone'),

    # === mogą ===
    ('moga', 'mogą'),

    # === reprezentować ===
    ('reprezentowac', 'reprezentować'),

    # === odtworzyć ===
    ('odtworzyc', 'odtworzyć'),

    # === wejść ===
    ('wejsc', 'wejść'),

    # === przekształcać ===
    ('przeksztalca', 'przekształca'),

    # === umożliwiać ===
    ('umozliwiajac', 'umożliwiając'),

    # === konsekwencja ===
    ('konsekwencje', 'konsekwencję'),

    # === szybkość ===
    ('Szybkosc', 'Szybkość'),
    ('szybkosci', 'szybkości'),
    ('szybkosc', 'szybkość'),

    # === zależeć ===
    ('zaleznoci', 'zależności'),  # typo fix
    ('zaleznosci', 'zależności'),
    ('zalezy', 'zależy'),

    # === także ===
    ('takze', 'także'),

    # === rozkład ===
    ('rozkladow', 'rozkładów'),
    ('rozkladu', 'rozkładu'),
    ('rozklady', 'rozkłady'),
    ('rozklad', 'rozkład'),

    # === bezpośredni ===
    ('Bezposrednio', 'Bezpośrednio'),
    ('bezposrednia', 'bezpośrednią'),
    ('bezposrednio', 'bezpośrednio'),
    ('bezposrednie', 'bezpośrednie'),

    # === różny ===
    ('roznych', 'różnych'),
    ('roznia', 'różnią'),
    ('rozniace', 'różniące'),

    # === różnorodność ===
    ('roznorodnosci', 'różnorodności'),
    ('roznorodnosc', 'różnorodność'),
    ('roznorodna', 'różnorodna'),

    # === artykuł ===
    ('artykulu', 'artykułu'),

    # === sposób ===
    ('sposob', 'sposób'),

    # === nauczyć ===
    ('nauczyc', 'nauczyć'),

    # === coresetów ===
    ('coresetow', 'coresetów'),
    ('coresetowa', 'coresetowa'),

    # === różnorodność (more forms) ===
    ('zroznicowany', 'zróżnicowany'),
    ('zroznicowane', 'zróżnicowane'),
    ('zroznicowana', 'zróżnicowana'),

    # === podzbiór ===
    ('podzbior', 'podzbiór'),
    ('podzbioru', 'podzbioru'),
    ('podzbiorow', 'podzbiorów'),

    # === ważność ===
    ('waznosci', 'ważności'),
    ('waznosc', 'ważność'),

    # === informatywność ===
    ('informatywnosc', 'informatywność'),

    # === liniowa ===
    ('liniowa', 'liniowa'),  # nominative is correct without change

    # === szczegółowość ===
    ('szczegolowocs', 'szczegółowość'),  # typo fix
    ('szczegolowy', 'szczegółowy'),
    ('szczegolnie', 'szczególnie'),
    ('szczegolnosc', 'szczególność'),

    # === większość ===
    ('Wiekszossc', 'Większość'),  # typo fix
    ('wiekszosci', 'większości'),
    ('wiekszosc', 'większość'),

    # === określać ===
    ('okresla', 'określa'),

    # === możliwy ===
    ('moze', 'może'),
    ('mozliwy', 'możliwy'),

    # === być ===
    ('byc', 'być'),

    # === który ===
    ('ktorych', 'których'),
    ('ktore', 'które'),
    ('ktora', 'która'),
    ('ktorej', 'której'),
    ('ktory', 'który'),
    ('ktorego', 'którego'),
    ('ktoremu', 'któremu'),

    # === każdy ===
    ('kazdej', 'każdej'),
    ('kazdym', 'każdym'),
    ('kazdy', 'każdy'),
    ('kazdego', 'każdego'),

    # === już ===
    ('juz', 'już'),

    # === między ===
    ('miedzy', 'między'),

    # === niezależnie ===
    ('niezaleznie', 'niezależnie'),

    # === przetwarzanie ===
    ('przetwarzania', 'przetwarzania'),

    # === funkcja ===
    ('funkcje', 'funkcje'),  # depends on context
    ('funkcja', 'funkcja'),  # correct

    # === fala ===
    ('fale', 'falę'),  # depends on context - "przez falę"

    # === podsumowuje ===
    ('podsumowuje', 'podsumowuje'),

    # === krawędź ===
    ('krawedzie', 'krawędzie'),
    ('krawedzi', 'krawędzi'),

    # === tęczówka ===
    ('Teczowka', 'Tęczówka'),
    ('teczowka', 'tęczówka'),

    # === soczewka ===
    ('soczewki', 'soczewki'),

    # === zmienność ===
    ('Zmiennosc', 'Zmienność'),
    ('zmiennosci', 'zmienności'),
    ('zmiennosc', 'zmienność'),

    # === ostrość ===
    ('ostrosci', 'ostrości'),
    ('ostrosc', 'ostrość'),

    # === niższa ===
    ('nizsza', 'niższa'),
    ('nizsze', 'niższe'),
    ('nizszych', 'niższych'),

    # === łagodny ===
    ('lagodny', 'łagodny'),

    # === również ===
    # Already covered

    # === właściwość ===
    ('wlasciwosci', 'właściwości'),
    ('wlasciwosc', 'właściwość'),

    # === łączenie ===
    ('laczymy', 'łączymy'),

    # === odległość ===
    ('odleglosc', 'odległość'),
    ('odleglosci', 'odległości'),

    # === pokrycie ===
    # 'pokrycie' - no diacritics needed

    # === więcej words ===
    ('rownanie', 'równanie'),
    ('Rown.', 'Równ.'),
    ('rown.', 'równ.'),

    # === klasteringu ===
    # loan word, OK

    # === średnich ===
    ('srednich', 'średnich'),
    ('srednioczestotliwosciowe', 'średnioczęstotliwościowe'),

    # === duplikaty ===
    ('duplikaty', 'duplikaty'),

    # === wierność ===
    ('wiernosc', 'wierność'),
    ('wiernosci', 'wierności'),

    # === jednorodność ===
    ('jednorodnosci', 'jednorodności'),
    ('jednorodnosc', 'jednorodność'),

    # === różniący ===
    # Already covered via rozniace

    # === redundancja ===
    ('redundancje', 'redundancję'),

    # === poniżej ===
    ('ponizej', 'poniżej'),

    # === inferencja ===
    ('inferencji', 'inferencji'),

    # === służyć ===
    ('Sluzy', 'Służy'),
    ('sluzy', 'służy'),

    # === masywny ===
    ('masywna', 'masywną'),

    # === mały ===
    ('malych', 'małych'),
    ('malymi', 'małymi'),
    ('male', 'małe'),
    ('maly', 'mały'),
    ('malego', 'małego'),
    ('malej', 'małej'),

    # === sygnał ===
    ('sygnal', 'sygnał'),
    ('sygnalu', 'sygnału'),

    # === gdyż ===
    ('gdyz', 'gdyż'),

    # === zmniejszyłoby ===
    ('zmniejszyloby', 'zmniejszyłoby'),

    # === efektywna ===
    ('efektywna', 'efektywną'),

    # === dostarczyć ===
    ('dostarcza', 'dostarcza'),

    # === cennego ===
    # no diacritics

    # === nieostrych ===
    ('nieostrych', 'nieostrych'),

    # === obsługi ===
    ('obslugi', 'obsługi'),

    # === odrzucić ===
    ('odrzucic', 'odrzucić'),

    # === redundantny ===
    # no diacritics

    # === klastry ===
    ('klastrow', 'klastrów'),
    ('klastrow', 'klastrów'),

    # === więcej ===
    ('wiecej', 'więcej'),

    # === położenie ===
    ('polozenie', 'położenie'),

    # === podczas ===
    # no diacritics

    # === większość ===
    # Already covered

    # === cztery ===
    # no diacritics

    # === różnica ===
    ('roznice', 'różnicę'),
    ('roznicy', 'różnicy'),
    ('roznica', 'różnica'),

    # === odpowiedź ===
    ('odpowiedz', 'odpowiedź'),

    # === właściwy ===
    # Already covered

    # === leży ===
    ('lezy', 'leży'),

    # === kwantyfikacja ===
    ('kwantyfikacji', 'kwantyfikacji'),

    # === wkład ===
    ('wkladu', 'wkładu'),

    # === jakość ===
    ('jakosci', 'jakości'),
    ('jakosc', 'jakość'),

    # === doświadczalny ===
    ('doswiadczalny', 'doświadczalny'),

    # === informacja ===
    ('informacje', 'informację'),

    # === celowi ===
    # no diacritics

    # === pełny ===
    ('pelnej', 'pełnej'),
    ('pelny', 'pełny'),
    ('pelna', 'pełną'),
    ('pelnego', 'pełnego'),
    ('pelne', 'pełne'),

    # === rozmaitość ===
    ('rozmaitosci', 'rozmaitości'),
    ('rozmaitosc', 'rozmaitość'),

    # === pokrycie ===
    # no diacritics needed

    # === oryginalny ===
    # no diacritics

    # === ortogonalny ===
    # no diacritics

    # === umożliwia ===
    ('umozliwia', 'umożliwia'),

    # === zaawansowany ===
    # no diacritics

    # === identyfikuje ===
    # no diacritics

    # === komplementarny ===
    # no diacritics

    # === priorytetyzacja ===
    # no diacritics

    # === trudność ===
    ('trudnosci', 'trudności'),
    ('trudnosc', 'trudność'),

    # === strata ===
    # no diacritics

    # === granica ===
    # no diacritics for base forms

    # === decyzyjny ===
    # no diacritics

    # === obejmuje ===
    # no diacritics

    # === minimaksowa ===
    # no diacritics

    # === usługowa ===
    ('uslug', 'usług'),
    ('uslugowa', 'usługowa'),

    # === submodularne ===
    # no diacritics

    # === luka ===
    # no diacritics... wait "luka" → "luka" is correct. But "lukę" (accusative)

    # === typowo ===
    # no diacritics

    # === niezmiennicze ===
    # no diacritics

    # === względem ===
    ('wzgledem', 'względem'),

    # === niższopoziomowy ===
    ('niskopoziomowych', 'niskopoziomowych'),
    ('niskopoziomowe', 'niskopoziomowe'),

    # === szum ===
    # no diacritics

    # === zawartość ===
    ('zawartosci', 'zawartości'),
    ('zawartosc', 'zawartość'),
    ('zawartoscia', 'zawartością'),

    # === odwzorowałby ===
    ('odwzorowalby', 'odwzorowałby'),

    # === bliskie ===
    # no diacritics

    # === ostry ===
    # no diacritics for base

    # === rozmyty ===
    # no diacritics

    # === dodanie ===
    # no diacritics

    # === korekcja ===
    # no diacritics

    # === niewystarczający ===
    ('niewystarczajacy', 'niewystarczający'),

    # === wzorców ===
    ('wzorcow', 'wzorców'),

    # === świadomy ===
    ('swiadoma', 'świadomą'),
    ('swiadomy', 'świadomy'),
    ('swiadome', 'świadome'),
    ('swiadomej', 'świadomej'),

    # === kuracja ===
    ('kuracje', 'kurację'),

    # === selekcję ===
    ('selekcje', 'selekcję'),

    # === zachowuje ===
    # no diacritics

    # === warunkiem ===
    # no diacritics

    # === koniecznym ===
    # no diacritics

    # === prawidłowy ===
    ('prawidlowej', 'prawidłowej'),
    ('prawidlowy', 'prawidłowy'),

    # === statystyczny ===
    # no diacritics

    # === dystrybuanta ===
    # no diacritics

    # === metryka ===
    # no diacritics

    # === niż ===
    # "niz" → "niż" - handled via regex

    # === również ===
    # Already covered

    # === żaden ===
    ('zaden', 'żaden'),

    # === jakakolwiek ===
    # no diacritics

    # === wyekstrahować ===
    ('wyekstrahowac', 'wyekstrahować'),

    # === moduł ===
    ('modulow', 'modułów'),
    ('moduly', 'moduły'),
    ('modulem', 'modułem'),
    ('modulami', 'modułami'),

    # === zubozone → zubożone ===
    ('zubozone', 'zubożone'),

    # === znajdzie ===
    # no diacritics

    # === ogranicza ===
    # no diacritics

    # === polega ===
    # no diacritics

    # === znalezienie ===
    # no diacritics

    # === małego ===
    # Already covered

    # === wytrenowaliśmy ===
    ('wytrenowalismy', 'wytrenowaliśmy'),

    # === regresor ===
    # no diacritics

    # === przewidywanie ===
    # no diacritics

    # === wyników ===
    ('wynikow', 'wyników'),

    # === ujawniają ===
    ('ujawniaja', 'ujawniają'),

    # === niemal ===
    # no diacritics

    # === równie ===
    ('rownie', 'równie'),

    # === obejmujące ===
    ('obejmujace', 'obejmujące'),

    # === chwytająca ===
    ('chwytajaca', 'chwytającą'),
    ('chwytajace', 'chwytające'),
    ('chwytaja', 'chwytają'),

    # === kierunkowość ===
    ('kierunkowosc', 'kierunkowość'),

    # === entropia ===
    # no diacritics

    # === pół- ===
    # no changes needed for "pol-"

    # === dlaczego ===
    # no diacritics

    # === typowa ===
    # no diacritics

    # === atypowa ===
    ('atypowa', 'atypową'),
    ('atypowe', 'atypowe'),

    # === nietypowe ===
    # no diacritics

    # === taki/takich ===
    # no diacritics

    # === dobrze ===
    # no diacritics

    # === oba ===
    # no diacritics

    # === wytłumaczyć ===
    ('wytlumaczyc', 'wytłumaczyć'),

    # === mechanizm ===
    # no diacritics

    # === zawierają ===
    ('zawieraja', 'zawierają'),

    # === odbiciu ===
    # no diacritics

    # === wąski ===
    ('waskich', 'wąskich'),

    # === odrzuca ===
    # no diacritics

    # === cennego ===
    # no diacritics

    # === ramek ===
    # no diacritics

    # === treningowego ===
    # no diacritics

    # === energię ===
    ('energie', 'energię'),
    ('energii', 'energii'),

    # === pojawia ===
    # no diacritics

    # === możliwość ===
    ('mozliwosc', 'możliwość'),
    ('mozliwosci', 'możliwości'),

    # === następny ===
    # no diacritics

    # === gałęzi ===
    ('galezi', 'gałęzi'),
    ('galeziowe', 'gałęziowe'),

    # === dwugałęziowe ===
    ('Dwugaleziowe', 'Dwugałęziowe'),
    ('dwugaleziowe', 'dwugałęziowe'),

    # === przestrzenne ===
    # no diacritics

    # === uwadze ===
    # no diacritics

    # === wewnątrz ===
    ('wewnatrz', 'wewnątrz'),

    # === skalowa ===
    # no diacritics base

    # === skałowa → no, it's "skalowa" from "skala" (scale) ===
    # no diacritics

    # === zachowująca ===
    ('zachowujaca', 'zachowującą'),

    # === granic ===
    # no diacritics

    # === pionowe ===
    # no diacritics

    # === diagonalne ===
    # no diacritics

    # === uczalne ===
    ('uczalnymi', 'uczalnymi'),

    # === wagami ===
    # no diacritics

    # === stopniowe ===
    # no diacritics

    # === poprawia ===
    # no diacritics

    # === demonstrując ===
    ('demonstrujac', 'demonstrując'),

    # === krytyczny ===
    # no diacritics

    # === optymalizacja ===
    # no diacritics

    # === formalizuje ===
    # no diacritics

    # === agregacja ===
    # no diacritics

    # === regularyzator ===
    # no diacritics

    # === penalizujący ===
    ('penalizujacym', 'penalizującym'),
    ('penalizujacy', 'penalizujący'),

    # === konwolucja ===
    # no diacritics

    # === stopniowo ===
    # no diacritics

    # === wygładzają ===
    ('wygladzaja', 'wygładzają'),

    # === innowacje ===
    # no diacritics

    # === fundamentalną ===
    ('fundamentalna', 'fundamentalną'),

    # === właściwość ===
    # Already covered

    # === dostępna ===
    ('dostepna', 'dostępną'),
    ('dostepna', 'dostępna'),

    # === mapach ===
    # no diacritics

    # === podczas ===
    # no diacritics

    # === adresują ===
    ('adresuja', 'adresują'),

    # === architekturze ===
    # no diacritics

    # === ortogonalnym ===
    # no diacritics

    # === komplementarnym ===
    # no diacritics

    # === podejściem ===
    ('podejsciem', 'podejściem'),
    ('podejscie', 'podejście'),
    ('podejscia', 'podejścia'),

    # === zapewnienie ===
    # no diacritics

    # === same ===
    # no diacritics

    # === wystarczającą ===
    ('wystarczajaca', 'wystarczającą'),

    # === spektralnie ===
    # no diacritics

    # === ograniczają ===
    ('ograniczaja', 'ograniczają'),

    # === niezależnie ===
    ('niezaleznie', 'niezależnie'),

    # === wzmocnienie ===
    # no diacritics

    # === zakres ===
    # no diacritics

    # === eksponuje ===
    # no diacritics

    # === tła ===
    # already has ł

    # === łączy ===
    ('laczy', 'łączy'),

    # === przyspieszając ===
    ('przyspieszajac', 'przyspieszając'),

    # === krytycznych ===
    # no diacritics

    # === lokalizacja ===
    # no diacritics

    # === Potwierdzają ===
    ('Potwierdzaja', 'Potwierdzają'),
    ('potwierdzaja', 'potwierdzają'),
    ('potwierdzajac', 'potwierdzając'),

    # === wideo ===
    # no diacritics

    # === generuje ===
    # no diacritics

    # === masywną ===
    # Already covered

    # === wstępne ===
    ('wstepne', 'wstępne'),

    # === usuwanie ===
    # no diacritics

    # === kosztowną ===
    ('kosztowna', 'kosztowną'),

    # === ekstrakcją ===
    ('ekstrakcja', 'ekstrakcją'),

    # === Służy ===
    ('Sluzy', 'Służy'),
    ('sluzy', 'służy'),

    # === podwójnemu ===
    ('podwojnemu', 'podwójnemu'),

    # === redukcji ===
    # no diacritics

    # === zapobieganie ===
    # no diacritics

    # === dominacji ===
    # no diacritics

    # === warunki ===
    # no diacritics

    # === zmniejszyłoby ===
    ('zmniejszyloby', 'zmniejszyłoby'),

    # === efektywną ===
    # Already covered

    # === korelacja ===
    # no diacritics

    # === trudnością ===
    ('trudnoscia', 'trudnością'),

    # === synergia ===
    # no diacritics

    # === naturalną ===
    ('naturalna', 'naturalną'),

    # === nietypowych ===
    # no diacritics

    # === tendencję ===
    ('tendencje', 'tendencję'),

    # === trudniejszych ===
    # no diacritics

    # === odwrotnie ===
    # no diacritics

    # === sformułowaniu ===
    ('sformulowaniu', 'sformułowaniu'),

    # === ważonego ===
    ('wazonego', 'ważonego'),

    # === odrebne ===
    ('odrebne', 'odrębne'),

    # === ponieważ ===
    ('Poniewaz', 'Ponieważ'),
    ('poniewaz', 'ponieważ'),

    # === każdego ===
    # Already covered

    # === wybierany ===
    # no diacritics

    # === priorytetyzowane ===
    # no diacritics

    # === jawnego ===
    # no diacritics

    # === ważenia ===
    ('wazenia', 'ważenia'),

    # === trudności ===
    # Already covered

    # === pozycjonuje ===
    # no diacritics

    # === szerszym ===
    # no diacritics

    # === krajobrazie ===
    # no diacritics

    # === kontekście ===
    ('kontekscie', 'kontekście'),

    # === nowosc ===
    ('Nowosc', 'Nowość'),
    ('nowosc', 'nowość'),

    # === badane ===
    # no diacritics

    # === augmentacji ===
    # no diacritics

    # === zastosowanie ===
    # no diacritics

    # === pozostaje ===
    # no diacritics

    # === niezbadane ===
    # no diacritics

    # === wypełnia ===
    ('wypelnia', 'wypełnia'),

    # === lukę ===
    ('luke', 'lukę'),

    # === kompaktowej ===
    # no diacritics

    # === użycie ===
    ('uzycie', 'użycie'),
    ('Uzycie', 'Użycie'),

    # === filtrowania ===
    # no diacritics

    # === świadomego ===
    ('swiadomego', 'świadomego'),

    # === empiryczne ===
    # no diacritics

    # === wykazanie ===
    # no diacritics

    # === zachowanie ===
    # no diacritics

    # === pełnego ===
    # Already covered

    # === pozycji ===
    # no diacritics

    # === Ekstrakcja ===
    # no diacritics for base

    # === Ekstrakcje ===
    ('Ekstrakcje', 'Ekstrakcję'),

    # === sygnatury ===
    # no diacritics

    # === chwytającej ===
    ('chwytajacej', 'chwytającej'),

    # === Motywuje ===
    # no diacritics

    # === motywuja ===
    ('motywuja', 'motywują'),

    # === nieproporcjonalna ===
    ('nieproporcjonalna', 'nieproporcjonalną'),

    # === informatywność ===
    ('informatywnosc', 'informatywność'),

    # === wskazuje ===
    # no diacritics

    # === takiej ===
    # no diacritics

    # === której ===
    # Already covered

    # === semantyczne ===
    # no diacritics

    # === kodują ===
    ('koduja', 'kodują'),

    # === korelacje ===
    ('korelacje', 'korelacje'),

    # === wytłumaczyć ===
    ('wytlumaczyc', 'wytłumaczyć'),

    # === typowo ===
    # no diacritics

    # === zawierają ===
    # Already covered

    # === albo ===
    # no diacritics

    # === atypową ===
    # Already covered

    # === odbicia ===
    # no diacritics

    # === zawartosc ===
    # Already covered

    # === wzorce ===
    # no diacritics for base

    # === brak ===
    # no diacritics

    # === zasłonięte ===
    ('zasloniete', 'zasłonięte'),

    # === chwytane ===
    # no diacritics

    # === 9-wymiarowy ===
    # no diacritics

    # === wektor ===
    # no diacritics

    # === Zachowanie === (as heading)
    # Already covered

    # === wiernie ===
    # no diacritics

    # === pomimo ===
    # no diacritics

    # === redukcji ===
    # no diacritics

    # === pierwotnego ===
    # no diacritics

    # === rozmiaru ===
    # no diacritics

    # === nieodróżnialny ===
    ('nieodrozniany', 'nieodróżnialny'),

    # === istotności ===
    ('istotnosci', 'istotności'),

    # === przekraczają ===
    ('przekraczaja', 'przekraczają'),

    # === oznacza ===
    # no diacritics

    # === sieci ===
    # no diacritics

    # === staje ===
    # no diacritics

    # === uczone ===
    # no diacritics

    # === wolniej ===
    # no diacritics

    # === środnich → średnich ===
    # Already covered

    # === wygladzaja ===
    # Already covered

    # === potwierdza ===
    # no diacritics

    # === potrzebe ===
    ('potrzebe', 'potrzebę'),

    # === wyłączenie ===
    ('wylaczenie', 'wyłączenie'),

    # === potwierdzajac ===
    # Already covered

    # === siec (standalone word) ===
    # Handled below via regex

    # === treści ===
    ('tresci', 'treści'),

    # === cenności ===
    # no diacritics for base

    # === doniosłość ===
    # no diacritics for base

    # === dlaczego ===
    # no diacritics

    # === podejście ===
    # Already covered

    # === Nasz ===
    # no diacritics

    # === pozycja ===
    # no diacritics

    # === równań → equations ===
    ('rownan', 'równań'),

    # === Rozważmy ===
    # Already has Polish chars

    # === zapewniajac ===
    ('zapewniajac', 'zapewniając'),

    # === Zapewnia ===
    # no diacritics

    # === przypisywane ===
    # no diacritics

    # === zapobiegajac ===
    ('zapobiegajac', 'zapobiegając'),

    # === jednorodności ===
    # Already covered

    # === informację ===
    ('informacje', 'informację'),  # depends on context: singular acc. or plural nom.
    # Let's skip this one as it's ambiguous

    # === wzajemną ===
    ('wzajemna', 'wzajemną'),

    # === zwiększa ===
    ('zwieksza', 'zwiększa'),

    # === poprawiając ===
    ('poprawiajac', 'poprawiając'),

    # === Pierwsza ===
    # no diacritics

    # === używa ===
    ('uzywa', 'używa'),

    # === specyficznych ===
    # no diacritics

    # === największy ===
    ('Najwiekszy', 'Największy'),
    ('najwiekszy', 'największy'),

    # === wideo ===
    # no diacritics

    # === rozszerzamy ===
    # no diacritics

    # === modyfikacja ===
    # no diacritics

    # === amplitudowego ===
    # no diacritics

    # === generalizacji ===
    # no diacritics

    # === międzydomenowej ===
    ('miedzy-domenowej', 'między-domenowej'),
    ('miedzydomenowej', 'międzydomenowej'),

    # === semantyka ===
    # no diacritics

    # === amplituda ===
    # no diacritics

    # === niskopoziomowe ===
    # no diacritics

    # === pokrycia ===
    # no diacritics

    # === embeddingowe ===
    # no diacritics

    # === Wybierając ===
    ('Wybierajac', 'Wybierając'),
    ('wybierajac', 'wybierając'),

    # === obejmujące ===
    ('obejmujace', 'obejmujące'),

    # === tła ===
    # Already correct (has ł)

    # === gladka ===
    ('gladka', 'gładką'),
    ('gladki', 'gładki'),
    ('gladkosc', 'gładkość'),

    # === tkanka ===
    # no diacritics

    # === obiektów ===
    ('obiektow', 'obiektów'),

    # === krawędzi ===
    # Already covered

    # === Zgodnie ===
    # no diacritics

    # === składowe ===
    # Already covered

    # === dużych ===
    ('duzych', 'dużych'),

    # === ostre ===
    # no diacritics

    # === łącznie ===
    ('lacznie', 'łącznie'),

    # === wykazanie ===
    # no diacritics

    # === empirycznego ===
    # no diacritics

    # === trenowana ===
    # no diacritics

    # === metodą ===
    ('metoda', 'metodą'),  # But "metoda" can be nominative too! Context-dependent.
    # Skip this one

    # === prostego ===
    # no diacritics

    # === treningowym ===
    # no diacritics

    # === kroku ===
    # no diacritics

    # === standardowych ===
    # no diacritics

    # === spełnia ===
    ('spelnia', 'spełnia'),

    # === również ===
    # Already covered

    # === Regularyzacja ===
    # no diacritics for base

    # === ujawnia ===
    # no diacritics

    # === akumulacje ===
    ('akumulacje', 'akumulację'),

    # === porównaniu ===
    ('porownaniu', 'porównaniu'),
    ('porownanie', 'porównanie'),
    ('Porownanie', 'Porównanie'),

    # === modeli ===
    # no diacritics

    # === działając ===
    # Already covered

    # === filtr ===
    # no diacritics

    # === uczone ===
    # no diacritics

    # === Adresuje ===
    # no diacritics

    # === złożeń ===
    # Already covered (zlozenia)

    # === monolityczne ===
    # no diacritics

    # === umożliwiając ===
    # Already covered

    # === szerokosc ===
    ('szerokoscia', 'szerokością'),
    ('szerokosc', 'szerokość'),
    ('szerokosci', 'szerokości'),

    # === pasma ===
    # no diacritics

    # === Przepuszczenie ===
    # no diacritics

    # === wejść ===
    # Already covered

    # === losowymi ===
    # no diacritics

    # === Implikacje ===
    # no diacritics

    # === Najnowsze ===
    # no diacritics

    # === zaawansowany ===
    # no diacritics

    # === Rozpoznane ===
    # no diacritics

    # === fale → falę ===
    ('fale', 'falę'),

    # === jawnie ===
    # no diacritics

    # === operacje ===
    # no diacritics

    # === Tabela ===
    # no diacritics

    # === metody ===
    # no diacritics

    # === mechanizm ===
    # no diacritics

    # === dokladnosc ===
    ('dokladnosci', 'dokładności'),
    ('dokladnosc', 'dokładność'),

    # === obj. ===
    # abbreviation, OK

    # === opartych ===
    # no diacritics

    # === dekompozycja ===
    # no diacritics

    # === falkowa ===
    # no diacritics

    # === uczalnymi ===
    # no diacritics

    # === oddzielnie ===
    # no diacritics

    # === ablacyjne ===
    # no diacritics

    # === pokazuja ===
    ('pokazuja', 'pokazują'),

    # === poprawia ===
    # no diacritics

    # === przekraczaja ===
    # Already covered

    # === potwierdza ===
    # no diacritics

    # === Zachowuje ===
    # no diacritics

    # === detale ===
    # no diacritics

    # === przestrzenne ===
    # no diacritics

    # === Znaczenie ===
    # no diacritics

    # === ograniczona ===
    # no diacritics

    # === ortogonalnym ===
    # no diacritics

    # === komplementarnym ===
    # no diacritics

    # === spektralnie ===
    # no diacritics

    # === polega ===
    # no diacritics

    # === znalezieniu ===
    # no diacritics

    # === wlasciwosci ===
    # Already covered

    # === uczenia ===
    # no diacritics

    # === pelnego ===
    # Already covered

    # === przeglady ===
    ('przeglady', 'przeglądy'),

    # === identyfikuja ===
    ('identyfikuja', 'identyfikują'),

    # === kryteria ===
    # no diacritics

    # === Priorytetyzacja ===
    # no diacritics

    # === informatywne ===
    # no diacritics

    # === lokalizację ===
    ('lokalizacje', 'lokalizację'),

    # === dopasowanie ===
    # no diacritics

    # === momentów ===
    ('momentow', 'momentów'),

    # === submodularne ===
    # no diacritics

    # === pojedynczej ===
    # no diacritics

    # === embeddingi/embeddingów ===
    ('embeddingow', 'embeddingów'),

    # === klasteringu ===
    # loan word

    # === semantyczne ===
    # no diacritics

    # === sygnatury ===
    # no diacritics

    # === komplementarna ===
    ('komplementarna', 'komplementarną'),

    # === oznaczając ===
    ('oznaczajac', 'oznaczając'),

    # === "sia" => fix for sie at end of sentence ===
    # Handled via regex below

    # === "rzedy" ===
    ('rzedy', 'rzędy'),

    # === wielkości ===
    ('wielkosci', 'wielkości'),

    # === niskoczęstotliwościowe ===
    ('niskoczestotliwosciowe', 'niskoczęstotliwościowe'),
    ('niskoczestotliwosciowy', 'niskoczęstotliwościowy'),
    ('niskoczestotliwosciowych', 'niskoczęstotliwościowych'),
    ('niskoczestotliwosciowa', 'niskoczęstotliwościowa'),

    # === wysokoczęstotliwościowe ===
    ('wysokoczestotliwosciowych', 'wysokoczęstotliwościowych'),
    ('wysokoczestotliwosciowej', 'wysokoczęstotliwościowej'),
    ('wysokoczestotliwosciowym', 'wysokoczęstotliwościowym'),
    ('wysokoczestotliwosciowe', 'wysokoczęstotliwościowe'),
    ('wysokoczestotliwosciowy', 'wysokoczęstotliwościowy'),
    ('wysokoczestotliwosciowa', 'wysokoczęstotliwościowa'),
    ('wysokoczestotliwosciowo', 'wysokoczęstotliwościowo'),

    # === "ostrych" ===
    # no diacritics

    # === "dętych" ===
    # not relevant

    # === "największy" ===
    # Already covered

    # === "również" ===
    # Already covered

    # === "określonych" ===
    ('okreslonych', 'określonych'),
    ('okresla', 'określa'),

    # === "między" ===
    ('miedzy', 'między'),

    # === "główny" ===
    ('glowny', 'główny'),
    ('glowna', 'główna'),
    ('glowne', 'główne'),
    ('Glowny', 'Główny'),

    # === "więcej" ===
    ('wiecej', 'więcej'),

    # === "można" ===
    ('mozna', 'można'),

    # === "dokładność" ===
    # Already covered

    # === "prawdopodobieństwo" ===
    ('prawdopodobienstwo', 'prawdopodobieństwo'),

    # === "połączenia" ===
    ('polaczenia', 'połączenia'),
    ('polaczenie', 'połączenie'),

    # === Extra abbreviation fixes ===
    ('Rown.~', 'Równ.~'),

    # === szczególnie ===
    # Already covered

    # === czubki ===
    # no diacritics

    # === injektory ===
    # no diacritics

    # === tworzą ===
    ('tworza', 'tworzą'),

    # === generując ===
    ('generujac', 'generując'),

    # === silną ===
    ('silna', 'silną'),

    # === zawartość ===
    # Already covered

    # === mokra ===
    # no diacritics

    # === rogówka ===
    ('rogowki', 'rogówki'),
    ('rogowka', 'rogówka'),

    # === szczyty ===
    # no diacritics

    # === skoncentrowana ===
    # no diacritics

    # === wąskich ===
    # Already covered

    # === kierunkowych ===
    # no diacritics

    # === torebka ===
    # no diacritics

    # === materiał ===
    ('material', 'materiał'),
    ('materialu', 'materiału'),

    # === różnią się ===
    # Already covered (roznia)

    # === zależności ===
    # Already covered

    # === zmienność ===
    # Already covered

    # === ostrości ===
    # Already covered

    # === trakcie ===
    # no diacritics

    # === tworzac ===
    ('tworzac', 'tworząc'),

    # === niższa ===
    # Already covered

    # === kolejne ===
    # no diacritics

    # === częstotliwościowej ===
    # Already covered

    # === efektywnie ===
    # no diacritics

    # === prawie-duplikaty ===
    # no diacritics

    # === kosztowną ===
    # Already covered

    # === ekstrakcją ===
    # Already covered

    # === wybranego ===
    # no diacritics

    # === znaczącą ===
    ('znaczaca', 'znaczącą'),

    # === Wysoki ===
    # no diacritics

    # === potwierdza ===
    # no diacritics

    # === równie ===
    # Already covered

    # === informatywne ===
    # no diacritics

    # === zwroty/zwrotów ===
    # no diacritics

    # === odwzorowałby ===
    # Already covered

    # === zajmują ===
    ('zajmuja', 'zajmują'),

    # === otrzymuje ===
    # no diacritics

    # === specularnych ===
    # no diacritics? loan word

    # === zdominowane ===
    # no diacritics

    # === gładkość ===
    # Already covered (gladkosc)

    # === ostrych ===
    # no diacritics

    # === zestawia ===
    # no diacritics

    # === Weryfikujemy ===
    # no diacritics

    # === dwu-próbkowego ===
    ('dwu-probkowego', 'dwu-próbkowego'),

    # === empirycznymi ===
    # no diacritics

    # === dystrybuantami ===
    # no diacritics

    # === pula (pulą) ===
    ('pula', 'pulą'),  # context-dependent. "nad pulą" vs "pula obrazów"
    # Skip - too ambiguous

    # === istotności ===
    # Already covered

    # === wierność ===
    # Already covered

    # === spektralną ===
    ('spektralna', 'spektralną'),  # context-dependent: "wierność spektralną" vs "wierność spektralna jest..."
    # Let me handle this carefully in context

    # === faz ===
    # no diacritics

    # === skąd ===
    ('skad', 'skąd'),

    # === więc ===
    ('wiec', 'więc'),

    # === później ===
    ('pozniej', 'później'),

    # === średnioczęstotliwościowe ===
    ('srednioczestotliwosciowe', 'średnioczęstotliwościowe'),

    # === zbalansowanej ===
    # no diacritics

    # === łagodnej ===
    ('lagodnej', 'łagodnej'),

    # === osiągnięcia ===
    ('osiagniecia', 'osiągnięcia'),

    # === usługowej ===
    ('uslugowa', 'usługową'),

    # === również ===
    # Already covered

    # === embeddingów ===
    # Already covered

    # === dodaj words at the end for completeness ===
    # 'sie' is handled via regex
    # 'sa' is handled via regex
    # 'siec' is handled via regex
    # 'niz' is handled via regex
]

# Regex-based replacements for short words that need word boundaries
REGEX_REPLACEMENTS = [
    # "sie" → "się" (standalone word)
    (r'\bsie\b', 'się'),
    # "sa" → "są" (standalone word, but careful with LaTeX commands)
    (r'\bsa\b', 'są'),
    # "siec" → "sieć" (standalone)
    (r'\bsiec\b', 'sieć'),
    # "niz" → "niż" (standalone)
    (r'\bniz\b', 'niż'),
    # "juz" → "już"
    (r'\bjuz\b', 'już'),
    # "ze" → "że" (conjunction, but careful - "ze" as preposition is correct)
    # Skip - too ambiguous
    # "byc" → "być"
    (r'\bbyc\b', 'być'),
    # "moze" → "może"
    (r'\bmoze\b', 'może'),
    # "mozna" → "można"
    (r'\bmozna\b', 'można'),
    # "rownan" → "równań"
    (r'\brownan\b', 'równań'),
    # abbreviation "czest." → "częst."
    (r'\bczest\.', 'częst.'),
    # "Rown." → "Równ."
    (r'\bRown\.', 'Równ.'),
    # "stale" → "stałe" (in context "pozostaje stałe")
    (r'\bstale\b', 'stałe'),
    # "fale" → "falę" only when preceded by "przez"
    (r'\bprzez fale\b', 'przez falę'),
]


def fix_polish(text):
    """Apply all Polish diacritical fixes to text."""
    # First apply word replacements (longer first to avoid partial matches)
    # Sort by length descending
    sorted_replacements = sorted(WORD_REPLACEMENTS, key=lambda x: len(x[0]), reverse=True)

    for ascii_form, polish_form in sorted_replacements:
        if ascii_form == polish_form:
            continue
        # Use word boundary matching for safety
        pattern = re.compile(r'\b' + re.escape(ascii_form) + r'\b')
        text = pattern.sub(polish_form, text)

    # Then apply regex replacements
    for pattern, replacement in REGEX_REPLACEMENTS:
        text = re.sub(pattern, replacement, text)

    return text


def main():
    for filepath in sys.argv[1:]:
        print(f"Processing: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        fixed = fix_polish(content)

        # Count changes
        import difflib
        orig_lines = content.splitlines()
        fixed_lines = fixed.splitlines()
        diff = list(difflib.unified_diff(orig_lines, fixed_lines, lineterm=''))
        changes = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        print(f"  Changed lines: {changes}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"  Saved: {filepath}")


if __name__ == '__main__':
    main()
