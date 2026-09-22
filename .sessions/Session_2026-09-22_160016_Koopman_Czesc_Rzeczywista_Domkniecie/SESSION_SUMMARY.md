# Session 2026-09-22 16:00 - Czesc rzeczywista modow Koopmana, niezmienniki po czasie i teoria domkniecia stanu

## Cel sesji

Runda trzecia tego samego dnia, po `Session_2026-09-22_131716_koopman_langevin_vit_dobor_klatek_etap1`
i `Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad`. Trzy zadania od autora:

1. Policzyc niezmienniki Koopmana na WYCINKU z ramki narzedzia, klatka po klatce, na wszystkich dostepnych
   klatkach z zaznaczonym narzedziem, i porownac je z niezmiennikami liczonymi po przesunieciu.
2. Napisac analize TEORETYCZNA: skoro w latencie rozdzielono polozenie, tozsamosc, przesuniecie i tlo, to czy
   da sie zmierzyc, ile brakuje do domkniecia modelu narzedzia, i czy da sie tym sterowac doborem klatek przez
   teorie Mori-Zwanziga, Koopmana i Langevina, czyli jadro pamieci, wymuszenie i szum.
3. Sprawdzic hipoteze autora, ze tozsamosc narzedzia siedzi w czesci rzeczywistej modow Koopmana, w tym
   mod po modzie i z osobnym podzialem na czesc dodatnia i ujemna.

Zadne zadanie nie wymagalo treningu detektora i zaden trening nie zostal uruchomiony.

## Kontekst

- Projekt grantowy: galaz `feature/koopman-tft-experiments`, bez zmian merytorycznych w tej sesji.
- Praca merytoryczna: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`, galaz
  `codex/eden-nano-training-20260914`. Moje commity tej rundy: `5b699ac`, `e5916f4`, `4dbcb5b`, `4bfa6fe`, `87a57ea`.
- W tej samej galezi rownolegla sesja zlozyla wlasne commity (`27342b2`, `b5be062`, `4a10f6e`: porownanie trzech
  drog do tozsamosci i rozdzielenie tozsamosci od polozenia). Nie sa moja praca i nie byly przeze mnie ruszane.
- Eden: kolejka pusta przez cala sesje, nic nie liczono. Do katalogu `code/` wgrano skrypt
  `niezmiennik_czasowy.py`, `patch_analysis.py` oraz `pool_manifest.jsonl`, ale ZADNEGO zadania nie zgloszono.
- Wszystkie pomiary tej rundy liczone lokalnie na nagraniach czastek `CZASTKI_b0500` i `CZASTKI_b2500`
  (16 gestych klipow, 4800 klatek), z prawdziwymi ramkami uzytymi WYLACZNIE do oceny.

## Wykonane kroki

1. **Niezmienniki po czasie na wycinku ramki** (`niezmiennik_czasowy.py`, nowy). Wycinek narzedzia opisany
   tokenami z prawdziwej ramki, przepisanymi na stala siatke trzy na trzy we wspolrzednych ramki, wiec opis nie
   zalezy od wielkosci ani polozenia ramki. Operator liczony na parach klatka do nastepnej, 4570 par, 4698 klatek
   z zaznaczonym narzedziem. Wynik: najtrwalsza wielkosc ma modul 0,989, ale przy czasie potasowanym wychodzi
   0,966, a przy parach laczacych rozne filmy 0,104. Prawie cala trwalosc bierze sie wiec z bezwladu latentu,
   dokladnie tak, jak wczesniej w projekcie grantowym. Niezmienniki czytaja wydluzenie ramki w 88 procentach,
   ale 74 procent ich zmiennosci to roznice miedzy filmami.
2. **Porownanie dwoch drog.** Kosinusy glowne miedzy kierunkami znalezionymi po czasie a znalezionymi po
   przesunieciu: 0,95, 0,90, 0,87 i 0,83 dla czterech pierwszych, potem 0,54 i nizej. Dwie niezalezne procedury
   trafiaja w te same cztery wektory latentu, co jest argumentem, ze sa one wlasnoscia reprezentacji.
3. **Dokument 42** `42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md` (analiza teoretyczna, bez
   pomiarow). Rozklada latent na czesc niezmiennicza i ekwiwariantna wzgledem dzialania przesuniecia, podaje
   trzy NIEROWNOWAZNE znaczenia domkniecia (grupowe, wymiarowe, Markowa) z estymatorem bez etykiet dla kazdego,
   przeklada je na trzy reguly doboru klatek, a na koncu pokazuje sufit: przy usuwalnych 23 procentach chybien
   kazda regula gra o tysieczne AP, wiec struktura niezmiennicza jest mocniejsza jako czlon STRATY niz jako
   regula DOBORU.
4. **Zlokalizowana dziura w przestrzeni stanow (wynik rachunkowy, z dokumentu 42).** Zwrot narzedzia, czyli po
   ktorej stronie jest koncowka, jest niezmienniczy wzgledem przesuniecia, wiec powinien byc w kodzie, a wychodzi
   zero. Powod: kod klatki liczymy jako SREDNIA po tokenach, a srednia jest symetryczna wzgledem permutacji
   tokenow i kasuje kazda wielkosc nieparzysta wzdluz osi narzedzia. Poprawka: dolozyc moment pierwszy, wazony
   polozeniem tokenu wzdluz osi ramki. Tego jeszcze nie policzono.
5. **Hipoteza o czesci rzeczywistej, poziom rodzin** (`czesc_rzeczywista_a_tozsamosc.py`, nowy; sekcja 4m
   dokumentu 38). W rodzinie obrotowej czesc rzeczywista niesie wyraznie wiecej niz urojona (0,809 wobec 0,650),
   a najlepsza pojedyncza postacia jest modul pary (0,844). Same czesci rzeczywiste trzech rodzin, 15 liczb, daja
   0,887 przy odroznieniu narzedzia od tla i 78,3 procent przy odczycie wydluzenia, wobec 0,902 i 79,3 dla pelnych
   20 liczb. Hipoteza broni sie wiec w wersji praktycznej: na jedna liczbe kodu czesci rzeczywiste sa wydajniejsze.
6. **Obalone wlasne przewidywanie.** Dokument 42 przewidywal, ze postaci kwadratowe (modul pary, kwadrat wartosci)
   pobija liniowe. Przy rownej liczbie kolumn jest odwrotnie: 0,893 wobec 0,902, a odczyt wydluzenia 73,1 wobec
   79,3. Powod: odczyt nie potrzebuje scislej niezmienniczosci, a modul wyrzuca faze niosaca poze. Utrzymuje sie
   tylko czesciowo, w rodzinie przemiennej, gdzie kwadrat bije wartosc surowa (0,723 wobec 0,686). Poprawka
   wpisana w dokument 42 w miejscu przewidywania.
7. **Hipoteza o czesci rzeczywistej, poziom pojedynczych modow** (`mod_po_modzie.py`, nowy; sekcja 4n dokumentu
   38). Kazdy z 32 modow osobno, jako jedna kolumna cech. Korelacja rangowa czesci rzeczywistej z odroznieniem
   narzedzia od tla wynosi -0,04, czyli zero. Najlepszy pojedynczy mod ma czesc rzeczywista 0,022 i wynik 0,772,
   a mod o najwiekszej czesci rzeczywistej (0,786) daje tylko 0,616. Podzial na dodatnie i ujemne nic nie daje
   (0,569 wobec 0,566), a wewnatrz ujemnych zaleznosc biegnie odwrotnie (-0,62).
8. **Rozdzielenie dwoch wlasnosci, dotad mieszanych.** Mod o czesci rzeczywistej najblizszej jedynki jest jednak
   najlepszy w dwoch innych miarach: przezywa przesuniecie najlepiej (0,77) i jako JEDYNY pojedynczy mod czyta
   ksztalt narzedzia powyzej poziomu zgadywania (58,6 procent). Modul wartosci wlasnej mierzy wiec trwalosc pod
   ruchem i idzie w parze z ksztaltem, a nie z odroznianiem metalu od tkanki. To pierwszy pomiar, ktory te dwie
   wlasnosci rozdziela.
9. **Powtorzenie na drugim checkpoincie** (uwaga autora w trakcie rundy: to jest sied wytrenowana wstepnie).
   Ten sam pomiar na wagach po 2500 klatkach: mod o najwiekszej czesci rzeczywistej nadal jedyny czyta ksztalt
   (58,4 procent) i nadal najlepiej przezywa przesuniecie (0,78), a korelacja czesci rzeczywistej z odroznieniem
   nadal zerowa (-0,02). Zmienia sie jedno: mod o wartosci wlasnej blisko zera, ktory na L0 byl najlepszy
   (0,772), po douczeniu znika. Tamten wynik nalezy wiec przypisac sieci po 500 klatkach, a nie reprezentacji.

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_160016_czesc_rzeczywista_modow_i_domkniecie_stanu\README.md` (nowy)
- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_160016_czesc_rzeczywista_modow_i_domkniecie_stanu\SESSION_SUMMARY.md` (nowy)

Odczyt bez zmian: `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\test\USTALENIA.md` (ustalenia U10, U12, U14
jako punkt odniesienia dla sekcji o pamieci i o podlodze potasowanego czasu).

### Zewnetrzne (poza projektem): repozytorium ViT

Galaz `codex/eden-nano-training-20260914`. Nowe pliki kodu:
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\niezmiennik_czasowy.py`
- `...\pipeline\koopman_langevin\czesc_rzeczywista_a_tozsamosc.py`
- `...\pipeline\koopman_langevin\mod_po_modzie.py`

Dokumenty:
- `...\docs\Analiza_ponownej_publikacji\42_DOMKNIECIE_PRZESTRZENI_STANOW_NARZEDZIA_A_DOBOR_KLATEK.md` (nowy)
- `...\docs\Analiza_ponownej_publikacji\38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md` (sekcje 4m i 4n)

Raporty dowodowe:
- `...\evidence_koopman_langevin_2026-09-21\REPORT_niezmiennik_czasowy_klipy.json`
- `...\evidence_koopman_langevin_2026-09-21\REPORT_czesc_rzeczywista.json`
- `...\evidence_koopman_langevin_2026-09-21\REPORT_mod_po_modzie.json`
- `...\evidence_koopman_langevin_2026-09-21\REPORT_mod_po_modzie_b2500.json`

Na Edenie wgrane, ale NIEURUCHOMIONE: `~/smallvit_runs/2026-09-21_koopman_langevin/code/niezmiennik_czasowy.py`,
`patch_analysis.py`, `kalman_state.py`, `pool_manifest.jsonl`.

## Otwarte problemy

- Niezmienniki po czasie na CALEJ puli (10 637 klatek, 118 filmow) nie zostaly policzone. Pula jest rzadka
  w czasie: mediana odstepu miedzy klatkami tego samego filmu to 20 klatek, a par o odstepie do 5 klatek jest
  tylko 1973. Sensowny pomiar wymaga albo tych rzadkich par, albo zgeszczenia puli.
- Moment pierwszy, czyli poprawka odzyskujaca zwrot narzedzia, jest wyprowadzony, ale niepoliczony.
- Miekka maska, czyli wazenie tokenow podobienstwem do wzorca narzedzia zamiast brania calego wnetrza ramki,
  jest wyprowadzona, ale niepoliczona.
- Deficyt pamieci (Mori-Zwanzig) na wspolrzednych niezmienniczych, po odjeciu ruchu kamery, niepoliczony.
- Straznik determinizmu rejestratora nadal nie przechodzi (0,131 jednostki wspolrzednej).
- Zaden trening ramion K5 ani zadnej nowej reguly doboru nie zostal uruchomiony i nie zostanie bez zgody autora.

## Nastepne kroki

1. Policzyc moment pierwszy i sprawdzic, czy odzyskuje zwrot narzedzia; to najtanszy test z listy w dokumencie 42.
2. Policzyc kod tozsamosci z miekka maska i sprawdzic, czy udzial zmiennosci miedzy filmami spada ponizej
   58 procent.
3. Przeliczyc ramiona K5 na poprawionym kodzie (15 albo 20 liczb, postac liniowa) i dopiero wtedy przedstawic
   autorowi plan etapu 2 do zatwierdzenia.
4. Sprawdzic defekt niezmienniczosci jako sygnal doboru i jego korelacje z pewnoscia detektora; jesli przekracza
   0,6, odrzucic regule bez treningu.

---

Uwaga o miejscu zapisu: ta sesja dotyczy pracy w repozytorium ViT i tutaj jest jej miejsce docelowe.
Kopia tego samego opisu lezy tez w archiwum projektu grantowego
(`F:\Studia\MCHTR\Grant\Projekt\.sessions\`), bo tam biegl zapis pierwotny.
