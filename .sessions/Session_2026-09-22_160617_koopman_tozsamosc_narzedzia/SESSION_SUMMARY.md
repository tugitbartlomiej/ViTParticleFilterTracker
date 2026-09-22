# Session 2026-09-22 16:06 - Wektor tozsamosci narzedzia w latencie detektora

## Cel sesji

Wydzielic w latencie detektora RF-DETR Nano wektor niosacy TOZSAMOSC narzedzia i oddzielic go od polozenia,
ruchu po kadrze i odcisku nagrania. Zbadac trzy drogi liczenia takiego wektora, porownac je jednym metrem
i rozstrzygnac cztery zarzuty autora: (1) badac czesc rzeczywista i urojona osobno, z podzialem na dodatnia
i ujemna, (2) liczyc niezmienniki na wycinku z ramki klatka po klatce, (3) uzyc deep Koopman, (4) skladowe
glowne nie sa uczciwym punktem odniesienia, bo nie wiadomo, co niosa.

## Kontekst

- Repozytorium: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker`, galaz `codex/eden-nano-training-20260914`.
- Poprzednia runda tego dnia: ramiona K5 i widmo bez symetryzacji (sesja zapisana takze w projekcie grantowym,
  `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad`).
- Dokumenty merytoryczne: `SmallViTAblation/docs/Analiza_ponownej_publikacji/38_KRYSTALIZACJA_...md`
  (sekcje 4j do 4r) oraz `39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md` (sekcja 5f).
- Eden: kolejka pusta przez cala sesje; jedyne zadania tego dnia to 1809407 i 1809408. Proba zgloszenia
  krotkiego zadania CPU odbita przez limit konta (`AssocMaxSubmitJobLimit`), wiec analize puli policzono
  lokalnie po sciagnieciu nagrania (731 MB).
- Sprzet lokalny: RTX 3070, uzyta tylko do deep Koopmana (male sieci, minuty).

## Wykonane kroki

1. **Ramiona K5 na calej puli.** Rejestrator czastek przeszedl 10 637 klatek puli uczacej pod detektorem L0
   (zadanie 1809407, 3 minuty na H100), a `identity_code_pool.py` policzyl kod tozsamosci kazdej klatki
   z 612 682 par tokenow (zadanie 1809408, 13 sekund). Zbudowano cztery ramiona doboru: k-centrum w kodzie (K5),
   plus klatki bez kodu (K5b), najblizej prototypu (K5c), najdalej (K5d). Zadne ramie etapu 1 nie roznilo sie
   od losu w tej przestrzeni; K5 zmniejsza najwieksza luke pokrycia z 2,914 do 1,335.
2. **Usterka K4 z etapu 1.** `select_batch(strategy='coverage')` liczy k-centrum kosinusowe i zaklada wektory
   jednostkowe, a dostal wspolrzedne tylko ustandaryzowane. Zapisane jako zastrzezenie w dokumencie 39.
3. **Co niesie kod tozsamosci.** Wydluzenie prawdziwej ramki 85,8 procent, pole 29,0, zwrot koncowki 0,1.
   Wycofano wczesniejsze zdanie o "otwarte czy zamkniete": w tym podzbiorze jest jedna klasa narzedzia.
4. **Sprawdzenie siatki i dopasowania tokenow.** Dziesiec tokenow najpodobniejszych do wzorca narzedzia lezy
   w prawdziwej ramce w 35 procentach (los 2, siatka transponowana 7). Dopasowanie tokenu miedzy klatkami po
   podobienstwie trzyma sie narzedzia w 79 procentach na osmiu klipach (los 3). Szesc rysunkow objasniajacych.
5. **Slad po przejsciu narzedzia (opis Eulera).** Kwadrat, ktory narzedzie opuscilo, wyglada narzedziowo
   z wynikiem 0,899, ale kwadrat, do ktorego narzedzie dopiero przyjdzie, tak samo (0,902). Efekt symetryczny
   w czasie, wiec to zasieg widzenia tokenu, nie pamiec. Zgodne z U14 projektu grantowego.
6. **Eksperymenty z niedzwiedziem (przeglad).** W linii DAVIS czastki niesione przeplywem optycznym trzymaja sie
   zwierzecia w 76,3 procentach po 80 klatkach (94,4 z CoTracker3, kontrola tla 7,0). Zysk dla dynamiki znikomy,
   a na drugim filmie przeplyw przegral z czastka nieruchoma (ICC 0,722 wobec 0,947); wygralo sledzenie
   w przestrzeni cech (40,8 wobec 27,6 procent), czyli odpowiednik naszego testu z rysunku 05.
7. **Widmo bez symetryzacji.** Mody dziela sie na trwale (Re dodatnia), przemienne (Re ujemna) i obrotowe
   (Im rozna od zera). Osobno licza sie tylko trwale (0,834), ale razem daja 0,893 odroznienia narzedzia od tla,
   wydluzenie rosnie z 71 do 80 procent, a wielkosc ramki z zera do 30.
8. **Niezmienniki wycinka klatka po klatce (pomysl autora).** Tokeny z prawdziwej ramki przepisane na stala
   siatke 3 na 3 we wspolrzednych wzglednych; operator miedzy kolejnymi klatkami. Pula: najtrwalszy mod 0,964
   wobec podlogi 0,841 na potasowanym czasie i 0,069 dla par miedzy filmami. Przeglad siatek od 2 na 2 do 5 na 5:
   gestsza nic nie doklada (najtrwalszy mod zmienia sie o 0,004), bo w ramce miesci sie okolo 25 tokenow.
9. **Deep Koopman na opisie ramki.** Enkoder podnosi opis do 8, 16 albo 32 liczb, w ktorych krok klatki ma byc
   mnozeniem przez macierz; podzial po filmach, wczesne zatrzymanie, trzy ziarna. Zaden model nie bije
   przewidywania "nic sie nie zmienia" (liniowy minus 0,7 procent, deep od minus 45 do minus 61) i deep nie
   odroznia sie od wlasnej podlogi z potasowanego czasu. Znajduje jednak te same kierunki co liniowy
   (kosinusy 0,98 i 0,94). Czesc modow ma modul powyzej jedynki, czyli wielkosci rosnace z klatki na klatke.
10. **Ksztalt z niezmiennikow: nie.** Odtwarzanie wycinka obrazu z szesnastu liczb kodu daje R2 minus 0,17,
    czyli gorzej niz jedna usredniona klatka. Te same liczby odtwarzaja 12 procent opisu tokenow, a szesnascie
    skladowych glownych 63. Rysunek `rysunki_k5/REPORT_ksztalt_z_niezmiennikow.png`.
11. **Ktory mod niesie stan narzedzia.** Najlepszy pojedynczy mod to ten o czesci rzeczywistej 0,956
    (44,6 procent wydluzenia sam jeden), ale 37 wolniejszych modow o dodatniej czesci rzeczywistej daje razem
    69,5, a mody obrotowe 74,7. Mody o ujemnej czesci rzeczywistej niosa glownie POLOZENIE (42,2 procent).
12. **Trzy drogi jednym metrem** (`porownanie_tozsamosci.py`, pula, 9 441 klatek, 118 filmow, po osiem liczb
    z czesci rzeczywistej): przesuniecie narzedzia 87,2 procent wydluzenia przy 4,1 procent przecieku polozenia;
    widmo czasu 85,7 przy 40,5 przecieku; deep 9,8. Osiem skladowych glownych: 88,0 przy 7,7, czyli nie gorzej
    niz Koopman. Wynik negatywny dla calej rodziny i zapisany jako taki.
13. **Rozklad przez odejmowanie (pomysl autora).** Reszta widma czasu po odjeciu kodu z przesuniecia niesie
    POLOZENIE (28,5 procent) i nie niesie stanu narzedzia (ponizej 3). Odwrotnie nic nie zostaje. Daje to czysty
    rozklad: tozsamosc z przesuniecia, polozenie z reszty czasu.
14. **Teza autora o statyce potwierdzona.** Rozklad kazdej wspolrzednej na srednia w filmie (statyka) i ruch
    wewnatrz filmu (dynamika): osiem modow o najwiekszej czesci rzeczywistej ma 64 procent statyki, osiem
    skladowych glownych 52, kierunki losowe 30. Zaleznosc monotoniczna po Re (73 procent przy 0,964, 47 przy
    0,857). Checkpoint po 2 500 klatkach daje ten sam obraz, wiec trening tej wlasnosci nie zmienia.

## Pliki zmienione / utworzone

### W projekcie (ViTParticleFilterTracker)

Nowe skrypty w `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin`:
`identity_code_pool.py`, `identity_cover_arm.py`, `co_niesie_kod.py`, `rysunki_k5.py`, `sledzenie_czastek.py`,
`widmo_tozsamosci.py`, `slad_po_przejsciu.py`, `niezmiennik_czasowy.py`, `deep_koopman_ramki.py`,
`ksztalt_z_niezmiennikow.py`, `ktore_mody_nioso.py`, `porownanie_tozsamosci.py`, `statyka_w_koopmanie.py`.

Listy ramion: `SmallViTAblation/curated/selection_matrix_stage1/matrix_K5*_b1500_sel42.json` i `K5_MANIFEST.json`.

Dokumenty: `SmallViTAblation/docs/Analiza_ponownej_publikacji/38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md`
(sekcje 4k do 4r) oraz `39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md` (sekcja 5f i nowy kontrargument).

Dowody: `SmallViTAblation/docs/Analiza_ponownej_publikacji/evidence_koopman_langevin_2026-09-21/REPORT_widmo_tozsamosci.json`,
`REPORT_slad_po_przejsciu.json`, `REPORT_niezmiennik_czasowy_gesty.json`, `REPORT_niezmiennik_czasowy_pula.json`
(plus warianty `_c2`, `_c4`, `_c5`), `REPORT_deep_koopman_gesty.json`, `REPORT_porownanie_tozsamosci_gesty.json`,
`REPORT_porownanie_tozsamosci_pula.json`, `REPORT_statyka_w_koopmanie_pula.json`, `_gesty_b0500.json`, `_gesty_b2500.json`.

Poza gitem: `SmallViTAblation/results/koopman_langevin_epochs/KOD_TOZSAMOSCI_pool_b0500.npz`,
`results/koopman_langevin_epochs/czastki/CZASTKI_pool_b0500/` (731 MB, sciagniete z Edena),
`results/koopman_langevin_epochs/rysunki_k5/` (osiem rysunkow PNG).

### Zewnetrzne (poza projektem)

- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad\`
  (zapis pierwszej rundy tego dnia).
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-bez-nazw-roboczych.md`
  (uzupelnienie: zwroty obrazowe wlasnego autorstwa tez wymagaja wyjasnienia przed uzyciem).
- Tylko do odczytu: `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\test\latki_lagrange.py`,
  `film_lagrange.py`, `sledzenie.py`, `USTALENIA.md`, `F:\Studia\MCHTR\Grant\Projekt\experiments\enkoder_z_pamiecia\RAPORT.md`,
  `RAPORT_SAM2.md`, `F:\Studia\MCHTR\Grant\Projekt\Research\Tezy\tezy\D104.md`.
- Eden: `~/smallvit_runs/2026-09-21_koopman_langevin/czastki/CZASTKI_pool_b0500`,
  `KOD_TOZSAMOSCI_pool_b0500.npz`, `run_niezm.slurm` (zadanie nie zgloszone, limit konta).

## Otwarte problemy

- Straznik determinizmu rejestratora czastek nadal nie przechodzi (0,131 jednostki wspolrzednej).
- Zwykle skladowe glowne nie sa gorsze od zadnej z trzech drog Koopmana w odczycie stanu narzedzia.
  Przewaga Koopmana lezy dzis tylko w czystosci (brak przecieku polozenia) i w uporzadkowaniu statyki.
- Operator nie bije przewidywania "nic sie nie zmienia" na nowych filmach; bez wymuszenia (ruch kamery)
  sterowanie doborem klatek przez ten model byloby sterowaniem po omacku.
- Pomiar "miedzy filmami" laczy narzedzie jako przedmiot z okiem, swiatlem i kadrem; nie rozdziela ich.
- Tokeny narzedzia pochodza z wnetrza ramki, wiec przy cienkim narzedziu czesc opisuje tlo.
- Zwrotu narzedzia (gdzie jest koncowka) nie niesie zadna rodzina modow.
- Zaden trening ramion K5 nie zostal uruchomiony i nie zostanie bez zgody autora.

## Nastepne kroki

1. Przeliczyc kod tozsamosci na trzech rodzinach modow i odbudowac ramiona K5.
2. Dolozyc wymuszenie (ruch kamery z polozenia ramki) do operatora i sprawdzic, czy wychodzi nad trwalosc.
3. Przedstawic autorowi plan etapu 2 do zatwierdzenia: B2, B3, K4 z poprawionym k-centrum, K5, K5c oraz ramie
   odniesienia ze skladowych glownych; trzy ziarna, budzet 1500, 12 epok. Koszt okolo godziny na A100
   przy pieciu procesach rownoleglych.
4. Naprawic straznik determinizmu rejestratora.
