# Session 2026-09-22 15:03 - Kod tozsamosci narzedzia: ramiona K5, widmo bez symetryzacji, slad po przejsciu

## Cel sesji

Runda druga tego samego dnia, kontynuacja sesji `Session_2026-09-22_131716_koopman_langevin_vit_dobor_klatek_etap1`.
Trzy zadania:

1. Domknac pomysl autora "przejscie z latentu do klatek": policzyc kod tozsamosci narzedzia dla calej puli
   uczacej i zbudowac z niego ramiona doboru klatek (rodzina K5), bez uruchamiania treningu.
2. Sprawdzic zarzut autora, ze w niezmiennikach Koopmana trzeba osobno badac czesc rzeczywista dodatnia,
   rzeczywista ujemna i urojona.
3. Sprawdzic drugi zarzut: w eksperymentach z niedzwiedziem (DAVIS) czastki mialy sledzic obiekt, a w opisie
   Eulera lata miala trzymac slad po przejsciu obiektu. Ustalic, co tam naprawde zmierzono, i powtorzyc
   pomiar na materiale chirurgicznym.

## Kontekst

- Projekt grantowy: galaz `feature/koopman-tft-experiments`, stan roboczy bez zmian merytorycznych
  (zmodyfikowane pozostaja `.gitignore`, `SpmProofs.lean`, `hot.md`, `assets/MANIFEST.json`,
  `outputs/rejestr/rodziny.json` - nie ruszane w tej sesji).
- Praca merytoryczna: `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`, galaz
  `codex/eden-nano-training-20260914`. Siedem commitow w tej rundzie, od `84cab8f` do `037e422`.
- Eden: katalog `~/smallvit_runs/2026-09-21_koopman_langevin`. Na koniec sesji kolejka pusta, zadne zadanie
  nie liczy. W tym samym repozytorium pracuje rownolegla sesja (dokument 37, zadania `token_extract`,
  `stan_prep`, `network_state_pilot`) - jej pliki nie byly ruszane.
- Reguly autora obowiazujace w tej sesji: kampanie i nowe ramiona najpierw opisac i czekac na zatwierdzenie
  (pamiec `feedback-plan-przed-treningiem`), raport po kazdym biegu, pelne sciezki, tryb prosty w odpowiedziach.

## Wykonane kroki

1. **Kod tozsamosci dla calej puli (Eden).** Zadanie 1809407 (3 minuty na H100) przepuscilo detektor L0
   (`ut_b0500_best.pth`) przez wszystkie 10 637 klatek puli uczacej `tracking_static36` i zapisalo tokeny;
   zadanie 1809408 (13 sekund, CPU) dopasowalo operator translacyjny na 612 682 parach tokenow narzedzia
   i policzylo kod oraz domkniecie kazdej klatki. Kod ma 9 685 klatek, 952 klatki sa bez kodu, bo L0 nie ma
   tam pewnej ramki. Plik `KOD_TOZSAMOSCI_pool_b0500.npz` (439 KB) pobrany na dysk lokalny, poza gitem.
   Strazniki rejestratora: kierunek martwy praktycznie zerowy, promien 1,0, ale straznik determinizmu nadal
   nie przechodzi (0,131 jednostki wspolrzednej).
2. **Cztery ramiona K5 zbudowane** skryptem `identity_cover_arm.py` przy budzecie 1500 i tym samym L0:
   k-centrum euklidesowe w kodzie (K5), K5 plus 20 procent klatek bez kodu (K5b), klatki najblizsze
   prototypu (K5c) i najdalsze (K5d). Zmierzone pokrycie przestrzeni kodu dla wszystkich ramion, takze
   z etapu 1: sam L0 zostawia najwieksza luke 2,914, ramiona etapu 1 leza w przedziale od 2,3 do 2,7
   (czyli na poziomie losu), a K5 schodzi do 1,335. Wspolne klatki K5 z ramionami etapu 1: od 9 do 13 procent.
3. **Usterka w ramieniu K4 z etapu 1.** `select_batch(strategy='coverage')` liczy k-centrum kosinusowe
   i zaklada wektory jednostkowe, a `selection_matrix.py` podal mu wspolrzedne ustandaryzowane. K4 bylo wiec
   regula "kierunek najbardziej przeciwny do juz wybranych", nie pokryciem. Wynik stoi (ramie deterministyczne),
   ale nazwa jest na wyrost; kontrola euklidesowa wchodzi do etapu 2. B3 tej wady nie ma.
4. **Co naprawde niesie kod tozsamosci** (`co_niesie_kod.py`, etykiety wylacznie do oceny, podzial po filmach,
   9 142 klatki, 117 filmow): wydluzenie prawdziwej ramki 85,8 procent, pole ramki 29,0, zwrot koncowki
   0,1 i 0,3, polozenie w kadrze 10,0 i 18,5. Udzial zmiennosci miedzy filmami 58 procent. Wczesniejsze
   sformulowanie w rozmowie ("osiem liczb mowi, czy narzedzie jest otwarte czy zamkniete") wycofane:
   w tym podzbiorze jest jedna klasa narzedzia (cystotome) i nie ma etykiety stanu.
5. **Szesc rysunkow objasniajacych** na prawdziwych klatkach (`rysunki_k5.py`, `sledzenie_czastek.py`),
   zapisanych poza gitem w `results/koopman_langevin_epochs/rysunki_k5`: podzial klatki na 576 kwadratow,
   parowanie kwadratow narzedzia po przesunieciu, co przezywa przesuniecie (zgodnosc 0,77 dla modu
   najstabilniejszego wobec minus 0,17 dla najmniej stabilnego), mapa stanow z wyborem K5, dopasowanie
   tokenow przez 21 klatek, oraz rysunek pokazujacy, ze sama czastka stoi.
6. **Dwa sprawdzenia poprawnosci.** Siatka: dziesiec tokenow najpodobniejszych do wzorca narzedzia lezy
   w prawdziwej ramce w 35 procentach przypadkow, przy 2 procentach dla losu i 7 procentach przy siatce
   transponowanej, wiec przypisanie token-miejsce jest poprawne. Dopasowanie tokenow miedzy klatkami po
   podobienstwie kosinusowym trzyma sie narzedzia w 79 procentach na osmiu klipach (97 na jednym klipie),
   przy 3 procentach dla losu. Sprostowano tez wlasne zdanie z rozmowy: kwadraty narzedzia pochodza
   z wnetrza ramki detektora, nie z obrysu narzedzia, wiec czesc z nich opisuje tlo.
7. **Widmo bez symetryzacji** (`widmo_tozsamosci.py`, nowa sekcja 4k dokumentu 38). Dotychczasowy kod uzywal
   operatora po symetryzacji, ktorego wartosci wlasne sa z definicji rzeczywiste, wiec czesc obrotowa
   przepadala. Po rezygnacji z symetryzacji mody dziela sie na trzy rodziny: trwale (16 modow, odroznienie
   narzedzia 0,834), przemienne o ujemnej czesci rzeczywistej (6 modow, 0,743) i obrotowe (10 modow, 0,593).
   Razem daja 0,893, wydluzenie ramki rosnie z 71 do 80 procent, a wielkosc ramki z zera do 30 procent.
   Kat obrotu jest znikomy (okres okolo 256 przesuniec). Zwrotu narzedzia nie niesie zadna rodzina.
   Wniosek: kod tozsamosci dla K5 przeliczyc na 18 liczbach zamiast 8 przed etapem 2.
8. **Przeglad eksperymentow z niedzwiedziem** (podagent Explore, projekt grantowy, tylko odczyt). Istnieja
   oba opisy. W linii DAVIS `latki_lagrange.py` czastki sa niesione przeplywem optycznym Farnebacka
   i po 80 klatkach 76,3 procent zasianych na niedzwiedziu wciaz na nim lezy (94,4 procent przy CoTracker3),
   przy 7,0 procent dla kontroli z tla. Zysk dla dynamiki byl jednak znikomy (R2 z 0,0762 na 0,0783),
   a na drugim filmie (Wilford) przeplyw optyczny przegral z czastka nieruchoma (ICC 0,722 wobec 0,947)
   i zostal zastapiony sledzeniem w przestrzeni cech (40,8 procent wobec 27,6 procent, teza D104).
   Odniesienie dopisane do dokumentu 39.
9. **Slad po przejsciu narzedzia** (`slad_po_przejsciu.py`, nowa sekcja 4l dokumentu 38). Kwadrat, ktory
   narzedzie opuscilo i do ktorego nie wroci, wyglada narzedziowo z wynikiem 0,899 po trzech klatkach,
   ale kwadrat, do ktorego narzedzie dopiero przyjdzie, wyglada tak samo (0,902). Efekt jest symetryczny
   w czasie, wiec to zasieg widzenia tokenu poza ramka, nie pamiec kwadratu. Zgodne z U14 projektu
   grantowego (jadro pamieci glebokosci jednego kroku, 5 do 6 procent przyrostu).

## Pliki zmienione / utworzone

### W projekcie (Grant/Projekt)

- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad\README.md` (nowy)
- `F:\Studia\MCHTR\Grant\Projekt\.sessions\Session_2026-09-22_150300_kod_tozsamosci_narzedzia_widmo_i_slad\SESSION_SUMMARY.md` (nowy)
- `C:\Users\bartl\.claude\projects\F--Studia-MCHTR-Grant-Projekt\memory\feedback-bez-nazw-roboczych.md` (uzupelnienie:
  zwroty obrazowe wlasnego autorstwa tez wymagaja wyjasnienia przed uzyciem)

Tylko do odczytu w tej sesji (bez zmian): `F:\Studia\MCHTR\Grant\Projekt\Research\DiagnosticTool\test\latki_lagrange.py`,
`film_lagrange.py`, `sledzenie.py`, `USTALENIA.md` (U10, U12, U14), `F:\Studia\MCHTR\Grant\Projekt\experiments\enkoder_z_pamiecia\RAPORT.md`,
`RAPORT_SAM2.md`, `F:\Studia\MCHTR\Grant\Projekt\Research\Tezy\tezy\D104.md`.

### Zewnetrzne (poza projektem): repozytorium ViT

Galaz `codex/eden-nano-training-20260914`, commity `84cab8f`, `aff568e`, `03fd517`, `76cd61f`, `14ba669`,
`a492bd4`, `037e422`.

Nowe pliki kodu:
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\pipeline\koopman_langevin\identity_code_pool.py`
- `...\pipeline\koopman_langevin\identity_cover_arm.py`
- `...\pipeline\koopman_langevin\co_niesie_kod.py`
- `...\pipeline\koopman_langevin\rysunki_k5.py`
- `...\pipeline\koopman_langevin\sledzenie_czastek.py`
- `...\pipeline\koopman_langevin\widmo_tozsamosci.py`
- `...\pipeline\koopman_langevin\slad_po_przejsciu.py`

Nowe listy ramion (wersjonowane):
- `...\curated\selection_matrix_stage1\matrix_K5_identity_cover_b1500_sel42.json`
- `...\curated\selection_matrix_stage1\matrix_K5b_identity_cover_plus_unseen_b1500_sel42.json`
- `...\curated\selection_matrix_stage1\matrix_K5c_closure_near_b1500_sel42.json`
- `...\curated\selection_matrix_stage1\matrix_K5d_closure_far_b1500_sel42.json`
- `...\curated\selection_matrix_stage1\K5_MANIFEST.json`

Dokumenty:
- `...\docs\Analiza_ponownej_publikacji\38_KRYSTALIZACJA_PRZESTRZENI_STANOW_W_LATENCIE_DETEKTORA.md` (sekcje 4k i 4l,
  odsylacz w 4j)
- `...\docs\Analiza_ponownej_publikacji\39_EKSPERYMENT_DOBORU_KLATEK_ETAP_1.md` (sekcja 5f z tabelami pokrycia,
  tego co niesie kod, sprawdzen siatki i odniesienia do niedzwiedzia; nowy kontrargument o K4)
- `...\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_widmo_tozsamosci.json`
- `...\docs\Analiza_ponownej_publikacji\evidence_koopman_langevin_2026-09-21\REPORT_slad_po_przejsciu.json`

Dane i rysunki poza gitem:
- `...\results\koopman_langevin_epochs\KOD_TOZSAMOSCI_pool_b0500.npz`
- `...\results\koopman_langevin_epochs\rysunki_k5\01_klatka_i_kwadraty.png` oraz 02 do 06

Na Edenie: `~/smallvit_runs/2026-09-21_koopman_langevin/czastki/CZASTKI_pool_b0500` i
`~/smallvit_runs/2026-09-21_koopman_langevin/KOD_TOZSAMOSCI_pool_b0500.npz`.

## Otwarte problemy

- Straznik determinizmu rejestratora czastek nadal nie przechodzi (0,131 jednostki wspolrzednej, podejrzenie:
  jadra zalezne od wielkosci partii). Porownan klatka po klatce nie wolno na tym opierac.
- Kod tozsamosci liczony jest na osmiu modach po symetryzacji, a sekcja 4k pokazuje, ze wlasciwe jest 18 liczb
  z trzech rodzin. Ramiona K5 trzeba przed etapem 2 przeliczyc.
- Tokeny narzedzia pochodza z wnetrza ramki detektora, wiec przy cienkim narzedziu czesc opisuje tlo.
  Wersja z maska (SAM lub obrys) nie byla probowana.
- Zwrotu narzedzia (gdzie jest koncowka) nie niesie zadna rodzina modow; przyczyna lezy prawdopodobnie
  w usrednianiu tokenow po calej ramce, nie w symetryzacji.
- Zaden trening ramion K5 nie zostal uruchomiony i nie zostanie bez zgody autora.

## Nastepne kroki

1. Przeliczyc kod tozsamosci puli na trzech rodzinach modow (18 liczb) i odbudowac ramiona K5.
2. Przedstawic autorowi plan etapu 2 do zatwierdzenia: B2, B3, K4 z poprawionym k-centrum, K5 i K5c,
   trzy ziarna treningu, budzet 1500, 12 epok; kontrast glowny K5 wobec rownomiernego w czasie, kontrast
   pomocniczy K5 wobec K5c (rzadkie stany wobec stanow juz rozumianych). Szacowany koszt: dziewiec biegow
   po okolo 25 minut na A100, okolo godziny przy pieciu procesach rownoleglych.
3. Naprawic straznik determinizmu rejestratora (stala wielkosc partii albo wylaczone autocasty).
4. Rozwazyc wersje tokenow narzedzia z maska zamiast ramki, zeby pary translacyjne nie mieszaly tla.

---

Uwaga o miejscu zapisu: ta sesja dotyczy pracy w repozytorium ViT i tutaj jest jej miejsce docelowe.
Kopia tego samego opisu lezy tez w archiwum projektu grantowego
(`F:\Studia\MCHTR\Grant\Projekt\.sessions\`), bo tam biegl zapis pierwotny.
