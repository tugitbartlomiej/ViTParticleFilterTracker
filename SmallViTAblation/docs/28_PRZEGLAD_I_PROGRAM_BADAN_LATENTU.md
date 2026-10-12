# 28. Przegląd i program badań: ruch latentu, tożsamość narzędzia i dobór klatek w małym ViT

Data: 2026-10-05. Status: przegląd wykonanych prac, literatura, cztery nowe pomiary bez treningu i PROPOZYCJA
dalszych badań. Nie uruchomiono żadnego uczenia, generowania obrazów ani zadania na Edenie. Każdy etap z rozdziału 9,
który uczy sieć, wymaga osobnego planu i zgody.

**Po co ten dokument.** Aparat, którym badamy latent (latent jako cząstka, operator Koopmana, równanie Langevina z
pamięcią, niezmienniki tożsamości, rozdział na prognozę uczciwą i warunkową, diagnostyka pod dobór danych), powstał
w projekcie grantowym na szeregach czasowych, a od 21 września jest przenoszony tutaj, na obrazy. Poprzedni przegląd
z 5 października leżał w projekcie grantowym i opierał się głównie na audycie eksperymentów z wyrocznią J128. Ten
dokument jest jego następcą osadzonym w `SmallViTAblation`: zbiera wszystko, co tu zrobiono, zestawia to z tym, jak
ta sama metoda zachowała się w szeregach czasowych, dokłada literaturę i mówi, jak badać to dalej w tym repozytorium.

Odpowiada na cztery pytania użytkownika:

1. Jak wykorzystać aparat do prognozowania tego, jak zachowa się wektor tożsamości narzędzia.
2. Jak dobierać zdjęcia, żeby przyspieszyć uczenie.
3. Czy szukanie ustawień latentu o najwyższym mAP i rysowanie z nich obrazu przez Stable Diffusion ma naukowe
   uzasadnienie, potrzebne do publikacji.
4. Jak podłączyć do tego równanie Langevina i operator Koopmana.

**Jak czytać.** Rozdział 0 to cała odpowiedź na jednej stronie. Rozdział 7 rozwija cztery odpowiedzi, rozdział 9 to
program badań z kodem, danymi i kryteriami. Rozdziały 2-6 to materiał dowodowy (aparat z szeregów czasowych, mapa
wykonanych eksperymentów, nowe pomiary), rozdział 8 to literatura, rozdział 12 to spis prac z odnośnikami.

**Materiał źródłowy.** Pięć streszczeń całej linii prac z liczbami i ścieżkami leży w
`docs/28_material_przeglad_2026-10-05/` (pliki 01-05). Poprzedni przegląd, audyt i plan sterowania z 4 października
są skopiowane do `docs/latent_sd_20261005/`.

<!-- DALEJ-0 -->

## 1. Słownik (zwykłymi słowami)

- **Latent**: liczby wewnątrz sieci opisujące jeden kawałek obrazu (token). Tutaj 768 liczb po funkcji GELU w
  ostatnim bloku małego ViT; obraz 480 × 320 ma 600 tokenów.
- **Głowa**: ostatnia część sieci, która z latentu robi ramkę narzędzia i pewność wykrycia.
- **Sondy**: 192 stałe obrazki (12 wycinków w 16 wersjach, 180 z narzędziem), których latent zapisujemy w trakcie
  uczenia. Latent sondy to średnia po tokenach narzędzia.
- **Cząstka**: jeden śledzony latent (sondy albo tokenu) traktowany jak punkt, który się porusza.
- **Zegar wideo**: kolejne klatki filmu przy zamrożonych wagach. **Zegar uczenia**: kolejne kroki optymalizatora przy
  tym samym obrazie. To dwa różne ruchy tej samej cząstki.
- **Wektor tożsamości**: kierunek w przestrzeni latentu oddzielający narzędzie od tła. **Kierunek stały**: wyznaczony
  raz, na końcu rozgrzewki (tak działa pętla z docs/26). **Kierunek odświeżany**: wyznaczany od nowa w każdej chwili.
  **Siła tożsamości**: rzut latentu narzędzia na ten kierunek.
- **J128**: 128 kanałów latentu wybranych na parach „klatka z narzędziem i ta sama klatka z narzędziem usuniętym”
  (400 par, usuwanie programem LaMa).
- **Paczka**: 32 obrazy jednego kroku uczenia. **Pchnięcie paczki**: o ile przesunie latent sond sama część kroku
  AdamW pochodząca z gradientu tej paczki (`S_nowa` w docs/23). **Pęd**: część kroku pochodząca z pamięci
  poprzednich gradientów (`S_ped`).
- **Tryb warunkowy (wyjaśnienie)**: przewidujemy krok, znając paczkę, która go tworzy. **Tryb uczciwy (prognoza)**:
  znamy tylko stany do chwili `t`, paczki do `t−1` i stan optymalizatora.
- **Umiejętność**: jaka część błędu prognozy „nic się nie zmieni” została usunięta (1 to prognoza idealna, 0 to tyle
  co brak zmiany). Podawana jako ułamek albo w procentach.
- **Funkcja odpowiedzi**: ile z jednorazowego pchnięcia zostaje w latencie po `h` krokach, w jednostkach pierwszego
  kroku.
- **Wyrocznia**: procedura, która przy zamrożonych wagach poprawia same liczby latentu tak, żeby głowa trafiła w
  prawdziwą ramkę. Zna etykietę, więc nie jest detektorem. **Idealny latent** to jej wynik.
- **Obrót**: 500 kroków uczenia, przed którymi ramię dokłada 134 klatki. **Ramię**: jeden wariant doboru klatek
  uczony od tego samego stanu po rozgrzewce. **Kotwica**: sonda użyta jako wzorzec dla obrazu ze Stable Diffusion.
- **Zbiór kontrolny**: oznaczone klatki, na których mierzymy błąd detektora, ale na których sieć się nie uczy.

## 2. Skąd przyszedł aparat: latent sieci TFT w szeregach czasowych

Źródła: `docs/28_material_przeglad_2026-10-05/04_aparat_w_szeregach_czasowych.md` oraz wskazane tam pliki projektu
grantowego (`F:\Studia\MCHTR\Grant\Projekt\experiments\tft_langevin_*`, `tft_deep_koopman_mz`,
`Research\syntheses\schemat-diagnostyki-latentu-koopman-langevin-2026-09-18.md`).

### 2.1 Obiekty

- **Cząstka**: jedna seria sklep-produkt ze zbioru Favorita (174 685 serii). **Położenie** `z_t`: latent enkodera TFT
  (192 liczby) dla okna 56 dni kończącego się w dniu `t`. **Zegar**: dni; wagi sieci zamrożone.
- **Wejścia** `u_t`: kalendarz, plan promocji i to, co zaszło danego dnia. Część jest znana z góry (promocje,
  święta), część nie (sprzedaż jutra).
- **Równanie Langevina w latencie**: `z(t+1) = z(t) + D + JP + W + N`, gdzie `D` to dryf zależny od położenia,
  prędkości i przyspieszenia, `JP` to jądro pamięci 56 dni, `W` to wymuszenie przez wejścia, `N` to przewidywalna
  część dawnych reszt. Cztery człony są osobne i każdy da się wyłączyć.
- **Koopman z wejściem**: `g(t+1) = K g(t) + B u(t)` po nieliniowym podniesieniu `g = φ(z)`, z opcjonalną pamięcią
  Mori-Zwanziga na opóźnieniach.

### 2.2 Wyniki, które mają znaczenie tutaj

| ustalenie w szeregach czasowych | liczby |
|---|---|
| Dwa tryby dają różne wyniki i nie wolno ich mylić | uczciwy (L3, jutro znane tylko z planu promocji i kalendarza) 79,8; warunkowy (L2, znany cały dzień jutra) 90,4; na 9074 nowych seriach 48,3 i 57,5-60,1 |
| Luka między trybami to nieznane wejście jutra | modele warunkowe z prognozą sprzedaży zamiast prawdziwej: 66-67; „około 23 punktów to nieprzewidywalna część jutrzejszej sprzedaży” |
| Wymuszenie niesie większość wyniku, pamięć mało | bez członu `W` spadek o 30-60 punktów; jądro pamięci 0,4-5 punktów; w Koopmanie pamięć „mierzalna, ale nie przewiduje” |
| Wysoki wynik na jeden krok nie znaczy, że model opisuje dynamikę | bieg swobodny Koopmana: 73 po 3 dniach, −43 po 7 dniach |
| Większość zmienności latentu to tożsamość, nie ruch | 92 procent wariancji to „adres” serii (jej średnia po czasie), 7,6 procent to ruch |
| Dobór danych według wielkości błędu szkodzi | ważenie serii energią reszty: mediana umiejętności per seria 78,3 → 65,8; te same wagi przetasowane: bez zmian |
| Wynik łączny bywa zdominowany przez garstkę przypadków | 1 procent serii niesie 73-85 procent mianownika; mediana per seria 76-78 przy wyniku łącznym od 48 do 80 |
| Kopia sieci to nie równanie | destylacja dała 95,1, ale to „przepisany TFT” (696 tysięcy parametrów); celem jest jawne równanie z wyłączalnymi członami |

### 2.3 Schemat diagnostyki, który stąd wynikł

Twierdzenie o rozkładzie błędu (dowód w Lean, `DiagnostykaInnowacji.lean`, 13 twierdzeń): dla każdego predyktora
błąd dzieli się na część nieusuwalną (to, czego z dostępnej historii przewidzieć się nie da) i usuwalną (brak w
modelu). Wnioski, które przenoszą się na obrazy bez zmian:

1. Sufit umiejętności wyznacza informacja dostępna w chwili prognozy, a nie klasa modelu. Podnosi go tylko bogatsza
   historia, czyli w praktyce przejście z trybu uczciwego do warunkowego.
2. Duża reszta modelu to suma zaskoczenia i braku w modelu; sama jej wielkość ich nie rozdziela. Dlatego dobór
   danych ma celować w część usuwalną i w przyczynę błędu, a nie w jego wielkość (reguła RHO-loss).
3. Część C schematu przenosiła aparat na zegar uczenia jako projekt, bez pomiaru. Pomiary z tego repozytorium są
   pierwszym sprawdzeniem tej części.

### 2.4 Lekcje metodologiczne zapisane w projekcie grantowym

- Tryb (uczciwy albo warunkowy) nazywać przy każdym wyniku; porównanie liczb z dwóch trybów to błąd kategorii.
- Podłogą jest „brak ruchu”; obok niej placebo, które niszczy tylko badany związek.
- Kryteria i wybór modelu zapisywać przed testem; walidację odsuwać w czasie.
- Raportować wynik łączny, medianę i wynik bez 1 procenta największych przypadków.
- Każdy plik wyników ma sekcję „Czego to NIE dowodzi”.

## 3. Przekład pojęć: szereg czasowy, wideo, uczenie

| pojęcie | szeregi czasowe (TFT) | obrazy, zegar wideo | obrazy, zegar uczenia |
|---|---|---|---|
| cząstka | latent jednej serii sklep-produkt | latent tokenu albo klatki | latent stałej sondy albo stałego tokenu |
| krok zegara | dzień | klatka filmu (albo kilka klatek) | krok AdamW |
| co się zmienia | dane wchodzące do okna | obraz | wagi sieci |
| co stoi | wagi | wagi | obraz |
| wejście | kalendarz, promocje, sprzedaż dnia | ruch ręki chirurga i kamery (nieobserwowany) | skład paczki i stan optymalizatora |
| co znane z góry | promocje, święta | nic | stan optymalizatora (pęd), harmonogram, paczka, jeśli sami ją wybieramy |
| tryb warunkowy | znany cały dzień jutra: 90,4 | nie istnieje (przyszłej klatki nie znamy) | znana paczka kroku: 0,95-0,98 na jeden krok |
| tryb uczciwy | plan promocji i kalendarz: 79,8 | historia klatek: błąd równania z pamięcią o 18 procent mniejszy niż „bez zmian” | historia i pęd: 0,64-0,68 na jeden krok, 0,21 na 10 kroków |
| „adres” (stała część) | średnia serii, 92 procent wariancji | odcisk operacji: 72 procent wariancji wolnych modów | środek, wokół którego latent sondy drga |
| liczba cząstek | 72 600 uczących | 4800-57 000 klatek | 180 sond, 218-7000 tokenów, 1-10 przebiegów |

Dwie różnice budowy, od których zależy wszystko dalej:

1. **W szeregach czasowych tryb warunkowy był tylko diagnozą, w zegarze uczenia jest zadaniem sterowania.** Sprzedaży
   jutra nie wybieramy, więc wynik 90,4 mówił tylko, ile dałaby wiedza, której nie mamy. Paczkę następnego kroku
   wybieramy sami. Tryb warunkowy opisuje więc dokładnie to, czym możemy sterować, a jego wysoki wynik na jeden krok
   jest podstawą doboru, a nie ciekawostką.
2. **W szeregach czasowych wejście przychodziło z zewnątrz, w uczeniu wejście zależy od stanu.** Promocja nie wie,
   gdzie jest latent. Pchnięcie paczki to gradient policzony w bieżącym stanie sieci, więc „wie”, gdzie sieć jest, i
   w dużej części cofa to, co zaszło przed chwilą. Tego sprzężenia w aparacie z szeregów czasowych nie było. Zmierzył
   je dopiero pomiar funkcji odpowiedzi z 5 października (rozdział 6.4).

## 4. Co zrobiono w SmallViTAblation: mapa prac od 21 września do 5 października

Pełne opisy z liczbami i ścieżkami: `docs/28_material_przeglad_2026-10-05/01_...md` (RF-DETR), `02_...md` (mały ViT
do 1 października), `03_...md` (mały ViT od 1 października), oraz dokument 24 dla okresu do 30 września. Tutaj tylko
to, co potrzebne do wniosków. Wszystkie wartości mAP i AP to walidacja, nie test.

### 4.1 Zegar wideo: zamrożony RF-DETR Nano (21-23 września)

Pytanie tej części: czy równanie ruchu latentu wzdłuż filmu wskaże klatki, na których warto uczyć.

| eksperyment | co mierzono | wynik | werdykt |
|---|---|---|---|
| Etap 1 (`pipeline/koopman_langevin`) | czy stan narzędzia powstaje w latencie; macierz 27 treningów z doborem | odczyt położenia z latentu: R² od −0,10 (wagi COCO) do 0,86; sześć wolnych modów Koopmana 0,73-0,87 wobec 0,04-0,45 dla PCA; dobór według reszty Langevina: +0,0027 AP wobec odniesienia i tylko +0,0006 wobec placebo (próg +0,004) | sygnał ma „wartość diagnostyczną, nie selekcyjną” |
| `krystalizacja_latentu` | jak latent układa się w uczeniu; dobór po siatce tożsamości | szczyt trwałości modów w epoce 5 w 32 z 36 ustawień, przy dołku straty walidacyjnej (epoki 6, 6, 7); 250 klatek: metoda 0,6296, losowe 0,6319 | sygnał końca uczenia tak, dobór nie |
| `koopman_kierunki_stale` | klatki, w których stałe kierunki Koopmana mocno się zmieniają | 0,6696 wobec 0,6820 dla losowych (3 ziarna) | wynik negatywny |
| `dobor_cech_adaptacyjny` | „brakujące cechy” z priorytetem Koopman-GLE | losowe 0,7131; glister 0,7072; placebo 0,6941; cechy 0,6903; z Koopman-GLE 0,6839 | „żadna metoda doboru nie pokonała losowania” |
| `dobor_klatek_stan_sieci` | równanie z pamięcią na tokenach; pilot doboru | błąd tokenu: „bez zmian” 0,410-0,414, Koopman 0,369-0,373, równanie z pamięcią 0,333-0,340; AP 0,673 wobec 0,655 (losowe), jedno ziarno | prognoza lepsza od „bez zmian”; zysk doboru bez dowodu statystycznego |
| zbiór testowy, diagnoza chybień | co wskazuje klatki, na których detektor chybia | zaskoczenie Koopmana 0,585; wektor tożsamości pod obrysem 0,761; pewność detektora 0,882 (pole pod krzywą ROC) | zwykła pewność detektora jest lepszym wskaźnikiem niż reszta równania |

Tylko plany albo kod bez uruchomienia: `wybieracz_tozsamosc_koopman`, `wlokno_pomylek`, `tozsamosc_czastki_bledy`,
`koopman_dynamika_uczenia`.

Ustalenie o tożsamości z tej części: wolne mody całej klatki są odciskiem operacji (72 procent wariancji między
filmami), a nie stanem narzędzia; zbiór ma jedną klasę narzędzia, więc „które to narzędzie” jest niemierzalne, mierzalne
jest „narzędzie czy tło” i kształt ramki.

### 4.2 Zegar uczenia: dostrajany RF-DETR Nano i Large (24-28 września)

Pytanie: czy ruch latentu stałych obrazów między punktami kontrolnymi da się przewidzieć i użyć do doboru.

| eksperyment | wynik | werdykt |
|---|---|---|
| `koopman_gesta_historia`, `koopman_osobno_obraz_token`, `koopman_jeden_wektor` | Koopman na samym położeniu gorszy od „bez zmian” (0,061 wobec 0,047 na 100 aktualizacji); „zero z 41 472 macierzy spełniło bramkę”; jeden wektor: przewaga 2,8 procent przy progu 5 | stała macierz nie widzi, że trajektoria zwalnia, a następny krok zależy od prędkości, nie od położenia |
| `koopman_z_pamiecia` | stan z prędkością: błąd kroku 0,0037 wobec 0,0103 bez pamięci; przewaga nad „ostatnim przyrostem” 3,9 procent przy progu 5 | „pamięć naprawia Koopmana”, ale ledwie dorównuje prostej kontroli |
| `koopman_tozsamosc_kierunki` | jeden skalar tożsamości: operator na 4 przyrostach 0,171 wobec 0,617 dla „ostatniego przyrostu” na 100 aktualizacji; kierunek LDA obraca się (cosinus z początkiem 0,73, potem 0,64) przy stałej jakości rozdzielenia 0,995-0,998; dobór 134 klatek: +0,0007 wobec +0,0009 dla losowych | prognoza działa dla jednego skalara; dobór nie różni się od losowego |
| `langevin_latent_detektora` | pełne równanie 0,151 wobec 0,171 operatora i 0,482 samego dryfu (wczesne okno); w późnym oknie 0,557 wobec 0,454 | bramka niespełniona |
| `petla_uczenia_koopman_fastgan` (Nano), `petla_od_poczatku_large` (Large, 3 ziarna) | Nano: 0,9055 / 0,9045 / 0,9072 (pętla, losowe, bez dokładania); Large: pętla minus losowe −0,0030 | kryteria niespełnione |
| diagnoza porażki na Large | prognoza gorsza od „latent się nie ruszy” w 12 z 12 przypadków; dostrajany kręgosłup zmienia się o 0,026 procent; latent kręgosłupa wobec mAP: korelacja rang 0,07 | latent siedział w części sieci, która prawie się nie uczy i nie wiąże się z wynikiem |
| `dekoder_sondy_large`, `sterowanie_langevin_kalman` | stan dekodera wobec mAP w czasie 0,86, między ramionami −0,07 i −0,51 | bramki niespełnione; pętla sterowania nie poszła na Eden |

Wniosek, który doprowadził do małego ViT: w dostrajanym detektorze z gotowym kręgosłupem nie ma czego przewidywać ani
czym sterować, bo tożsamość jest odziedziczona, a kręgosłup stoi. Potrzebna była sieć uczona od zera, w której latent
naprawdę się rusza.

### 4.3 Mały ViT od zera: pętle doboru klatek (28 września - 1 października)

Sieć 2,93 mln parametrów, baza 1500 klatek, pula około 9 tysięcy, obrót to 500 kroków i 134 dołożone klatki.
Miara: pole pod krzywą mAP@0,5 (średnia ocen po rozgrzewce).

| wersja | ziarna | ramiona i pole pod krzywą | wobec losowych | wobec tej samej pętli bez prognozy |
|---|---:|---|---|---|
| v1 (lista neuronów, FastGAN) | 11 | pętla 0,4426; losowe 0,4298; bez dokładania 0,4044 | +0,013, 9 z 11 | brak takiego ramienia; prognoza równa „bez zmian” |
| v2 (wektor 768) | 10 | 0,4432; 0,4341; 0,3999 | +0,009, 8 z 10 | prognoza wektora lepsza od „bez zmian” w 0 z 6 obrotów |
| v3 (siła tożsamości) | 10 | pętla 0,4432; ta sama bez prognozy 0,4428; losowe 0,4327; kolejne 0,4255; bez 0,4030 | +0,0105, 7 z 10 | +0,0004, 5 z 10 |
| v3, iteracja 2 | 20 | 0,478; 0,473; 0,468; 0,463; 0,416 | +0,010, 15 z 20 | 13 z 20, na końcu −0,003 |
| 4.3 (Langevin z jądrem, Stable Diffusion) | 10 | metoda 0,4883; ta sama ze starszą prognozą (Koopman/Langevin wersji 3) 0,4874; SD bez prognozy 0,4871; losowe 0,4720; kolejne 0,4646; bez 0,4219 | +0,0163, 10 z 10 | +0,0012, 5 z 10 |

Co z tego wynika, zgodnie z dokumentem 24 (sekcja E):

- Dokładanie klatek wyszukanych jako podobne do wycinków narzędzia uczy szybciej niż klatki losowe i kolejne. To
  powtórzyło się w pięciu kampaniach.
- **W żadnym porównaniu, w którym prognoza jest jedyną różnicą, prognoza nie dała nic.** Trafniejsza prognoza (28,7
  procent wobec 16,2 procent umiejętności na 500 kroków) nie zmieniła doboru.
- Wkład generatora nie jest rozdzielony: nie było ramienia „wyszukanie z prawdziwej klatki kotwicy, bez obrazu z
  generatora”.
- Przewaga nad losowymi maleje pod koniec uczenia (na końcu 4.3: +0,022, 7 z 10).

### 4.4 Mały ViT: równanie ruchu latentu (30 września - 3 października)

| pytanie | wynik | źródło |
|---|---|---|
| Z czego składa się jeden krok ruchu latentu? | zmiana wag daje 97,8 procent przyrostu; równanie `Δz(t+1) = 0,83·Δz(t) + 0,98·S(t)` (bezwładność plus siła paczki) daje 85,0 (od 80,5 do 89,2 w 10 ziarnach); siła paczki i pędu policzone osobno 94,9 (od 93,9 do 95,8); placebo (paczka z innego kroku) 59,1, czyli tyle co bez paczki | `maly_vit_rownanie_latentu`, docs/23 |
| Ile da się przewidzieć bez paczki bieżącej? | sam pęd 64-67; bezwładność 56-57; uczone macierze 768 × 768 50-52; sieci z 11 mln parametrów 39-41 (i wymuszenie z przeszłych paczek równe placebo) | `maly_vit_rownanie_macierze`, `maly_vit_tozsamosc_jeden_krok` |
| Na ile kroków trzyma równanie kroku? | siły liczone w prawdziwych wagach każdego kroku: 97-98 na 1-5 kroków i co najmniej 90 do 150 kroków; siły liczone w wagach z chwili startu przy znanych przyszłych paczkach: 98, 98, 88, 6 na 1, 2, 3, 5 kroków; sam pęd: 62, 38, 20 | `maly_vit_horyzont_rownania_krok` |
| Ile z samej historii na wiele kroków? | najlepszy model (Langevin z jądrem 20 i pędem): 69, 58, 45, 26, 21 na 1, 2, 3, 5, 10 kroków i 10-28 dalej; Koopman z 20 opóźnieniami daje te same liczby | `maly_vit_prognoza_tozsamosci_z_historii` |
| Czy to wina losowych paczek? | paczka 32, 128, 320: prognoza na 10 kroków 0,20, 0,24, 0,24; wielkość kroku bez zmian | `maly_vit_gesta_historia_paczka` |
| Jaki jest sufit na 100 kroków? | z samej historii 0,31-0,35; na 10 kroków z historią, pędem i zanikiem wag 0,22; z prawdziwą pełną siłą każdego kroku 0,985 | `maly_vit_sufit_przewidywalnosci_100` |

To jest najmocniejszy pozytywny wynik całej linii: **jeden krok ruchu latentu jest w 95 procentach wyjaśniony
jawnym, dwuczłonowym równaniem bez dopasowywanych macierzy**, a współczynniki przenoszą się między ziarnami. I
najważniejsze ograniczenie: to wyjaśnienie przy znanej paczce, nie prognoza.

### 4.5 Mały ViT: tożsamość narzędzia

| definicja | jak liczona | co o niej wiadomo |
|---|---|---|
| lista neuronów (v1) | neurony „zapalone” na narzędziu w większości wersji sondy | rdzeń wspólny wszystkim wersjom: 0 neuronów w każdym ziarnie; lista kurczy się z 80-88 do 33 |
| kierunek `stary` (v3, 4.x) | LDA, narzędzie przeciw tłu sond, stały od kroku 1000 | rozdzielenie doskonałe na sondach, ale wobec tego samego miejsca bez narzędzia tylko 2,0 odchylenia; sieć nieuczona też przechodzi test; siła wzdłuż niego spada w uczeniu (6,13 → 5,30 bez dokładania, → 4,36 z dokładaniem) |
| kierunek `nowy` | narzędzie przeciw temu samemu miejscu bez narzędzia | rozdzielenie 16,6; cosinus ze `starym` 0,28; w pętli nieużyty; nie jest lepiej przewidywalny |
| 8 kierunków z par | podprzestrzeń z 400 par „z narzędziem i bez” (LaMa), krok 1000 | trafność zbilansowana 88,6 procent; jedyna definicja z próbą przyczynową: podmiana tych kierunków zmienia pewność razy pokrycie ramki z 0,362 na 0,343 (przedział zmiany poniżej zera), dla AP50 bez rozstrzygnięcia |
| J128 | 128 kanałów z tych samych par, wybór na kroku 500 | trafność zbilansowana 92,8 procent, przedział od 88,2 do 97,0; sprawdzona tylko na kroku 500; próby przyczynowej dla samych J128 nie było |

Stabilność: cosinus kierunków między dwoma ziarnami 0,085; cosinus prostego wektora z wektorem końcowym rośnie w
uczeniu od 0,20 po 11 epokach do 0,83 (kolejne pomiary: 0,20; 0,41; 0,59; 0,75; 0,83); osie o największym udziale
zmieniają się między krokiem 500 a 1000. Kierunek tożsamości nie jest więc ani wspólny dla przebiegów, ani stały w
jednym przebiegu.

### 4.6 Mały ViT: cząsteczki, operatory i sieci Koopmana (1-3 października)

| eksperyment | wynik |
|---|---|
| gęsta historia tokenów (1001 stanów co krok, 22 GB) | tokeny narzędzia ruszają się mocniej niż tło (0,76), ale nie w swoistym kierunku (0,52); 192 osie niosą 90 procent energii ruchu |
| równanie Langevina na J128 i na 8 kierunkach z par | jeden krok: 48-60 procent i 72-79 procent zgodnych znaków; 10 kroków: −4 do 9 procent; stała prędkość wygrywa na jednym kroku w J128 |
| cel 85 procent zgodnych znaków na 500 kroków (99 konfiguracji) | najlepszy model 56-62 procent; „nie osiągnięto celu” |
| EDMD 128 × 128 i sieć NNDMD 768 (34 obrazy), NNDMD na 1500 obrazach (Eden) | wszystkie gorsze od „bez zmian” na każdym horyzoncie; NNDMD na 1500 obrazach: błąd większy o 15 procent |
| identyfikacja z historycznymi paczkami | 500 kroków: 22,65 wobec 22,64 procent, czyli historia paczek nic nie dodaje |

### 4.7 Wyrocznia J128 i obraz ze Stable Diffusion (4-5 października, kod w projekcie grantowym)

Eksperymenty `vit_latent128_maksymalizacja`, `vit_latent128_obraz_sd` i pokrewne leżą w
`F:\Studia\MCHTR\Grant\Projekt\experiments\`; opisuje je audyt w `docs/latent_sd_20261005/`. Wynik: przy zamrożonych
wagach poprawienie samych liczb latentu (128 kanałów w 600 tokenach, z etykietą) podnosi AP50 z 0,44 do 0,88, głównie
przez dokładniejszą ramkę (AP75 z 0,02 do 0,77). Obraz narysowany pod ten latent zmienił latent o 1,35 procent drogi
do celu, w kierunku bez związku z celem (cosinus 0,03). Liczby i przyczyna w rozdziale 6.1.

### 4.8 Bilans w pięciu zdaniach

1. Jeden krok ruchu latentu małego ViT jest zrozumiany: pęd plus pchnięcie paczki, 95 procent.
2. Wszystko powyżej kilku kroków bez wiedzy o paczkach kończy się na 0,2-0,35, niezależnie od modelu (Koopman,
   Langevin z jądrem, EDMD, sieci), paczki i definicji tożsamości.
3. Dobór przez wyszukiwanie klatek podobnych do narzędzia przyspiesza mały ViT o 0,009-0,016 pola pod krzywą wobec
   losowych; prognoza nie ma w tym udziału, a na dostrajanym RF-DETR nic nie pokonało losowania.
4. Wektor tożsamości w definicji używanej w pętli mierzy głównie scenę, obraca się w uczeniu i jest inny w każdym
   ziarnie.
5. Idealny latent z wyroczni pokazuje, czego brakuje głowie (dokładnej ramki), ale leży poza zasięgiem obrazów.

## 5. Co powtórzyło się między szeregami czasowymi a obrazami, a co nie

| ustalenie | szeregi czasowe | obrazy | zgodność |
|---|---|---|---|
| wymuszenie niesie wynik, pamięć mało | bez wymuszenia −30 do −60 punktów; jądro 0,4-5 punktów | siła paczki +26 punktów; jądro około 4 punktów ponad bezwładność | tak |
| luka między trybem warunkowym i uczciwym to nieznane wejście | 90,4 wobec 79,8; sufit bez jutra 66-67 | 0,95 wobec 0,65 na krok; 0,985 wobec 0,22 na 10 kroków | tak, w obrazach luka jest dużo większa |
| proste równanie bije duże modele | uczone macierze i destylacja przeuczają się (94 na cząstkach uczących, 62 na nowych) | skalar 56-57 bije sieci (39-41) i macierze (50-52); NNDMD gorsze od „bez zmian” | tak |
| wysoki wynik na krok nie daje biegu na wiele kroków | Koopman: 73 po 3 dniach, −43 po 7 | równanie kroku przy zamrożonych wagach: 98, 88, 6 na 2, 3, 5 kroków | tak |
| stała część dominuje nad ruchem | „adres” serii 92 procent wariancji | odcisk operacji 72 procent; latent sondy drga wokół środka | tak |
| dobór po wielkości błędu albo po reszcie równania nie pomaga | mediana 78,3 → 65,8 | reszta Langevina równa placebo; „żadna metoda nie pokonała losowania” na RF-DETR | tak |
| reszta równania jako wskaźnik trudności | zaskoczenie zmienia kierunek kroku, nie długość (korelacja 0,08) | zaskoczenie Koopmana 0,585 wobec 0,882 dla pewności detektora | tak |
| tożsamość jest stabilna | adres serii stały; odtworzony z zachowania bez identyfikatora | kierunek obraca się w uczeniu (0,20 → 0,83) i jest inny w każdym ziarnie (0,085) | **nie** |
| wejście jest zewnętrzne | promocje nie zależą od latentu | pchnięcie paczki zależy od stanu sieci i cofa poprzedni ruch | **nie** |
| tryb warunkowy da się wykorzystać | nie, jutrzejszej sprzedaży nie wybieramy | tak, paczkę wybieramy sami | **nie**, i to jest szansa |
| liczba niezależnych cząstek | 72 600 serii | 180 sond z 12 wycinków, 1-10 przebiegów | **nie**, tutaj próba jest o rzędy wielkości mniejsza |

Najkrócej: wszystko, co aparat mówił o granicach prognozy, potwierdziło się na obrazach ostrzej niż w szeregach
czasowych. Nie przeniosły się dwie rzeczy, na których opierał się pomysł pętli: stała tożsamość i zewnętrzne wejście.
Doszła jedna rzecz, której w szeregach czasowych nie było: możliwość wyboru wejścia.

## 6. Nowe pomiary z 5 października (wszystkie bez treningu, na zapisanych plikach)

Każdy pomiar ma własny katalog z planem zapisanym przed uruchomieniem, kodem, wynikami i sekcją „Czego to NIE
dowodzi”. Dwa pierwsze (6.1, 6.2) są opisane szerzej w `docs/latent_sd_20261005/vit-idealny-latent-sd-langevin-dobor-danych-2026-10-05.md`.

### 6.1 Idealny latent z wyroczni leży poza zasięgiem obrazów

Źródło: 400 obrazów × 600 tokenów × 128 kanałów z eksperymentu `vit_latent128_maksymalizacja`; liczby w
`docs/latent_sd_20261005/` (plik przeglądu, rozdział 2) i w `OSIAGALNOSC.json` obok skryptu w projekcie grantowym.

| wielkość | prawdziwe tokeny | cele wyroczni |
|---|---:|---:|
| typowa zmiana jednej współrzędnej, w naturalnych odchyleniach kanału | | 22 |
| współrzędne zmienione o ponad 10 odchyleń | | 17 procent |
| nietypowość tokenu (odległość od środka chmury prawdziwych tokenów; 1 to „typowy”): mediana i maksimum | 0,90 i 4,6 | 26 i 593 |
| cele bardziej nietypowe niż najbardziej nietypowy z 240 tysięcy prawdziwych tokenów | | 99,1 procent |
| obraz ze Stable Diffusion: przebyta część drogi do celu i cosinus z kierunkiem celu | | 1,35 procent i 0,03 |

Obraz zmienił latent o wielkość naturalną (17 procent normy), ale cel leży kilkadziesiąt razy dalej i w kierunku, z
którym naturalna zmienność obrazów nie ma związku. Dla porównania: w pętli 4.3 cel dla Stable Diffusion leżał blisko
(latent kotwicy przesunięty wzdłuż kierunku tożsamości o przewidzianą zmianę) i był pokrywany w 88 procentach.
Różnica między 88 a 1,3 procent to różnica między celem w zasięgu i poza nim.

### 6.2 Siła tożsamości wzdłuż kierunku stałego nie idzie w parze z mAP (kampania 4.3)

Źródło: telemetria 10 ziaren × 6 ramion × 14 obrotów z `experiments/maly_vit_petla_sd_rownanie/wyniki_eden43`.

| ramię | średnia siła tożsamości | pole pod krzywą mAP@0,5 |
|---|---:|---:|
| bez dokładania | 4,88 | 0,422 |
| kolejne | 4,52 | 0,465 |
| losowe | 4,51 | 0,472 |
| metoda | 4,41 | 0,488 |
| metoda ze starszą prognozą | 4,30 | 0,487 |
| SD bez prognozy | 4,19 | 0,487 |

Kolejność jest odwrócona (korelacja rang −0,77). W czasie tak samo: siła spada z 5,26 do 3,95, mAP rośnie z 0,36 do
0,53. Po odjęciu średniej z ramion w każdej parze (ziarno, obrót): zmiana siły wobec zmiany mAP 0,006 (przedział od
−0,07 do 0,08), przewidziane pchnięcie wobec zmiany siły w obrocie 0,21 (od 0,12 do 0,32), przewidziane pchnięcie
wobec zmiany mAP −0,06. Lukę tego obliczenia (kierunek był stały) zamyka pomiar 6.6.

### 6.3 Próg 85 procent jest spełniony na jednym kroku ze znaną paczką, w każdym obrocie kampanii

Źródło: 840 rekordów tej samej telemetrii; plik `docs/28_material_przeglad_2026-10-05/TELEMETRIA43_DWA_TRYBY.json`.
W każdym obrocie pętla sprawdzała równanie jednego kroku na prawdziwych krokach.

| co przewidywane | co znane | zgodność znaku (mediana) | umiejętność (mediana) |
|---|---|---:|---:|
| zmiana siły tożsamości, 1 krok | paczka kroku i pęd | 0,946 (najgorszy obrót 0,919; wszystkie 840 powyżej 0,85) | 96,3 procent |
| cały latent sond, 1 krok | paczka kroku i pęd | | 96,5 procent |
| zmiana siły tożsamości, 1 krok | sam pęd | | 68,0 procent |
| zmiana siły tożsamości, 500 kroków | historia (Langevin z jądrem) | 0,78 (na większej połowie zmian) | 28,7 procent |
| zmiana siły tożsamości, 500 kroków | historia (Koopman/Langevin wersji 3) | 0,72 | 15,4 procent |

Cel „ponad 85 procent zgodnych znaków” jest więc od dawna osiągnięty, ale w trybie warunkowym i na jednym kroku. W
trybie uczciwym na 500 kroków nie jest i, jak pokazują 6.4 i 6.5, być nie może.

### 6.4 Funkcja odpowiedzi: ile zostaje z pchnięcia jednej paczki

Katalog: `experiments/maly_vit_odpowiedz_latentu_na_paczke/` (plan, poprawka po wyniku, `WYNIKI.md`). Dane: dwa
nagrania co krok z siłami (ziarna 47 i 48, kroki 1000-3000). Dla każdego horyzontu `h` regresja przesunięcia latentu
180 sond po `h` krokach na pchnięcie bieżącej paczki, pęd, zanik wag i odchylenie od średniej z ostatnich 101 kroków.

| horyzont (kroki) | 1 | 2 | 3 | 5 | 10 | 20 | 50 | 100 | 500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| odpowiedź na pchnięcie paczki (w jednostkach pierwszego kroku) | 0,98 | 1,97 | 2,72 | 3,22 | 2,40 | 1,23 | 0,64 | 0,51 | 0,39 |
| to samo na drugim ziarnie | 0,98 | 1,98 | 2,73 | 3,17 | 2,41 | 1,32 | 0,66 | 0,47 | 0,30 |
| gdyby pchnięcie tylko powtarzało się przez pęd (β1 = 0,9) | 1,00 | 1,90 | 2,71 | 4,10 | 6,51 | 8,78 | 9,95 | 10,0 | 10,0 |
| ekstrapolacja równania z docs/23 | 0,98 | 1,79 | 2,47 | 3,49 | 4,87 | 5,63 | 5,76 | 5,76 | 5,76 |
| uczciwa prognoza całego ruchu (pęd, zanik, odchylenie) | 0,67 | 0,55 | 0,42 | 0,26 | 0,21 | 0,26 | 0,29 | 0,27 | 0,2 |
| o ile podnosi ją znajomość bieżącej paczki | +0,29 | +0,32 | +0,31 | +0,22 | +0,06 | +0,01 | +0,002 | +0,002 | +0,001 |

Pierwszy wiersz: ziarno 47, kroki 1000-2499, przedziały 95 procent w `WYNIKI.md` eksperymentu (na przykład 3,17-3,27
dla 5 kroków, 0,56-0,71 dla 50, 0,31-0,45 dla 500; powyżej 50 kroków przedziały są zaniżone). Dwa ostatnie wiersze:
ziarno 47, kroki 2000-2940; na ziarnie 48 różnice do 0,03.

Co widać:

- **Trzy pierwsze kroki to czysta mechanika optymalizatora**: pchnięcie powtarza się przez pamięć pędu dokładnie tak,
  jak wynika z β1 = 0,9.
- **Od czwartego kroku działa siła zwrotna.** Szczyt 3,2 w piątym kroku, potem powrót. Po 50 krokach zostaje 0,64
  pierwszego kroku, po 500 około 0,4 (dla części wspólnej wszystkim sondom około 0,7, dla części właściwej jednej
  sondzie około 0,24).
- **Skąd siła zwrotna.** Pchnięcia z następnych 2-4 kroków są skierowane przeciwnie do kroku, który właśnie zaszedł
  (korelacja −0,30 do −0,33), przeciwnie do pędu (−0,31 i −0,35) i przeciwnie do odchylenia od średniej (−0,21).
  Autokorelacja samych pchnięć: +0,23 po jednym kroku, −0,14 do −0,19 po 3-5 krokach, zero od siódmego. To obraz
  drgań o okresie 8-10 kroków. Wszystko powtarza się na drugim ziarnie z dokładnością do 0,02.
- **Pchnięcie jest prawie w całości nową informacją**: pęd i odchylenie wyznaczają 5,5 procent pchnięcia, razem z
  wcześniejszymi pchnięciami 11-15 procent.
- **Składanie pchnięć ze stałą funkcją odpowiedzi nie działa**: nawet gdy znamy wszystkie paczki w oknie i każde
  pchnięcie liczymy w jego prawdziwym stanie, na 10 kroków wychodzi 0,63 (kryterium 0,85 niespełnione). Z prawdziwym
  pędem każdego kroku wychodzi 0,985. Brakujący składnik to pęd policzony w stanie, w którym pada.

Znaczenie dla doboru: ocena paczki po jednym kroku mierzy głównie fazę drgań; 87 procent szczytowego skutku jest
potem cofane. Liczy się część, która zostaje.

### 6.5 Wagi błądzą losowo ze słabym dryfem, latent sond drga w miejscu

Katalog: `experiments/maly_vit_kregoslup_trajektorii_wag/`. Dane: 101 zapisów wag co 10 kroków i latent sond z biegu
gęstej historii (ziarno 47, kroki 0-1000), miary od kroku 300.

| wielkość | wagi całej sieci | wagi głowy | latent sond | błądzenie losowe |
|---|---:|---:|---:|---:|
| udział pierwszego kierunku głównego w kwadracie przesunięcia | 0,886 | 0,906 | 0,762 | 0,811 |
| przesunięcie netto podzielone przez długość drogi | 0,176 | 0,180 | 0,039 | 0,120 |
| cosinus przesunięć w sąsiednich oknach 100 kroków | 0,06 | 0,15 | −0,36 | 0 |
| cosinus przesunięć w oknach 100 kroków odległych od siebie (udział trwałego dryfu) | 0,10 | 0,17 | 0,02 | 0 |
| to samo dla okien 200 kroków | 0,16 | 0,25 | 0,05 | 0 |

- Udział pierwszego kierunku (0,886) nie dowodzi „kręgosłupa”: błądzenie losowe daje 0,81. Kryterium z planu było
  postawione za nisko, co opisuje poprawka w katalogu eksperymentu.
- W wagach jest słaby trwały dryf: około 10 procent energii przesunięcia w oknie 100 kroków, 26 procent w oknie 350
  kroków. Najsilniejszy w końcowej normalizacji i w głowie, najsłabszy w łodydze.
- Latent sond nie oddala się, tylko drga wokół wolno płynącego środka: sąsiednie okna są przeciwne (−0,36; wielkość,
  która tylko drga, dałaby −0,5), a trwały dryf to 2-5 procent energii przesunięcia.

**Skąd bierze się sufit prognozy.** Jeśli wielkość drga wokół środka, najlepsza uczciwa prognoza to sam środek. Jej
błąd to jedna wariancja drgań, a błąd prognozy „bez zmian” to dwie, więc umiejętność nie przekroczy 0,5. Trwały dryf,
który mógłby dodać resztę, to w latencie kilka procent. Stąd 0,2-0,35 we wszystkich dotychczasowych próbach, bez
względu na model. Cel 85 procent na całym ruchu latentu w horyzoncie 10-500 kroków jest poza zasięgiem z powodu
budowy tego ruchu.

### 6.6 Tożsamość liczona kierunkiem odświeżanym: pełna od 50. kroku, nie rośnie, kierunek stały się starzeje

Katalog: `experiments/maly_vit_tozsamosc_odswiezana_a_map/`. Dane: latent 192 sond co 10 kroków z kampanii pętli v2,
10 ziaren, trzy ramiona, kroki 0-4500. Ten sam przepis na kierunek co w pętli (`sila.wektor`), zastosowany raz w
kroku 1000 (kierunek stały) i od nowa co 50 kroków (kierunek odświeżany, sprawdzany na wycinkach nieużytych do
dopasowania). Kontrola: siła stała odtwarza liczby z planu pętli v2 (6,13; 5,30; 4,36).

| krok | mAP@0,5 (bez / losowe / koopman) | rozdzielenie odświeżane | rozdzielenie wzdłuż kierunku stałego | siła stała (bez / losowe / koopman) | cosinus kierunku odświeżanego z kierunkiem z kroku 1000 |
|---:|---|---:|---:|---|---:|
| 50 | 0,000 | 10,7 | 3,0 | 3,41 | 0,01 |
| 300 | 0,10 | 8,5 | 3,0 | 5,18 | 0,15 |
| 950 | 0,32 | 7,3 | 9,2 | 5,70 | 0,74 |
| 1000 | 0,32 | 7,3 | 27,4 (chwila dopasowania) | 6,13 | 1,00 |
| 1050 | 0,32 | 7,3 | 8,6 | 5,92 | 0,77 |
| 1500 | 0,34 / 0,38 / 0,37 | 7,0 | 5,0 | 5,18 / 5,41 / 5,44 | 0,45 |
| 2500 | 0,40 / 0,43 / 0,44 | 7,2 | 3,1 | 5,06 / 4,65 / 4,77 | 0,25 |
| 4500 | 0,42 / 0,49 / 0,50 | 6,8 | 1,8 | 5,30 / 4,36 / 4,37 | 0,12 |

- **Rozdzielenie „narzędzie czy tło” jest najwyższe na samym początku** (10,7 w kroku 50 przy mAP równym zero) i
  potem już tylko lekko maleje; pole pod krzywą ROC wynosi 1,00 w każdej chwili. Kryterium K1 (wzrost w 8 z 10
  ziaren) niespełnione: 4 z 10. Ta definicja tożsamości nie mierzy postępu detektora.
- **Między ramionami związek z mAP jest słaby**: wyższe rozdzielenie odświeżane niż w ramieniu bez dokładania ma
  `koopman` w 9 z 10 ziaren i `losowe` w 6 z 10 (oba mają wyższe mAP w 10 z 10); korelacja wewnątrz chwili 0,12
  (przedział od 0,00 do 0,25). Dla siły stałej korelacja jest ujemna (−0,14), jak w kampanii 4.3.
- **Kierunek stały to migawka jednej chwili.** W chwili dopasowania daje rozdzielenie 27, 50 kroków obok 9, po 500
  krokach 5, po 3500 krokach 2. Spadek siły stałej to więc starzenie się kierunku, a nie gorsze zakodowanie narzędzia.
- **Przewidywalne jest tylko samo starzenie.** Prognoza zmiany na 500 kroków z krzywej pozostałych dziewięciu ziaren
  (uczciwa: nie używa przyszłości tego biegu): cosinus z kierunkiem z kroku 1000: 0,53; siła stała: −0,11;
  rozdzielenie odświeżane: −0,11; siła każdej sondy osobno: −0,11; mAP: 0,11.
- **Między ziarnami kierunki są prostopadłe** (cosinus 0,00-0,01), a podobieństwo układu sond (CKA) maleje w
  uczeniu: 0,78, 0,55, 0,39 w krokach 1000, 2500, 4500.
- **mAP skacze**: między ocenami co 100 kroków zmienia się średniokwadratowo o 0,04, a przez 500 kroków rośnie
  średnio o 0,013-0,022. Pojedyncza ocena ma błąd rzędu różnicy między ramionami.

### 6.7 mAP drga razem z wagami, a wagi uśrednione są o 0,05-0,11 lepsze

Katalog: `experiments/maly_vit_srednia_wag_a_map/`. Sama ocena zapisanych wag na lokalnej karcie, 3000 klatek
walidacji, ten sam kod co w biegach (zgodność z zapisanymi ocenami biegu do 0,002). Wagi uśrednione to średnia
parametrów z kilku ostatnich zapisów; uczenie się nie zmienia.

| ziarno 47, krok | surowe wagi | średnie mAP surowych w oknie 100 kroków | wagi uśrednione z 100 kroków | wagi uśrednione z 200 kroków |
|---:|---:|---:|---:|---:|
| 600 | 0,256 | 0,237 | 0,307 | 0,290 |
| 700 | 0,314 | 0,250 | 0,312 | 0,319 |
| 800 | 0,320 | 0,308 | 0,382 | 0,357 |
| 900 | 0,348 | 0,314 | 0,407 | 0,405 |
| 1000 | 0,310 | 0,337 | 0,419 | 0,426 |

| ziarno 48, krok (zapisy co 100 kroków) | 1400 | 1800 | 2200 | 2600 | 3000 |
|---|---:|---:|---:|---:|---:|
| surowe wagi | 0,387 | 0,407 | 0,393 | 0,415 | 0,385 |
| wagi uśrednione z 5 zapisów (400 kroków) | 0,400 | 0,442 | 0,468 | 0,485 | 0,490 |

- **mAP pojedynczego stanu sieci skacze**: między zapisami co 10 kroków zmienia się średniokwadratowo o 0,041, przy
  postępie 0,026 na 100 kroków; w oknie 100 kroków rozstęp wynosi 0,09-0,15 (na przykład 0,294, 0,160 i 0,314 w
  krokach 660, 680, 700).
- **Uśrednienie wag usuwa drgania i daje model lepszy niż którykolwiek stan chwilowy w oknie.** Kryterium ustalone
  przed wynikiem (przewaga co najmniej 0,01 nad średnią okna w 4 z 5 chwil i nad końcem okna w 4 z 5) spełnione:
  5 z 5 (od +0,062 do +0,093) i 4 z 5. Na drugim ziarnie, w krokach 1400-3000: 9 z 9 chwil, średnio +0,054.
- **Skala.** Zysk z doboru klatek w kampanii 4.3 to 0,016 pola pod krzywą. Wagi uśrednione ziarna 48, bez
  dokładania klatek, mają w kroku 2800-3000 mAP 0,49-0,50, czyli tyle, ile surowe wagi najlepszych ramion z dołożonymi
  938 klatkami w kroku 4500 (średnia z 10 ziaren kampanii v2).
- Statystyki BatchNorm nie mają tu znaczenia (uśrednione albo przeliczone: różnice do 0,01).

Zastrzeżenia: dwa ziarna, baza bez dokładania, walidacja, uśrednianie po fakcie z zapisów. Nie zmierzono, czy zysk z
doboru klatek utrzymuje się na wagach uśrednionych; to jest teraz pierwsze pytanie programu (rozdział 9).

### 6.8 Obraz całości

Siedem pomiarów składa się w jeden obraz, zgodny z literaturą o AdamW na brzegu stabilności (rozdział 8.1):

1. **Sieć nie stoi w punkcie, tylko krąży wokół wolno płynącego środka.** W wagach wygląda to jak dyfuzja ze słabym
   dryfem, w latencie sond i w mAP jak drgania. Środek (wagi uśrednione) jest lepszym detektorem niż stan chwilowy.
2. **Trzy skale czasu.**
   - *Szybka, 1-4 kroki:* mechanika optymalizatora. Krok to pęd plus pchnięcie paczki, ze współczynnikami 0,98;
     pchnięcie powtarza się przez pęd. Przewidywalne w 0,95 przy znanej paczce i w 0,65 bez niej.
   - *Średnia, 5-50 kroków:* siła zwrotna. Pchnięcie osiąga 3,2 pierwszego kroku i wraca; zostaje 0,6 po 50 krokach.
     Z historii przewidywalny jest tylko powrót do środka (0,2-0,3; granica 0,5).
   - *Wolna, setki kroków:* dryf środka. W latencie to kilka procent energii ruchu w oknie 100-200 kroków, w wagach
     10-26 procent. To jest uczenie się we właściwym sensie i to na nie wpływa skład zbioru.
3. **Pojedyncza paczka steruje skalą szybką, skład zbioru skalą wolną.** Jedna paczka jest widoczna przez około 10
   kroków, a jej trwały ślad to 0,4 pierwszego kroku. Dołożenie 134 klatek zmienia średnie pchnięcie wszystkich
   następnych paczek, czyli przesuwa środek.
4. **Tożsamość w sensie „narzędzie czy tło” jest pełna od początku** i dlatego nie mówi nic o postępie; brakuje
   dokładnej ramki, co pokazała wyrocznia.

## 7. Odpowiedzi na cztery pytania

### 7.1 Jak zachowa się wektor tożsamości i co z tego da się prognozować

**Co wiemy o trzech definicjach.**

| definicja | co mierzy | zachowanie w uczeniu | związek z mAP |
|---|---|---|---|
| kierunek stały z końca rozgrzewki (pętle v3 i 4.3) | migawkę jednej chwili; w dwóch trzecich stan drgań z tej chwili | starzeje się: cosinus z kierunkiem aktualnym 0,75 po 50 krokach, 0,45 po 500, 0,10 po 3500; siła wzdłuż niego skacze o 0,5 w 50 kroków | brak albo odwrotny (−0,14; w 4.3 kolejność ramion odwrócona) |
| ten sam przepis, odświeżany | „narzędzie czy tło i scena” | pełne rozdzielenie od 50. kroku, potem płasko | słaby dodatni między ramionami (0,12), żaden w czasie |
| parowa: J128 albo 8 kierunków z par „z narzędziem i bez” | obecność narzędzia przy tej samej scenie | nieznane: sprawdzona w jednym punkcie kontrolnym; najważniejsze kanały zmieniają się między krokiem 500 a 1000 | nieznany; jedyna z próbą przyczynową (wpływ na pewność razy pokrycie ramki) |

**Co da się prognozować.**

1. *Jeden krok przy znanej paczce*: 0,95-0,98, także dla siły tożsamości, ze zgodnością znaku 0,95 w każdym obrocie
   kampanii. To działa i to jest tryb, w którym sami wybieramy paczkę.
2. *Skutek wybranej paczki na kilka-kilkanaście kroków*: z funkcji odpowiedzi (6.4). Średnio 3,2 pierwszego kroku po
   5 krokach, 2,4 po 10, 0,6 po 50. Jak ściśle trzyma to dla różnicy między dwiema paczkami, zmierzy etap E1.
3. *Starzenie się kierunku*: krzywa cosinusa jest powtarzalna między ziarnami (0,53 z krzywej innych biegów).
   Praktyczny wniosek: kierunek odświeżać co obrót i liczyć z uśrednionych zapisów, nie z jednej chwili.
4. *Powrót do środka*: jedyna rzecz przewidywalna z historii w horyzoncie 10-500 kroków, z granicą 0,5. Wynik 28,7
   procent z kampanii 4.3 to właśnie on.

**Czego prognozować się nie da i nie warto.** Siły wzdłuż kierunku stałego za 500 kroków (−0,11 nawet z krzywej
innych ziaren), rozdzielenia odświeżanego (płaskie), współrzędnych latentu na 10-500 kroków bez wiedzy o paczkach
(sufit z budowy ruchu, 6.5). Zgadza się to z literaturą: przewidywalne są wielkości zbiorcze i kolejność, a nie
współrzędne reprezentacji (8.2, 8.3).

**Co proponuję zamiast tego** (różnice wobec metody z docs/26 nazwane wprost; żadnej nie wprowadzono):

- *Obiekt:* latent liczony na wagach uśrednionych, bo tam nie ma drgań, a model jest lepszy (6.7).
- *Kierunek:* zamiast „gdzie jest narzędzie” kierunek „czego potrzebuje głowa”: gradient straty detekcji względem
  latentu na klatkach kontrolnych, przeliczany w każdym obrocie. Jest związany z jakością detekcji z definicji, a
  wyrocznia pokazała, że brakuje dokładnej ramki, nie obecności.
- *Jeśli tożsamość ma zostać celem:* w definicji parowej, odświeżanej, liczonej na większej próbie. Najpierw tani
  test bez treningu, czy w ogóle wiąże się z mAP (Z7 w rozdziale 9).
- *Prognoza:* skutku paczki albo zestawu klatek, który sami wybieramy, a nie całego ruchu.

### 7.2 Jak dobierać zdjęcia, żeby przyspieszyć uczenie

**Co działa dziś.** Dokładanie klatek wyszukanych jako podobne do wycinków narzędzia: +0,009 do +0,016 pola pod
krzywą wobec losowych w pięciu kampaniach małego ViT. Prognoza nie ma w tym udziału, wkład generatora nie jest
rozdzielony, a na dostrajanym RF-DETR nic nie pokonało losowania.

**Dlaczego dobór przez prognozę nic nie zmieniał.** Przewidywana wielkość (siła wzdłuż kierunku stałego) nie wiąże
się z mAP; na 500 kroków prognoza to powrót do środka; lista najbliższych klatek prawie nie zależy od małej zmiany
celu (54 z 60 wspólnych).

**Co wynika z pomiarów z 5 października.**

1. *Najpierw to, co darmowe.* Wagi uśrednione dają +0,05 do +0,11 mAP przy tej samej liczbie kroków. To kilka razy
   więcej niż cały dotychczasowy zysk z doboru. Każde dalsze porównanie ramion trzeba prowadzić na wagach
   uśrednionych, bo dopiero tam widać, czy dobór dokłada coś ponad to.
2. *Właściwa skala sterowania to skład zbioru, nie pojedyncza paczka.* Jedna paczka zostawia trwale 0,4 swojego
   pierwszego kroku i jest widoczna przez około 10 kroków. Zestaw 134 klatek zmienia średnie pchnięcie na 500 kroków.
3. *Kandydatów oceniać po skutku trwałym.* Ocena po jednym kroku mierzy fazę drgań: 87 procent szczytowego skutku
   jest cofane. Miarą powinien być spadek straty kontrolnej po 20-50 krokach, liczony na wagach uśrednionych albo
   jako średnia z okna.
4. *Cel doboru: strata detekcji na zbiorze kontrolnym*, ewentualnie kierunek potrzebny głowie; nie siła tożsamości.
5. *Kontrole, bez których wynik nie przejdzie recenzji:* losowe z przetasowaniem; wyszukanie z prawdziwej klatki
   kotwicy bez generatora; te same ramiona na wagach uśrednionych; harmonogram z malejącym współczynnikiem uczenia.

**Reguła, którą proponuję sprawdzić** (etapy E1 i E2 w rozdziale 9): spośród kandydujących zestawów klatek wybierać
ten, dla którego zmierzony na próbę spadek straty kontrolnej po 20-50 krokach jest największy, z zachowaniem części
losowej zestawu (25-50 procent) jako bezpiecznika. Równanie Langevina służy tu do dwóch rzeczy: tanio odsiewa
kandydatów (pchnięcie z jednego wirtualnego kroku przemnożone przez funkcję odpowiedzi) i mówi, po ilu krokach
skutek mierzyć.

### 7.3 Czy „latent o najwyższym mAP, potem obraz ze Stable Diffusion” ma naukowe uzasadnienie

**Pierwsza połowa tak.** Szukanie latentu, przy którym głowa trafia, to znana konstrukcja: cechy wyprowadzone z
etykiet jako cel pośredni dla sieci [Hao20, Zha21p, Kim24k] i cele warstw ukrytych liczone ze straty wyjścia
[Lee14, Meu20]. Jako diagnoza „czego głowa potrzebuje od reszty sieci” jest poprawna, a jej wynik jest konkretny:
głowa umie trafić (AP50 0,44 → 0,88), brakuje jej dokładnej ramki (AP75 0,02 → 0,77).

**Druga połowa w obecnej postaci nie.**

- Cel leży poza zasięgiem obrazów (6.1): typowa współrzędna jest przesunięta o 22 odchylenia kanału, 99 procent celów
  jest bardziej nietypowych niż najbardziej nietypowy prawdziwy token. Literatura mówi to samo: obraz z zadanych cech
  powstaje tylko wtedy, gdy cechy pochodzą z naturalnego obrazu [Shi25f].
- Obraz, który by ten cel realizował, uczyłby mało: przy idealnym latencie strata jest 11 razy mniejsza, więc sygnał
  uczący jest bliski zera. Prace o generowaniu danych pod kontrolą uczonego modelu zgodnie pokazują, że pomagają
  obrazy poprawne i umiarkowanie trudne [Hem23, Hem25, Wan24m, Fri25].
- Idealny latent klatki różni się od prawdziwego nie treścią obrazu, tylko błędem enkodera na tej samej treści. Żeby
  go wykorzystać, trzeba nauczyć enkoder dawać ten latent dla TEJ SAMEJ klatki, a nie szukać innej klatki.

**Jak to obronić w publikacji.**

1. *Idealny latent jako cel pośredni na prawdziwych klatkach uczących* (dodatkowy człon straty, jak LabelEnc):
   użycie etykiety jest uczciwe, bo uczenie i tak z niej korzysta. Warunki z literatury: cel z ograniczonym zasięgiem
   (1-2 odchylenia kanału), rozgrzewka, cel odświeżany względem bieżącej głowy. To jest najbliższe intuicji
   użytkownika i ma precedens z zyskiem około +2 AP.
2. *Stable Diffusion tam, gdzie pula nie wystarcza*: jako źródło poprawnych wariantów o regulowanej trudności dla
   braków, których prawdziwe klatki nie pokrywają, z celem w zasięgu. Zawsze z kontrolą „wyszukanie bez generatora”,
   bo wyszukane prawdziwe obrazy bywają lepsze od syntetycznych [Gen24, Bur23].
3. *Czego nie pisać*: że obraz, przy którym detektor jest najlepszy, jest najlepszym obrazem uczącym; że 0,88 to
   wynik detektora (to interwencja z etykietą).

### 7.4 Jak podłączyć równanie Langevina i operator Koopmana

**Postać równania.** Pomiary dają równanie Langevina z bezwładnością i siłą zwrotną, czyli tę samą klasę modelu,
którą [Kun21] opisuje dla SGD z pędem. W zapisie dyskretnym, dla latentu sondy `z`:

    krok:            z(t+1) − z(t) = P(t) + S(t)                         zmierzone: współczynniki 0,98 i 0,98
    pęd:             P(t+1) ≈ β1 · [P(t) + S(t)],  β1 = 0,9             zmierzone: zgodne przez 3 kroki
    pchnięcie:       S(t) = F[z(t) − c(t), historia] + ξ_B(t)            F: siła zwrotna z opóźnieniem 2-4 kroków
    środek:          c(t+1) = c(t) + dryf zależny od składu zbioru       wolny; w latencie 2-10 procent energii ruchu

`P` to część kroku z pamięci pędu, `S` to pchnięcie paczki, `c` to wolny środek (latent na wagach uśrednionych),
`ξ_B` to część pchnięcia właściwa wylosowanej albo wybranej paczce. Trzy rzeczy są w tym równaniu inne niż w wersji
z szeregów czasowych: wejście zależy od stanu (człon `F`), bezwładność jest znana z optymalizatora, a nie dopasowana,
i istnieje wejście, które wybieramy (`ξ_B`).

**Co jest znane, a co trzeba zmierzyć.**

| składnik | stan | jak go mieć |
|---|---|---|
| krok i pęd | znane, bez dopasowania | wirtualny krok AdamW (`kierunki` w `nagranie.py`, `czesci_kroku` w `rownanie.py`) |
| średnia odpowiedź na pchnięcie `R(h)` | zmierzona z nagrań (6.4) | `experiments/maly_vit_odpowiedz_latentu_na_paczke` |
| odpowiedź na różnicę dwóch paczek w tym samym stanie | niezmierzona | pary przebiegów, etap E1 |
| jądro siły zwrotnej `F` | znane jakościowo (korelacje z 6.4) | dopasowanie z nagrań, pomiar Z9 |
| przełożenie składu zbioru na dryf środka | niezmierzone | pary obrotów, etap E2 |

**Rola Koopmana.** Operator z wejściem [Pro14, Kor16] jest liniową postacią tego samego modelu: stan to odchylenie
od środka, pęd i środek; wejście to pchnięcie wybranej paczki albo średnie pchnięcie dołożonego zestawu. Trzy uwagi
z pomiarów:

- Operator dopasowany do surowych trajektorii latentu bez wejścia (EDMD, NNDMD, macierze 768 × 768) nie ma czego
  uchwycić: sześć prób dało wyniki gorsze od „bez zmian”. Struktura jest już znana z optymalizatora; nieznane są
  tylko funkcja odpowiedzi i przełożenie.
- Wzmocnienie wejścia zależy od stanu (K3 w 6.4: stała odpowiedź daje 0,63, prawdziwy pęd 0,985), więc właściwa jest
  postać biliniowa [Bru20, Iac22b] albo po prostu liczenie pchnięcia w bieżącym stanie.
- Sensowne miejsce dla widma Koopmana to diagnoza: para zespolona o okresie 8-10 kroków to drgania optymalizatora,
  mody wolne to dryf środka. To da się sprawdzić na nagraniach co krok bez treningu.

**Trzy zastosowania, od najpewniejszego.**

1. *Tani odsiew kandydatów* (skala szybka): pchnięcie z wirtualnego kroku razy `R(h)`.
2. *Wybór zestawu na obrót* (skala wolna): przewidziana zmiana środka = dryf własny + przełożenie razy średnie
   pchnięcie zestawu, policzone na wagach uśrednionych. Tu prognoza jest warunkowa i dotyczy tego, czym sterujemy.
3. *Osiągalność* [Yeu17c, Liu17]: które kierunki latentu pula klatek w ogóle potrafi pchnąć. Część braku poza
   zasięgiem puli to jedyne uzasadnione miejsce dla generatora.

<!-- DALEJ-3 -->

## 8. Co mówi literatura

Odczyt: P to pełny tekst przeczytany przez pomocnika Undermind, S to streszczenie i metadane. Odnośniki w rozdziale
12. Trzy wyszukiwania pogłębione uruchomione dla tego dokumentu (131, 133 i 184 prace) i dziesięć wcześniejszych są
wymienione w rozdziale 13. Tabele dotyczące idealnego latentu i generowania danych są w pełnej postaci w
`docs/latent_sd_20261005/`; tutaj ich skrót.

### 8.1 Drgania pod AdamW: to znany reżim, nie usterka

| praca | co ustala | odczyt | co to znaczy tutaj |
|---|---|---|---|
| [Coh22] Adaptive Gradient Methods at the Edge of Stability | Adam pracuje na „adaptacyjnym brzegu stabilności”: krzywizna po uwzględnieniu skalowania Adama rośnie do progu `(2+2β1)/(η(1−β1))`, czyli `38/η` dla β1 = 0,9, a iteracje oscylują wzdłuż najbardziej stromych kierunków; przy małych paczkach krzywizna zostaje nieco poniżej progu; pokazane też dla transformerów | P | nasz obraz (pchnięcia zmieniają znak po 3-5 krokach, ruch wraca) to opisany mechanizm; dla η = 5e-4 próg to 76 000, u nas niezmierzony |
| [Kun21] The Limiting Dynamics of SGD | SGD z pędem jako równanie Langevina z bezwładnością; w stanie ustalonym sieć ma stałą prędkość chwilową i krąży w przestrzeni fazowej położenie-prędkość wzdłuż kierunków krzywizny; współczynnik tłumienia `(1−β)/√(2η(1+β)(ρ+λ))`: większy pęd albo krok daje drgania słabo tłumione; „deterministyczna prognoza wag psuje się w czasie, statystyki ruchu pozostają przewidywalne” | P | to jest gotowy język dla naszego równania: bezwładność, siła zwrotna, szum; i gotowe ostrzeżenie o granicy prognozy |
| [Kun23b] Noise Is Not the Main Factor... | przewaga Adama nad SGD nie bierze się z szumu paczek; w pełnej paczce rośnie; Adam zachowuje się jak spadek po znakach gradientu z pędem, więc długość kroku prawie nie zależy od wielkości gradientu ani paczki | P | wyjaśnia wynik z paczkami 32, 128, 320: większa paczka nie zmniejsza kroku ani drgań |
| [Son24f] Does SGD really happen in tiny subspaces? | większość gradientu leży w kilku stromych kierunkach, ale uczenie tylko w tych kierunkach nie obniża straty; uczenie tylko w pozostałych (łagodnych) obniża ją tak samo jak zwykłe; składowe strome to oscylacja; pęd i Adam przyspieszają uczenie właśnie w kierunkach łagodnych | P | duże, szybko cofane pchnięcia to nie uczenie; uczenie jest w części, która zostaje |
| [Coh24b] Understanding Optimization in Deep Learning with Central Flows | uśredniona w czasie trajektoria optymalizatora oscylującego na brzegu stabilności daje się opisać gładkim równaniem i przewidzieć na długi horyzont; analiza dla uczenia bez losowych paczek | S | przewidywalny jest środek drgań, nie same drgania |

### 8.2 Wolny dryf, uśrednianie wag i to, co niezmienne

| praca | co ustala | odczyt | co to znaczy tutaj |
|---|---|---|---|
| [Xu26g] Optimizer-Induced Low-Dimensional Drift... | w transformerze uczonym AdamW (51 mln parametrów) wagi mają jeden trwały kierunek dryfu („kręgosłup”) i drgania w poprzek; gradient pojedynczej paczki jest do niego prawie prostopadły (cosinus 4e-4), skumulowany krok z 200 kroków ma cosinus 0,15-0,34; długa ekstrapolacja zawodzi, gdy zmienia się cel uczenia (kierunek zgina się o 71 stopni) | P | u nas trwały dryf jest, ale w 1000 kroków nie dominuje nad dyfuzją (6.5); miary liczone na jednej trajektorii trzeba porównywać z błądzeniem losowym; dokładanie klatek co 500 kroków to właśnie zmiana celu, przy której ekstrapolacja zawodzi |
| [Ant18] PCA of high dimensional random walks... | składowe główne błądzenia losowego wyglądają jak gładkie krzywe i niosą udziały malejące jak `1/k²` | S | duży udział pierwszej składowej nie dowodzi kręgosłupa; trzeba mierzyć stałość kierunku |
| [Wan26m] Measuring Structured Predictability... | przewidywalność przesunięć parametrów jest niska w dużych macierzach i skupia się w przejściowych „kieszeniach” (około 6 procent obserwacji); parametry pomocnicze (normalizacje, przesunięcia) są wyraźnie bardziej przewidywalne; zanik wag ją podnosi, pęd obniża; przewidywalność nie idzie w parze z jakością (korelacja z trafnością −0,71) | P | zgodne z naszym wynikiem: dryf najtrwalszy w końcowej normalizacji i głowie |
| [Mor24d] Exponential Moving Average of Weights... | średnia ruchoma wag to odszumiony punkt w sąsiedztwie trajektorii, a nie krok naprzód; zmniejsza rozchwianie przewidywań (z 18,8 do 11,7 procent zmienionych decyzji); przy długim oknie statystyki BatchNorm trzeba przeliczyć | P | sposób na „środek drgań”: latent liczony na uśrednionych wagach jest naturalnym obiektem prognozy i oceny paczek |
| [Pas23] SGD-induced drift of representation... | reprezentacja dryfuje jak błądzenie losowe w kierunkach, które nie zmieniają straty; odczyt i wzajemne położenie reprezentacji (kąty, odległości) są niezmienne; „prognoza pojedynczych współrzędnych jest niewykonalna, prognoza wielkości niezmiennych tak” (model liniowy, paczka 1) | P | prognozować rozdzielenie i geometrię, nie rzut na stałą oś |

### 8.3 Co w uczeniu jest przewidywalne

| praca | co ustala | odczyt |
|---|---|---|
| [Hac19] Let's Agree to Agree | kolejność, w jakiej sieć uczy się przykładów, jest prawie ta sama między ziarnami (korelacja 0,99) i podobna między architekturami (0,87-0,97); znika na danych syntetycznych | P |
| [Mao23] The training process of many deep networks explores the same low-dimensional manifold | w przestrzeni przewidywań trajektorie 151 tysięcy modeli leżą na rozmaitości o kilku wymiarach (3 wymiary to 76 procent); ta sama architektura idzie tą samą drogą, różni się postępem | P |
| [Bal21b], [Pau21], [Ton18] | trudność przykładu (głębokość, na której sieć go rozstrzyga; norma błędu po kilku epokach; liczba zapomnień) jest powtarzalna i przewiduje późniejsze zachowanie | P, S, S |
| [Kye22] TiDAL | mały moduł uczy się przewidywać przebieg uczenia przykładu i wybiera dane do oznaczenia; zysk 10-20 procent wobec losowania w klasyfikacji | P |
| [Kwo24], [Wu20], [Oka23] | rankingi trudności są szumne między architekturami; zysk z kolejności uczenia często znika wobec losowej; losowanie z przetasowaniem to mocny punkt odniesienia | S, S, P (audyt) |

Wspólny wniosek: powtarzalne i przewidywalne są wielkości zbiorcze i kolejność (kto, kiedy, jak trudny), najlepiej
w przestrzeni przewidywań. Nie ma pracy, która pokazuje prognozę współrzędnych reprezentacji stałego obrazu na wiele
kroków naprzód.

### 8.4 Śledzenie kierunku konceptu przez uczenie

| praca | co ustala | odczyt |
|---|---|---|
| [Her20] What shapes feature representations? | cechy bywają odczytywalne już w sieci nieuczonej; uczenie wzmacnia potrzebne i tłumi zbędne; odczyt w sieci nieuczonej to własność architektury, nie nauki | P |
| [Ran23c] On the special role of class-selective neurons in early training | selektywność klasowa neuronów skacze w pierwszej epoce i potem opada w warstwach pośrednich; tłumienie jej od początku psuje uczenie, od epoki 1 lub 5 już nie (tylko sieci ResNet) | P |
| [Kap25b] Bridging Critical Gaps in Convergent Learning | zbieżność reprezentacji między ziarnami zachodzi prawie w całości w pierwszej epoce; w ViT nie ma wyróżnionych osi, więc kierunki trzeba porównywać po dopasowaniu obrotem; jednostki po funkcji typu ReLU mają osie wyróżnione | P |
| [Jar26] Feature Evolution and Migration during ViT Training | w ViT-Tiny (wymiar 192, jak nasz) cechy wędrują między warstwami, głównie we wczesnym uczeniu; tę samą cechę w różnych punktach kontrolnych trzeba dopasowywać, a nie zakładać stałego kierunku | P |
| [Par22c] ConceptEvo | degradacja konceptów wyprzedza spadek trafności | P |
| [Var25] Class Separation Dynamics in ViTs | rozdzielenie klas w ViT poprawia się wykładniczo, ale pod Adamem niemonotonicznie; prawa załamują się dla wymiaru 256 i więcej | P |
| [Hua25i] Causality ≠ Decodability... | w ViT sygnał łatwo odczytywalny bywa nieużywany przez sieć, a słabo odczytywalny bywa używany | S |

Zalecany protokół z tych prac: (1) śledzić dwie krzywe, rzut na kierunek pierwotny i na kierunek dopasowany od nowa;
(2) mierzyć obrót kątami głównymi i dopasowaniem ortogonalnym; (3) trudne negatywy i sieć nieuczona jako kontrola;
(4) odczytywalność i użycie przez głowę badać osobno, interwencją.

### 8.5 Skutek jednej paczki w czasie i dobór paczek

| praca | co ustala | odczyt |
|---|---|---|
| [Liu26p] Finite-Horizon Input-Output Dynamics of Minibatch Perturbations in AdamW | zaburzenie jednej paczki wraca przez pęd ze szczytem po 7,5-9 krokach; model liniowy zachowuje kolejność kandydatów (korelacja rang 0,76-0,89 w horyzoncie 8-12 kroków), wielkości słabo; wskaźniki jednego kroku nie przewidują opóźnionego skutku | P |
| [Guo26] Delayed Optimizer-State Transport... | uwzględnienie 8 kroków transportu przez stan AdamW zmienia wybór w 60 z 96 okien; wymaga znajomości przyszłych paczek i kosztuje około 10,6 raza więcej | P |
| [Den26b] How Faithful Is Trajectory-Based Data Attribution? | prognozy pierwszego rzędu psują się z horyzontem i krokiem uczenia; AdamW trzeba liczyć dokładnie jak optymalizator | P |
| [Guu23] Simfluence | strata przykładu testowego jako prosty układ liniowy napędzany kolejnymi przykładami uczącymi; korelacja rang z prawdą około 0,89 | P |
| [Wu18c] Understanding Short-Horizon Bias | ocena decyzji kilka kroków naprzód systematycznie faworyzuje szybki, płytki zysk | P |
| [Min22], [Hu21] | celować w stratę usuwalną (strata minus strata modelu odniesienia); wybór po samej stracie zawodzi przy szumie | P (audyt), S |
| [Liu17], [Qiu22b], [Les18b] | nauczanie jako sterowanie: nauczyciel wybiera przykład, który najbardziej zbliża ucznia do celu; z puli jest tak szybki jak przy tworzeniu przykładów tylko wtedy, gdy pula pokrywa potrzebny kierunek | P |

Nasz pomiar 6.4 jest zgodny z [Liu26p] co do kształtu (narastanie przez pęd i powrót w skali kilku do kilkunastu
kroków) i daje to, czego tam nie ma: odpowiedź reprezentacji, a nie straty, w detektorze uczonym od zera.

### 8.6 Idealny latent i obraz z niego (skrót)

- **Cele latentu z etykiet mają precedensy**: LabelEnc [Hao20] (około +2 AP, pod warunkiem uczenia celu razem z
  siecią i rozgrzewki), LGD [Zha21p], LabelDistill [Kim24k], cele warstw ukrytych liczone ze straty wyjścia
  [Lee14, Meu20]. Warunek wspólny: cel blisko tego, co sieć umie wytworzyć; teoria [Meu20] działa tylko dla małych
  przesunięć celu.
- **Obraz z zadanych cech da się narysować, gdy cechy pochodzą z naturalnego obrazu** [Shi25f, Bor21, Neu25]; dla
  celów spoza tego zbioru realistyczny obraz nie powstaje [Shi25f]. Niewidocznym zaburzeniem pikseli można nadać
  obrazowi cudzą reprezentację [Sab15], ale to nie są obrazy, na których warto uczyć.
- **Generowanie danych pod kontrolą uczonego modelu pomaga, gdy obraz jest poprawny i umiarkowanie trudny**
  [Hem23, Hem25, Wan24m, Zha22i, Hay25], także w chirurgii [Fri25, Ven25e]; obrazy „najpewniejsze” dają najmniej,
  a dane syntetyczne bez dopasowania do zadania potrafią szkodzić [Ven25e]. Wyszukane prawdziwe obrazy bywają
  lepsze od syntetycznych [Gen24, Bur23].
- **Przykłady łatwe pomagają przy małych danych i na początku** [Sor22, Guo23f, Lia24e]; w detekcji próbki dobrze
  pokrywające obiekt ważą więcej niż najtrudniejsze [Cao19].

### 8.7 Prognozowanie wag i reprezentacji operatorem Koopmana i pokrewnymi

| praca | co ustala | odczyt |
|---|---|---|
| [Dog20], [Tan20b] | prognoza wag operatorem Koopmana albo DMD w małych sieciach, do pomijania kroków uczenia | S |
| [Kny24], [Jan23c], [Sho25b] | sieci uczone do przewidywania wag kilka kroków naprzód („nowcasting”); wymagają tysięcy zapisanych trajektorii i działają na krótkim horyzoncie | S |
| [Lin21b], [Uy21], [Kam18] | pamięć Mori-Zwanziga i współrzędne opóźnień jako uogólnienie Koopmana przy niepełnym stanie | S |
| [Has22], [Lu20c] | mała reszta dopasowania nie gwarantuje trafnej prognozy operatora | S |

Wyszukiwania z 24 września i 2 października nie znalazły pracy, która przewiduje reprezentacje stałych obrazów w
przyszłych punktach kontrolnych bez przyszłych gradientów; prognozy wag są dowodem pokrewnym, nie bezpośrednim.

<!-- DALEJ-9 -->
