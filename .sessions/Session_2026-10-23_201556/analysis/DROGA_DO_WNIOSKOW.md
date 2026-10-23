# Jak doszliśmy do wniosków (2026-10-23, 17:30–20:15)

Zapis kolejności pytań, odpowiedzi, pomiarów i poprawek. Ważne: kilka wersji po drodze było błędnych i zostało obalonych pomiarem;
zaznaczam je, żeby nie wróciły.

## 1. Punkt wyjścia: co mówi płaszczyzna krok × podkrok

Z eksperymentów z nocy (s47, 806 checkpointów; replikacja 3047, 46 checkpointów, 10 klatek TEST): uwaga jest na starcie czystą
translacją, po uczeniu gwiazdą na tokenach narzędzia (tokeny narzędzia niosą 56–77 % wpływu w pełnym jakobianie uwagi, sąsiedzi
na siatce 0 %); MLP 1 rozpycha, MLP 4 rośnie do końca; front rozdziału (Fisher) w bloku 4; każdy podkrok to siodło. Użytkownik
pytał o wnioski dla uczenia i o obronę przed „po co to robiłeś”.

## 2. Pytanie: „nadajnik kontekstu, gwiazda” i czy uwaga ustawia tokeny pod MLP 4

Odpowiedź: gwiazda = każdy z 600 tokenów bierze wartości od tych samych kilkunastu tokenów narzędzia (wzbogacenie 11–24× na 3047,
40× na s47), więc wszyscy dostają prawie ten sam wektor. Interpretacja użytkownika („uwaga przestawia token tam, gdzie MLP 4 może
go rozdzielić”) uznana za zasadniczo trafną z zastrzeżeniami: część wspólna (translacja) nic nie rozdziela, liczy się część własna;
MLP działa punktowo; rozdział budują bloki 1–3 stopniowo. **Błąd w tej odpowiedzi (poprawiony w pkt 7):** napisałem, że lasery
MLP 4 „zapalają się dla tokenów z dużym sygnałem narzędzia i pchają je w swoją stronę”.

## 3. Pytanie: czy uwaga bierze pod uwagę pole MLP; gdzie MLP „nie widzi”; jak miesza się strata; czym jest uczenie

- W przejściu taśma jest jednokierunkowa: uwaga nie zna wag MLP 4. W uczeniu zna je przez gradient (przeszedł przez jakobian MLP 4).
  Współadaptacja, nie jednostronne ustawianie.
- Pole czyta LN(x), kierunek koloidu; translacja wspólna zmienia LN(x+t) różnie dla każdego tokenu, więc nie jest „pusta”.
- „MLP nie widzi” = koloid po ciemnej stronie wszystkich progów (dostaje tylko b₂). Pomiar z zapisanych danych: odsetek zapalonych
  laserów na token spada z 41–45 % (krok 0) do 4–19 % (koniec), efektywna liczba laserów 340–360 → 60–220 z 768. Fisher po
  podkrokach (3047, 60 ep): uwaga 3 → 0,92, MLP 3 → 1,60, uwaga 4 → 1,71, MLP 4 → 3,61: **rozdział robią MLP, uwaga zmienia to,
  co MLP widzą.**
- Strata w przejściu nie występuje; działa tylko przy skoku wag po paczce. Życzenie koloidu = gradient straty po jego stanie,
  cofany przez transponowane jakobiany; lasery obracają się o sumę życzeń oświetlanych koloidów. Trzy konsekwencje zgodne z
  pomiarami: zamiana łodygi (pole powstaje tylko tam, gdzie koloidy bywają), ruch prosty → zygzak (jakobiany ≈ I na starcie → to samo
  życzenie na każdym podkroku), gaszenie laserów widzących obie klasy (sprzeczne życzenia się znoszą).

## 4. Pytanie: strata, energia, superpozycja, pole lokalne wokół ognisk

Dwa szybkie pomiary rozstrzygnęły „pułapka czy nie”:
- asymetria jakobianów (z `jakobiany_3047.npz`): ‖A‖/‖S‖ = 0,86–1,0 w każdym podkroku i czasie → pola nie są potencjalne;
- cos(v_c, w_c) z wag (3 checkpointy 3047): średnio ≈ 0 (|cos| 0,09–0,22 wobec 0,06 losowo); blok 1 +0,07 (rozpycha), bloki 5–6
  ujemne (siła przywracająca, ściskanie) — spójne z r_po/r_przed.
Wniosek: laser działa lokalnie (czapa wokół w_c na sferze LN), ale przenosi w bok, nie do ogniska: **odchylacz, nie pułapka**.
Strata = potencjał na 5 odczytach głowy, zdegenerowany w 192D (płaski w ~187 kierunkach); koloidy płyną do przeciwobrazów
(obszarów, nie punktów); strata ogniskowa „wypuszcza” tło. Superpozycja = rzadki kod (43 % → ~10 %) jako skutek, nie cel.
Sprawdzone w `trening.py`: LR stałe 5e-4 po 300 krokach rozgrzewki, WD 0,05 → temperatura gaśnie nie przez harmonogram.
Z tego wyszła predykcja: „temperatura widoczna gaśnie z IoU, niewidoczna wolniej” (pkt 8).

## 5. Pytanie: czy pole MLP 1 „zostaje” i dodaje się do uwagi 2

Nie: po każdym podkroku dalej idą tylko położenia; x₃ = x₂ + Uwaga₂(x₂), MLP 1 siedzi w argumencie, nie w sumie. Czapa ≠ pułapka.

## 6. Użytkownik zlecił dwa pomiary agentami w tle (18:34, 18:38)

- `lasery_progi_miedzy_klasami_20261023__18-34-55`: wybiórczość laserów klasowa.
- `temperatura_widoczna_dla_glowy_20261023__18-38-56`: rozkład drgania na podprzestrzeń jakobianu głowy; predykcja zapisana
  przed pomiarem (3 progi).

## 7. Wynik 1 obalił moją wersję z pkt 2

Lasery wybiórcze są niemal wyłącznie **tłowe** (blok 4, 400 ep: 40,5 tylko tło, 2 tylko narzędzie, 9,5 obie; zero: 0); narzędzie jest
ciemniejsze niż tło; 48 laserów wybiórczych daje 73–91 % skoku Fishera w MLP 4 (losowy podzbiór: 0); lasery „tylko tło” odsuwają tło
od narzędzia (cos −0,23…−0,28). Poprawiona wersja: po uwadze 3–4 koloidy narzędzia lądują po ciemnej stronie progów, a strumień
unosi tło. Rozdział powstaje z ruchu ≈580 koloidów, nie dwudziestu. Spójne z mechaniką sprężyn: 580 sprężyn tła przeważa.

## 8. Wynik 2: predykcja nie potwierdzona

Obie składowe (widoczna/niewidoczna) stygną razem do ~30 ep i są płaskie do 400 ep; stosunek spadków 0,8–0,9 (próg ≥ 3: nie);
udział widoczny 0,08–0,10 przy zerze 0,023 → kierunki czytane przez głowę są 3–4× gorętsze na wymiar. Wniosek: po 30 ep ruch to
stacjonarny szum stałego LR; strata nigdy nie wiotczeje do końca; „sieć uczy się w kierunkach, których strata nie widzi” — w tej postaci
bez poparcia. Zostaje słabsze zdanie: dokładność i obie temperatury nasycają się razem ~30 ep, profil pól zmienia się jeszcze 370 ep.
Brakująca kontrola: pokrycie kierunków głowy z 5 głównymi kierunkami chmury.

## 9. Pytanie: życzenie i okno odczytu fizycznie

Ekran na końcu linii, głowa = optyka rzutująca koloid na 5 odczytów, etykieta = znacznik, strata = luźna sprężyna między obrazem
a znacznikiem; koloid = koralik sztywno osadzony na odkształcalnych szynach (polach), siła sprężyny przenosi się na szyny
(τ = Jᵀ F, statyka manipulatora = reguła łańcuchowa). Okno odczytu = cień: rura promieni rzutu padających na znacznik; kierunki
niewidoczne = ruch wzdłuż promienia. Tło ma 1 warunek (ciemno), środek narzędzia 5.

## 10. Alegoria całości: od teatru cieni do mikrosoczewek

- Użytkownik: „sprężyna słaba, bo u mnie światło”. Zaproponowany teatr cieni z 600 okienek (cień koloidu zawsze na własnej komórce).
- Użytkownik: „soczewka ze szkła z koloidami, światło od dołu skupia się na ramce”. Ocena: jedna soczewka kłamie (mieszanie na
  ekranie, którego nie ma; zachowanie światła, którego nie ma); **matryca mikrosoczewek** jest wierna (transmisja = ciepło, ognisko =
  przesunięcie, plamka = rozmiar; po doczytaniu modelu: l, t, r, b od środka komórki). Soczewka zbiorcza istnieje tylko w gwieździe uwagi.
- Użytkownik: układ całościowy (obraz na dole przez filtry CNN → koloidy = mikrosoczewki formowane przez lasery i uwagę → cień
  prostokąta na górze). Trzy poprawki: obraz nie przechodzi na ekran, lecz **naświetla** półfabrykaty przez łodygę (stan 0); na ekranie
  plamka + odczyt, nie wspólny prostokąt; lampa zapala się po 12. podkroku (sonda = logit lens, do oznaczenia).

## 11. Implementacja okienka M (18:50–20:00)

Rozpoznanie aplikacji (sfera: 768 reflektorów, szklana sfera, wiązka reflektor → załamanie → koloid wzdłuż d_c; dok, GLView, SSBO,
przejścia na dysku); agent Explore zmapował cykl życia doku, dane i testy. Scena: zdjęcie (z = −76), 4 płyty łodygi (PCA-RGB z
przejścia), promienie naświetlenia do koloidów, druga sfera (wspólne opcje z dużą), linki koloid → soczewka, matryca 30 × 20 (z = 60),
lampa z wiązkami, ekran (z = 78) z cieniem / różnicą / kalką i ramkami. Klucz `glowa` (13 × 600 × 5) w `history.podkroki`;
stare przejścia bez klucza liczone od nowa. Zrzuty sprawdzone, kamera poprawiona, testy zielone.

## 12. Ostatnie pytania teoretyczne

- Odczyt A (koloid = soczewka, lasery = maszyna) i odczyt B (stany = światło, bloki = szkło warstwowe = D²NN, Lin i in. 2018);
  różnice: progi (GELU), gwiazda (mieszanie zależne od pola — pasywne szkło tego nie ma), uczenie przepisuje wszystkie warstwy naraz
  (różnicowanie funkcji w czasie, nie kolejne nakładanie). Życzenie = pole sprzężone (in situ backpropagation, Hughes i in. 2018).
- SLA / polimeryzacja dwufotonowa: zgodny próg, niezgodny kierunek (laser przesuwa stan wzdłuż v_c, nie utwardza miejsca).
- Uwaga jako wtrącenie Eshelby’ego / STZ w szkle: dalekozasięgowe pole od 20 cząstek, ale sprzężenie po podobieństwie, nie po odległości.
- Tory przez 12 podkroków = rzeźbienie jednego półfabrykatu; ścieżki przez checkpointy = historia maszyny.
- „Super soczewka” jest stała (wszystkie wagi), mikrosoczewki z okienka M to ślad zdjęcia przechodzącego przez nią; soczewka
  nieliniowa i samonastawna, ale jedna dla wszystkich zdjęć (na 10 klatkach TEST IoU ~0,53).
