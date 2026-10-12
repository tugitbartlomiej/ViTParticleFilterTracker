# Czy cząsteczki w latencie ViT zachowują się jak szkło?

**Werdykt: potwierdzone jest częściowe podobieństwo kinematyczne do szkła; nie jest potwierdzony mechanizm szklisty. Prosta hipoteza, że podczas treningu cała populacja systematycznie zamarza, jest sprzeczna z tymi pomiarami.** Nie oznacza to matematycznego wykluczenia wszystkich możliwych reżimów szklistych w sieciach.

Data: 2026-10-10. Audyt na żądanie użytkownika, w pełnych 768 współrzędnych, bez PCA. Użyte skille: statistical-analysis, data-validation, data-visualization. Kryteria interpretacji sprawdzono w pełnych tekstach przez Undermind.

![Wynik audytu](wniosek_768d.png)

## 1. Co rzeczywiście przeliczono

- **Gęsty bieg 35 000 zdjęć, 21 epok:** 1533 zapisane stany, 8 stałych zdjęć, 600 tokenów na zdjęcie, 768 współrzędnych tokenu. Co 1 krok do 1094, potem co 50 do 22950. Przejście 1094→1100 ma 6 kroków i nie zostało potraktowane jak równy odstęp. Główne pomiary zaczynają się w kroku 300; rozgrzewkę analizowano oddzielnie.
- **Osobny bieg 35 000 zdjęć, 60 epok:** ponowna inferencja tych samych ośmiu zdjęć przez **806 istniejących checkpointów**, do kroku 65640. Nie wykonano nowego treningu. Siatka regularna po kroku 1000 ma odstęp 100; dodatkowe końce epok wykorzystano w kontroli FP32. Okna przecinające wznowienie po kroku 22974 wykluczono z analizy starzenia.
- Bieg gęsty i bieg 60 epok to dwie różne trajektorie z tej samej receptury i ziarna 47. Nie sklejano ich i nie traktowano jako niezależnych losowych ziaren. Różnica ocen przy wspólnych krokach dochodzi do 0,0455 mAP.
- Osiem sond to wiersze puli 144, 448, 922, 2741, 2786, 3200, 3438, 3698. Pochodzą z **7 filmów**, wszystkie mają etykietę narzędzia cystotome i należą do treningu. Są pierwszymi ośmioma z uporządkowanej listy 100 sond. To nie jest reprezentatywna próba wszystkich klas ani walidacja generalizacji.
- Potwierdzono SHA-256 bajtów ośmiu obrazów podczas odtwarzania, zgodność wymiarów i ścisłe wczytanie wag. Obrazy przygotowano tym samym skalowaniem INTER_AREA do 480×320.

## 2. Definicje

Cząsteczka to reprezentacja **tego samego tokenu tego samego obrazu** w kolejnych krokach uczenia. Wspólna przestrzeń 768 osi pozostaje układem odniesienia. Odjęcie średniego przesunięcia jest dodatkową kontrolą ruchu zbiorowego, nie definicją globalnego układu.

MSD jest średnią wartości `||z(t_w + tau) - z(t_w)||²` po 600 tokenach i ośmiu obrazach. Podane jednostki to kwadrat jednostek aktywacji; nie dzielono wartości przez 768.

Zawracanie mierzono znormalizowaną autokorelacją **iloczynu skalarnego przyrostów**, a osobno średnim cosinusem kierunków i udziałem ujemnych iloczynów. Długość przyrostu oraz odstęp pomiędzy przyrostami to dwie różne zmienne. Przyrostów co 1, 10, 50 i 100 kroków nie utożsamiano.

Pomiary wykonano w ruchu surowym, po odjęciu średniego ruchu populacji, po odjęciu ruchu centroidu obrazu i po odjęciu średniego ruchu sześciu sąsiadów. Sąsiadów wybierano według pełnej odległości 768D wewnątrz obrazu. Główne krzywe mają sąsiadów ustalonych w kroku 300; dodatkowa kontrola wyznacza ich ponownie w każdym początku pomiaru. Jest to **sąsiedztwo reprezentacji**, nie wykazane fizyczne oddziaływanie ani potwierdzona klatka mechaniczna.

## 3. Zawracanie: potwierdzone, zależne od skali czasu

Okno 300–1094 gęstego biegu; średnia po obrazach.

| Długość sąsiednich przyrostów | Surowy ruch 768D | Ruch względem sąsiadów |
|---:|---:|---:|
| 1 krok | +0,678 | +0,477 |
| 5 kroków | −0,110 | −0,057 |
| 10 kroków | −0,299 | −0,264 |
| 20 kroków | −0,419 | −0,387 |
| 50 kroków | −0,401 | −0,398 |

Na pojedynczych krokach dominuje podtrzymanie kierunku. Dopiero po kilku krokach pojawia się przeciwny składnik ruchu. W dodatkowym pomiarze tokenowym około 83 procent par przyrostów dziesięciokrokowych ma ujemny iloczyn skalarny (do 64 równomiernie rozłożonych początków na obraz). To nie znaczy, że każdy token wraca dokładnie do wcześniejszego punktu.

Zawracanie jest zatem realne w zapisanej geometrii 768D. Nie jest samo w sobie sygnaturą szkła: taki efekt daje również ograniczony ruch stochastyczny z pamięcią.

## 4. Plateau: częściowo potwierdzone, ale nieswoiste

We wczesnym oknie po rozgrzewce:

- Przy opóźnieniach 1–5 kroków wykładnik MSD wynosi 1,53 dla ruchu surowego: ruch jest uporządkowany w czasie.
- Przy opóźnieniach 10–100 wynosi 0,259 w surowym ruchu oraz 0,325 względem sąsiadów.
- W węższym zakresie 35–80 spada odpowiednio do 0,184 i 0,242. Widać wyraźne spłaszczenie, ale nie ścisłe zatrzymanie.

W późniejszych oknach 10000–15000 i 17500–22500 MSD względem sąsiadów jest płaskie w szerszym zakresie: wykładniki dla 200–2000 kroków wynoszą około **0,169 i 0,181**. W długim biegu, dla szerokiego zakresu wieku 24000–65600, wykładnik dla opóźnień 500–5000 wynosi **0,189** względem sąsiadów, ale **0,295** dla surowego ruchu. Istnieje więc rzeczywiste, zależne od układu odniesienia ograniczenie wzrostu przemieszczenia.

Próg audytu „|alpha| ≤ 0,2 przez dekadę opóźnień i wzrost MSD nie większy niż 10^0,2” jest **konwencją diagnostyczną**, nie uniwersalną definicją szkła. Wczesna krzywa go nie spełnia, część późnych krzywych względem sąsiadów go spełnia. Policzono także słabszy próg 0,35. Braku tego progu nie interpretujemy jako dowodu nieistnienia szkła.

Spłaszczenie i późniejszy wzrost odtwarza także prosty model **OU z pamięcią szumu plus wolna dyfuzja**, bez klatek tworzonych przez wzajemne oddziaływanie sąsiadów. Dla wczesnej krzywej względem sąsiadów RMSE log MSD na pominiętych podczas dopasowania opóźnieniach wynosi 0,021, czyli około 2 procent błędu multiplikatywnego. Zwykły OU daje 0,148. Model z pamięcią przewiduje też korelację kolejnych przyrostów co 10 kroków −0,256 wobec zmierzonego −0,264.

To jest sprawdzian **nieswoistości sygnatury**, nie dowód, że prawdziwy proces jest OU. Opóźnienia tej samej krzywej są skorelowane; to nie walidacja na niezależnym treningu. Dopasowanie gorzej odtwarza najkrótsze skale, szczególnie surowy ruch co 1 krok.

## 5. Starzenie: zależy od fazy, skali i układu odniesienia

**Na początku następuje uspokojenie.** Między oknami wieku 300 i 700 energia pojedynczego kroku maleje w ruchu surowym o około 51 procent, a względem sąsiadów o 36 procent. Stały learning rate nie oznacza stałych statystyk ruchu.

**Potem lokalne przemieszczenia rosną.** Przy stałym opóźnieniu 1000 kroków i sąsiadach odświeżanych na początku pomiaru:

| Bieg | Wiek początku | MSD względem odświeżanych sąsiadów |
|---|---:|---:|
| Gęsty, 21 epok | 1000 | 2,250 |
| Gęsty, 21 epok | 19000 | 4,945 |
| Długi, 60 epok | 1000 | 2,212 |
| Długi, 60 epok | 24000 | 5,416 |
| Długi, 60 epok | 40000 | 5,355 |
| Długi, 60 epok | 50000 | 5,936 |
| Długi, 60 epok | 60000 | 5,049 |

W gęstym biegu wzrost od 1000 do 19000 wynosi **2,20 raza**. Występuje na każdym zdjęciu: indywidualne stosunki 1,81–3,01. W długim biegu po 24000 ruch lokalny waha się w pobliżu podobnego poziomu; od pierwszego do ostatniego z tych wieków spada o około 7 procent. Nie znika.

**Duże przemieszczenia surowe mogą jednocześnie zwalniać.** Dla wspólnego wysokiego progu MSD około 91,7 w gęstym biegu czas dojścia rośnie z około 2271 kroków przy wieku 1000 do 7430 przy wieku 10000. Względem sąsiadów odpowiednie testy wspólnych progów często pokazują odwrotną tendencję. Istnieją więc różne skale relaksacji; nie wolno opisać ich jednym wskaźnikiem starzenia.

**W bardzo późnym treningu jest umiarkowane spowolnienie części ruchu.** W długim biegu, między wiekiem 24000 a 60000, surowe MSD przy opóźnieniu 1000 maleje z 60,22 do 44,16, czyli o 27 procent. Przy opóźnieniu 5000 maleje już tylko o około 4 procent. Czasy do wysokich progów nie rosną monotonicznie. Nie ma stabilnego prawa starzenia obejmującego wszystkie fazy i skale.

Przetestowano trzy wspólne progi przemieszczenia i zaznaczono progi nieosiągnięte jako cenzurowane. Malejąca amplituda `MSD(t_w, tau)` przy stałym `tau` nie jest tym samym wykładnikiem, co wzrost czasu relaksacji `t*(t_w)`. W szczególności wcześniejszego wykładnika spadku amplitudy nie wolno bezpośrednio utożsamiać z wykładnikiem subaging z literatury.

## 6. Kontrola skali, BatchNorm i jakości modelu

Po znormalizowaniu każdego tokenu do długości 1, w FP32, dla stałego opóźnienia **5 epok**:

| Warunek | Wiek 25 epok | Wiek 50 epok |
|---|---:|---:|
| Standardowe statystyki BatchNorm: MSD kierunków | 0,3481 | 0,5104 |
| BatchNorm zamrożone na statystykach kroku 300: MSD kierunków | 0,3950 | 0,4048 |

Początki uśredniono po trzech sąsiednich epokach. Bez kontroli BN ruch kierunków rośnie o około **47 procent**. Po zamrożeniu statystyk pozostaje niemal stały: około **+2,5 procent**. Kierunki reprezentacji nadal zmieniają się również przy zamrożonych statystykach. To nie wspiera późnego, systematycznego zamarzania i pokazuje istotną wrażliwość części ruchu na BatchNorm. Zamrożenie na wczesnych statystykach jest kontrolą wrażliwości, nie ścisłym rozkładem przyczynowym ani ponownym treningiem.

BF16 i FP32 dają dla sprawdzonych przemieszczeń dziesięciokrokowych różnice średniego MSD do **0,36 procent**. Dla sprawdzonych długich odcinków różnice są również małe. Nie jest to pełna kontrola FP32 każdego pojedynczego kroku historycznego gęstego biegu.

Bieg 60 epok jest faktycznie dostępny i został użyty. Nie jest jednak coraz lepszym detektorem tylko dlatego, że jest dłuższy: najlepszy koniec epoki ma **mAP50 = 0,6533 w epoce 26**, epoka 21 ma 0,6524, a epoka 60 ma 0,6084. Średnie walidacji po plateau są podobne. Ta historia jest silniejsza jako obserwacja **długotrwałej dynamiki**, nie jako dowód szkła ani rosnącej jakości.

## 7. Co potwierdzamy, a czego nie

**Potwierdzamy:** rzeczywisty ruch w pełnej przestrzeni, krótką pamięć kierunku, późniejsze zawracanie, subdyfuzyjny wzrost MSD w określonych zakresach, częściowe plateau i zależność od wieku treningu. To nie jest wyłącznie złudzenie rzutu PCA.

**Odrzucamy prosty obraz:** starszy model coraz mocniej i jednostajnie więzi wszystkie cząsteczki. Lokalny ruch najpierw rośnie, później pozostaje aktywny; tempo zmian zależy od skali czasu. Kontrole kierunków i BN nie pokazują trwałego późnego zamarzania.

**Nie potwierdzamy:** wielociałowego mechanizmu szkła, przejścia szklistego ani diagnozy „ciecz przechłodzona”. Te obserwacje są zgodne również z ograniczonym ruchem z pamięcią, zmiennym wymuszeniem przez wspólne wagi, wolną przebudową reprezentacji i zmianą statystyk normalizacji. Stabilne sąsiedztwo w latencie może wynikać również ze stałych różnic reprezentacji między tokenami; samo nie dowodzi klatki mechanicznej.

Brak spójnej diagnozy szkła nie wyklucza słabszych analogii ani innych reżimów w innych modelach. Obecne dane nie zawierają kontrolowanej zmiany temperatury/szumu, testu barier ani interwencji izolującej wzajemne oddziaływanie tokenów. Nie można na ich podstawie udowodnić albo globalnie wykluczyć fazy szklistej. Określenie, które odpowiada pomiarom, to: **dynamika reprezentacji z pamięcią, ograniczonym ruchem i wieloma skalami relaksacji**.

## 8. Walidacja i materiały

- Macierze Grama policzono z pełnych współrzędnych, z centrowaniem czasowym i akumulacją float64. Maksymalny względny błąd wobec bezpośrednich odległości float64 w sprawdzonych parach wyniósł poniżej **0,003 procent**.
- Niezależne sprawdziany wzorów MSD, VACF i odjęcia centroidu przeszły. Analityczny stacjonarny OU daje plateau i ujemne korelacje bez starzenia: kontrola ujemna jest poprawna.
- Przedziały w załączniku są bootstrapem po 7 filmach, warunkowo względem danego treningu. Tysiące tokenów nie są traktowane jako niezależne powtórzenia. Nie raportujemy pozornej istotności statystycznej dla „fazy szkła”.
- Oryginalne pliki pozostawiono bez zmian. Tymczasowy plik latentów długiego biegu, około 5,9 GB, usunięto po obliczeniach. Zachowano macierze, kontrole FP32, wyniki, źródła i kod odtworzenia.

[Pełne tabele i odtworzenie obliczeń](WYNIKI_LICZBOWE.md) · [Diagnostyka gęstego biegu](dense_diagnostics.png) · [Diagnostyka 60 epok](long_diagnostics.png) · [Kontrola skali](epoch_scale_controls.png) · [Walidacja](VALIDATION.json) · [Pochodzenie i hashe kodu](AUDIT_PROVENANCE.json).

Kod i plan: `../../experiments/audyt_szkla_768d_20261010/`.

## Literatura sprawdzona w Undermind

1. Winter, Janssen, *Glassy dynamics in deep neural networks: A structural comparison*. [Tekst](https://arxiv.org/abs/2405.13098). Sekcje III.B, III.D i dodatki F/H rozróżniają ograniczenie w dołku od klatki, a częściowe podobieństwa od pełnej fenomenologii szkła.
2. Baity-Jesi i in., *Comparing dynamics: deep neural networks versus glassy systems*. [DOI](https://doi.org/10.1088/1742-5468/ab3281). Równanie 3 oraz rysunki 3–5: dwuczasowe MSD, starzenie, późna dyfuzja i różne reżimy parametryzacji.
3. Herrera Segura i in., *Subaging in underparametrized deep neural networks*. [DOI](https://doi.org/10.1088/2632-2153/ac8f1b). Starzenie mierzone przez wzrost czasu dojścia do wspólnego progu, z czułością na próg i skończone okno.

Prace te mierzą głównie **wagi** i inne optymalizatory. Przeniesienie ich interpretacji na aktywacje tokenów AdamW jest hipotezą badawczą, nie gotowym dowodem.
