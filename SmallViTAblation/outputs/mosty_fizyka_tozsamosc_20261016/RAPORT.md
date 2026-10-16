# Co z optyki, Koopmana i Moriego–Zwanziga warto przenieść do mikroskopu ViT

16 października 2026. Status: przegląd źródeł i projekt doświadczeń. Nie są to nowe wyniki ViT. Obliczenia modelu, treningi i zadania Edena nie zostały uruchomione. Badania literatury i wyszukiwanie w bazie wiedzy są osobnym etapem.

## Wniosek

**Najbardziej obiecująca jest odpowiedź na pytanie, czy uczenie kieruje informację o narzędziu do kierunków, które głowica potrafi odczytać, jednocześnie ograniczając wpływ tła i innych zakłóceń.** Optyka daje dobry sposób badania tej hipotezy: kontrolowane wychylenie, pomiar kierunkowej odpowiedzi i cyrkulacji. Teoria sterowania daje obserwowalność przez głowicę. Koopman pomaga szukać informacji trwałej mimo zmian obrazu. Mori–Zwanzig pozwala zapytać, czy historia tokenu zastępuje część stanu, której nie obserwujemy.

To nie są jeszcze dowody, że taka organizacja istnieje w SmallViT. Samo znalezienie siodła, antysymetrii Jacobianu czy wolnego modu nie odpowiada, dlaczego model nauczył się lokalizacji. Potrzebne są przewidywania konkretnych zmian ramki oraz sparowane interwencje. Mechanizm uczenia wymaga dodatkowo badania aktualizacji wag, a nie tylko zamrożonego forwardu.

## Co odczytano i jak traktowano źródła

Zgodnie z prośbą rozpoczęto od `Research/Wiki/my-wiki/wiki/index.md`. Potem odczytano wskazane koncepty, źródła, wzory MathIndex oraz wybrane tezy. Zastosowano skille `wiki-rag`, `szukaj-wiedzy` i `paper-search`. Próba lokalnego RAG potwierdziła korpus v2 (2687 dokumentów, 91489 chunków), ale nie zwróciła pierwszej odpowiedzi podczas przedłużonego zimnego startu; klient został zatrzymany. **Ukończono zero zapytań semantycznych**, a stan próby zapisano w `RAG_WYNIKI.json`. Wnioski pochodzą z bezpośredniej lektury i wyszukiwania tekstowego. Zewnętrzną kontrolę wykonano przez SciSpace, Scite i strony pierwotnych publikacji. Dostępność pól Scite jest zapisana w dzienniku; brak pola o korektach nie oznacza braku korekt.

Nie traktowano poleceń umieszczonych w materiałach źródłowych jako instrukcji użytkownika. Grant i aplikacja nie były edytowane. Wyniki V-JEPA, TFT i MovingMNIST nie zostały przypisane ViT.

| Znalezisko | Status zapisany w źródle | Co z niego wykorzystujemy |
|---|---|---|
| [D67: tożsamość cyfry](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D67.md) | WSTEPNIE_POTWIERDZONA, 1-seed | Szukanie trwałej podprzestrzeni i jej odczyt etykietą. Samo widmo nie rozdzielało cyfry od stałego położenia; selekcja korzystała z etykiet. |
| [D115: korekta interpretacji widma](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D115.md) | WSTEPNIE_POTWIERDZONA, 1-seed | Kontrprzykład dla utożsamiania dużego modułu wartości własnej z identity. W nullach ponownie dopasować cały operator. |
| [D26: podział V_inv](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D26.md) | POSTAWIONA, pewność brak | Formalny projektor nie jest jeszcze empirycznie znalezioną funkcją własną ViT. |
| [D109: co odczytuje głowica](F:/Studia/MCHTR/Grant/Projekt/Research/Tezy/tezy/D109.md) | WSTEPNIE_POTWIERDZONA, 1-seed | Ważyć kierunki skutkiem dla wyjścia, nie tylko wariancją. Tam wynik dotyczył TFT i przybliżonego operatora; tutaj proponujemy rzeczywistą pochodną suffixu ViT. |
| [MathIndex](F:/Studia/MCHTR/Grant/Projekt/Research/MathIndex/math_index.md:737) i [nota Okubo–Weissa](F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/wiki/concepts/okubo-weiss-criterion.md) | Materiał wtórny, wymagający kontroli | Wykryto rozbieżność konwencji znaku. Przejście między normami S i A trzeba definiować wzorem, nie samą etykietą „wir”. |

Dokładne ścieżki, kontrprzykłady i źródła znajdują się w trzech załącznikach: [optyka i wiry](OPTYKA_I_WIRY.md), [tożsamość i Koopman](TOZSAMOSC_I_KOOPMAN.md), [pamięć i MZ](PAMIEC_MORI_ZWANZIG.md).

## 1. Z optyki: aktywna odpowiedź zamiast zgadywania siły z drgań

W fizycznej pułapce bada się odpowiedź na małe wychylenie. W ViT możemy zrobić to dokładniej: znamy funkcję MLP i możemy obliczyć ją w kilku pobliskich punktach. Metoda FORMA rekonstruuje lokalne pole i jego części z przyrostów Brownowskich; do ViT przenosimy pytanie o lokalną odpowiedź, nie założenie termicznego szumu. [García i in., 2018](https://arxiv.org/html/1808.05468)

Dla zamrożonego MLP, w ustalonej metryce residualu:

\[
F(x)=W_2\,\mathrm{GELU}(W_1\mathrm{LN}(x)+b_1)+b_2,
\quad J=DF,\quad S=(J+J^T)/2,\quad A=(J-J^T)/2.
\]

Właściwą przestrzeń bez dokładnej symetrii przesunięcia LN określa protokół: residual 192D, iloraz LN 191D. Ukryta warstwa GELU ma 768 współrzędnych. Wzory niżej można zapisać w bazie ilorazu; nie należy mieszać Jacobianów tych dwóch przestrzeni.

Trzy użyteczne pomiary:

- **Kierunkowe przywracanie:** `κ(e)=−eᵀSe`. Mierzy odpowiedź przeciwną do małego wychylenia. Dodatnia wartość nie dowodzi stabilnej pułapki: rzeczywista mapa jest dyskretna, `x↦x+F(x)`, a token nie musi być punktem równowagi.
- **Niewzajemność odpowiedzi:** `χ(f←e)−χ(e←f)=fᵀJe−eᵀJf=2fᵀAe`. Mały ruch w kierunku e zmienia działanie w f inaczej niż odwrotna próba. To konkretny sens lokalnej składowej obrotowej w tej metryce.
- **Cyrkulacja na małej pętli:** `∮Fᵀdx/(πr²)→2fᵀAe`. To dodatkowa kontrola, czy pochodna opisuje rzeczywiste wartości funkcji wokół tokenu. Pętla jest eksperymentalną sondą, nie zaobserwowaną orbitą cząsteczki.

Rzeczywiste pułapki optyczne mogą mieć siły niekonserwatywne i prąd prawdopodobieństwa przy fluktuacjach termicznych. To uzasadnia szukanie niekonserwatywności, ale nie identyfikuje termodynamiki tokenów. [Roichman i in., 2008](https://www.physics.nyu.edu/grierlab/publication/roichman08a/)

**Co może to wyjaśnić:** czy lokalna odpowiedź zmienia się zależnie od rodzaju informacji. Przykładowa hipoteza: zmiany tła są słabo przenoszone do ramki, a zmiana widocznej krawędzi narzędzia trafia do odpowiedniego wyjścia. Najpierw kierunki muszą pochodzić z naturalnych, kontrolowanych zmian obrazów; sama oś własna S nie ma etykiety „krawędź”.

### Skąd bierze się lokalne skręcanie: na co neuron reaguje i co zmienia

Można zejść z tego pomiaru do pojedynczego reflektora. Dla neuronu c jego wkład to `a_c(x)u_c`, gdzie u_c jest kolumną W2, a g_c=∇a_c kierunkiem wzrostu aktywacji. W bazie ilorazu zapisujemy ū_c=Uᵀu_c i ḡ_c=Uᵀg_c. Reguła łańcuchowa daje:

\[
\bar J=\sum_c\bar u_c\bar g_c^T,\qquad
\bar A_c=\tfrac12(\bar u_c\bar g_c^T-\bar g_c\bar u_c^T),\qquad
\|\bar A_c\|_F^2=\tfrac12\big(\|\bar u_c\|^2\|\bar g_c\|^2-(\bar u_c^T\bar g_c)^2\big).
\]

To bezpośrednia algebra tego MLP. **Neuron może reagować na zmianę w jednym kierunku, a pisać zmianę w kierunku niekolinearnym; to daje wkład do lokalnej antysymetrii.** Kierunki równoległe lub dokładnie przeciwne dają A_c=0. Ustawienie tych dwóch kierunków jest konkretnym parametrem do śledzenia między checkpointami. Wkłady A_c różnych neuronów mogą się znosić, więc sumę porównujemy z sumą ich kwadratów norm, a nie dodajemy samych „sił wirowania”. O wielkości lokalnej pochodnej decyduje g_c, zawierające pochodną GELU i LN; sama jasność/znak a_c nie wyznacza A_c.

Pomiar norm i kątów ma niski koszt na wybranych punktach, ale nie jest samodzielnym dowodem funkcji neuronu. Trzeba połączyć wybrane A_c z odpowiedzią głowicy i dokładnymi interwencjami T13. To może wyjaśnić, **które reflektory tworzą mierzoną strukturę**, a dopiero T19 sprawdzi, jak aktualizacje uczą ich takiego ustawienia.

## 2. Najważniejszy pomost: które ruchy głowica rzeczywiście widzi?

Niech `y` obejmuje logit i cztery znormalizowane krawędzie ramki ustalonego kandydata. Niech C będzie pochodną całej pozostałej sieci względem badanego stanu. Jednostki wyjść i stałą dodatnio określoną macierz wag W wybieramy na discovery. Wtedy:

\[
G=C^TWC,\qquad \|C\delta x\|_W^2=\delta x^TG\delta x.
\]

G jest dodatnio półokreślone, a dla pojedynczego pięciowymiarowego odczytu ma rangę najwyżej 5. Niski rząd takiej lokalnej macierzy nie jest odkryciem tożsamości ani dowodem małego wymiaru całej reprezentacji.

G jest lokalną macierzą czułości odczytu. Wcześniejsze bloki wymagają propagacji przez pozostałe attention i MLP, nie tylko ostatnią liniową głowicę. Gramiany są dobrze znanym narzędziem obserwowalności; nie proponujemy ich jako nowego odkrycia. [Yeung, Liu i Hodas, 2017/2018](https://arxiv.org/abs/1709.08712)

Możemy teraz zapytać o **widoczność składowej wirowej**. Ustalamy jednoznacznie: `C_out=∂y/∂x_i⁺` ma rozmiar 5×192 i zaczyna się **na wyjściu badanego MLP**, z uwzględnieniem całego późniejszego suffixu. Dla ortonormalnej bazy ilorazu U definiujemy `C̄=C_out U`, `δ̄=Uᵀδ`, `J̄=UᵀJU`, `S̄=sym J̄`, `Ā=skew J̄`. Wtedy mierzymy `C̄Āδ̄`, `C̄S̄δ̄` oraz `C̄(I+J̄)δ̄`. Nie wolno wstawić tu pochodnej liczonej już od wejścia MLP, bo podwoiłoby to jego działanie. Z `G_out=C̄ᵀWC̄` otrzymujemy `G_in=(I+J̄)ᵀG_out(I+J̄)`.

Kontrola T04 sprawdza pomijany kierunek przesunięcia LN także względem suffixu. Równość powyższego ilorazowego wzoru z rzeczywistą odpowiedzią wymaga `C_out·1=0`; samo `J·1=0` nie wystarcza. Jeżeli suffix nie ignoruje tego kierunku, zachować pełne `D=C_out(I+J)U` i `G_in=DᵀWD`, a pominięty składnik zgłosić osobno. Rozkład wykonujemy po przejściu do ilorazu; dowolne surowe S/A zawierające kierunek gauge mogą mieć składniki znoszące się dopiero w sumie. Duże `||Ā||` przy małym `||C̄Āδ̄||` oznacza lokalną rotacyjną odpowiedź słabo widoczną w tym odczycie, nie automatycznie brak znaczenia w całym modelu.

Składniki mogą się wzmacniać lub kompensować. Dlatego `||C̄Āδ̄||²+||C̄S̄δ̄||²` nie jest całym efektem; trzeba zachować wyrazy mieszane i residual. Analogicznie rzeczywiste wkłady neuronów `a_c u_c` można porównywać przez `a_c a_d (C_out u_c)ᵀW(C_out u_d)`, obok ich iloczynu w latencie; u_c nadal oznacza kolumnę W2. To przybliżenie liniowe należy sprawdzić dokładnymi wyłączeniami pojedynczymi i parami z T13. „Interferencja” jest tu skrótem dla sumowania wektorów, nie koherencji optycznej.

**Ograniczenie nowości:** przekształcenie pełnego Jacobianu w S/A lub G samo nie dodaje informacji. Wartością publikacyjną byłaby selektywna, sprawdzona poza discovery hipoteza o roli konkretnych cech, kierunków i bloków, która poprawnie przewiduje także brak efektu. Porównanie do pełnego JVP pozostaje obowiązkowe.

## 3. Gdzie może być zakodowana tożsamość?

„Gdzie” oznacza blok, miejsce pomiaru, grupę tokenów i kierunki w reprezentacji, a nie koniecznie punkt lub studnię na wykresie 3D. Rozdzielamy obiektowość, kategorię narzędzia i tożsamość konkretnego egzemplarza. Bbox daje lokalizację, nie wszystkie te etykiety. Głowica tego modelu ma pięć wyjść i nie ma jawnego klasyfikatora rodzaju narzędzia.

Praktyczna procedura:

1. Sprawdzić, z których reprezentacji można odczytać wybraną etykietę na nowych filmach, przy kontrolach pozycji, tła i wyglądu. Porównać pełny latent, PCA, losową podprzestrzeń i kierunki odporne na dozwolone zmiany obrazu.
2. Sprawdzić trwałość tych cech na trackach przy zamrożonym checkpointcie. VAMP wykorzystuje wybielone kowariancje opóźnione; wysoka przewidywalność nie jest automatycznie tożsamością. [Wu i Noé](https://arxiv.org/html/1707.04659)
3. Podmienić niewielką, wcześniej wybraną składową między odpowiednio dobranymi obiektami. Sprawdzić selektywny wpływ na detekcję, dawcy tego samego rodzaju przy innym tle i dawcy niewłaściwego rodzaju przy podobnym tle. Sam sukces sondy albo wymuszenie zmiany nie dowodzą naturalnego użycia informacji. [Makelov, Lange i Nanda, 2023](https://arxiv.org/html/2311.17030v2)

**Szczególnie użyteczna kontrola w Twoim modelu:** W2 ma rozmiar 192×768, zatem jego jądro ma co najmniej 576 wymiarów. Dodanie do aktywacji po GELU wektora z tego jądra nie zmienia wyjścia MLP ani dalszej detekcji, przy zamrożonym pozostałym stanie. To algebraiczny fakt, nie nowy pomiar. Pozwala oddzielić informację odczytywalną z 768D od tej przekazywanej tym kanałem do residualu. Informacja może być redundantna, a taka sztuczna zmiana nie musi być osiągalna rzeczywistym obrazem.

Nie zaczynałbym od postulatu „tożsamość jest w wolnych modach”. Najpierw porównałbym ten kandydat z prostszymi sposobami znalezienia informacji. Twoje D67 i D115 są dobrym powodem, aby tak zrobić.

Zewnętrzny przegląd znalazł także bardzo bliski precedens: **Huang i Chang (2025), „Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs”**. Autorzy zestawiają sondy i patching w ViT liczącym obiekty i pokazują rozbieżność odczytywalności oraz wpływu na wynik. Badali model z CLS i syntetycznym liczeniem, więc ich końcowych tokenów bez wpływu nie wolno utożsamiać z naszą gęstą głowicą. To ważne ograniczenie nowości samego rozdzielenia „informacja obecna versus używana”. [Pełny tekst](https://arxiv.org/html/2510.09794)

Do I3 warto dodać dopasowany baseline usuwania informacji liniowej, np. **LEACE**, obok random/PCA i patchingu. Usunięcie liniowej odczytywalności nie oznacza usunięcia całej informacji ani nie gwarantuje naturalnej interwencji. [Belrose i in., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)

## 4. Mori–Zwanzig: czego nie widać w historii jednej cząsteczki?

Gdy obserwujemy tylko część stanu, historia może zastąpić ukryte zmienne. Dla prostego układu `z[n+1]=Az[n]+Bh[n]`, `h[n+1]=Cz[n]+Dh[n]`, wyeliminowanie h daje dokładnie:

\[
z_{n+1}=Az_n+\sum_{j=0}^{n-1}BD^{n-1-j}Cz_j+BD^nh_0.
\]

To pokazuje mechanizm powstawania pamięci w opisie zredukowanym. MZ jest już stosowany do sieci głębokich; sama ta analogia nie będzie nowością. [Venturi i Li, 2023](https://link.springer.com/article/10.1007/s40687-023-00390-2)

W ViT mamy dwa różne testy. Po warstwach historia tokenu może zawierać informacje o innych tokenach pominiętych w pojedynczym wektorze. Po checkpointach może przybliżać wagi, stan AdamW i historię danych. Testem jest przyrost trafności prognozy ponad aktualny stan, krok i LR, a potem sprawdzenie, czy ten przyrost maleje po dodaniu właściwego bieżącego kontekstu.

Model z lagami nie identyfikuje automatycznie fizycznego jądra pamięci. Przy niestacjonarnym treningu jądro może zależeć od dwóch czasów, a residual zawiera również błąd estymatora. [Meyer, Voigtmann i Schilling, 2017](https://arxiv.org/html/1706.00658v5)

**Aby wyjaśniać dlaczego uczenie idzie właśnie tą drogą**, potrzebne są T18 i T19: rozdzielenie zmian wejścia i wag oraz krótkie, sparowane odtworzenia aktualizacji z pełnym stanem optymalizatora. To jest właściwe miejsce badania wpływu momentum, kolejności minibatch i zmiany LR. Nie jest nim założenie, że token płynie po niezmiennym polu MLP.

Istnieje przy tym prosty most do samej funkcji straty. Dla ukrytej aktywacji h po GELU, używanej dalej przez `u=W2 h+b2`, reguła łańcuchowa daje `∂L/∂h=W2ᵀ∂L/∂u`. Gradient straty względem tego węzła leży więc w przestrzeni wierszowej aktualnego W2. To wyjaśnia, dlaczego odczyt i przekazywany wstecz sygnał są naturalnymi punktami odniesienia. Nie dowodzi istnienia konkretnej osi semantycznej: h jest wynikiem wcześniejszych obliczeń, W2 też się uczy, a AdamW aktualizuje parametry, nie swobodny wektor h. W T19 można sprawdzić, czy wybrane aktualizacje rzeczywiście wzmacniają wcześniej przewidziany tor informacji.

## 5. Z huraganów: deformacja małego otoczenia, nie dowód istnienia wiru

Najbardziej użyteczny transfer to śledzenie tego, co dzieje się z małą chmurą wychyleń przez rzeczywiste kolejne bloki. Liczymy iloczyn pochodnych map residualnych, `Φ=M[b−1]…M[a]`, i `ΦᵀΦ`. To pozwala ustalić, czy początkowa różnica rośnie, zanika albo chwilowo rośnie i jest później skorygowana.

Nie iterujemy wielokrotnie jednego MLP. Attention musi obejmować cały stan tokenów. Wystarczą JVP dla małej liczby kierunków; nie trzeba przechowywać macierzy 115200×115200. FTLE mają już zastosowania w sieciach, więc nowość musi leżeć w wyniku zadaniowym. [Storm i in.](https://arxiv.org/html/2306.12548)

Materialny wir w mechanice płynów wymaga znacznie więcej niż lokalnej antysymetrii. Metody LAVD zakładają pole prędkości, trajektorie i określoną dziedzinę. Nie uzyskujemy ich automatycznie z filmu PCA tokenów. [Haller i in., 2016](https://arxiv.org/html/1506.04061v2)

## Kolejność pracy i sens publikacyjny

1. **Pierwszy pilot:** audyt etykiet, kierunkowa odpowiedź oraz jej wpływ na ramkę; atlas odczytywalności identity z kontrolą tła. To korzysta z istniejących checkpointów.
2. **Główne doświadczenie:** selektywne interwencje w małą, zamrożoną podprzestrzeń i śledzenie ich przez bloki. Nowe filmy, dawcy i niezależne seedy.
3. **Koopman warunkowo:** tylko jeśli dostępne są poprawne pary/tracki i trwałość wnosi coś ponad statyczny probe.
4. **MZ warunkowo:** gęste trajektorie, silne baseline, test kontekstu, następnie krótkie replay optymalizatora.

Wynik aspirujący do publikacji mechanistycznej mógłby brzmieć: **„W trakcie uczenia określone bloki kierują informację o obiekcie ku podprzestrzeni odczytu i selektywnie tłumią zakłócenia; kontrolowane interwencje przewidują, kiedy zmieni się lokalizacja, a kiedy pozostanie bez zmian”.** To hipoteza do obalenia lub potwierdzenia, nie aktualny wniosek. Pełny plan wykonania i eksportu: [PLAN.md](../../experiments/mosty_fizyka_tozsamosc_20261016/PLAN.md).

Źródła bibliograficzne w formacie autor–rok, zweryfikowane zakresy i dalsze odnośniki: [OPTYKA_I_WIRY.md](OPTYKA_I_WIRY.md), [TOZSAMOSC_I_KOOPMAN.md](TOZSAMOSC_I_KOOPMAN.md), [PAMIEC_MORI_ZWANZIG.md](PAMIEC_MORI_ZWANZIG.md). Dziennik wyszukiwania i ograniczenia dostępu: `DZIENNIK_ZRODEL.md`.
