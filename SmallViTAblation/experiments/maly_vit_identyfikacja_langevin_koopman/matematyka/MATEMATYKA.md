# Dyskretna pamięć po eliminacji ukrytego stanu

Zweryfikowano dokładną eliminację w skończonym liniowym modelu. Jest to wynik o zadanym modelu matematycznym, a nie dowód, że latent ViT spełnia liniową rekurencję, jest Markowowski, obserwowalny lub sterowalny.

## Założenia i indeksowanie

Niech x_k ma wymiar n, h_k wymiar m, a u_k wymiar p. Macierze mają zgodne wymiary: A n×n, B n×m, C m×n, E m×m, D n×p i F m×p. Model działa od k=0:

$$x_{k+1}=A x_k+B h_k+D u_k,$$
$$h_{k+1}=C x_k+E h_k+F u_k.$$

Wszystkie sumy poniżej biegną od j=0 do k−1. Dla k=0 są puste i równe odpowiednim wektorom zerowym. Przyjmujemy E⁰=I. Nie zakładamy odwracalności ani stabilności E.

Iteracja ukrytej rekurencji daje

$$h_k=E^k h_0+\sum_{j=0}^{k-1}E^{k-1-j}(C x_j+F u_j).$$

Dowód przez indukcję: dla k=0 zostaje h_0. Pomnożenie wzoru dla k przez E i dodanie C x_k+F u_k daje wyrazy z potęgami E^{k-j} dla j<k i ostatni składnik z E⁰ dla j=k. To dokładnie wzór dla k+1. Kolejność mnożenia macierzy pozostaje ustalona; nie wolno zakładać przemienności.

Po podstawieniu:

$$x_{k+1}=A x_k+\sum_{j=0}^{k-1}B E^{k-1-j}C x_j+D u_k+\sum_{j=0}^{k-1}B E^{k-1-j}F u_j+B E^k h_0.$$

W zapisie opóźnień l=k−j definiujemy K_l=B E^{l−1} C oraz L_l=B E^{l−1} F dla l≥1. Wtedy

$$x_{k+1}=A x_k+\sum_{l=1}^{k}K_l x_{k-l}+D u_k+\sum_{l=1}^{k}L_l u_{k-l}+B E^k h_0.$$

Najbliższy historyczny stan x_{k−1} ma współczynnik BC, nie BEC. Bezpośrednie u_k ma współczynnik D, a historyczne u_{k−1} współczynnik BF. Dla k=0 otrzymujemy x_1=A x_0+D u_0+B h_0. Dla k=1 otrzymujemy x_2=A x_1+BC x_0+D u_1+BF u_0+BE h_0.

Skończony wymiar ukrytego stanu nie gwarantuje skończonej długości pamięci x. Jeżeli E^r=0, jądra zerują się po r opóźnieniach. Jeżeli promień spektralny E jest mniejszy od 1, odpowiedź zanika asymptotycznie; odcięcie jej po ustalonej liczbie opóźnień jest wtedy aproksymacją.

## Sprawdzenie SymPy

`verify_memory.py` sprawdza oddzielnie h_k i x_{k+1}, porównując bezpośrednią symulację z sumą eliminacyjną. Używa wyłącznie arytmetyki symbolicznej lub wymiernej:

- Pełny symboliczny skalar: k=0,…,6, wszystkie współczynniki, wejścia i warunki początkowe niezależne.
- Pełne symboliczne nieprzemienne macierze 2×2: k=0,…,2, niezależne wejścia i warunki początkowe.
- Dokładne macierze wymierne 2×2: k=0,…,8, stabilne E.
- Dokładne macierze wymierne 2×2: k=0,…,8, nilpotentne E.

Łącznie przeszło 28 par sprawdzeń. Każda reszta obu tożsamości jest dokładnie zerowa. Wynik to `sympy_wynik.json`. Kontrole skończonej liczby przykładów nie stanowią formalnego dowodu wszystkich macierzy i wszystkich k.

## Zakres formalnego dowodu Lean

`DiscreteMemory.lean` importuje wyłącznie `Init`. Kompiluje się lokalnie przez istniejący Lean 4.28.0, bez Mathlib, instalacji, sieci ani chmury. Udowodniono cztery twierdzenia:

1. `hidden_representation`: dla dowolnego k i całkowitych skalarów, h_k jest sumą propagowanego h_0 i pamięci zdefiniowanej rekurencyjnie jako M_0=0, M_{k+1}=e M_k+q_k.
2. `propagated_power`: propagowany składnik jest równy e^k h_0 dla każdego k.
3. `observed_elimination`: skalarna rekurencja x po podstawieniu q_j=c x_j+f u_j jest równa a x_k+b e^k h_0+b M_k+d u_k dla każdego k.
4. `memory_three`: M_3=e²q_0+e q_1+q_2, czyli jawna suma dla trzech historycznych wymuszeń.

To formalny dowód rekurencyjnej reprezentacji skalarnej nad Int oraz ograniczonej rozwiniętej sumy. Nie jest to Lean dowód ogólnej sumy macierzowej nad R. Ta ostatnia ma wyprowadzenie indukcyjne powyżej i kontrole SymPy. Typ Int obejmuje współczynniki ujemne, ale nie obejmuje dowolnych współczynników rzeczywistych.

W końcowym źródle nie ma `sorry`, `admit` ani deklaracji `axiom`. Audyt `#print axioms` wykazuje wyłącznie standardowe `propext`, używane przez upraszczanie w Lean. Nie dodano założeń specyficznych dla zadania poza jawnie przekazanymi równaniami początkowymi i rekurencjami. Kod wyjścia kompilatora wynosi 0.

## Obserwowalność i sterowalność rozszerzonego modelu

Dla stanu s_k=[x_k;h_k] i pomiaru y_k=x_k:

$$s_{k+1}=T s_k+G u_k,\quad T=\begin{bmatrix}A&B\\C&E\end{bmatrix},\quad G=\begin{bmatrix}D\\F\end{bmatrix},\quad H=[I_n\;0],\quad y_k=H s_k.$$

Przy N=n+m obserwowalność tego skończonego liniowego modelu oznacza rangę N macierzy O=[H;HT;…;HT^{N−1}]. Ranga N macierzy sterowalności R=[G,TG,…,T^{N−1}G] oznacza możliwość osiągnięcia dowolnego rozszerzonego stanu za pomocą dowolnie wybieranych wejść rzeczywistych w skończonej liczbie kroków. Kryteria dotyczą zadanych macierzy. W przypadku wejścia opisującego obraz lub zdarzenie z danych nie oznaczają możliwości fizycznego ustawiania każdego u_k.

SymPy sprawdza dwa dokładne przykłady z n=m=p=1. Dla T=[[1,1],[1,0]], G=[0;1] i H=[1,0] obie rangi wynoszą 2. Dla T=diag(1,1/2), G=[1;0] i tego samego H obie rangi wynoszą 1. Samo dopisanie h do modelu nie zapewnia żadnej z tych własności. Estymowany model wymaga oddzielnej analizy rang, wartości singularnych, skalowania i identyfikowalności z dostępnych wejść. Nie sprawdzano tych cech dla ViT.

## Gaussian colored AR nie oznacza white noise

Niech eta_{k+1}=rho eta_k+epsilon_k, 0<rho<1 i q>0. Załóżmy niezależne innowacje Gaussian o wariancji q oraz niezależny stacjonarny eta_0 o wariancji q/(1−rho²). Wtedy

$$\operatorname{Var}(\eta_k)=\frac{q}{1-\rho^2},\qquad \operatorname{Cov}(\eta_{k+1},\eta_k)=\frac{\rho q}{1-\rho^2}\ne0.$$

SymPy oblicza dokładnie wariancję eta_1 i kowariancję eta_1 z eta_0 ze wspólnych niezależnych standardowych zmiennych Normal. Dla rho=1/2 i q=3/4 otrzymuje kowariancję 1/2. Zatem te marginesy Gaussian tworzą proces kolorowy. Gaussian opisuje rozkład, a white noise wymaga zerowej kowariancji między różnymi chwilami. Dla rho=0 proces może być biały; ten przypadek jest wyłączony w powyższym kontrprzykładzie.

W deterministycznej eliminacji składnik B E^k h_0 jest warunkiem początkowym. Staje się zmienną losową dopiero po określeniu rozkładu h_0; sama eliminacja nie daje white noise ani stacjonarności. Jeśli do rekurencji h dodamy innowację w_k, eliminacja tworzy dodatkową sumę sum_{j<k} B E^{k−1−j} w_j, zwykle skorelowaną w czasie nawet dla niezależnych w_j. Weryfikacja AR jest osobnym kontrprzykładem, nie diagnozą rzeczywistych reszt ViT.

## Reprodukcja i artefakty

Uruchomienie ze środowiska Windows:

```powershell
& 'F:/Python/3.13/python.exe' 'F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/maly_vit_identyfikacja_langevin_koopman/matematyka/run_verification.py'
```

`run_verification.py` zapisuje `verification_result.json`: dokładne komendy, wersję Lean, stdout, stderr, kody wyjścia i SHA-256 trzech źródeł wykonawczych. `all_checks_passed=true` dotyczy wyłącznie opisanych kontroli algebraicznych. Nie uruchamiano testów na danych prognozy, eksperymentów ViT, usług płatnych ani zadań cloud. Wszystkie nowe artefakty leżą w tym katalogu `matematyka`.
