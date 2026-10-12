# Wspólna identyfikacja dryfu, pamięci i wejścia obrazowego

2026-10-01. Niezależny przegląd gpt-6-astra, xhigh. Kontekst: `PLAN.md`, `MODEL.md`, `kod/koopman_obserwator.py`. Zakres nowego etapu: wspólne dopasowanie modelu na lokalnych danych CPU. Stanem pozostaje obserwowany skalar LDA; nie identyfikujemy pełnego stanu sieci. Filtrację całkowicie odkładamy.

## Rekomendacja równania i indeksów

Niech `u_k=phi(B_k)` opisuje obrazy i zrealizowane augmentacje źródłowej paczki k, która powoduje `S[k] -> S[k+1]`. Przy prognozie ze stanu `S[t]` znane są `S[0:t]` oraz `u[0:t-1]`. Prawdziwe `u[t]` jest niedostępne.

Dla sondy i ustalamy w chwili t referencję `a_it`, np. średnią jej ostatnich 500 stanów. W całym przykładzie uczącym, także w etykietach, używamy tej samej referencji:

\[
z_{i,r}^{[t]}=\psi(s_{i,r}-a_{it}),\qquad
v_{i,r}^{[t]}=z_{i,r}^{[t]}-z_{i,r-1}^{[t]}.
\]

Pierwszy pilot powinien wspólnie estymować **wszystkie** współczynniki równania

\[
\boxed{\;
z_{i,t+H}^{[t]}-z_{i,t}^{[t]}
=D_Hz_{i,t}^{[t]}
+\sum_{j=0}^{p-1}K_{H,j}v_{i,t-j}^{[t]}
+\sum_{\ell=0}^{q-1}G_{H,\ell}u_{t-1-\ell}
+b_H+\varepsilon_{i,t,H}.
\;}
\tag{1}
\]

Bloki oznaczają odpowiednio dryf w liftingu, pamięć stanu i pamięć wykonanych wejść obrazowych. Operator bieżącego stanu wynosi `I+D_H`. Równanie można skompresować, zastępując poszczególne opóźnienia u ustalonymi funkcjami jego historii. Nie uczyć najpierw autonomicznego dryfu, aby następnie dołączyć do jego skalarnego błędu dowolny blok u.

Równanie (1) aproksymuje warunkową średnią przyszłego liftingu przy dostępnej historii i polityce próbkowania. Przyszłe nieznane wymuszenia są uśredniane statystycznie w etykietach. **Nie identyfikuje ono współczynnika natychmiastowej reakcji na dowolnie zadane przyszłe u.** Taką interpretację trzeba odróżnić od użytecznej prognozy z historii. Przy `psi(x)=(x,x²)` odczyt s jest liniowy, więc średnia pierwszej współrzędnej bezpośrednio daje prognozę średniego stanu. Nie trzeba dekodować średniej przez nieliniowy autoenkoder.

Addytywne G jest ograniczeniem pierwszego modelu: ten sam obraz może inaczej oddziaływać na sondy w różnych stanach. Małe rozszerzenie słownika o `x_it * u_(t-1)` pozwala dopuścić taki efekt, nadal dopasowując wszystkie bloki wspólnie. Przy czterech współrzędnych obrazu kosztuje cztery kolumny. Jest to osobna, wcześniej ustalona ablacja; sama nazwa „operator kontrolowany” nie zapewnia wystarczalności addytywnego wejścia ani nie ustanawia zależności przyczynowej.

## Dlaczego pamięć wejścia należy identyfikować razem z pamięcią stanu

Przykład z ukrytym stanem h wyjaśnia strukturę, ale nie stanowi identyfikacji rzeczywistych zmiennych ViT:

\[
z_{k+1}=Az_k+Bu_k+Ch_k,\qquad
h_{k+1}=Eh_k+Fz_k+Ju_k.
\]

Po eliminacji h:

\[
z_{k+1}=Az_k+Bu_k+
\sum_{j=1}^{k}CE^{j-1}Fz_{k-j}+
\sum_{j=1}^{k}CE^{j-1}Ju_{k-j}+CE^kh_0.
\tag{2}
\]

Ukryta pamięć propaguje wcześniejsze stany **i wcześniejsze wejścia**. Ich skończone aproksymacje należy estymować jednocześnie. W modelu obserwowanym pominięcie drugiej sumy może przenieść jej efekt do oszacowanego dryfu lub jądra stanu. Współczynniki FIR są wystarczającym pierwszym celem; rozkład jądra na konkretne ukryte macierze `C,E,F,J` może być niejednoznaczny.

Wariant rzeczywiście kontrolowany, jednokrokowy, ma postać

\[
z_{k+1}=Az_k+\sum_{j=1}^{p}\widetilde K_jz_{k-j}
+Bu_k+\sum_{\ell=1}^{q}\widetilde G_\ell u_{k-\ell}+c+\eta_{k+1}.
\tag{3}
\]

W historycznej identyfikacji można używać `u_k`, jeśli para kończąca się w `k+1` jest już ukończona. Dla prognozy z t równanie (3) wymaga jednak modelu rozkładu `u_t,u_(t+1),...` i jego marginalizacji. Wstawienie rzeczywistych przyszłych paczek daje inne zadanie. Ustawienie przyszłego u na zero jest poprawne dla średniej tylko przy odpowiednich założeniach o jego warunkowej średniej i zależnościach; samo wycentrowanie treningowego u tych założeń nie zapewnia. Dlatego (1) jest prostszym pierwszym pilotem bez przyszłych wejść, a (3) osobnym protokołem identyfikacji i rollout'u.

Zastosowanie przyrostów w (1) jest jawną zmianą bazy pamięci. Dla skończonej historii:

\[
A z_t+\sum_{j=1}^{p}\widetilde K_jz_{t-j}
=\Big(A+\sum_{j=1}^{p}\widetilde K_j\Big)z_t
-\sum_{\ell=0}^{p-1}\Big(\sum_{j=\ell+1}^{p}\widetilde K_j\Big)(z_{t-\ell}-z_{t-\ell-1}).
\]

Już ta równoważność pokazuje, że nazwa i wartość współczynnika „dryfu” zależą od przyjętej konwencji pamięci. Należy zamrozić tę konwencję przed porównaniem mechanizmów.

## Wspólna estymacja i regularizacja

Zbudować jeden wiersz cech `X_it=[z_it, v_it,...,v_(it-p+1), historia u, 1]`. Dla horyzontów `H={1,3,5,10,20,50}` połączyć cele `Delta z_H` w macierz wielowyjściową. Współczynniki wszystkich bloków dopasować na tych samych wierszach, w tej samej skali i z tymi samymi wagami obserwacji:

\[
\min_{\{D_H,K_{H,j},G_{H,\ell},b_H\}}
\frac1n\sum_{i,t,H}\alpha_H
\left\|W_H\big(\Delta z_{i,t,H}-\widehat{\Delta z}_{i,t,H}\big)\right\|_2^2
+\lambda_D\sum_H\|D_H\|_F^2
+\lambda_K\sum_{H,j}w_j\|K_{H,j}\|_F^2
+\lambda_G\sum_{H,\ell}\gamma_\ell\|G_{H,\ell}\|_F^2.
\tag{4}
\]

`W_H` ustala skalę wyjść na treningu; zapobiega dominacji kwadratowej obserwabli lub jednego horyzontu. Stałe i niemal stałe współrzędne usunąć albo jawnie zabezpieczyć. Wyraz wolny pozostawić niekarany. W pierwszym pilocie wystarczy jedna wspólna lambda po standaryzacji wszystkich bloków; dodatkowe kary blokowe badać dopiero po prostym wyniku. Dopuszczalna, z góry ustalona mocniejsza kara dla dalekich opóźnień jest założeniem o jądrze, nie wnioskiem z danych. Ridge należy stosować do przyrostu operatora `D_H`, jeśli punktem odniesienia ma być brak zmiany.

Bez dodatkowej kary łączącej H, wielowyjściowy ridge nadal daje algebraicznie osobne oszacowania współczynników poszczególnych horyzontów. Wspólne są macierz cech, wiersze treningowe, transformacje i zasada doboru regularizacji. To wystarcza do uczciwej wspólnej identyfikacji **bloków** równania. Nie dowodzi zgodności `K_H=K_1^H`. Wymuszenie wspólnej semigrupy zmieniłoby hipotezę modelową; nie jest potrzebne na tym etapie.

Jeśli używamy referencji ruchomej `a_it`, wszystkie opóźnione stany i cały cel muszą być wyrażone względem **tej samej** `a_it`, a nie osobnych średnich z ich chwil. Wybór ten daje model lokalny z referencją utrzymywaną przez horyzont. Gdy dynamika zależy także od absolutnego poziomu, samo `s-a_it` może być niewystarczające; można z góry przewidzieć kontrolę z `a_it` jako dodatkową cechą. Nie tworzyć jej po obejrzeniu testu.

## Joint a poprawna residualizacja

Niech Z zawiera bieżący lifting, a W pamięć stanu i wejścia; Y jest całym celem liftingu. Przy zwykłych najmniejszych kwadratach twierdzenie Frisch–Waugh–Lovell prowadzi do

\[
M_Z=I-ZZ^+,\qquad
\widehat C=\arg\min_C\|M_ZY-M_ZWC\|_F^2,
\qquad
\widehat A=Z^+(Y-W\widehat C).
\tag{5}
\]

Trzeba więc usunąć część wyjaśnianą przez Z **zarówno z celu, jak i z W**, a potem ponownie wyliczyć współczynnik bieżącego stanu. Sama regresja reszty autonomicznego dryfu na surowym W nie jest równoważna wspólnemu dopasowaniu. Kolejność „najpierw dryf” definiuje wówczas arbitralną projekcję. Równoważność (5) dotyczy wspólnego zestawu wierszy i wag; uczenie etapów na różnych fragmentach czasu ją znosi.

Dla ridge, po osobnym usunięciu niekaranego wyrazu wolnego, rozważmy dokładnie cel

\[
n^{-1}\|Y-ZA-WC\|_F^2+\lambda_Z\|A\|_F^2+\lambda_W\|C\|_F^2.
\]

Jego redukcja daje

\[
S_Z=I-Z(Z^\top Z+n\lambda_ZI)^{-1}Z^\top,
\quad
\widehat C=(W^\top S_ZW+n\lambda_WI)^{-1}W^\top S_ZY,
\]
\[
\widehat A=(Z^\top Z+n\lambda_ZI)^{-1}Z^\top(Y-W\widehat C).
\tag{6}
\]

Zwykła regresja `S_ZY` na `S_ZW` używałaby `S_Z²`, a więc innego kryterium. To istotna pułapka regularizowanej residualizacji. W praktyce stosować rozwiązanie blokowe przez małe macierze Grama; **nie budować macierzy n × n**. Poprawne (5)/(6) jest alternatywnym sposobem obliczenia tego samego modelu joint, a nie niezależną rodziną wymagającą konkursu wyników.

W obecnym `koopman_obserwator.py` funkcje `drift_fit` i `complete_fit` używają różnych przedziałów czasu, a druga dopasowuje skalarną resztę do nieoczyszczonej historii. To poprawnie opisany predyktor dwuetapowy, lecz nie równanie (4), (5) ani (6). Jeśli zachować go jako kontrolę, trzeba wyrównać dostępną historię i zakres danych; różnica wyniku nie powinna mieszać zmiany estymatora z podwojeniem okresu uczenia.

## Kiedy rozdział jest nieidentyfikowalny

Jeśli `X=[Z,W]` ma niezerowe jądro, istnieje wektor v taki, że `Xv=0`. Wtedy dla dowolnego wektora wyjściowego a macierze `Theta` i `Theta+v a^T` dają identyczne prognozy, mimo innego podziału na dryf, pamięć i wejście. Przy prawie zależnych kolumnach przewidywania mogą być stabilne, a poszczególne bloki bardzo niestabilne. Dodatni ridge wybiera rozwiązanie zależne od kary; nie dostarcza brakującej informacji o mechanizmie.

Sprawdzać SVD pełnej macierzy cech oraz wejście po usunięciu części wyjaśnianej przez stan i pamięć. Małe wartości osobliwe tej ostatniej macierzy oznaczają, że dane słabo rozdzielają dodatkowy wpływ obrazów od historii stanu. Uzupełnić to o porównanie bloków czasowych, wielkości współczynników i prognoz po ablacji. Bogate paczki lub różne identyfikatory nie gwarantują persistent excitation ich zredukowanych cech. Powtarzanie tego samego opisu paczki w 180 wierszach sond nie tworzy 180 niezależnych realizacji wejścia.

Nawet pełna ranga identyfikuje najwyżej współczynniki danej projekcji statystycznej przy jej założeniach. Niewidziane wagi i pamięć optymalizatora mogą korelować z wejściem oraz resztą. Ponadto `paczka_lat` jest funkcją obrazów **i aktualnego encodera**. Jego historia jest przyczynowo dostępna, ale nie stanowi automatycznie zewnętrznego, niezależnie zadawanego sterowania. Współczynników takiego u nie wolno bez dodatkowych założeń przedstawiać jako skutku interwencji na treści obrazu.

Oddzielić treść/embedding obrazu i augmentacje od N/P, gradientów, strat oraz zmierzonych pchnięć. Główny pilot obrazu nie powinien zawierać tych ostatnich; są osobną kontrolą. Według aktualnego `PLAN.md` PCA obrazów wykazuje istotną zmianę rozkładu między s47 i s48. Kolejny ridge jej nie usuwa. Wspólny zamrożony encoder, jeśli jego cechy już istnieją lokalnie, byłby lepszą podstawą transferu. Bez niego wynik należy oznaczyć jako prognozę z historycznym embeddingiem zależnym od sieci. Metadane augmentacji i udział narzędzia mogą stanowić dodatkową prostą kontrolę, lecz nie zastępują pełnej zawartości obrazu.

## Konkretny mały pilot CPU

1. **Jeden stały lifting:** `psi(x)=(x,x²)` ze skalą z treningu, bez kolejnego uczenia autoenkodera. Cały lifting celu jest etykietą. Włączenie liniowej współrzędnej s zapewnia jednoznaczny odczyt prognozy skalarnej.
2. **Wspólne przykłady wszystkich H:** s47, starty `1000..1750` co 5, końce wszystkich etykiet `<=1800`; walidacja `1800..2000` co 5. Po wyborze refit startów `1000..2450` z końcami `<=2500`. s48 starty `2100..2950`; późny s47 od 2500. Wszystkie transformacje i PCA dopasować ponownie tylko na odpowiednim prefiksie. Ograniczenie `t+max(H)<=cut` zapobiega przypadkowemu dołączeniu niedojrzałych etykiet w macierzy wielowyjściowej.
3. **Pamięć:** p=5 albo 20 przyrostów liftingu. Wejście obrazowe: PCA4 istniejących opisów paczek, dopasowane na s47. Zastosować trzy rozłączne grupy historii: ostatnie `u_(t-1)`, średnia `u_(t-5)..u_(t-2)` i średnia `u_(t-20)..u_(t-6)`. Są to grupy 1, 4 i 15 wykonanych paczek. Rozłączne grupy ograniczają redundancję nakładających się średnich. Zachować diagnostykę skali i zakresu cech na obu biegach.
4. **Mała macierz:** przy p=20 powstaje 2+40+12+1=55 kolumn oraz 12 wyjść dla sześciu H. Zastosować jeden fit joint, standaryzację treningową, niekarany intercept i lambdy `{0.001,0.1,10}`. Nie zamrażać osobno autonomicznego operatora przed estymacją pozostałych bloków.
5. **Ablacje na identycznych wierszach:** brak zmiany, istniejący prosty predyktor ponownie dopasowany na tych samych wierszach, dryf joint bez pamięci i wejścia, dryf+pamięć stanu, dryf+pamięć stanu+tylko ostatni obraz, pełna pamięć stanu i obrazów. Porównanie ostatnich dwóch izoluje wartość pamięci wejścia. N/P dołączyć ewentualnie jako oddzielnie nazwaną kontrolę, nie jako składnik głównego modelu obrazowego.
6. **Jedna reguła wyboru:** z góry wybrać średnią po H z walidacyjnego `MSE_H / MSE_brak_zmiany,H`, z równą wagą horyzontów. Wybrać p i lambdę wyłącznie na s47. Raportować każdy H także osobno, wszystkie przegrane ablacje oraz błędy w czasie. Ten wspólny wybór różni się od wcześniejszego dobierania innego zwycięzcy dla każdego H; należy go jawnie oznaczyć.
7. **Przyczynowość i budżet:** test podmiany surowego sufiksu stanów od `t+1` i paczek od `t`, po którym przebudowuje się cechy przy zamrożonych transformacjach. Osobno sprawdzić, że tablica przesunięta `q[t]` zawiera wyłącznie paczkę źródłową `t-1`. Jeden wątek CPU, limit pięciu minut dopasowań po przygotowaniu małych cech, artefakty poniżej 100 MB. Nie uruchamiać dodatkowych poszukiwań architektury po obejrzeniu s48.

Ten pilot sprawdzi, czy wspólne oszacowanie bloków i jawna pamięć obrazów poprawiają prognozę w ustalonej klasie modeli. Dane s48 pozostają eksploracyjne; poprawa nie przywróci niezależności testu. Przewaga predykcyjna ani ranga regresji nie będą dowodem semantycznej tożsamości cząsteczek lub identyfikacji pełnego ViT.
