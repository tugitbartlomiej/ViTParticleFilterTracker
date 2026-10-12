# Wspólny ruch cząsteczek i dodatkowy stan w równaniu Langevina

Data: 2026-10-02, Europe/Warsaw. Rodzina: `maly_vit_langevin_wspolny_ruch`. Klasa: badawcza, eksploracyjna. Użytkownik zlecił obliczenie proponowanego modelu i poszukiwanie lepszej prognozy. Plan zapisany przed nowym dopasowaniem i oceną wariantów. Wykorzystujemy procedurę skilla `eksperyment`, z lokalnymi deklaracjami w repozytorium SmallViT; CLI rejestru projektu Grant nie należy do tego repozytorium.

## Pytanie i kryterium wyniku

Czy uwzględnienie wspólnego ruchu cząsteczek narzędzia, reprezentacji obszaru poza maską i kilku kierunków spoza podprzestrzeni tożsamości poprawia autonomiczną prognozę ośmiu współrzędnych tożsamości względem wcześniejszego modelu korzystającego z historii każdej cząsteczki osobno?

Główny wybór modelu: średnia redukcja MSE względem persystencji na horyzontach 1, 10 i 50 aktualizacji na walidacji. Każdy horyzont ma równą wagę. Osobno wybieramy model dla jednej aktualizacji. Dla dyskretyzacji co10 analogicznie średni wynik H10/H50 i model dla H10. Nie wybieramy zwycięzcy po sprawdzeniu testu. Wynik ujemny lub brak poprawy jest pełnoprawnym wynikiem.

Przewaga będzie oceniona wobec wcześniejszego zamrożonego GLE oraz odpowiednich prostszych wariantów na dokładnie tych samych originach i celach. Wariant z dodatkowymi stanami ma większy dostęp do historii pomiarowej; jest to badany czynnik. Ablacje oddzielą korzyść sprzężenia od korzyści dodatkowych kierunków. Nie twierdzimy, że sam większy model identyfikuje prawdziwe siły.

## Dane i granice interpretacji

- Ten sam seed47, 1001 stanów po aktualizacjach 0..1000; pełne tokeny co1, wagi co10. Źródło `maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/dane/LATENT_GESTY.npy`, shape1001×12×600×768, tylko mmap i małe wycinki w RAM.
- Główny obraz: indeks2, `cataract_lmm/tracking/TR_0057_S1_P03/00001431`, 31 cząsteczek maski, source-role fit/train. Maska i pokrycie pozostają stałe.
- Dodatkowy obraz dla transferu operatora: indeks8, źródłowa rola holdout. Parametrów nie dopasowujemy na jego trajektoriach. Nie jest to nowy bieg ani niezależne ziarno. Jego identyfikator i liczba cząsteczek zostaną zweryfikowane w metadanych.
- U500, rank8, z wcześniejszego eksperymentu pozostaje zamrożone. U1000 nie jest wejściem tej analizy. Cel q=U500^T z ma identyczną definicję dla wszystkich wariantów.
- Obszar `background` pochodzi z maski źródłowej. Nie zakładamy, że brak tam innych narzędzi lub obiektów: jest to kontekst poza maską badanego narzędzia.
- Dla kontekstu wyznaczamy średnią pełnych wektorów wszystkich tokenów tej maski, następnie projektujemy ją tą samą bazą [U500,V]. Nie tworzymy niezależnej definicji przestrzeni dla tła.
- Pomiar w eval obejmuje zmiany wag i statystyk BatchNorm. Brak wykonanej historii batch-ID, augmentacji i gradientów tego biegu. Nie wprowadzamy fikcyjnego sterowania z innych biegów.
- Cała historia i wcześniejszy test800..1000 były już oglądane. To nowa analiza algorytmicznie korzystająca z przeszłości, lecz test eksploracyjny, nie nietknięte potwierdzenie. Transfer na obraz8 zwiększa zakres sprawdzianu, ale nie zastępuje nowych niezależnych treningów.

## Dodatkowe kierunki V

Nie używamy wszystkiego z768D w swobodnej dużej regresji. Budujemy niewielką bazę V ortogonalną do U500. Stan pojedynczej cząsteczki:

```math
s_{i,t}=[U_{500}^T(z_{i,t}-z_{i,500}),\ V^T(z_{i,t}-z_{i,500})].
```

Przestrzeń U opisuje docelowe współrzędne tożsamości; V dostarcza pomocniczych zmiennych dynamiki. Początkowi kandydaci: rank0,4,8. V wyznaczamy wyłącznie na zakończonych parach fit do700, bez obrazów/stanów walidacji i testu:

1. PCA reszt przyrostów poza U: `(I-UU^T)(z_t-z_(t-1))`, z centrowaniem i ważeniem pokryciem.
2. Baza predykcyjna PLS/SVD: lewostronne kierunki kowariancji pomiędzy tą resztą w chwili t i przyszłym przyrostem q w chwili t+1, tylko dla par z końcem <=700. Jest to nadzorowana selekcja cech na treningu. Nie nazywamy jej nienadzorowaną PCA ani dokładnymi obserwablami Koopmana.

Zapisać dokładne indeksy, centrowanie, wagi, wartości osobliwe, wybrane rangi, U^T V i ortonormalność. Bazy dostępne są najpóźniej w700. Po selekcji nie refitujemy U/V na teście; refit operatorów do799 może zachować bazę wyznaczoną do700. Różne rangi biorą początkowe kolumny tej samej bazy. Wyższy rząd lub dodatkowy wariant wolno dołączyć tylko po jawnej decyzji na walidacji, przed jakąkolwiek oceną nowych modeli na teście.

## Sprzężony model

Średnia narzędzia `f_t=sum_i w_i s_i,t`, w_i proporcjonalne do pokrycia maską. Kontekst `g_t` jest projekcją średniej tokenów poza maską, względem własnego anchora500. Model cząsteczek:

```math
\Delta s_{i,t+dt}=b+A s_{i,t}+C_f f_t+C_g g_t
  +\sum_{l=0}^{L-1}\{K_l\Delta s_{i,t-l dt}
                   +M_{f,l}\Delta f_{t-l dt}
                   +M_{g,l}\Delta g_{t-l dt}\}+\epsilon_{i,t+dt}.
```

Kontekst ma osobny, dopasowany na tym samym prefiksie operator przyrostu z jego historią i historią f. Wszystkie parametry danego operatora dopasowujemy wspólnie przez ridge. Niepenalizowany intercept, skale tylko z fit, jawny powrót do oryginalnych jednostek. Model jest empirycznym dyskretnym GLE/afiniczną aproksymacją Koopmana w rozszerzonym stanie. Pamięć dotyczy skończonej historii przyrostów.

Przy każdym kroku autonomicznym:

1. Wyliczyć aktualne f z bieżących stanów wszystkich prognozowanych cząsteczek.
2. Obliczyć przyrosty wszystkich s_i i g z tego SAMEGO bieżącego stanu/historii.
3. Zaktualizować jednocześnie s_i oraz g, przesunąć historie o własne przewidziane przyrosty.

W przyszłości nie podajemy prawdziwych f, g, q, V^Tz, obrazów, gradientów ani reszt. Dodatkowe stany również muszą być przewidziane. Wagi pokrycia i anchor znane z przeszłości są stałe. Sprzężenie przez średnie ma niski rząd przestrzenny; korelacja nie dowodzi fizycznego oddziaływania cząsteczek. W tej architekturze wspólne człony dodają ten sam przyrost wszystkim cząsteczkom. Po odjęciu średniej narzędzia kasują się, więc nie modelujemy jeszcze odmiennej reakcji różnych tokenów na ten sam kontekst. To ograniczenie pierwszego testu, nie ogólny model wszystkich interakcji.

## Rodziny i strojenie

- Persystencja i stała ostatnia prędkość, identyczne oryginalne współrzędne celu.
- Dokładny poprzedni GLE U8 bez sprzężenia, zamrożone macierze i ustawienia; kontrola odtwarzania jego wyniku.
- Obowiązkowo lokalny U8 w NOWEJ siatce, z taką samą normalizacją i procedurą jak warianty sprzężone. Nowe skalowanie każdego lagu osobno i nowa siatka ridge różnią się od starego GLE. Dlatego sprzężony U8 porównujemy również z tą ablacją, aby oddzielić sprzężenie od zmiany estymatora.
- Sprzężony U8 bez dodatkowych kierunków.
- Lokalny U8+V bez sprzężenia: kontrola samej dodatkowej obserwacji.
- Sprzężony U8+V dla PCA i PLS, r=4/8.

Startowa siatka: L0/2/4/8, dodatkowo L1 dla dt10; ridge0.001/0.01/0.1/1/10. Rozsądnie współdzielić statystyki treningowe i unikać duplikatów. Rejestrować pełny wynik walidacyjny, liczbę parametrów, kondycję, zakres parametrów, czas i ewentualne eksplozje. Nie usuwać nieudanych prognoz; niestabilny model otrzymuje jawnie dyskwalifikujący wynik walidacyjny. Jeżeli nie działa baseline reprodukcji, naprawić implementację przed testem.

## Chronologia

- dt1: fit target501..700; validation origin700..749, H1/10/50, target<=799. Początkowe lagi sprzed500 są dozwolone jako znana historia. Dla wszystkich L te same targety fit. Po wyborze refit operatorów z targetami<=799.
- Główny test: origin800..950 dla H1/2/5/10/20/50, identyczne listy dla modeli. H100 opcjonalnie oddzielna diagnostyka800..900 bez mocnego CI.
- dt10: fit target510..700co10; validation origin700..740co10, H10/50; refit do790. Test origin800..950co10, H10/20/50. Model dt1 dodatkowo oceniony na tych samych originach.
- Główne modele i najlepsze warianty krótkiego horyzontu wybieramy na walidacji, zapisujemy ich konfigurację/hashe przed testem. Przed testem niezależny audyt Astry. Po teście nie dostrajamy wariantów do jego wyniku.
- Transfer na obraz8: te same U/V i operatory, jego maska, znany anchor500 oraz historia do originu. Żadnego refitu, doboru hiperparametrów lub wyboru cząsteczek według wyniku tego obrazu.

## Metryki i kontrole

Główny MSE dotyczy WYŁĄCZNIE ośmiu docelowych q w ich oryginalnej skali, z wagami coverage, średnią po originach i współrzędnych. Pomocnicze r i g nie zmieniają mianownika metryki. Raportować skill względem persistence, względną zmianę MSE wobec starego GLE, cosine wektorów przemieszczenia, znak wszystkich składowych i osobno filtrowany znak z progiem z fit. Używać identycznych progów oceny q dla porównywanych modeli. Filtr według rzeczywistej wielkości przyszłej zmiany jest analizą po poznaniu celu, nie pewnością dostępną online.

Paired circular block bootstrap originów, wspólne bloki dla modeli i wszystkich cząsteczek, blok minimum50 aktualizacji, 300 powtórzeń. Przy151 originach daje około3 efektywne bloki; CI pozostają opisowe. Nie przedstawiać cząsteczek jako31 niezależnych doświadczeń. Transfer8 raportować oddzielnie.

Kontrole konieczne: odtwarzalność starego GLE, zgodność surowych/sklowanych macierzy, synchroniczność rollout, skończoność wyników, prefix/poison także dla kontekstu i pomocniczych stanów, niezmienność U/V/hashów po selekcji. Dopasowanie V nie może korzystać z końca trajektorii. Reszty: osobny ciąg one-step, czasowe ACF/Ljung–Box i współzależność tokenów; gaussowskość osobno, bez certyfikatu białego szumu i bez FDT dla AdamW.

## Zapis i odpowiedzialność

Root: PLAN.md i PODSUMOWANIE.md. Worker plan_poprawek: kod, deklaracja WYNIKI.json, małe artefakty, WYNIKI.md, README i figury. Astra: AUDYT_ASTRA.md. Nie zmieniamy dawnych eksperymentów.

Nowy katalog <=150 MB, zachować minimum1 GB wolnego dysku F. Pełne tokeny i pośrednie macierze w RAM; zapisywać małe projekcje, bazy, operatory, metryki, wybrane prognozy, wersje/hash kodu i źródeł. Python F:/Python/python.exe, BLAS4 wątki. Bez instalacji, pobierania, kasowania, treningu ViT, zleceń na Edenie, commitów i modyfikacji wiki. Opisać wyniki dodatnie i ujemne zaraz po biegu. Raportować status pilotażu, nie gwarantować80% ani ogólnej identyfikowalności całego układu.
