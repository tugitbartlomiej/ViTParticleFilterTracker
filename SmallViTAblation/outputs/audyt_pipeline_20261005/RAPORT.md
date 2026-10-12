---
title: "Konfiguracja pipeline'u ViT ukierunkowana na skrócenie treningu"
date: 2026-10-05
status: "analiza zakończona; rekomendacja do walidacji prospektywnej"
poprzednie_audyty: [AUD-20261004-VIT-PRZYSPIESZENIE-PIPELINE]
powiazane_ustalenia: [AU59, AU60, AU61, AU62, AU63]
---

# Co wybrać po dotychczasowych eksperymentach

**Do praktycznego treningu wybieram teraz oryginalny checkpoint ViT, rzeczywiste oznaczone zdjęcia, odnawianą losową pulę z kontrolą powtórzeń, ciągłość AdamW i zatrzymywanie według walidacyjnej jakości detekcji.** Pierwszą badaną poprawką będzie tani dobór trudnych i różnorodnych przykładów. SD, dopasowywanie sztucznego latentu i Langevin nie są obowiązkowymi etapami tej receptury. Langevina zachowuję jako osobny wariant krótkiego planowania; SD jako źródło propozycji dopiero po wykazaniu korzyści ponad wyszukiwanie bez generatora.

To wybór najlepiej uzasadniony obecnymi dowodami, **nie empirycznie wyznaczone globalne optimum**. Nie mamy kompletnego porównania czasu dojścia do tej samej jakości dla wszystkich wariantów. Parametry startowe w towarzyszącym JSON-ie są propozycją testu, a nie nowym wynikiem treningu. Raport nie zmienia trwających eksperymentów.

## 1. Zakres i źródła

Przegląd obejmuje dzisiejsze rodziny interwencji J128, wspólnych regulatorów, CAV/TCAV, inwersji SD, wyszukiwania obrazów, dopasowania wag Lorentza oraz prognoz paczek Gibbsa. Starsze badania odczytano w raportach źródłowych, a szerszą historię powiązano z [audytem z 4 października](../../docs/latent_sd_20261005/audyt-vit-przyspieszenie-pipeline-2026-10-04.md), który zinwentaryzował 59 rodzin. Nie wykonano ponownie wszystkich historycznych treningów ani odczytów wszystkich checkpointów.

Sprawdzono również aktualny kod: residual w [core.py](../../experiments/vit_optymalizacja_latentu128/core.py), błąd zegara i zależność kontrolnego ramienia od prognozy w [petla4.py](../../experiments/maly_vit_petla_sd_rownanie/kod/petla4.py), sposób liczenia sił w [rownanie_original.py](../../experiments/vit_langevin_cztery_paczki_sd/rownanie_original.py), sterowanie nową generacją w [run.py](../../experiments/vit_sd_szum_nowy_checkpoint/run.py). Bieżący kod nie zastępuje historycznego snapshotu; tam, gdzie wynik pochodzi z wcześniejszego audytu, jego zakres pozostaje jawny.

Opis sesji: [zapis Boyle'a](../../../.sessions/2026-10-05-latent-vit-sd-langevin.md). Dane w tym raporcie są migawką z 5 października, po ukończeniu nowego obrazu SD i pełnego przestrzennego CAV Ohma. Nowe planowanie sekwencji Gibbsa nie ma tutaj wyniku końcowego.

## 2. Co rzeczywiście osiągnięto

| Etap | Wynik | Znaczenie dla szybszego treningu |
|---|---|---|
| Mały ViT, historyczna pętla SD 4.3, 10 ziaren | Średnia AP50 po ocenach: 0,4883 wobec 0,4720 dla losowania; 10/10 wygranych. Końcowe AP50 0,566 wobec 0,544; kontrast końcowy p=0,13 | Najmocniejszy dodatni sygnał dla efektywności aktualizacji. Nie wykazano oszczędności czasu ani osobnego wkładu SD |
| Ramię SD bez przesunięcia celu w tej kampanii | 0,4871 wobec 0,4883 z prognozą | Brak potwierdzonego zysku z przesunięcia celu; kontrola nadal używała prognozy do wyboru kotwic |
| RF-DETR, adaptacyjna selekcja, 3 ziarna | Najlepsza walidacyjna AP50:.95: random 0,7131; GLISTER 0,7072; cechy+Koopman/GLE 0,6839 | Selekcja gradientowa ani model dynamiki nie są automatycznie lepsze od losowania; dotyczy innego modelu i tej implementacji |
| RF-DETR, pilot doboru stanu sieci | 1500 wybranych zdjęć: AP 0,67317; random1500 0,65547; random3000 0,67122 | Korzyść jakościowa jednego biegu, ale wybór+trening trwały około 27,12 min wobec 20,38 min dla random3000 |
| Indywidualna korekta latentu, 400 zdjęć | AP50 0,437555 → 0,884748; AP50:.95 0,120881 → 0,770525 | Etykieta każdego obrazu sterowała jego własną korektą. To możliwości interwencji, nie wynik samodzielnie działającego detektora |
| Wspólny regulator J128 | Wybrany adapter: test AP50 0,367857 wobec 0,457589; prosty affine 0,465775, przedział zmiany obejmuje zero | Duży wynik indywidualnej korekty nie przeniósł się na jedną potwierdzoną wspólną regułę |
| Wspólny regulator przestrzenny | AP50 0,425228 wobec 0,457589; AP50:.95 0,121785 wobec 0,114275 | Mieszany wynik różnych metryk, bez potwierdzonej przewagi; koszt kampanii 1510,8 s |
| Wspólny skalar pełnego CAV | AP50 0,466715 wobec 0,457589; CI różnicy [-0,002226; 0,024335] | Niewielka, niepewna zmiana. Nie ma podstaw do włączenia CAV jako głównego regulatora |
| Duże przesunięcia CAV, jedno zdjęcie | 817 wartości, najlepsze AP50:.95 pozostało 0,8; duże przesunięcia szkodziły | Jeden kierunek CAV nie zastępuje indywidualnego dostrojenia wielu współrzędnych |
| Pełny przestrzenny CAV 600×768, 400 par | Separator: BA 75,88%, AUROC 0,8584. Dodatnia pochodna na 7/85 obrazów, ujemna na 78/85; kontrole nieistotne po korekcji dwóch porównań | Pełny kierunek konceptu nie jest wspólnym kierunkiem wzrostu odpowiedzi detektora. Nie potwierdzono poprawy treningu |
| Lorentz: uczenie wag na jednym zdjęciu | Zgodność wartości J128: 64,48% po 1000, 68,33% po 2000, 74,25% po 6402; świeży forward potwierdzony | Ręczną korektę częściowo przeniesiono do wag dla tego obrazu. Nie zbadano poprawy na nowych filmach |
| Nowy obraz SD z checkpointem Lorentza | Zgodność J128 62,83%, cosine 0,9295, MSE640 0,05649; widoczne artefakty | Inwersja liczb poprawiła się, lecz jakość obrazu i użyteczność treningowa nie zostały potwierdzone |
| Cztery znalezione paczki × 100 aktualizacji | Redukcja MSE128: 0,5222%, 0,7818%, 0,0481%, 0,4438%; random 0,3001%. Każda paczka obniżyła IoU i confidence sondy | Najbliższe zdjęcia nie okazały się skuteczną drogą do celu; małe zbliżenie latentu nie oznacza poprawy detekcji |
| Langevin, krok pierwszy | Skill przyrostu pełnego latentu 88,80–98,60% wobec braku zmiany | Dobry opis lokalnego kroku przy policzonej sile; nie jest procentem osiągnięcia sztucznego celu |
| Langevin, stała początkowa siła na 100 kroków | Prognoza nie odtworzyła rzeczywistej trajektorii | Tego wariantu nie używać do wyznaczania czasu dojścia do celu |

Źródła: [SD 4.3](../../experiments/maly_vit_petla_sd_rownanie/WYNIKI.md), [adaptacyjny wybór](../../experiments/dobor_cech_adaptacyjny/WYNIKI_KAMPANII_2026-09-22.md), [pilot i koszty](../../experiments/dobor_klatek_stan_sieci/WYNIKI_PILOT.md), [oracle](../../experiments/vit_latent128_maksymalizacja/WYNIKI.md), [wspólny regulator](../vit_latent128_wspolna_regulacja/RAPORT.md), [przestrzenny](../vit_latent128_regulator_przestrzenny/RAPORT.md), [CAV](../vit_cav_maksymalizacja_detekcji/RAPORT.md), [pojedynczy obraz CAV](../vit_cav_jedno_zdjecie_duze_przesuniecia/RAPORT.md), [pełny przestrzenny CAV](../../experiments/vit_cav_pelny_latent_pary400/WYNIKI_TCAV.md), [Lorentz](../vit_wyuczenie_latentu_jedno_zdjecie_do85/READY_FOR_SD.json), [nowe SD](../vit_sd_szum_nowy_checkpoint/RAPORT.md), [cztery paczki](../vit_langevin_cztery_paczki_sd/REPORT.md).

## 3. Gdzie rozchodzi się obecny łańcuch rozumowania

### Korzystny stan wewnętrzny nie określa automatycznie korzystnej paczki

Dla sondy x mamy z=fθ(x), a sztuczna interwencja wyznacza z*. Wyszukanie obrazu u z fθ(u) bliskim z* odpowiada na pytanie o podobieństwo reprezentacji. Trening zależy jednak od gradientu straty obrazu u i pamięci optymalizatora. W małym kroku:

\[
\Delta z(x;B)\approx J_\theta f_\theta(x)\,\Delta\theta(B;\theta,m,v).
\]

Obraz już łatwy może mieć latent bardzo bliski celowi, ale mały gradient. Podobny obraz może też dać aktualizację szkodzącą sondzie. Potrzebujemy więc miernika skutku **uczenia na obrazie**, a nie wyłącznie podobieństwa jego aktywacji. Wynik czterech paczek jest bezpośrednim ostrzeżeniem przed utożsamieniem tych wielkości.

### W tym modelu „pełne 768” nie oznacza całego stanu potrzebnego głowie

Mierzony tensor 600×768 leży po GELU wewnątrz ostatniej MLP. Kod dalszej części używa również residualu:

\[
h=\operatorname{Norm}(r+\operatorname{MLP}_{tail}(z)),\qquad \hat y=\operatorname{Head}(h).
\]

Sztuczna korekta testowana była przy residualu oryginalnego zdjęcia. Nowy obraz zmienia również r. Nawet identyczne z może zatem dać inne h i inną detekcję. Dla badania funkcjonalnego celu należy kontrolować parę (r,z) albo stan po dodaniu residualu, a jako wynik końcowy mierzyć detekcję. Samo przejście na taki cel nie gwarantuje udanej generacji.

### Dopasowanie checkpointa do celu zmienia warunki eksperymentu

Lorentz pokazał częściową realizowalność celu **w zmodyfikowanym modelu** na jednym znanym obrazie. Nie pokazał, że pierwotny checkpoint sam dojdzie do tego stanu przez standardowy trening na odnalezionych zdjęciach. Zgodność SD przed i po liczono przez różne checkpointy, dla różnych generacji i ustawień. Wzrost z około 0,96% do 62,83% nie jest izolowanym pomiarem skutku jednej zmiany. Stałe ograniczenie osiągalności starego checkpointa nie obowiązuje automatycznie po zmianie wag.

### Koncept jest odczytywalny, ale nie musi być wąskim gardłem zadania

TCAV bada lokalną wrażliwość wyniku, a CAV opisuje kierunek wyznaczony przez separator. Kontrole TCAV również dawały wysokie wyniki, więc swoistość efektu nie została potwierdzona. Historyczny kontrast narzędzie–tło bywał łatwy do odczytania już przy słabym detektorze. To nie dowodzi bezużyteczności J128 z par LaMa; pokazuje, że nie wolno używać samej „siły tożsamości” jako miary ukończenia treningu. Osobno potrzebne są lokalizacja, rozróżnianie narzędzi i kontrola fałszywych trafień. [TCAV](../vit_tcav_tozsamosc_narzedzia/RAPORT.md), [odświeżany odczyt a mAP](../../experiments/maly_vit_tozsamosc_odswiezana_a_map/WYNIKI.md).

## 4. Rekomendowana konfiguracja praktyczna

### R0 — wariant do użycia jako punkt odniesienia

- Oryginalny checkpoint wytrenowany standardowym zadaniem detekcji; dla odtworzenia tych testów k1000 z autentycznym stanem AdamW. Model Lorentza pozostaje diagnostyką jednego zdjęcia.
- Batch 32, dotychczasowa geometria wejścia 480×320. Nie zmieniać jednocześnie rozdzielczości, augmentacji, LR i selekcji, bo zgubi się przyczyna zysku.
- Zachować optimizer, momenty, scheduler i numer aktualizacji przy wznowieniu. Nie zaczynać nowego warmupu przy każdym dołożeniu zdjęć.
- Uczyć na prawdziwych klatkach z prawdziwymi etykietami. Zacząć od puli roboczej 1500 klatek, odnawianej z całego dozwolonego train. To budżet startowy porównania, nie obowiązek przejrzenia całego train w każdej rundzie.
- Losować z kontrolą powtórzeń i pokrycia filmów. Wariant z jawnym równoważeniem filmów porównać także z dotychczasowym losowaniem, bo zmienia rozkład próbkowania.
- Walidacja w stałych odstępach aktualizacji, zapis best i last. Zatrzymywanie po braku poprawy ma być regułą ustaloną wcześniej, nie wyborem najlepszej epoki po obejrzeniu całego biegu.
- SD, CAV, stały sztuczny cel i długi rollout Langevina: wyłączone w R0.

Wybór R0 jest ostrożną decyzją operacyjną: losowanie jest silnym konkurentem w istniejących kampaniach, a dodawane moduły mają niepotwierdzony koszt względem korzyści. Nie twierdzę, że to jakościowo najlepszy wariant po dowolnej liczbie kroków.

### R1 — pierwsza poprawka do sprawdzenia

Co 256 aktualizacji ocenić rotującą pulę maksymalnie 2048 kandydatów treningowych. Użyć bieżącej straty detekcji, błędu lokalizacji i zapisanej informacji o pokryciu danych. Aktualizować część puli roboczej, zachowując eksplorację.

Startowe proporcje paczki: 16 zdjęć losowych, 8 trudnych po odrzuceniu podejrzanych etykiet, 8 z niedostatecznie reprezentowanych warunków. Limit dwóch z tego samego filmu stosować tylko, gdy pula na to pozwala; relaksację zapisywać. Duplikaty i niemal sąsiednie identyczne klatki ograniczać. Te liczby są propozycją, a ich przewaga wymaga testu.

Nie wybierać wyłącznie największej straty: może ona oznaczać błędną anotację, klatkę niemożliwą do rozpoznania albo wielokrotnie powtórzony przypadek. Nie traktować wszystkich innych instrumentów jako fałszywych detekcji, jeżeli etykietowana jest tylko wybrana klasa narzędzia.

Tani indeks wyszukiwania budować na stałym encoderze i istniejącym cache, o zgodnych wagach, preprocessingu i źródłach. Aktualnym ViT przeliczać tylko krótką listę. Przy zmianie checkpointa nie porównywać bezpośrednio surowych latentów zapisanych w innej przestrzeni modelu.

## 5. Jak zachować Langevina z konkretną rolą

**W tym wariancie nie ma CAV.** Wejściami są bieżący stan modelu, stan optymalizatora i kandydacka paczka. Model dynamiki przewiduje ruch sond po kroku uczenia. CAV/TCAV pozostają niezależną analizą reprezentacji.

Najpierw sprawdzić ranking maksymalnie czterech paczek, z horyzontem jednej aktualizacji. Dwie propozycje mogą pochodzić z trudnych/różnorodnych zdjęć, jedna z niedostatecznie pokrytych warunków, jedna z losowania. Sondy powinny obejmować wiele obrazów i filmów, zarówno trudnych, jak i łatwych. Zbiór sterujący jest częścią uczenia doboru; nie wolno nazywać go nietkniętym testem.

Dla paczki B kryterium docelowe:

\[
U(B)=\frac{L_{ctrl}(\theta)-L_{ctrl}(\theta+\widehat{\Delta\theta}_B)}{t_{score}(B)+t_{train}(B)}.
\]

Strata kontrolna ma używać tych samych składników detekcji co model, z odpowiednimi etykietami; rzeczywistą jakość potwierdza AP. Przy różnym koszcie kandydatów mianownik musi być zmierzony. Przy jednakowych kosztach ranking sprowadza się do przewidywanego spadku straty.

Prostszy przybliżony licznik to -∇LctrlᵀΔθB. Dla AdamW ΔθB musi uwzględniać aktualne momenty, LR, clipping i decay. Sam iloczyn surowych gradientów SGD nie jest tym samym. Kod obecnej siły już zawiera gradient oraz wirtualne przejścia przez model: dobry wynik prognozy nie oznacza darmowej oceny.

W gałęzi badawczej można także mierzyć spadek odległości do lokalnego celu latentu, ale jego waga w głównej funkcji wyboru powinna początkowo wynosić zero. Dopiero przewaga nad wyborem według detekcji uzasadni nadanie mu roli sterującej. W przeciwnym razie można bardzo sprawnie optymalizować nieodpowiedni cel.

Po prawdziwej aktualizacji porównać prognozę z odczytem, skorygować model, dopiero potem planować następny krok. Wydłużenie horyzontu do 5 lub 10 wymaga osobnego potwierdzenia. Informacje z przyszłych rzeczywistych aktualizacji nie mogą wejść do wcześniejszej prognozy.

Istniejący nowy test Gibbsa minimalizujący odległość jednej sondy nadal warto dokończyć: rozstrzyga hipotezę użytkownika o osiąganiu sztucznego celu. Nie będzie jednak sam w sobie dowodem przyspieszenia detektora. Obecny kod liczy cztery gradienty kandydatów na wybór i przejścia sondy, więc koszt musi zostać porównany z poświęceniem tego czasu na zwykły trening.

Kolejność paczek wpływa na uczenie. Permutacja elementów tej samej paczki nie zmienia średniej straty w zwykłym deterministycznym batchu; losowe augmentacje i implementacyjne efekty kolejności trzeba kontrolować. Liczba checkpointów to częstość zapisu. Jednostką dynamiki jest aktualizacja optymalizatora.

W bieżącym odczycie nowy program Gibbsa faktycznie wykonuje już wybory i aktualizacje. Wynik końcowy pozostaje otwarty. Raportowane podczas biegu `gain128` i `agreement128` są ułamkami, więc nie można ich przepisać jako procentów bez mnożenia przez 100. Bieżąca polityka minimalizuje odległość jednej sondy; nie przenosić jej automatycznie do głównego treningu bez kontroli jakości detekcji.

## 6. Rola SD po tej analizie

Nie porzucam generatora jako kierunku badań. Odkładam jego użycie w głównej pętli przyspieszania, ponieważ brak dowodu, że daje korzyść ponad tańsze wyszukiwanie z prawdziwej kotwicy.

Najpierw osobna kontrola inwersji: ten sam szum i ustawienia, bez sterowania; naturalny latent prawdziwego zdjęcia; stopniowe przejście do sztucznego celu. Dopiero to rozdzieli jakość bazowego SD, trudność sztucznego celu i wpływ guidance. Ostatni bieg z nowym checkpointem nie miał takiej kontroli, więc przyczyna artefaktów pozostaje hipotezą. Nie można kategorycznie przypisać ich wyłącznie guidance.

Jeśli inwersja naturalnego celu działa, porównać cel w stanie po residualu lub kontrolę residual+MLP, ograniczyć siłę korekt i zachować realizm obrazu. Wytrenowanie adaptera warunkującego SD wymaga wielu par obraz–latent, stałego checkpointa i odłożonych filmów; teraz byłby to dodatkowy koszt bez wykazanej amortyzacji.

Następnie SD może służyć jako rzadko uruchamiany generator różnorodnych zapytań do indeksu. Prawdziwe odnalezione zdjęcia zachowują swoje GT. Nie przenosić ramki z oryginału na zniekształcony obraz syntetyczny bez kontroli geometrii. Wygenerowane artefakty nie są dowodem, że takie tekstury należy wprowadzić do treningu.

Artykuł [Show Me What You Don't Know](https://arxiv.org/html/2603.21782v1) bada próbkowanie obrazów o podobnej reprezentacji i niezmienniczości modelu, z generatorem jako priorem. To uzasadnia diagnostykę reprezentacji. Nie dowodzi, że trening na takich obrazach przyspiesza detektor ani że dowolny sztuczny tensor jest osiągalnym celem. Warto zachować osobne pomiary zgodności cech i naturalności obrazów.

## 7. Koszt i przechowywanie danych

Warunek przyspieszenia przy podobnej jakości:

\[
T_{selection}+T_{generation}+T_{forecast}+T_{train}+T_{eval}+T_{I/O}<T_{baseline}.
\]

Jeżeli narzut selekcji i pozostałych modułów wynosi c razy koszt właściwego treningu, liczba aktualizacji musi spaść poniżej 1/(1+c) liczby bazowej, zakładając jednakowy czas kroku. Przy narzucie 10% potrzeba ponad 9,1% mniej kroków; przy narzucie 100% ponad połowę. Koszt jednorazowy można amortyzować wyłącznie przez faktyczną liczbę porównywalnych zastosowań cache, a nie hipotetyczne przyszłe biegi.

Nie trzeba przechowywać pełnego 600×768 dla wszystkich 310257 obrazów: float32 to około **571,9 GB** bez metadanych. Kompaktowy wektor 768 float16 na obraz zajmuje około **476,6 MB**, ale traci strukturę tokenów i nadaje się wyłącznie do preselekcji. Pełną reprezentację zachować dla sond i krótkiej listy, a dokładny ranking policzyć na nich. Rozmiary są obliczeniem pojemności, nie pomiarem pliku.

Praktyczne zmiany o małym ryzyku:

1. Zegary `perf_counter`, oddzielne nazwy dla czasu i top-k, osobny koszt każdego etapu. Czasy GPU mierzyć zdarzeniami CUDA lub synchronizować granice pomiaru; zapisanie czasu wywołania asynchronicznego nie wystarcza.
2. Dane dekodować/cache'ować raz dla konkretnego preprocessingu. Na Windows dobrać liczbę workerów po pomiarze; więcej workerów nie zawsze pomaga. Indeks i manifesty zamiast kopii całych zbiorów.
3. Mixed precision tam, gdzie zgadza się z dotychczasową recepturą i nie psuje stabilności; diagnozę dokładnych latentów utrzymać w ustalonej precyzji. Nie obiecywać zysku bez krótkiego benchmarku. Kompilacja jest opcjonalna, ze wliczonym czasem kompilacji, szczególnie przy krótkich biegach.
4. Rzadziej wykonywać pełną walidację; tania kontrola służy do monitorowania, pełna do wyboru modelu. Wszystkie ramiona mają identyczny harmonogram ocen i zapisy predykcji.
5. Do uczciwych pomiarów czasu jedna kampania GPU naraz. Współbieżne SD i trening na RTX zniekształcają porównanie; nie zatrzymywać cudzych zadań, tylko zaplanować okno pomiarowe.

Zalecenia techniczne odpowiadają [oficjalnemu przewodnikowi PyTorch](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) i [AMP](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html). Nie przenosimy opublikowanych przyspieszeń innych modeli na ten ViT.

## 8. Porównanie rozstrzygające

| Ramię | Zmieniany element | Pytanie |
|---|---|---|
| R0 | Losowanie, zachowana receptura | Jak szybki jest obecny uczciwy baseline? |
| R1 | Tani dobór trudności i różnorodności | Czy poprawia czas do jakości po doliczeniu selekcji? |
| R2 | R1 + rzeczywista kotwica → retrieval bez SD | Czy samo wyszukiwanie przynosi zysk? |
| R3 | R2 + lokalne planowanie Langevina / wpływu paczki | Czy prognoza pomaga ponad prosty wybór i próbny krok? |
| R4 | R3 + SD jako zapytanie do retrieval | Czy generator daje dodatkowy zysk ponad identyczną selekcję bez SD? |

Nie uruchamiać od razu dużej macierzy pięciu drogich ramion. Najpierw R0/R1, następnie tylko rozszerzenia, których minimalne kontrole przechodzą. Przy badaniu R3 dodać porównanie z dokładnym wirtualnym krokiem tego samego AdamW i z prostą regułą kosztowo porównywalną. R4 musi używać tych samych kotwic co jego kontrola, a ramiona bez prognozy nie mogą korzystać z niej pośrednio.

Protokół startowy:

- Osobno SmallViT od zera i dostrajanie z checkpointa; nie łączyć ich wyników z RF-DETR.
- Podział według pacjenta, jeżeli wiarygodna identyfikacja jest dostępna; w przeciwnym razie minimum film i jawne ograniczenie. Dotychczasowe wielokrotnie oglądane 400 zdjęć to benchmark rozwojowy.
- Dtrain, zbiór sterujący do wyboru danych, walidacja do hiperparametrów i nietknięty końcowy test mają różne role. Dobór danych na etykietach sond jest elementem uczenia, nawet gdy wagi aktualizowane są na innej paczce.
- Trzy sparowane ziarna na pilot, co najmniej pięć na potwierdzenie; przedziały po ziarnach i filmach. To plan startowy, nie gwarancja mocy statystycznej.
- Dwa budżety: równe aktualizacje i równy pełny czas. Zachować ekspozycje, liczbę unikalnych zdjęć i rozkład źródeł w logach.
- Główna miara przyspieszenia: czas do z góry zamrożonego progu AP50:.95; AP50 raportowane dla ciągłości historycznej. Próg ustalić na odrębnej kalibracji baseline, a nie na końcowym teście. Podawać także krzywe AP względem czasu i liczby aktualizacji.
- Proponowana bramka praktyczna: co najmniej 20% mniej czasu do progu, bez pogorszenia końcowego AP50:.95 większego niż 0,005. Dla silnego twierdzenia przedział różnicy czasu i niegorszości jakości musi spełnić ustalone granice. Te progi są decyzją projektową do zamrożenia przed kampanią.
- Jeżeli ramię nie osiągnie progu w budżecie, zapisać „nie osiągnięto” wraz z budżetem. Nie liczyć średniej wyłącznie z udanych biegów i nie zgadywać dalszych epok.

## 9. Retrospekcja audytu

Wykorzystano istniejące identyfikatory; nie utworzono duplikatów AU. Ten raport i konfigurację zapisano zgodnie z instrukcją użytkownika pod SmallViTAblation. Globalny rejestr projektu Grant odczytano, ale nie zmieniano jego plików.

| Ustalenie | Ocena na obecnym materiale | Konsekwencja |
|---|---|---|
| AU59, P1: czas | Otwarte. Bieżący petla4 nadal przypisuje `tk` zarówno czas, jak i top-k; nowa generacja ma własny czas, lecz nie naprawia starej kampanii | Historyczne AP zachowane, twierdzenie o skróceniu czasu nadal niepotwierdzone |
| AU60, P1: ablacje | Otwarte. Kotwice kontrolnego SD nadal zależą od d[j]; nowy test czterech paczek nie izoluje wartości generatora | Potrzebne ramiona bez SD oraz bez prognozy także w doborze kotwic |
| AU61, P1: oracle → naturalny forward | Częściowo rozstrzygnięte eksperymentalnie: Lorentz wyuczył 74,25% dla jednego obrazu; SD 62,83%. Nadal brak generalizacji i udanej naturalnej rekonstrukcji | Nie przedstawiać jako potwierdzonego przyspieszenia; kontrolować residual |
| AU62, P1: latent jako użyteczność | Otwarte, nowe paczki dały bezpośredni kontrprzykład: mały spadek MSE przy gorszej detekcji | Główny cel musi dotyczyć detekcji |
| AU63, P2: prognoza | Nowy test rozdziela przyczynowo prognozę od przyszłych pomiarów; lokalny krok dobry, stała siła na 100 kroków zawiodła | Lepszy protokół pomiaru nie dowodzi dobrej długiej prognozy; sekwencje jeszcze w badaniu |

Brak podstaw do oznaczenia pełnego rozwiązania któregokolwiek z pięciu historycznych problemów. Nowych globalnych ustaleń: 0; rozpatrzonych: 5 (4 P1, 1 P2). Nie wprowadzano napraw kodu ani nie odnawiano historycznych wyników.

Odczytowy lint globalnego rejestru wykazał cztery wcześniejsze błędy poza tą analizą: trzy referencje do brakującej syntezy przy AU21/starym raporcie oraz nieznany stan `ZAMKNIETE` w AU31. Nie zmieniano tych niepowiązanych plików. To nie jest błąd nowej konfiguracji.

## 10. Co wnosi literatura i co pozostaje hipotezą

[GRAD-MATCH](https://proceedings.mlr.press/v139/killamsetty21a.html) uzasadnia wybór podzbiorów przez dopasowanie gradientu do celu treningowego lub walidacyjnego. [GLISTER](https://ojs.aaai.org/index.php/AAAI/article/view/16988) wiąże wybór danych z jakością walidacyjną. To trafniejsze podstawy dla selekcji niż samo podobieństwo aktywacji, ale nie gwarancja sukcesu u nas: lokalna wcześniejsza implementacja GLISTER nie pokonała losowania. Koszt, aproksymacja optymalizatora i utrata różnorodności muszą być kontrolowane.

Najbardziej wartościowa hipoteza badawcza z Twojego pipeline'u brzmi teraz: **czy informacja o lokalnej dynamice reprezentacji poprawia dobór rzeczywistych danych ponad samą stratę i różnorodność, tak aby skrócić czas do tej samej jakości?** To nadal wykorzystuje pomysł sterowania uczeniem. Wymaga sprawdzenia zysku z każdego członu, a nie dojścia do arbitralnego procentu zgodności z jednym sztucznym tensorem.

## 11. Dostarczone pliki i granica wykonania

- Ten raport: analiza i rekomendacja, bez nowych treningów uruchomionych w ramach tego przeglądu.
- [Konfiguracja startowa i ramiona](../../configs/pipeline_przyspieszenie_20261005.json): poprawny JSON specyfikacji eksperymentalnej; nie jest automatycznie wczytywany przez istniejące skrypty.
- Kontynuowane wcześniej zadania pozostają odrębne. Końcowy pełny CAV uwzględniono po odczycie raportu; przyszły końcowy wynik planowania Gibbsa może zmienić ocenę R3.

Ostateczny wybór na teraz: **R0 jako domyślna receptura, R1 jako pierwszy kandydat do poprawy; R3 i R4 dopiero po przejściu porównań kosztu i jakości.** Nie ma obecnie dowodu, że maksymalizacja sztucznego latentu albo zwiększanie CAV jest najlepszą drogą do szybkiego detektora.
