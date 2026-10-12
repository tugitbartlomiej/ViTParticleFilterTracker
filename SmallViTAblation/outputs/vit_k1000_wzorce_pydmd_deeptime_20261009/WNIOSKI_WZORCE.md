---
title: Wspólne wzorce ruchu K1000 po rzeczywistym PyDMD i deeptime
date: 2026-10-09
status: zakonczone_trzy_biegi_i_zweryfikowane_wyniki
model: gpt-6-astra
reasoning_effort: xhigh
poprzednie_audyty:
  - outputs/vit_k1000_analiza_rezimow_astra_20261009/RAPORT.md
  - outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/RAPORT_WYKONANIA.md
scope: prognoza i opis trajektorii optymalizacji oracle; bez testu zastosowania korektora do detekcji
---

# Wspólne wzorce ruchu: wykonanie i wnioski

**Werdykt:** istnieją powtarzalne kierunki i przewidywalna dynamika ruchu między obrazami. Najmocniej potwierdzono to dla tokenów innych niż cel i modeli korzystających z niedawnego ruchu. Nie odkryto jeszcze reguły, która z naturalnego obrazu bez GT poprawia detekcję. Wspólny populacyjny DMDc z normami gradientu/momentów daje wynik mieszany i nie zapewnia wiarygodnej ogólnej przewagi nad persistence. Token celu, zwłaszcza przy porażce i po zmianie fazy, pozostaje najsłabszym punktem.

Wszystkie trzy joby **1825171,1825172,1825187 zakończyły się COMPLETED, exit0:0**. Istniejący collector PID51580 pobrał i zweryfikował trzy foldery. Synteza ponownie przeliczyła ich hashe; nie uruchamiano dodatkowego collectora, ponowienia ani dostrajania modeli. Wynik negatywny suplementu jest prawidłowo zakończonym eksperymentem, nie awarią.

Najważniejsze rozstrzygnięcia:

1. Populacyjny DMD prędkości na h5 zmniejsza RMSE768 względem persistence o około 54 procent, ale względem prostego przedłużenia ostatniego ruchu tylko o 3,61 procent w fit i 5,65 procent w min_norm.
2. Populacyjny DMDc stanu z wejściem startowym daje łącznie tylko 0,78 procent niższy RMSE niż persistence. Iloraz 0,99223, 95% bootstrap po filmach[0,97830;1,01213], nie daje przekonującego ogólnego efektu. W min_norm i dla targetów jest gorzej.
3. PCA32 zachowuje około 78–81 procent energii ruchu innych tokenów, ale tylko 31–33 procent ruchu targetu. Duża wyjaśniona wariancja stanów nie oznacza dobrego opisu naprawy narzędzia.
4. Nieudane targety oscylują: 24,96 procent aktywnych par ruchów co 5 ma cos<0, wobec 0,86 procent dla udanego fit. Równocześnie pozostałe tokeny poruszają się gładko nawet przy porażce.
5. Następny rozstrzygający krok to **mały corrector zależny od z 0 i kontekstu, oceniony przez rzeczywistą głowicę na odłożonych filmach**, z równoległą podłogą bezpośredniej wspólnej delty Hubble. Dokładny projekt i ograniczenia podano w sekcji 9. Ten test nie został tutaj wykonany.

## 1. Co rzeczywiście zmierzono

Nie analizujemy kolejnych klatek filmu jako kolejnych chwil dynamiki. Czas oznacza **kolejne sztuczne aktualizacje aktywacji tego samego tokenu tego samego obrazu** przy zamrożonym K1000. Film jest jednostką podziału train/validation/test. „Cząsteczka” oznacza wektor tokenu w jednej wspólnej przestrzeni 768D, a nie średnią chmury. Dane są mierzone po GELU w MLP ostatniego bloku, przed `fc2:768→192`, residualem zależnym od obrazu, LayerNorm i głowicą. Wybranie tego miejsca nie usuwa zależności wyniku od obrazu i kontekstu.

| Zakres | Dane rzeczywiście użyte | Dopasowanie | Ocena/opis |
|---|---|---|---|
| Panel `1825171` | 64 obrazy × 16 tokenów = 1024 trajektorie, 501 stanów pełnych 768D, g/m1/m2 przed każdym przejściem | PCA16 przemieszczeń i skale wyłącznie train; osobne DMD, DMDc i VAMP dla fit/min_norm | Prognozy 1/5/20 aktualizacji; błędy bezpośrednio w 768D; role i statusy |
| Populacja `1825172` | Wszystkie 1500 × 600 tokenów, 101 rzeczywistych stanów co 5 kroków w stałej analitycznej PCA32 oraz deskryptory policzone w 768D | Próbka 160 obrazów train, do 24 interwałów i 24 pozycji + tokeny wyróżnionych ról; deeptime VAMP/DMD osobno dla faz | Etykiety wszystkich 90 mln interwałów; prognozy następnych 5/20 kroków z wcześniej zaobserwowanego ruchu |
| Suplement `1825187` | Ta sama pełna populacja PCA32; normy gradientu/momentów na początku interwału, LR, czas | Jeden wspólny operator A/B PyDMD DMDc; próbka train | Wszystkie przejścia co 5 i okna 20 kroków; osobne wyniki faz/ról/statusów |

**To nadal nie jest zapis wszystkich 900 000 wektorów 768D w każdym z 501 kroków.** Pełne kierunki co krok są tylko w panelu; dla populacji mamy projekcję kierunków co pięć kroków i wybrane skalary pełnej geometrii. PCA3 wizualizacji nie trafia do żadnego fit tej rodziny.

Źródła: [provenance panelu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/PROVENANCE.json), [provenance populacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/PROVENANCE.json), [kontrakt](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/KONTRAKT_DANYCH.md), [odczyt metadanych producenta z hashami](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/PRODUCER_METADATA_VERIFIED.json).

### Rozdzielenie danych i kontroli

| Zbiór populacyjny | Obrazy | Filmy | Sukces oracle | Wyczerpany budżet |
|---|---:|---:|---:|---:|
| train | 914 | 70 | 780 | 134 |
| validation | 293 | 23 | 248 | 45 |
| test | 293 | 25 | 255 | 38 |

Podział panelu obejmuje 40/12/12 rozłącznych filmów, po jednym obrazie z każdego. Panel train ma 39 sukcesów i tylko jedną porażkę; test ma 9 sukcesów i 3 porażki. Panel jest celowo wzbogacony o target/top1/konkurentów, więc jego średnia nie reprezentuje losowej cząsteczki z 900k. W populacji 38 porażek test pochodzi z 17 filmów. Filmy z sukcesami i porażkami mogą się pokrywać wewnątrz jednego splitu.

PCA32 producenta jest dopasowana na **501 obrazach z 40 filmów train**, 16 przestrzennie stratyfikowanych tokenach i stanach 0/100/500: 24 048 wektorów. Jedna średnia float64, jedna ortonormalna baza, bez dopasowania na validation/test. Udział wariancji stanów próbki train wynosi 92,276959 procent. To nie jest udział energii każdego ruchu ani każdego targetu. Panelowa PCA16 jest inną bazą, dopasowaną do przemieszczeń, i zachowuje 87,1544 procent wariancji własnej próbki train.

Ponowna kontrola po pobraniu sprawdza: hashe wszystkich plików, rozłączność filmów, przynależność obrazów fit PCA/operatorów do train, zgodność manifestu PCA oraz wszystkie 1500 tablic etykiet. 89 381 400 interwałów otrzymało reżim, 618 600 mieszanych interwałów faz otrzymało 255: razem dokładnie 90 000 000. Mieszane to 0,6873 procent. Nie potraktowano ich jak brakujących porażek.

„Test” oznacza tutaj holdout **analizy dynamiki i jej modeli**, nie automatycznie nowy test niezależny od wcześniejszego uczenia samego K1000. Żaden hiperparametr/rank nie został zmieniony po obejrzeniu wyników testowych. Wagi ViT nie były trenowane. [Splity](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/INPUTS_SPLITS.json), [fit operatorów](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/FIT.json), [niezależne podsumowanie i kontrola](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json).

## 2. Rozliczenie wcześniejszych diagnoz

| Wcześniejsze stwierdzenie/problem | Stan po tym wykonaniu | Zakres dowodu |
|---|---|---|
| „Nie było DMD/Koopmana” | **USTALENIE_OBALONE** dla starego biegu; nowe wykonanie rozszerza zakres | Stary job 1824880 wykonał DMD/DMDc ridge NumPy; teraz faktycznie wykonano PyDMD oraz deeptime |
| DMDc stworzyło korektę B | **USTALENIE_OBALONE** | Stare B powstało ze statycznych reżimów; diagnostyka DMDc nie była wejściem eksportu B |
| Brak wszystkich targetów w starym panelu128 | **NAPRAWA_POTWIERDZONA w nowym zakresie** | Nowy panel ma target każdego obrazu; pełna populacja obejmuje wszystkie tokeny |
| Pełny timeline 768D wszystkich900k co krok | **CZESCIOWO_NAPRAWIONE**, nadal brak pełnego zapisu | Populacja PCA32 co 5 + skalary768; panel768 co 1 |
| Algebraiczne jedynki VAMP w smoke v1 | **NAPRAWA_POTWIERDZONA w v2 i produkcji** | Usunięto nakładające się cechy pozycja/przyrost; używany jest przyrost |
| Katastrofalny learnedB smoke128 test-fit | **Wynik historyczny zachowany** | RMSE1,648296 wobec 0,024792 persistence; nowy panel to inny zakres danych, nie rehabilitacja tamtego testu |
| Mieszanie porażek z sukcesami w średniej endpointów A | **POTWIERDZONE, nie zmieniano A/B** | Poprzedni audyt: 217 porażek wnosi 99,564080 procent energii wszystkich końcowych delt |
| Ruch targetu zawsze przeciwny599 pozostałym | **NIEROZSTRZYGAJĄCE jako teza uniwersalna** | Znak i spójność zależą od fazy/roli; poniżej liczby zamiast ogólnej deklaracji |
| 74,6 procent nullspace dla oracle1500 | **USTALENIE_OBALONE dla przeniesienia liczby** | Liczba dotyczyła innego celu; obecna rodzina nie policzyła globalnego rozkładu rowspace/kernel |
| Wspólna poprawa IoU dzięki temporalnym wzorcom | **BRAK_DANYCH / OTWARTE** | Nie wykonano aplikacji nowego correctora ani oceny detekcji |

Stary oracle osiągnął oba progi90 na 1283/1500; A i B na 0/1500. mAP naturalny0,245496238, A0,239411249, B0,244481144, oracle0,861131039. Statyczna średnia końców i przypisanie końcowych reżimów z naturalnych cech nie testowały całego pomysłu użytkownika. Nowe wyniki poniżej sprawdzają temporalną część hipotezy, lecz jeszcze nie przeniesienie do korekty dostępnej bez GT. Historyczne liczby, miejsca podmiany, hashe i linie źródeł: [poprzedni audyt](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_analiza_rezimow_astra_20261009/RAPORT.md). Dokładna diagnoza smoke/OOD oraz zachowany wynik v1: [raport wykonania smoke](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/RAPORT_WYKONANIA.md).

## 3. Wspólne ruchy w całej populacji

Poniższa tabela dotyczy **testu populacyjnego**, wszystkich tokenów w czystych interwałach faz. Norma oznacza przemieszczenie netto przez pięć aktualizacji, nie sumę długości pięciu kroków. Cosinus jest liczony z dwóch kolejnych aktywnych przemieszczeń tego samego tokenu, w pełnych768D. Aktywność: obie normy >1e−4; pary przecinające fazę są wyłączone. To nie jest cosinus między różnymi cząsteczkami.

| Status/faza/rola | Interwały | Średnia norma768 | Energia ruchu zachowana w32D | Cos kolejnych ruchów768 | Odwrócenia, cos<0 |
|---|---:|---:|---:|---:|---:|
| sukces / fit / target | 4 095 | 2,13885 | **33,31%** | 0,78791 | 0,859% |
| sukces / fit / inne | 2 452 905 | 3,05505 | 77,78% | 0,98033 | 0% |
| porażka / fit / target | 3 800 | 1,02115 | **31,52%** | **0,40434** | **24,960%** |
| porażka / fit / inne | 2 276 200 | 1,59978 | 78,70% | 0,99430 | 0% |
| sukces / min_norm / target | 21 193 | 0,21675 | **32,13%** | 0,98478 | 0,057% |
| sukces / min_norm / inne | 12 694 607 | 0,69059 | 80,93% | 0,60932 | 2,239% |

**Potwierdzony wzorzec:** długie, gładkie ruchy tokenów innych niż target występują także w nieudanych optymalizacjach. Trudność targetu nie jest tym samym co brak przewidywalności wszystkich cząsteczek. Nieudany target często oscyluje. Zjawisko występuje także na train/validation: odwrócenia co 5 odpowiednio 26,13/27,56 procent; dla udanych targetów fit 0,53/1,06 procent. To opis korelacji ze statusem, nie dowód, że samo usunięcie oscylacji naprawi detekcję.

**Ważne ograniczenie PCA:** pomimo92,28 procent wariancji stanów train baza przechowuje tylko około jednej trzeciej energii ruchu targetu. Wynik jest podobny na wszystkich splitach. Główna baza analityczna jest wystarczająca do badania wspólnego rzutu, lecz nie do ogłoszenia pełnego rozwiązania ruchu narzędzia. Brakująca energia to energia poza PCA, **nie** automatycznie nullspace głowicy. PCA wybiera wariancję, a rowspace `fc2` wybiera kierunki widoczne downstream; to różne podprzestrzenie.

Cos0,609 innych tokenów w min_norm dotyczy6 754 078 aktywnych par, a nie wszystkich12,69mln interwałów. Miernik nie jest ważony energią i jest wrażliwy na bardzo małe ruchy. Panel co 1 daje dla swoich innych tokenów znacznie większą gładkość; różne próbkowanie tokenów, lag i próg aktywności nie pozwalają utożsamić obu średnich.

Źródła: [statystyki reżimów z geometrią768](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/REGIME_SUMMARY.json), [agregaty i weryfikacja liczności](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json), [statystyki per obraz](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/OCCUPANCY_PER_IMAGE.json).

### Reżimy: wspólne kierunki są, ale etykieta nie wyznacza naprawy

VAMP jest dopasowany do **par kolejnych przyrostów**, potem kmeans8 do jego cech, osobno dla fit/min_norm. Jest to analiza temporalna: embedding zależy od kowariancji opóźnionej. Końcowe przypisanie etykiety opisuje już zaobserwowany interwał; nie jest predykcją właściwej poprawki z samego naturalnego z 0. Osiem klas ustalono przed testem, nie odkryto ośmiu uniwersalnych praw ruchu. Dokumentacja wyjaśnia estymację kowariancji i model Koopmana: [deeptime VAMP](https://deeptime-ml.github.io/latest/api/generated/deeptime.decomposition.VAMP.html).

W populacyjnym fit reżim0 zawiera95,90 procent ruchów udanych targetów i 96,61 procent nieudanych. **Sam numer reżimu praktycznie ich nie rozdziela.** Odpowiednie średnie kierunki train/test w PCA32 mają cos0,99647 i 0,99156. Dowodzi to powtarzalnego rzutu średniego ruchu między filmami, ale wspólny rzut występuje również w porażkach i nie obejmuje większości energii targetu.

Stosunek `||średni przyrost32|| / średnia ||przyrost768||` wynosi 0,451 dla udanych targetów w fit-reżimie0,0,374 dla nieudanych. Jest to dolna granica analogicznej spójności średniego wektora pełnych768D, a nie średni cos wszystkich par. Dla innych tokenów udanych fit wartości w klasach wynoszą około 0,74–0,84: istnieją wspólne duże kierunki ruchu także poza pojedynczym obrazem.

W min_norm reżim0 obejmuje99,41 procent targetów i 95,38 procent pozostałych obserwacji tokenów w interwałach testu. Dla innych tokenów te95,38 procent odpowiada zaledwie **9,02 procent energii**. Rzadkie większe ruchy w pozostałych klasach wnoszą około 90,98 procent energii. Zatem etykieta „dominujący spokojny reżim” nie wyjaśnia sama energetycznie ważnych przejść.

Średnie przyrosty innych tokenów w fit oraz min_norm mają w tej samej PCA32 cos−0,99995; targetów +0,97259. Średnie target/inne mają cos+0,52294 w udanym fit, +0,80508 w nieudanym fit i−0,56750 w min_norm. **Interpretacja:** powrót pozostałych tokenów i podtrzymywanie innego ruchu targetu tworzą różne wzorce fazowe. Są to cosinusy rzutów średnich, nie twierdzenie, że każdy token lub każdy pełny wektor zachowuje się tak samo. Wnioskowanie o części poza PCA wymaga panelu lub dodatkowego zapisu.

Prawdopodobieństwo pozostania w tej samej etykiecie na następnym interwale testu wynosi 98,26 procent dla targetu-fit i 99,47 procent targetu-min_norm; dla innych93,96/98,29 procent. Ta trwałość nie dowodzi metastabilności: momentum Adama, płynny drift i dominacja klasy0 same tworzą długie serie etykiet. Nie wykonano testu przeciw surogatowi AR(1), stabilności liczby klas ani niezależnego dowodu reżimów fizycznych. [Przejścia](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/TRANSITIONS.json), [fit i singular values](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/FIT.json).

### Co potwierdza pełne768 panelu

W panelowym fit-reżimie7 udanych targetów średnie kierunki train/test pełnych768 mają cos **0,95253**, przy 520 interwałach test i 2230 train. Średnia norma kroku wynosi 0,46587 test wobec 0,45889 train, a spójność średniego wektora względem średniej długości0,56737. To bezpośredni dowód wspólnego kierunku w pełnych768 w tym zakresie. Dla porażek analogiczny cos wynosi 0,33393, a średnia skala0,59299 wobec 0,35344 train. Rozkład nie jest identyczny.

Nie wolno przenosić panelowego numeru7 na populacyjny numer7: inne bazy, lag i osobne dopasowania. Tak samo reżim0 w fit nie jest automatycznie tym samym reżimem0 w min_norm.

Panelowe targety porażek mają cos kolejnych kroków0,57648 i 10,89 procent odwróceń co 1; sukcesy-fit 0,89334 i 0 procent. Mniejsza częstość niż w populacji co 5 nie jest sprzecznością: porównywane są inne interwały i inna próbka filmów. Panel nie służy jako dowód dla 900k. [Wektory średnich768](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/REGIME_DIRECTIONS_768D.npz), [miary kierunków](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/REGIME_MOTION.json), [zmiany kierunku](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/DIRECTION_CHANGES.json).

## 4. Prognozy: populacyjny DMD z ostatniego ruchu

Model dostaje zakończony pięciokrokowy przyrost tego samego tokenu, przewiduje następny, a przy h20 sumuje cztery kolejne prognozy. Nie dostaje przyszłych gradientów. Okna mieszczą się w jednej znanej fazie; wybór takich okien to warunkowa ocena, nie rozwiązany predyktor przełączenia fazy. Maksymalnie12 początków na obraz/fazę/h, wszystkie600 tokenów. Ocena dotyczy validation/test; przypisanie reżimów obejmuje całą populację.

RMSE współrzędnych w768D dla h5 na test:

| Faza | Persistence | Ostatni przyrost | Średni przyrost train | Fazowy skurcz prędkości | deeptime DMD prędkości |
|---|---:|---:|---:|---:|---:|
| fit | 0,116630 | 0,055682 | 0,087875 | 0,055122 | **0,053669** |
| min_norm | 0,162191 | 0,079788 | 0,160076 | 0,079446 | **0,075277** |

DMD zmniejsza RMSE względem persistence o 53,98/53,59 procent, ale względem przedłużenia ostatniego przyrostu o **3,61/5,65 procent**. To jest właściwa skala dodatkowego efektu operatora ponad prostą pamięć ruchu. Jest lepszy od ostatniego przyrostu na 25/25 filmach fit i 23/25 min_norm. Opisowy bootstrap całych filmów,2000 replik, daje95-procentowe przedziały ilorazu RMSE DMD/ostatni-przyrost: **[0,96063;0,96729]** oraz **[0,92527;0,96189]**. To nie jest bootstrap milionów pozornie niezależnych obserwacji tokenów i kroków.

Dla h20 błąd32 i granice pełnego błędu768:

| Faza | Metoda | RMSE32 | Dolna–górna RMSE768 |
|---|---|---:|---:|
| fit | persistence | 1,90965 | 0,39870–0,43385 |
| fit | ostatni przyrost | 0,69483 | 0,16471–0,23747 |
| fit | fazowy skurcz | 0,57523 | 0,14422–0,22375 |
| fit | DMD prędkości | **0,36161** | **0,11163–0,20427** |
| min_norm | persistence | 1,44999 | 0,30246–0,33300 |
| min_norm | ostatni przyrost | 2,00991 | 0,41497–0,43773 |
| min_norm | fazowy skurcz | 1,12967 | 0,23885–0,27652 |
| min_norm | DMD prędkości | **0,87636** | **0,18941–0,23514** |

Przedłużanie ostatniego ruchu przez 20 kroków w min_norm jest gorsze od bezruchu; trzeba modelować zmianę skali/kierunku. Fazowy skurcz ma współczynnik0,91469 w fit i 0,83847 w min_norm. Promienie spektralne DMD prędkości wynoszą0,93429/0,86908, a condition designu13,83/16,29. To nie jest ten sam model co DMD bez historii ruchu w suplemencie.

Granice nie muszą rozstrzygać wszystkich par modeli. Jednak każda metoda utrzymuje identyczny residual stanu startowego poza bazą. Jego nieznany błąd **znosi się w sparowanej różnicy MSE**. DMD−skurcz dla h20 daje różnicę pełnego MSE −0,008338 w fit,95% bootstrap[−0,008987;−0,007724], i−0,021173 w min_norm,[−0,030835;−0,011739]. Nie wymaga to zmyślania dokładnego absolutnego RMSE768. [Metryki](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/DYNAMICS_SUMMARY.json), [per obraz](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k/DYNAMICS_PER_IMAGE.json), [bootstrap](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json).

## 5. Prognozy panelu: średnia poprawia się, target pozostaje trudny

Pełny RMSE768 na test, te same okna dla każdej metody. „Stały” to średni przyrost wyuczony na train; „skurcz” to fazowy model bazowy z train. DMDc learnedB na h5/20 otrzymuje prawdziwe przyszłe wejścia oracle.

| Faza | h | Persistence | Stały przyrost | Skurcz | deeptime DMD | PyDMD learnedB | PyDMD knownB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fit |1|0,018802|0,015929|0,018252|0,014138|0,014004|0,010443|
| fit |5|0,088381|0,073897|0,086226|0,065107|0,064375|0,046898|
| fit |20|0,292693|0,244944|0,294146|0,221235|0,217359|0,145897|
| min_norm |1|0,010708|0,010502|0,010278|0,009058|0,009396|0,006324|
| min_norm |5|0,052950|0,051835|0,049812|0,041317|0,042451|0,030665|
| min_norm |20|0,165243|0,160524|0,147805|0,111562|0,112975|0,083049|

DMDc poprawia persistence w średniej, lecz różnica od DMD jest mała i w min_norm ma przeciwny znak: prostszy DMD wygrywa. KnownB to kontrola odtwarzania **analitycznego kroku Adama po clamp**, z B=I po projekcji. Obejmuje g/m1/m2/LR i prawdziwy bieżący stan. Maksymalna różnica od rzeczywistego kroku wynosi 9,54×10⁻⁷ przed redukcją. To prawie znajomość właściwego ruchu, a nie autonomiczny sposób znalezienia poprawki bez GT; przewaga jest oczekiwana.

Rozbicie **samego targetu**, h1:

| Status/faza | Persistence | DMD | learnedB | Norma rzeczywista / przewidziana learnedB | Cos prognozy do ruchu |
|---|---:|---:|---:|---:|---:|
| sukces / fit |0,018905|0,021173|**0,017291**|0,45963 /0,37365|0,54483|
| porażka / fit |**0,014166**|0,016762|0,024366|0,28000 /0,58856|0,15820|
| sukces / min_norm |**0,010041**|0,011897|0,015715|0,06279 /0,35801|0,05664|

**Główna pułapka średniej:** dla targetów łącznie learnedB ma h1-fit 0,019754 wobec 0,017575 persistence, a min_norm0,015715 wobec 0,010041. Ruchy innych tokenów są liczniejsze i przewidywalniejsze. W min_norm targetu model mocno przeszacowuje skalę, około 5,70 raza dla średniej normy. Średnie norm i RMSE mają inne ważenie, więc nie należy z tego ilorazu wnioskować o każdym kroku.

W nowym panelu design learnedB ma rangę67/67 w obu fazach, condition1241,70/151,15, promień spektralny1,00605/1,00015. Nie jest zagwarantowana stabilność nieskończonego rolloutu. Współrzędne kontroli test mieszczą się pod globalnym maksimum train: maksymalna wartość skalowana test 137,41, train 153,54. To poprawa pokrycia wobec starego smoke199,79 przy train 36,66, lecz nie dowód pełnego pokrycia wspólnego rozkładu wejść. Nowy zestaw danych, większy panel, obecność targetów i nowa baza zmieniają warunki; nie przypisuję poprawy samej bibliotece ani nie zamykam starego negatywnego wyniku.

Źródła: [metryki panelu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/SUMMARY_METRICS.json), [role/status/filmy](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/METRICS_BREAKDOWN.json), [pełne wiersze prognoz](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/METRICS_PER_TRAJECTORY.csv), [operator/rank](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/MODEL_DIAGNOSTICS.json), [OOD](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/CONTROL_SCALE_DIAGNOSTICS.json).

### Czas absolutny i zdarzenie sukcesu

Faz nie synchronizowano sztucznie wspólnym numerem kroku. W dodatkowym wyrównaniu do pierwszego sukcesu średnia norma jednego kroku po 16 tokenach obrazu test wynosi 0,29156 w chwili−1,0,37598 w chwili0,0,85633 w+5 i 1,09053 w+9, następnie maleje do 0,70568 w+20. Dla tych punktów jest9 obrazów. To przejściowa dynamika po zmianie celu, zgodna z istnieniem pamięci optymalizatora; nie dowód wyizolowanego wpływu samego momentum. Okna porażek nie mają zdarzenia sukcesu i nie zostały sztucznie do niego wyrównane. [Okna zdarzeń](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/panel64/SUCCESS_EVENT_ALIGNED.json).

Nie wynika z tego, że min_norm pogarsza cel albo że jego optymalna korekta ma rosnącą normę. Wielkość pojedynczego kroku, odległość od z 0, kierunek powrotu i wybrany najlepszy odwiedzony stan `after` to różne wielkości. `After` nie musi być stanem500.

## 6. Populacyjny DMDc ze sterowaniem skalarnym

**Wykonany job 1825187:** realne `PyDMD.DMDc.fit`,122 046 par ze160 obrazów train, jedna wspólna para A/B dla wszystkich obrazów/tokenów/faz. X to stan PCA32 w chwili t, Y stan t+5. U zawiera `||g_t||`, `||m1_przed||`, `||m2_przed||`, LR pierwszej aktualizacji (`telemetry_learning_rate[t+1]`), t/500 i bias. Nie używa średniego/końcowego gradientu interwału, przyszłej delty, target-ID ani statusu sukcesu jako wejścia. Są to **mierzone amplitudy**, nie wektory sterowania. Clamp nie jest osobnym kierunkowym wejściem tego uproszczonego modelu; jest częścią obserwowanej odpowiedzi układu.

Dla h5 oceniono wszystkie90 000 000 przejść populacji. Dla h20 wszystkie97 początków na każdą z 900 000 cząsteczek, czyli87 300 000 okien per metoda. Mieszane fazy są ocenione osobno, nie usunięte. Naturalne top1/konkurenci/target to maski diagnostyczne. Odrębna kontrola splitów i liczności po pobraniu przeszła. [Provenance](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/PROVENANCE.json), [próbka fit](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/TRAIN_SAMPLING.json), [kod wykonanej wersji i hash](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SUPPLEMENT_DMDC_SUBMITTED.json).

### Causal startU: jeden krok operatora oznacza pięć aktualizacji

Pełny RMSE768 test, h5:

| Faza | Liczba przejść | Persistence | deeptime DMD stanu | PyDMD DMDc startU |
|---|---:|---:|---:|---:|
| fit |4 737 000|0,106204|0,092068|**0,086734**|
| min_norm |12 715 800|**0,104018**|0,110147|0,109737|
| przejście fazy |127 200|0,255060|0,252054|**0,241903**|

W fit DMDc poprawia persistence o 18,33 procent; w min_norm pogarsza o 5,50 procent. Łączny RMSE jest0,78 procent mniejszy od persistence, lepiej na 14/25 filmach. Filmowy95% bootstrap ilorazu obejmuje1. Względem DMD stanu dodatkowe U daje małą, bardziej spójną korzyść: iloraz0,98432,[0,97792;0,99287], lepiej na 18/25 filmach. **Zmierzone amplitudy wnoszą informację, lecz ten liniowy wspólny model nie rozwiązuje całej dynamiki.**

Nie wolno porównywać0,086734 z 0,053669 z sekcji 4 jako czystej ablacji „dodać kontrolę”. Tamten model ma ostatni rzeczywisty przyrost, osobne operatory fazowe i inne okna oceny; ten ma absolutny stan i jeden wspólny operator na wszystkie fazy. W każdej tabeli baseline widzi te same okna, ale między tabelami nie ma identycznego eksperymentu.

### Role rozstrzygają interpretację

RMSE768 targetów test, h5:

| Status/faza | Persistence | DMD stanu | DMDc startU |
|---|---:|---:|---:|
| sukces / fit |0,088868|**0,088287**|0,103063|
| porażka / fit |**0,059125**|0,063267|0,071160|
| sukces / min_norm |**0,034856**|0,041598|0,045902|
| sukces / przejście fazy |0,174196|**0,171719**|0,178198|

DMDc pogarsza target względem persistence w każdej pokazanej grupie. Dla pozostałych ról, przy rozłącznym priorytecie target→natural-top1→konkurent→reszta:

| Rola bez targetu | fit: persistence / DMDc | min_norm: persistence / DMDc |
|---|---:|---:|
| naturalny top1 |0,074037 /0,072609|0,025306 /0,033083|
| konkurenci |0,081129 /0,069385|0,053974 /0,060881|
| pozostałe |0,106820 /0,087127|0,105075 /0,110792|

Role są właściwościami tokenów wyznaczonymi przez producenta. „Pozostałe” nie znaczy, że potwierdzono semantyczne czyste tło każdego patcha. Target jest kotwicą celu oracle, a nie maską wszystkich pikseli narzędzia.

Średni cos prognozowanego do prawdziwego przyrostu32 dla całego test-fit wynosi 0,55948; dla min_norm −0,04337, tylko w aktywnych parach prognoza/ruch. Średnia norma32 wynosi 1,52103 wobec prawdziwej2,07564 w fit i 0,61279 wobec 0,54730 w min_norm. Sama zgodność średniej skali w drugiej fazie nie oznacza zgodności kierunku. [Metryki fazowe](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/SUMMARY_METRICS.json), [pełne stratyfikacje](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/METRICS_STRATIFIED.json), [agregaty ról](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json).

### h20: zamrożone U kontra prawdziwe przyszłe U

`frozen_startU` powtarza wszystkie składniki wejścia startowego przez cztery kroki operatora, także LR i czas. `futureU_oracle` podaje właściwe wejścia z każdego przyszłego początku interwału. Nie są to wejścia przewidziane przez model. Nawet startowy gradient zależy od GT oracle, więc „dostępny przed przejściem” nie znaczy „dostępny bez GT przy wdrożeniu”.

| Faza | Metoda | Dolna–górna RMSE768 h20 |
|---|---|---:|
| fit | persistence |0,35935–0,39471|
| fit | DMD stanu |0,30118–0,34260|
| fit | DMDc frozen_startU |**0,27322–0,31830**|
| fit | DMDc futureU oracle |0,28114–0,32513|
| min_norm | persistence |**0,22208–0,24734**|
| min_norm | DMD stanu |0,26371–0,28531|
| min_norm | DMDc frozen_startU |0,25824–0,28026|
| min_norm | DMDc futureU oracle |0,25170–0,27425|
| zmiana fazy | persistence |1,04544–1,13604|
| zmiana fazy | DMD stanu |1,05194–1,14203|
| zmiana fazy | DMDc frozen_startU |1,02122–1,11380|
| zmiana fazy | DMDc futureU oracle |**1,00842–1,10207**|

FutureU nie poprawia każdej grupy: w fit jest gorsze od frozenU. Łącznie poprawia jego RMSE32 o 1,12 procent; iloraz0,98875,[0,98216;0,99374]. To kontrolowany pomiar wartości informacji przyszłej w zamrożonym modelu, nie wynik osiągalny bez tej informacji. W porównaniu do persistence oba warianty mają przedziały sparowanej różnicy pełnego MSE obejmujące0. Zachowanie w min_norm pozostaje słabe także z prawdziwymi przyszłymi amplitudami.

### Identyfikowalność, OOD i stabilność

Design ma rangę38/38, condition4,25075 przy z góry ustalonym progu1e−8. Problem nie przypomina numerycznej katastrofy starego smoke; nie ma podstaw naprawiać go post hoc doborem ranku. Promień spektralny A wynosi 1,0069537, więc nie ma gwarancji długookresowej stabilności. Stan bez momentów kierunkowych, fazy i historii nie jest pełnym stanem Markowa optymalizatora. Norma m2 nie określa dzielenia współrzędna po współrzędnej w Adamie.

Na całym test wartości skalowane g/m1/m2 osiągają62,91/61,54/38,75. Odsetki przekroczenia zakresu próbki train-fit wynoszą odpowiednio 0,001581/0,0000512/0,0004608 procent przejść. Rzadkość tych przekroczeń nie dowodzi poprawnego wspólnego rozkładu wejść, lecz nie wspiera wyjaśnienia całej porażki masowym wyjściem poza jednowymiarowy zakres. Train poza próbką fit również zawiera przekroczenia. LR/czas mieszczą się w zakresach fit. [OOD](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/CONTROL_OOD.json), [rank i operator](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/population900k_dmdc_supplement/MODEL_DIAGNOSTICS.json).

Gradient jest sprzężeniem zwrotnym zależnym od stanu, obrazu i GT. Formalna pełna ranga pozwala zidentyfikować regresję na tej próbce; nie daje automatycznie przyczynowych, niezależnych „sił” A i B. Jeżeli kontrola jest funkcją stanu, rozdzielenie dynamiki swobodnej od działania wejścia wymaga dodatkowych założeń lub zróżnicowanego wymuszenia. Tutaj scalarny U jest poprawnym wejściem predykcyjnym o ograniczonym zakresie. Nie znamy fizycznego B mapującego sześć amplitud na 32 kierunki, więc **nie wymyślono knownB** dla populacji. KnownB ma uzasadnienie dopiero w panelowej kontroli analitycznego kroku. [Pierwotne DMDc: Proctor, Brunton, Kutz](https://arxiv.org/abs/1409.6358), [API PyDMD, known/learned B](https://pydmd.github.io/PyDMD/dmdc.html).

## 7. Jak interpretować błędy768 i ograniczenia temporalne

Niech P ma 32 ortonormalne kolumny, a r będzie częścią stanu poza P. Dla prognozy utrzymującej residual stanu startowego:

\[
\mathrm{MSE}_{768}=\frac{\|\widehat{\Delta z}_{32}-P^T\Delta z\|^2+\|\Delta r\|^2}{768}.
\]

Dla jednego interwału co 5 norma Δr jest zapisana. Pełny błąd jest identyfikowalny do dokładności projekcji/zapisu float32, chociaż nie zapisano kierunku residualnego. Dla20 kroków znamy normy czterech residualnych przyrostów, nie ich wzajemne kąty. Dolna granica wykorzystuje nierówność odwrotną trójkąta i różnicę norm residuów stanów; górna sumę norm przyrostów oraz sumę norm stanów końcowych. Nie są to przedziały ufności. Są deterministycznymi granicami geometrycznymi.

Przy identycznych oknach i identycznie utrzymanym residualu nieznany składnik znosi się w **różnicy MSE metod**, także przy h20. Nie znosi się w ilorazie absolutnych RMSE768. Dlatego bootstrap ilorazu32 jest jawnie opisany jako32D, a różnica MSE768 ma jawne założenie wspólnego residualu. [Implementacja granic populacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/analiza_populacji.py), [granice suplementu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/supplement_population_dmdc/population_dmdc_supplement.py).

Przy zachowaniu tylko 33,31 procent energii ruchu udanego targetu-fit nawet idealna prognoza rzutu32 i zerowy ruch poza nim zostawiają około 66,69 procent jego energii błędu względem persistence. To około 81,7 procent wyjściowego RMSE, a nie dowód niemożności poprawy przez model pracujący w bogatszej przestrzeni. Obecne DMDc nie osiąga nawet tej podłogi.

Co pozostaje nierozstrzygnięte:

- Cała populacja co 5 nie rozstrzyga zmian kierunku każdego z pięciu wewnętrznych kroków. Ruch netto może ukryć oscylację lub pętlę.
- Nie znamy pełnych kierunków residualnych900k, ich sum dla dowolnego h ani globalnej zgodności kierunków768 między filmami. Panel dostarcza bezpośrednich przykładów, nie kompletnego dowodu populacyjnego.
- Nie przeprowadzono nowego rozdziału ruchu na rowspace/kernel `fc2` dla całej populacji, ani detekcyjnej ablacji nullspace. Mały błąd latentu nie jest równoważny małemu błędowi głowicy.
- Nie zidentyfikowano przyczynowo osobnego wpływu g, momentum, LR, clamp i przełączenia celu. Adam wykorzystuje kierunkowe momenty i nieliniowe skalowanie; skalarne normy usuwają część informacji. [Kingma i Ba, Adam](https://arxiv.org/abs/1412.6980).
- Długie utrzymywanie etykiety i wysokie singular values VAMP nie potwierdzają samoistnie metastabilności. VAMP daje temporalne cechy operatora, lecz ich sens zależy od stanu, próbkowania i walidacji. [Wu i Noé, VAMP](https://arxiv.org/abs/1707.04659).
- Nie ma testu naturalnej osiągalności poprawionego latentu przez obraz. Spełnienie dolnego ograniczenia GELU nie oznacza przynależności do rozmaitości naturalnych aktywacji. Można ocenić korektor jako interwencję na sieci bez wymagania obrazu odwrotnego, ale nie należy mieszać tych dwóch pytań.
- Pooled RMSE waży liczbą dostępnych okien/tokenów. Bootstrap filmów respektuje główną zależność próbek, jednak nie usuwa wyboru jednego datasetu, jednej bazy i jednej konfiguracji ani wielokrotnych opisowych porównań. Miliony przejść nie zastępują liczby niezależnych filmów.

## 8. Dlaczego końcowa wspólna delta mogła zawieść mimo tych wzorców

**Potwierdzone:** mieszanie faz i statusów zmienia to, co jest uśredniane. Po sukcesie pozostałe tokeny wracają, zaś porażki pozostają długo w fit i dominują energię endpointów. W nowej analizie widać wspólne kierunki powrotu i osobny ruch targetu. Uśrednienie końców może więc zachować niepożądany duży ruch porażek albo znieść ruchy właściwe dla różnych faz. To wyjaśnia nieadekwatność samej średniej jako testu hipotezy użytkownika, ale nie stanowi zakończonej interwencyjnej ablacji przyczyn starego A/B.

**Potwierdzone:** wspólny numer patcha nie gwarantuje roli targetu, a ta sama klasa ruchu populacyjnego obejmuje sukcesy i porażki. Jeden tensor600×768 narzuca identyczną poprawkę danej pozycji dla wszystkich obrazów. Wykorzystanie wspólnych kierunków wymaga jeszcze rozpoznania, komu, kiedy i z jaką skalą je zastosować. Eksperyment z bezpośrednią wspólną deltą Hubble jest osobnym testem kompromisu zadaniowego; nie zmieniano jego kodu ani nie dopowiadano wyniku.

**Wniosek poparty liczbami:** reguła uwzględniająca aktualny ruch jest lepsza od stałego średniego przyrostu, a jeden operator stanu+amplitudy bez rozdzielenia faz słabo radzi sobie w min_norm. Samo „użyć Koopmana” nie rozwiązuje wyboru stanu obserwacji, kierunku i skali. Porównanie tych dwóch modeli nie izoluje przyczyny w kontrolowanej ablacji, bo równocześnie różnią się wejściami, fazami i oknami.

**Hipotezy do testu, nie ustalone przyczyny:** nieuwzględniony kontekst obrazu/residualu, niedostateczne reprezentowanie kierunków targetu w PCA32 oraz utrata informacji o momentach i clamp mogą ograniczać predykcję. Nie da się przypisać procentu winy każdemu czynnikowi z tych trzech analiz.

## 9. Jak przejść od wzorców do korektora bez GT

Pomysł użytkownika ma sens: nowe dane wykazują powtarzalną strukturę ruchu, której statyczne endpointy A/B nie testowały. **Nie ma jeszcze dowodu, że struktura da poprawę IoU po zastosowaniu od naturalnego z 0.** Operator przewidujący trajektorię rozpoczętą przez oracle nie jest automatycznie generatorem jej właściwego pierwszego kroku. Populacyjny DMD prędkości potrzebuje wcześniejszego ruchu, którego naturalny obraz sam nie dostarcza; populacyjny DMDc potrzebuje GT-zależnego gradientu.

Minimalny rozstrzygający etap, proponowany, niewykonany w tej rodzinie:

1. Zachować obecne odłożone filmy, a końcowy test poprawy detekcji zamknąć na dodatkowych nieużytych dotąd filmach, bo obecny test został już szczegółowo przeanalizowany. Hiperparametry i wybór wariantu rozstrzygać na train/validation.
2. Zbudować mały słownik kierunków na train, z osobnym wagowaniem targetów, pozostałych tokenów i faz. Najpierw porównać bazę ruchu z pełnego768 panelu oraz rowspace `fc2`, nie dobierać po obejrzeniu testu większego PCA. Sprawdzić zachowaną energię targetu na validation. Endpoint teacher wybierać spośród udanych odwiedzonych stanów z małą normą; porażki zachować do diagnozy/OOD, nie włączać bez rozróżnienia jako pożądane delty.
3. Porównać stałą deltę per patch z małą regułą `delta_j = amplituda(z0_j,kontekst) × suma_k w_k(z0_j,kontekst) kierunek_k`. W inference używać naturalnego latentu, pozycji, kontekstu i własnych naturalnych score/bbox. GT-target, sukces oracle, faza przyszła i prawdziwy ruch nie mogą być wejściami bramki. Oddzielny próg „nie ruszaj” chroni przypadki, dla których ruch nie jest wiarygodny. Ważenie kierunku i normy to osobne zadania; wynik target-min_norm pokazuje, dlaczego.
4. Podłogi na identycznych obrazach: naturalny K1000, dawne A/B, średnia udanych delt z kontrolą skali, bezpośrednio optymalizowana wspólna delta Hubble, prosty korektor zależny od z 0/kontekstu bez VAMP i jego wariant z temporalnym słownikiem. Oracle pozostaje granicą nadzorowaną GT. Równy dostęp do informacji przy inference; porównać także koszt i wielkość modeli.
5. Zastosować poprawkę dokładnie w tym samym postGELU, przeprowadzić oryginalny downstream z residualem danego obrazu i zmierzyć AP50/AP75/mAP50:95, IoU oraz confidence poprawnego/top1 trafienia, oba90, FP/recall, pogorszenia per film. Ocenić pełny krok i kontrolowaną skalę tylko według reguły wybranej na validation. Bootstrap po filmach. Osobno sprawdzić normę ruchu, rowspace/kernel i kalibrację score.

Test wspiera hipotezę dopiero wtedy, gdy **wariant temporalny poprawia zadanie na nowych filmach względem prostego korektora i wspólnej delty**, bez GT w wejściu. Jeżeli poprawia tylko MSE trajektorii albo tylko confidence kosztem IoU/FP, hipoteza o użytecznej korekcie pozostaje niepotwierdzona. Jeżeli prosty model naturalnych cech wygrywa, nie należy utrzymywać Koopmana dla samej nazwy.

W tym etapie nie ma potrzeby odtwarzać od razu terabajtowej historii. Można zacząć od istniejących pełnych endpointów1500 i bogatego panelu, kontrolując ich zakres. Nowy trening/korektor lub nowy replay wymaga osobnego zlecenia; tutaj ich nie uruchomiono.

## 10. Brakujące zapisy, koszt i to, co już uzyskano

Pełne stany float32 wszystkich1500×501×600×768 wymagają **1 385 164 800 000 bajtów**, czyli1,385TB dziesiętnego, około 1,260TiB. Gradient i dwa momenty co krok wymagają dodatkowych około 4,147TB dla 500 przejść; z pozycjami około 5,532TB bez kompresji. Nie utworzono takiego zapisu.

Producent wykazał16 770 472 466 bajtów replay oraz 6 295 646 282 panelu, razem23 066 118 748 bajtów kontenerów tensorów. Same pozycje PCA32 co 5 wymagają11 635 200 000 bajtów bez kompresji. Pliki analizy pobrane przez collector mają razem około 25,39MB; nie pobierano tutaj całego replay. [Walidacja producenta i rozmiary](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/PRODUCER_METADATA_VERIFIED.json).

Gdyby potrzebna była dokładna dalsza analiza:

| Cel | Co jeszcze zachować | Co alternatywa traci |
|---|---|---|
| Każda zmiana kierunku wszystkich tokenów co 1 | Streaming pełnych768 norm, iloczynów, cos i fazy dla każdego kroku; nie trzeba przechowywać wszystkich wektorów | Same skalary nie odtwarzają wspólnego kierunku między obrazami |
| Wspólne operatory bliższe pełnej geometrii | Train-only streaming covariances/lagged moments lub szersza baza ruchu z odpowiednim udziałem targetów; projekcje wszystkich tokenów co 1/co 5 | Traci się informację poza bazą; jej energię trzeba kontrolować per rola/faza |
| Dokładny h20 błąd768 bez całej historii | Normy residualnych przemieszczeń dla lag20 albo odpowiednie iloczyny/Gram między przyrostami | Dowolne inne h nadal nie są odtwarzalne bez dodatkowych iloczynów |
| Kierunkowe sterowanie populacją | Projekcje przedkrokowych g/m1/m2, LR, czas, maska clamp; bogaty stan/historyczne przyrosty | Redukcja przed nieliniowym działaniem Adama nie zachowuje całej operacji dzielenia/clamp |
| Test wpływu na detekcję | Pełny poprawiony latent lub delta plus właściwy kontekst/residual i oryginalny downstream | Same PCA/geometryczne miary nie zastępują IoU/mAP |

Pełny panel768 co 1, zwłaszcza dobrany według ról i stanów sukces/porażka, pozostaje najtańszą kontrolą jakości redukcji. Nie należy używać go jako substytutu opisu900k. Streaming moments ograniczają pamięć i dają możliwość fit bez zapisuTB, ale nie pozwalają później odtworzyć każdej ścieżki lub zmienić dowolnie pytania badawczego.

## 11. Dowód wykonania, wersje i odtwarzalność

| Job | Snapshot na Edenie | Status/elapsed SLURM | Pobrane pliki |
|---|---|---|---:|
|1825171|`code_frozen_production/`|COMPLETED0:0,1m34s|19|
|1825172|`code_frozen_production/`|COMPLETED0:0,3m53s|1510|
|1825187|`code_frozen_population_dmdc_supplement/`|COMPLETED0:0,14m30s|10|

Root analiz Eden: `/home2/faculty/bpiotrowski/smallvit_runs/vit_k1000_wzorce_pydmd_deeptime_20261009`. Prywatne środowisko tej rodziny: `.venv`, Python3.11, **PyDMD2025.8.1, deeptime0.4.5**, NumPy2.2.6, SciPy1.15.3, scikit-learn1.6.1. Runtime zapisuje wersje i hashe importowanych implementacji. Samo zainstalowanie biblioteki nie stanowi dowodu jej wykonania; tutaj backendy, modele, logi, provenance i test znanego układu potwierdzają rzeczywisty fit.

Wspólny adapter buduje pary `(obraz,token,t)→(ten sam obraz,ten sam token,t+lag)`. PyDMD dostaje technicznie `[X,Y]` z parametrem lag równym liczbie par, aby jego publiczne API wycięło dokładnie X/Y bez sztucznych mostków. Fizyczny lag pozostaje1 w panelu i 5 w suplemencie. Nie używa się zegara biblioteki jako czasu fizycznego. Test odzyskania znanych A/B sprawdził ten adapter. Deeptime DMD/VAMP otrzymują jawne pary. Nie zastąpiono PyDMD własną regresją NumPy. [Adapter](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/temporal_core.py), [dokumentacja PyDMD](https://pydmd.github.io/PyDMD/dmdc.html).

SHA256 głównych wykonanych źródeł:

```text
analiza_temporalna.py
dfc6dbeb53f6e9176f5a49b417bcf22164fd12dac9be72ecd82a5e066f79f515
analiza_populacji.py
c2a0ac2f9f3427949b080039c455e63492e08cf60e6d942b345112c71bf92577
temporal_core.py
ace40b2635a0973de5c4aa18f0b50f317993281ae0e220266b07dde4e1d7fd2a
population_dmdc_supplement.py
93b388612055ad0a71f28b23ef465ec3b4b7a9bbf00fe7643f43e59095a767c2
imported pydmd/dmdc.py
12d7935cf2192f400f87e8f2dabc6096749463d0851d0ded14434d57ba40dafe
imported deeptime/decomposition/_vamp.py
4872c21c506e97d02f43e9490c810954b942f1f70fb039b4128ccd3c635fcd5b
PCA32 basis semantic hash
8eb0de248d837cbcd81945c94e7210bf27165c3e077f1df1f78831ea3783845b
```

Pierwotny snapshot i marker pozostały bez zmian przy dodawaniu suplementu. Nie wymieniano źródeł już oczekujących jobów. Gate produkcji:13 testów PASS; gate suplementu:7 testów PASS, w tym rzeczywiste API, kontrola wejść startowych/LR, izolacja frozenU od przyszłości, residualny błąd768 i liczności. [Testy produkcji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TESTS_READY_PRODUCTION.txt), [testy suplementu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TESTS_READY_POPULATION_DMDC_SUPPLEMENT.txt), [pierwsze zgłoszenie](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/PRODUCTION_SUBMITTED.json), [osobny suplement](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SUPPLEMENT_DMDC_SUBMITTED.json).

Dowody transferu zawierają SHA256 każdego pliku: [panel](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TRANSFER_panel64.json), [populacja](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TRANSFER_population900k.json), [suplement](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TRANSFER_population900k_dmdc_supplement.json). Końcowy [stan collectora](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/TRANSFER_WYNIKOW.json) ma `complete`, bez pending. Dodatkowa lokalna synteza ponownie zweryfikowała 1539 plików. Jednostką bootstrapu pozostał film, nie pojedynczy token ani przejście. Wszystkie dokładne liczby i post hoc bootstrapy są w [SYNTEZA_DIAGNOSTYKA.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/SYNTEZA_DIAGNOSTYKA.json).

Odtworzenie postprocessingu, bez fit i bez nowych jobów, z katalogu SmallViT:

```powershell
python -B experiments/vit_k1000_wzorce_pydmd_deeptime_20261009/synteza_wynikow.py outputs/vit_k1000_wzorce_pydmd_deeptime_20261009
```

Zastosowano zasady dowodowe [audit-project-correctness](F:/Studia/MCHTR/Grant/Projekt/.agents/skills/audit-project-correctness/SKILL.md) i izolację rodziny [eksperyment](F:/Studia/MCHTR/Grant/Projekt/.agents/skills/eksperyment/SKILL.md). Zgodnie z jawnym ownership nie tworzono rejestrów AU ani plików ViT w Grant. Wszystkie nowe zapisy tego etapu należą do własnej rodziny. Model wykonawcy: **gpt-6-astra, reasoning effort xhigh**.

Końcowy odczyt SLURM i porównanie **18 plików obu frozen snapshots** z hashami zgłoszenia: [FINAL_EXECUTION_EVIDENCE.json](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/vit_k1000_wzorce_pydmd_deeptime_20261009/FINAL_EXECUTION_EVIDENCE.json). Wszystkie hashe zgodne. Maksymalny RSS batch wynosił odpowiednio 1 449 732 / 567 644 / 459 808 KiB; każdy konsument miał 4 CPU. Nie było naprawy ani rerunu.
