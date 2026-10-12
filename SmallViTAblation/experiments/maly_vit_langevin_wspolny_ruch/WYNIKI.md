# Wspólny ruch cząsteczek, kontekstu i kierunki poza tożsamością

Zakończono 2026-10-02T01:35:48.465909+02:00. Analiza eksploracyjna tej samej historii 1001 stanów (0..1000), czyli 1000 aktualizacji; test czasowy był oglądany wcześniej. Bez treningu ViT, pobierania lub refitu na drugiej sondzie.

Walidacja wybrała **localPLS8, L=2, ridge=1**. Na H1 skill względem braku zmiany to **49.68%**, wobec 60.24% starego GLE; na H10 **14.88%**, na H50 **0.18%**. To zamrożony wybór walidacyjny, nie najlepsza rodzina wybrana na teście. Samo sprzężenie i dodatkowe kierunki mają osobne ramiona poniżej.

## Dane i zamrożone podprzestrzenie

Główna sonda 2: `cataract_lmm/tracking/TR_0057_S1_P03/00001431`, źródłowa rola `fit`, split `train`; 31 cząsteczek narzędzia. Transfer 8: `cataract_lmm/tracking/TR_0134_S1_P03/00000341`, rola `holdout`, split `train`; 7 cząsteczek. Transfer ma własne maski/coverage i anchor 500, lecz dokładnie te same U,V oraz operatory i skale. Nie jest nowym biegiem ani dowodem generalizacji populacyjnej.

U500 (768×8) pochodzi wyłącznie z wcześniejszych 240 par train na k500. Cel każdej cząsteczki to pełny q=Uᵀz w 8D. Dodatkowe r=Vᵀz służą tylko jako pomocnicze współrzędne; nie zastępują celu i nie oznaczają semantycznej tożsamości. V ma 4 albo 8 kierunków poza U. Wszystkie q/r oraz średnie tła w przyszłości są prognozowane autonomicznie.

PCA: ważona covariance wycentrowanych residualnych przyrostów (I−UUᵀ)Δz, targets 501..700. PLS/SVD: crosscov residualnego Δz_t (t 500..699) z Δq_(t+1), zakończone targets 501..700. Oba centrowania i wagi coverage×1/200 tylko fit głównej sondy. PLS jest tu liniowym SVD crosscov, bez autoenkodera. Rangi 4/8 to prefiksy bazy 8. V ponownie ortogonalizowano względem U i metodą QR; obie bazy zamrożono przed walidacją i nie refitowano przy operatorach do 799.

Maksymalne |UᵀV|: PCA 5.98e-17, PLS 8.18e-17. Surowe 768D wszystkich tokenów narzędzia i pełne średnie tła odczytano przez mmap, wyłącznie do RAM. Dysk przechowuje małe projekcje, nie kopię 22 GB. Zgodność nowych q i dokładnych starych prognoz ma maks. błąd 0.

## Równanie i identyfikacja

```text
s_i(t)=[Uᵀz_i(t),Vᵀz_i(t)] − [Uᵀz_i(500),Vᵀz_i(500)]
g_fg(t)=sum_i w_i s_i(t)
g_bg(t)=[U,V]ᵀ mean_background z(t) − jego wartość w 500
Δs_i(t+dt)=b+A s_i(t)+C_fg g_fg(t)+C_bg g_bg(t)
              +sum_l [K_l Δs_i(t−l dt)+F_l Δg_fg(t−l dt)+G_l Δg_bg(t−l dt)]
              +epsilon_i(t+dt)
Δg_bg(t+dt)=d+E_bg g_bg(t)+E_fg g_fg(t)
              +sum_l [M_l Δg_bg(t−l dt)+N_l Δg_fg(t−l dt)]+eta(t+dt)
```

Wariant lokalny usuwa C/F/G i operator tła. CoupledU8 ma V puste; pozostałe ramiona dodają PCA/PLS. g_fg ZAWSZE wynika z aktualnych prognoz cząsteczek, nie ma osobnego fitu średniej. Aktualizacje foreground/bg są synchroniczne: oba przyrosty z tego samego snapshotu, dopiero potem podmiana stanów i pamięci. Wszystkie macierze foreground są wspólne dla cząsteczek. To niskorzędowe sprzężenie ze wspólnym ruchem, nie identyfikacja indywidualnych sił między tokenami.

Dokładna dekompozycja deterministycznej części tego modelu: Δg_fg=b+(A+C_fg)g_fg+C_bg g_bg+średnia pamięci; Δ(s_i−g_fg)=A(s_i−g_fg)+pamięć własnych odchyleń. Wspólny człon kontekstu znosi się w różnicach między cząsteczkami. Model nie różnicuje reakcji tokenów na ten sam kontekst; A/C_fg nie są osobno udowodnionymi mechanizmami fizycznymi.

Foreground fit: każda zakończona aktualizacja ma tę samą wagę, wewnątrz niej coverage-weighted tokeny. Background fit: jedna średnia na aktualizację, równe wagi czasu. Wszystkie poziomy centrowane tokenowym anchor 500. Każda cecha/laga ma osobny historyczny SD. Ridge wspólnie estymuje dryf, sprzężenie i pamięć; gamma=lambda×trace(Gram)/liczba_cech, intercept bez kary. To inne skalowanie i siatka niż dawny GLE; dlatego nowy localU8 jest obowiązkową kontrolą, a oldGLE pozostaje dokładną historyczną bazą.

W zapisie wierszowym F@B, B w pliku jest już fizyczne: B=diag(1/s_x) B_std diag(s_y), b=mean_y−mean_x B. Kolumnowe A/C/K są transponowanymi blokami B zgodnie z layout_fg; operator tła analogicznie layout_bg. Poziomy i przyrosty mają jednostki projekcji aktywacji; B jest bezwymiarowe dla kroku dt, b/reszta są przyrostami na ten krok. W forecast odwrócono wszystkie skale. epsilon_q w metadanych jest progiem znaku, nie realizacją reszty równania.

Nie zapisano faktycznych minibatch-ID/augmentacji/gradientów każdej aktualizacji, więc brak jawnego historycznego sterowania. Nie podano przyszłych paczek, obrazów, stanów, r, średnich ani reszt. Źródłowy eval obejmuje wagi i biegnące statystyki BatchNorm. Nie zakładamy Brownowskiego szumu, FDT dla AdamW ani dokładnego Mori–Zwanziga/globalnej liniowości. Dodatniego wyniku nie utożsamiamy z semantyczną swoistością.

## Wybór przed testem

10 nowych rodzin, ridge .001/.01/.1/1/10, L0/2/4/8 dla dt1 oraz L0/1/2/4/8 dla dt10. Każda rodzina wybiera swoją konfigurację wyłącznie na walidacji; również dokładny oldGLE może być głównym wyborem. Cel wyboru to średni skill fizycznego 8D q, równoważnie po H1/10/50 (dt1) albo H10/50 (dt10). Osobny best_short maksymalizuje H1/H10. Bez wyboru rodziny na teście.

dt1: fit targets 501..700; validation origins 700..749, cele<=799; operator refit do 799, V bez refitu. Test origins 800..950 dla wszystkich H1/2/5/10/20/50, 151 originów. dt10: targets 510..700, origins 700..740 co 10, refit do 790; test800..950 co 10, 16 originów. Porównania dt1/dt10 używają zapisanej sekcji common_co10, tych samych 16 originów. Nie dodawano H100 ani nowych siatek po teście.

Prefix oraz zatrucie przyszłego q/r/bg przez NaN dają identyczny rollout przy origin 800/950 dla każdej nowej rodziny i obu sond; globalne średnie powstają wewnątrz rollout z prefixu. U/V są już zamrożone, a projekcja/maskowanie w przygotuj.py jest pointwise w czasie. SHA modeli i danych są sprawdzane przed testem. Pełne wyniki wyboru, również duże/niestabilne wartości, pozostają w WYBOR_DT*.json.

## Sonda główna: zamrożone rodziny

| Rodzina | L/ridge dt1 | H1 skill | H10 skill | H50 skill | H1 znak wszystkich |
|---|---:|---:|---:|---:|---:|
|oldGLE|stary zamrożony|60.24%|9.11%|-5.03%|78.61%|
|localU8|4/0.01|60.24%|9.11%|-5.03%|78.61%|
|coupledU8|2/1|49.81%|19.08%|10.58%|75.87%|
|localPCA4|2/0.01|59.95%|10.63%|-3.47%|77.87%|
|localPCA8|2/0.01|59.87%|7.39%|-3.55%|77.89%|
|localPLS4|2/1|48.04%|14.54%|-0.28%|75.81%|
|localPLS8|2/1|49.68%|14.88%|0.18%|75.87%|
|coupledPCA4|2/1|50.91%|16.17%|8.97%|74.87%|
|coupledPCA8|2/1|51.51%|14.29%|12.50%|75.32%|
|coupledPLS4|4/1|51.81%|18.92%|14.57%|76.06%|
|coupledPLS8|2/1|52.60%|14.80%|14.19%|75.18%|
|best_short|2/0.01|59.95%|10.63%|-3.47%|77.87%|

## Wybrany model: błąd, kierunek i kontrast do starego GLE

| H | MSE | Skill vs brak zmiany | Znak wszystkich | Znak po odsiewie celu | Cosine zmiany | Δskill vs oldGLE, CI95 [pp] |
|---:|---:|---:|---:|---:|---:|---|
|1|0.003746|49.68%|75.87%|78.49%|0.6865|-10.56 pp; [-12.61, -8.98] pp|
|2|0.016855|34.75%|70.71%|71.89%|0.5642|-9.53 pp; [-11.68, -7.20] pp|
|5|0.082001|14.74%|62.72%|63.06%|0.3514|2.40 pp; [-2.21, 6.70] pp|
|10|0.150215|14.88%|63.87%|64.17%|0.3526|5.77 pp; [2.01, 8.73] pp|
|20|0.222476|16.51%|65.44%|65.74%|0.3970|7.11 pp; [1.22, 11.84] pp|
|50|0.289143|0.18%|65.41%|65.67%|0.3751|5.21 pp; [1.21, 9.55] pp|

Średni testowy skill równych H1/10/50: nowy 21.58%, stary 21.44%, różnica 0.142 punktu procentowego. Walidacyjna przewaga tego kryterium praktycznie się nie przeniosła. Na H1 MSE jest większy o 26.55% od starego, z opisowym CI pogorszenia [21.01, 34.81]%.

Wszystkie metody ocenione według fizycznego q, z identycznymi wagi coverage, originami, celem i progami dawnego fit do 799. Znak to poprawność składowych, nie całych wektorów; cosine dotyczy wektora zmiany. Persistence ma zerowy wektor zmiany i nieokreśloną cosine. Odsiew abs(rzeczywistej przyszłej zmiany)>próg jest warunkową metryką dopiero po poznaniu celu, nie bramką pewności online.

## Transfer na sondę 8 bez refitu

| Rodzina | H1 skill | H10 skill | H50 skill | H1 znak |
|---|---:|---:|---:|---:|
|oldGLE|57.47%|-29.72%|-128.19%|76.32%|
|localU8|57.47%|-29.73%|-128.20%|76.32%|
|coupledU8|39.72%|-21.49%|-75.19%|71.24%|
|localPCA4|51.76%|-44.86%|-118.98%|75.18%|
|localPCA8|41.12%|-45.79%|-107.45%|72.62%|
|localPLS4|32.85%|-3.42%|-44.85%|71.63%|
|localPLS8|30.52%|-5.13%|-32.45%|70.93%|
|coupledPCA4|34.37%|-31.84%|-62.17%|70.51%|
|coupledPCA8|30.04%|-35.27%|-57.81%|68.63%|
|coupledPLS4|34.63%|-0.87%|-66.63%|71.35%|
|coupledPLS8|28.38%|-39.69%|-58.59%|68.29%|
|best_short|51.76%|-44.86%|-118.98%|75.18%|

## Skala co 10: wspólne originy

| H | dt1 selected, co 10 | oldGLE dt1, co 10 | dt10 selected | oldGLE dt10 |
|---:|---:|---:|---:|---:|
|10|18.73%|12.27%|15.58%|16.12%|
|20|13.67%|5.90%|21.27%|20.12%|
|50|0.44%|-4.25%|21.07%|13.28%|

dt10 walidacja wybrała coupledPLS4, L=0, ridge=10. L0 oznacza brak jawnych lagów przyrostów w rozszerzonym stanie. Eliminacja pomocniczych r/bg może nadal dawać efekty pamięci w samym q. Wszystkie rodziny dt10, transfer i kontrasty są w METRYKI_DT10.json.

## Ranga, stabilność i ograniczenia identyfikacji

| dt / model / operator | Targets refit | Cecha count | Gram rank | Condition z ridge | Błąd raw/skal |
|---|---:|---:|---:|---:|---:|
|1 / localPLS8 / fg|299|48|48|7.23|3.62e-16|
|1 / coupledU8 / fg|299|72|72|15.4|2.94e-16|
|1 / coupledU8 / bg|299|48|48|11.1|1.71e-16|
|10 / coupledPLS4 / fg|29|36|36|1.87|3.47e-16|
|10 / coupledPLS4 / bg|29|24|24|1.65|1.6e-16|
|10 / coupledU8 / fg|29|24|24|1.51|2.36e-16|
|10 / coupledU8 / bg|29|16|16|1.4|1.18e-16|

localPLS8, dt1: spectral radius odchyleń = 0.999264; średniaFG/bg = None.

coupledU8, dt1: spectral radius odchyleń = 0.997740; średniaFG/bg = 0.9974689653576001.

coupledPLS4, dt10: spectral radius odchyleń = 0.998324; średniaFG/bg = 0.9989083766765029.

coupledU8, dt10: spectral radius odchyleń = 0.993411; średniaFG/bg = 0.9957013972407944.

Ablation: przy dt1 coupledU8 poprawia H50 względem nowego localU8 (10.58% vs −5.03%), kosztem H1 (49.81% vs 60.24%). LocalPLS8 vs coupledPLS8 na H50: 0.18% vs 14.19%; oba są skonfigurowane na walidacji, ale nie wybieramy nowego zwycięzcy po tych wynikach. Przy dt10 przewaga coupledU8 na sondzie 2 nie przenosi się na sondę 8: H50 −42.79% vs −0.29% localU8.

Operator tła ma tylko 200/20 targetów pierwotnego fit i 299/29 po reficie, niezależnie od liczby tokenów. Przy dużym D/L liczba cech przekracza liczbę czasowych obserwacji; ridge daje predyktor, nie jednoznaczną identyfikację dużej macierzy. Powtarzane globalne cechy w foreground nie tworzą nowych niezależnych próbek. Same parametry nie dowodzą obserwowalności/sterowalności rzeczywistego procesu.

Wyłącznie fit 501..700: udział energii wspólnego przyrostu meanfg = 82.10%, względnej = 17.90%; tożsamość dekompozycji ma relative error 1.63e-15. Średni cosine przyrostów meanfg/bg = 0.7385. To podział energii, nie trafność prognozy ani semantyczna odpowiedzialność.

CI to sparowany circular moving block bootstrap 300, bloki co najmniej 50 aktualizacji, wspólne originy/tokens/models. Przy 151/16 originach około 3 efektywne bloki: niepewność opisowa, jeden bieg i dwa obrazy, nie tokeny jako replikacje. Powtórnie analizowany test pozostaje eksploracyjny. Kolorowość reszty i gaussowskość są odrębne; wcześniejszy audyt szumu nie jest dowodem białości ani odzyskania ukrytego sygnału przez nowe sprzężenie.

## Nowe reszty wybranych modeli

Reszty one-step obliczono po dopasowaniu z ciągłej historii target 801..1000 (dt1, 200 próbek) oraz 810..1000 (dt10, 20 próbek). Wejście każdej reszty ma rzeczywistą historię do originu; są to diagnostyki niedomknięcia, odrębne od autonomicznych błędów H10/H50 i niewykorzystane do strojenia.

| dt / sonda / model | Szeregi | Ljung–Box raw p<.05 | Jarque–Bera raw p<.05 | Udział wspólnej energii reszty |
|---|---:|---:|---:|---:|
|1 / probe2_localPLS8|256|256|46|71.72%|
|1 / probe2_oldGLE|256|134|52|72.83%|
|1 / probe8_localPLS8|64|64|22|73.33%|
|1 / probe8_oldGLE|64|50|19|73.58%|
|10 / probe2_coupledPLS4|256|27|8|58.10%|
|10 / probe2_oldGLE|256|18|9|59.99%|
|10 / probe8_coupledPLS4|64|12|2|66.89%|
|10 / probe8_oldGLE|64|7|1|68.52%|

ACF/Ljung–Box są liczone osobno w czasie dla każdej cząsteczki/współrzędnej i średniej, bez spłaszczania tokenów. Normalność badana oddzielnie. Raw p bez korekty wielokrotnych porównań i liczby dopasowanych parametrów; dt10 ma bardzo mało próbek. Nowy dt1 zachowuje silne korelacje reszty (256/256 odrzuceń Ljung–Box), więc nie wykazano wybielenia szumu ani wydobycia jego całego przewidywalnego składnika. Pełne ACF, korelacje między tokenami i wariancje połów zapisano w SZUM_DT*.json. Brak odrzucenia normalności nie dowodzi gaussowskości ani białości.

![Porównanie rodzin](POROWNANIE_HORYZONTOW.png)

![Transfer bez refitu](TRANSFER_SONDA8.png)

![Ruch średnich](RUCH_FG_BG.png)

![Prognozy cząsteczek](PROGNOZY_INDYWIDUALNE.png)

## Artefakty i odtwarzanie

F:/Python/python.exe przygotuj.py; analiza.py --stage validation; audyt Astry; analiza.py --stage test; reszty.py; raport.py. Przygotuj wymaga PLAN.md. Nie uruchamia żadnego forward/treningu ViT. DANE_PROJEKCJE.npz zawiera U,V i małe projekcje obu sond; MACIERZE_DT*.npz oraz MODELE_DT*.json to fizyczne operatory, skale i jawny layout. WYBOR_DT*.json zachowuje całą siatkę; METRYKI_DT*.json wszystkie rodziny, per-token/per-coordinate błędy, transfer, common_co10 i kontrasty z CI. PROGNOZY_DT*.npz zapisuje tylko wybrane prognozy i bazę. AUDYT_PREFIX_DT*, ZGODNOSC_STARY_GLE.json, RUCH_WSPOLNY_FIT.json i AUDYT_ASTRA.md dokumentują weryfikację. PLAN/PODSUMOWANIE należą do roota.
