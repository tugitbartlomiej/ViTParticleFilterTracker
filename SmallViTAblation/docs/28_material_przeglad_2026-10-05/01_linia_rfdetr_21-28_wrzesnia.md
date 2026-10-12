# Raport: linia Koopman–Langevin na RF-DETR Nano i Large (21–28.09.2026)

**Konwencje.** `R` = `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation`. `DOC24` = `R\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md`, wiersze 408–2383. Foldery leżą w `R\experiments\`.

- Pola wpisów: P pytanie, Z zegar, O obiekt, M metoda, K kontrole, L liczby (plik), W werdykt, S status.
- BZ = „brak zmian", OP = „ostatni przyrost".
- „Wal." = 25 operacji, 12 043 klatki; „test" = 25 operacji testowych.
- Liczby przepisałem z plików .md i kilku małych .json. Surowych raportów JSON, na które te pliki się powołują, w większości nie otwierałem.

## A. Eksperymenty

**0. Poza folderami: Etap 1 na Nano** (kod `R\pipeline\koopman_langevin`, DOC24 §1.2–1.22)
- P: czy Koopman i Langevin pokazują budowanie stanu narzędzia w latencie i czy pomagają dobierać klatki.
- Z: czas wideo przy stałych wagach (16 klipów, 4800 klatek), śledzony po wagach (oś budżetu, oś epok). Operator translacyjny ma krok „przesunięcie narzędzia w kadrze" (inne).
- O: `query`, `box_pool`, `global_pool` (po 256 liczb); tokeny 24×24×384.
- M: EDMD, rozszerzone równanie Langevina (regresje grzbietowe, historia 8 kroków), odporny filtr Kalmana.
- K: BZ, przetasowane kroki, PCA, losowe rzuty, ramiona B0–B4, placebo K3.
- L (DOC24 §1.3, §1.9): R2 położenia z `box_pool` −0,10 (COCO) → 0,86 / 0,83 / 0,83; wolne mody Koopmana wobec PCA 0,84 / 0,04. Macierz 27 treningów (wal., najlepszy punkt): B2 0,6866 ± 0,0042, K1 0,6844 ± 0,0020, K4 0,6885 ± 0,0019, B0 0,6672 ± 0,0030. Kontrasty: K1−B3 +0,0027, K1−K3 +0,0006, K4−B3 +0,0068 (próg +0,004).
- W: sygnał ma „wartość diagnostyczną, nie selekcyjną".
- S: zmierzone (pilot).

**1. krystalizacja_latentu** (22–23.09)
- P: jak latent zmienia się w uczeniu, czy się krystalizuje, czy wskaże klatki poprawiające model. 23.09 dodatkowo: zajętość „przestrzeni tożsamości", wektor narzędzia, ocena na teście.
- Z: operator kroku klatki (czas wideo) liczony na każdym punkcie kontrolnym i śledzony w czasie uczenia; sonda po każdym kroku uczenia (337 kroków).
- O: bloki `query`, `box_pool`, `global_pool`; redukcja do 32, 8 modów, 4800 klatek, 4784 pary, 11 punktów (COCO + epoki 2…29). Tokeny 24×24 (profil 32) z `ut_b0500`, pula 9683 klatki.
- M: EDMD; „trwałość" = średnia Re ośmiu najwolniejszych λ; participation ratio; CKA; LDA na tokenach.
- K: 36 ustawień; PCA i kierunki losowe; 250 klatek losowych i równomiernych; pewność detektora; zwykła zmiana klatki.
- L (`README.md`): participation ratio `query` 8,6 → 16,4. Szczyt trwałości w epoce 5 w 32 z 36 ustawień. Dołek straty wal. w epokach 6, 6, 7. Sygnał kosztuje 0,004 zamiast 0,030 mAP. 250 klatek (wal.): kwadraty 0,6296 ± 0,0011, losowo 0,6319 ± 0,0023, równomiernie 0,6143 ± 0,0096.
- L (DOC24 §2.13, 2.16, 2.18): AUC wektora 0,976 z etykiet i 0,975 bez. Test: mAP 0,681 (najlepsza epoka) i 0,654 (ostatnia). AUC chybień na teście: zaskoczenie Koopmana 0,585, wektor 0,761, pewność 0,882.
- W: latent ustawia się po pierwszej epoce i dryfuje; szczyt trwałości wypada przy szczycie jakości; dobór po siatce tożsamości remisuje z losowym.
- S: zmierzone.

**2. koopman_dynamika_uczenia**
- P: czy ruch latentu klatek uczących w pierwszych 2 epokach przewidzi, które klatki sieć opanowała.
- Z: czas uczenia.
- O: 250 klatek; latent zapytania i wektor tożsamości po każdym kroku.
- M: wspólny Koopman z(k+1) = K z(k) z grzbietem; Langevin dzielący resztę na pamięć i szum paczek.
- K: mapa zbioru danych, liczba zapomnień, losowe.
- L (`PLAN.md`): tylko próg R² > 0,8.
- S: tylko plan („Nic nie uruchomiono").

**3. koopman_gesta_historia** (24–25.09, sesja równoległa)
- P: „czy dłuższa, gęstsza historia wolniejszego dostrajania pozwoli przewidywać latent?"
- Z: czas uczenia.
- O: bieg 1810662 (Nano od COCO, 1500 klatek, 8 epok × 100 aktualizacji, zapis co 5: 160 punktów, 161 stanów). Pełny tensor 576×384. 72 sondy (48 fit, 24 holdout).
- M: macierz afiniczna na położeniu; K = I + B; macierz na każde przejście; stan [z, v] z grzbietem 0,1.
- K: BZ, trend liniowy, OP.
- L (DOC24 §3.2): 100 aktualizacji: Koopman 0,0609, BZ 0,0474, trend 0,0476.
- L (`AUDYT_LITERATURY_6_KOOPMAN.md`): stan z pamięcią 0,0045273 wobec OP 0,0042506 (5 aktualizacji) i 0,2873054 wobec 0,1382725 (100 aktualizacji).
- W: żaden wariant nie pokonał prostych kontroli.
- S: zmierzone (diagnostycznie). Paleta 256 teł, test wpływu kierunków, EDMD i autoenkoder niewykonane.

**4. koopman_jeden_wektor** (25.09)
- P: czy jeden pełny wektor tokenu ma prognozę lepszą od obu kontroli.
- Z: czas uczenia.
- O: token 299 (384 liczby), 24 obrazy holdout, 744 pary.
- M: v̂ = α v_t (mediana α 0,9091).
- K: BZ, OP.
- L (`WYNIKI_TEST.md`): 5 aktualizacji 0,00525611 / OP 0,00540581 / BZ 0,01662728. 100 aktualizacji 0,13243375 / 0,21922456 / 0,30740036. Przewaga nad OP 2,77 procent przy bramce 5.
- W: bramka niezaliczona.
- S: zmierzone.

**5. koopman_kierunki_stale** (23.09)
- P: czy klatki, w których zwykle stałe kierunki Koopmana mocno się zmieniają, poprawiają trening.
- Z: czas wideo (pary klatek filmu odległe o najwyżej 30 klatek, 6761 par; wagi `ut_b0500`).
- O: tokeny w ramce etykiety, 3×3×32, wybielone do 48.
- M: EDMD; 6 z 48 kierunków o |λ| ≥ 0,85; błąd w tych kierunkach w jednostkach typowego szumu.
- K: 1500 losowych z tej samej puli; kierunki szybkie i losowe; pewność.
- L (`PLAN.md`): R² kontrastu wektora: stałe 0,25, szybkie 0,00, losowe 0,12, wszystkie 0,59. AUC chybień 0,64 (zwykła zmiana latentu 0,64, pewność 0,89). Wal., 3 ziarna: 0,6696 ± 0,0015 wobec 0,6820 ± 0,0008.
- W: „Wynik negatywny".
- S: zmierzone; wybór używa etykiet.

**6. koopman_osobno_obraz_token** (25.09)
- P: czy osobna macierz 384×384 dla każdej pary obraz-token przewidzi późniejsze punkty.
- Z: czas uczenia.
- O: 41 472 jednostki, kroki 5–365, dopasowanie 48 par, test 24 przejść.
- M: Koopman na samym położeniu.
- K: BZ, OP, model α.
- L (`DIAGNOZA.md`): jeden krok 0,0103307 / OP 0,0038589 / BZ 0,0082256 / α 0,0037180. 100 aktualizacji 0,2321807 / 0,1169686 / 0,1362229 / 0,0710764.
- W: „Zero z 41472 macierzy spełniło pełną ustaloną bramkę".
- S: zmierzone.

**7. koopman_tozsamosc_kierunki** (25–26.09)
- P: czy Koopman przewiduje elementy latentu identyfikujące narzędzie i czy obraz przewidzianego latentu (FastGAN) wskaże klatki poprawiające detektor.
- Z: czas uczenia.
- O: kierunek LDA w tokenach (384) i skalar tożsamości obrazu; 72 sondy; 73, potem 160 punktów.
- M: wspólny operator na 4 ostatnich przyrostach (L4); inwersja FastGAN i krok Gaussa-Newtona; wyszukiwanie DINOv2/FAISS; kontynuacja od kroku 245.
- K: BZ, OP, α, podprzestrzenie losowe; ramiona bez dodania, losowe, trudne.
- L (`WYNIKI.md`): AUC 0,995 (krok 5) i 0,998 (kroki 245, 800). L4 na holdout: 0,272 (1 krok) i 0,171 (100 aktualizacji) wobec OP 0,284 i 0,617. Przyrost ogólny 0,063 rozrzutu. L1 RGB 0,001–0,009. 134 klatki, 1 epoka: +0,0007 wobec losowych +0,0009 mAP@0,5 (wal.). 5 epok: −0,0005 / +0,0010 / trudne +0,0016.
- W: prognoza działa dla jednego skalara; dobór nie różni się od losowego.
- S: zmierzone.

**8. koopman_z_pamiecia** (25.09)
- P: czy stan [z_t, v_t] naprawia prognozę.
- Z: czas uczenia.
- O: 41 472 jednostki × 384, bez PCA.
- M: rodziny alpha, ar2, diag, vel, delay; wybór na kalibracji.
- K: BZ, OP, poprzedni Koopman.
- L (`WYNIKI.md`): jeden krok 0,003708 / OP 0,003859 / BZ 0,008226 / poprzedni 0,010331. 100 aktualizacji 0,0762 / 0,1170 / 0,1362 / 0,2322. Przez bramkę 7 445 z 41 472. Rodzina z położeniem −37,6 %.
- W: „Pamięć naprawia Koopmana"; przewaga 3,9 % poniżej progu 5 %.
- S: zmierzone.

**9. koopman_fastgan_petla** (24.09, sesja równoległa)
- P: czy prognoza zmian latentu da zapytanie wizualne wskazujące użyteczną klatkę.
- Z: czas uczenia.
- O: deskryptor sondy 1536 liczb, 8 kierunków; pretrening 400 aktualizacji, zapis co 20.
- M: wspólny afiniczny Koopman, horyzont całej epoki.
- K: BZ, trend, potasowana kolejność.
- L (`WYNIK_LOCAL_1810448.md`): Koopman 0,218159; BZ 0,137424; trend 0,185775.
- W: bramka prognozy niezaliczona.
- S: zmierzone; FastGAN i gałęzie treningowe nie ruszyły.

**10. langevin_latent_detektora** (26.09)
- P: czy równanie z dryfem Koopmana, pamięcią, wymuszeniem, sterowaniem i szumem bije L4.
- Z: czas uczenia (krok = 5 aktualizacji).
- O: skalar tożsamości 72 sond.
- M: D + JP (8 przyrostów) + W + S + N (AR(q)).
- K: L4, sam dryf, BZ.
- L (`WYNIKI.md`): od kroku 245, h = 20: 0,151 / L4 0,171 / sam dryf 0,482; 1 krok 0,416 / 0,407. Od kroku 485, h = 20: 0,557 / 0,454.
- W: bramka niespełniona.
- S: etap 1 zmierzony; sterowanie S(u) tylko plan.

**11. dobor_cech_adaptacyjny** (22.09, sesja równoległa)
- P: czy dobór według „brakujących cech" z GLISTER i priorytetem Koopman–GLE bije losowanie.
- Z: czas wideo przy zamrożonych wagach danej epoki; wybór ponawiany co epokę.
- O: tokeny 24×24, 384 → 24 kanały, okno 8 obserwacji co 5 klatek.
- M: GLE (D+JP+W+Z+N) i Koopman.
- K: random, glister, placebo.
- L (`WYNIKI_KAMPANII_2026-09-22.md`, wal., 3 ziarna): random 0,7131 ± 0,0042; glister 0,7072 ± 0,0017; placebo 0,6941 ± 0,0033; cechy 0,6903 ± 0,0083; z Koopman–GLE 0,6839 ± 0,0095.
- W: „Żadna metoda doboru nie pokonała losowania."
- S: zmierzone.

**12. dobor_klatek_stan_sieci** (22.09, sesja równoległa)
- P: czy zysk odczytu narzędzie/tło z tokenów przewidzianych z historii wybiera lepsze klatki.
- Z: czas wideo (8 obserwacji co 5 klatek); punkty kontrolne tylko do diagnostyki.
- O: token; 57 073 klatki, 3 foldy po 20 000 trajektorii.
- M: EDMD i pełne GLE, predyktor 0,5 K + 0,5 GLE.
- K: persistence, losowe 1500 i 3000, błąd odczytu, placebo.
- L (`WYNIKI_SELEKCJA.md`): MSE tokenu: persistence 0,40985–0,41401, Koopman 0,36903–0,37312, GLE 0,33297–0,33975.
- L (`WYNIKI_PILOT.md`): AP ostatniej epoki (wal.): 0,67317 wobec 0,65547 (losowe 1500), 0,65220 (placebo), 0,67122 (losowe 3000).
- W: bramka spełniona w jednym biegu; bez dowodu statystycznego i bez przyspieszenia (27,12 wobec 20,38 min).
- S: zmierzone, jedno ziarno.

**13. domykanie_tozsamosci_large** (27.09)
- P: jak zdefiniować domknięcie latentu tożsamości i wybierać klatki, które je przyspieszają.
- Z: czas uczenia.
- O: latent kręgosłupa DINOv2-B, 72 sondy (12 obrazów × 6 wariantów), skalar a i 5 liczb.
- M: miary W1–W3; projekt: s(x) = ⟨∇L_det(x), ∇J⟩.
- K: poziom przypadku 0,08; planowane ramię augmentacji.
- L (`PLAN.md`): R 1,30 → 0,91 → 0,79. Rozpoznanie 0,29 → 0,33 → 0,31. Obrót −3,4, żółte tło +3,5 (rozrzut 6,1). Min a 3,0 → 5,5.
- W: latent staje „w stanie niedomkniętym".
- S: diagnoza zmierzona; metoda tylko plan.

**14. dekoder_sondy_large** (28.09; folder bez pliku .md)
- P: czy stan dekodera wiąże się z mAP.
- Z: czas uczenia.
- O: q = max(pewność × IoU) dla 72 sond, wagi k00–k06, 3 ziarna.
- M: korelacje rang; Koopman, OU, Langevin, Kalman na q (Nano).
- K: BZ; skalar kręgosłupa.
- L (`wyniki\ANALIZA_DEKODERA.json`): Spearman z mAP: dekoder 0,856, kręgosłup 0,406.
- L (`wyniki\BRAMKI_LOKALNE.json`): między ramionami Nano ρ = −0,068 i −0,515; modele 1,00–1,77 × BZ.
- W: bramki G1 i G2 niespełnione.
- S: zmierzone.

**15. petla_od_poczatku_large** (27–28.09)
- P: czy pętla Koopman + FastGAN przyspiesza uczenie RF-DETR LargeDeprecated od początku dostrajania.
- Z: czas uczenia.
- O: tokeny 40×40×768, opis sondy 3072 → 5 liczb; 6 obrotów × 3 ziarna × 3 pętle (53 z 54).
- M: operator na 8 przyrostach, rollout 3 obroty; top-90 DINOv2 i ocena RHO; 134 klatki na obrót.
- K: pętle losowe i bez.
- L (`WYNIKI.md`, wal., mAP@0,5): pole pod krzywą: koopman 0,9018 / 0,9033 / 0,9001, losowe 0,9053 / 0,9075 / 0,9013. Koopman − losowe −0,0030 (sd 0,0033; 3/18 > 0). Obroty 1–3, koopman − bez: −0,0023.
- W: „Kryterium niespełnione w obu częściach."
- S: zmierzone.

**16. petla_uczenia_koopman_fastgan** (26–27.09)
- P: czy pętla prognoza → FastGAN → wyszukanie → uczenie przyspiesza Nano.
- Z: czas uczenia.
- O: skalar i S_4; 8 obrotów po 165 aktualizacji od kroku 245; ziarno 42.
- M: operator na 8 przyrostach; top-45 na kotwicę.
- K: losowe, bez.
- L (`WYNIKI.md`): mAP@0,5 po 8 obrotach: koopman 0,9055, losowe 0,9045, bez 0,9072. Top-134 wal.: 0,593 / 0,563 / 0,627. Prognoza obrotu 1: 0,127 wobec zmierzonych 0,111.
- W: kryterium „Formalnie spełnione na jednym ziarnie", ale „Najlepsza jest pętla bez dodawania".
- S: zmierzone (pilot).

**17. sterowanie_langevin_kalman** (28.09)
- P: czy pętla sterowania (Langevin z hamowaniem, Kalman, FastGAN) na stanie dekodera przyspieszy uczenie.
- Z: czas uczenia.
- O: q sond.
- M: Δx = −K(x − x*) + pamięć + W f + C u + ε.
- K: bramki G1–G4.
- L (`PLAN.md`): G1 ρ = −0,07 (p = 0,61) i −0,51 (p = 0,01, „znak odwrotny"). G2: 1,00 / 1,77 / 1,64 × BZ.
- W: „G3 i G4 w tej postaci nie idą na Eden."
- S: plan; G1 i G2 zmierzone lokalnie.

**18. tozsamosc_czastki_bledy** (23.09, sesja równoległa)
- P: czy rozkład przestrzenny sygnału narzędzia i jego dynamika wskazują błędy zamrożonego detektora lepiej niż pewność.
- Z: czas wideo (krok 3 klatki, historia 3).
- O: tokeny `CZASTKI_*`, maska `cystotome_seg`.
- M: EDMD w słowniku [s, s²], pamięć, nominalny Kalman.
- K: drabina U…F i przesunięcie cykliczne P.
- S: tylko kod („niczego nie uruchomiono").

**19. wlokno_pomylek** (23.09)
- P: jakie obrazy detektor widzi tak samo jak cystotom.
- Z: brak (statyczne podobieństwo latentu).
- O: latent tożsamości klatki, pula 9683.
- M: strata włókna ‖φ(x) − φ(x̃)‖², kierunki błędów.
- K: losowe 250.
- S: tylko plan.

**20. wybieracz_tozsamosc_koopman** (23.09)
- P: czy wybieracz z wektora tożsamości, zaskoczenia Koopmana i poprawki Langevina bije losowanie.
- Z: czas wideo.
- M: Koopman w układzie narzędzia, pamięć 1–3 kroków.
- K: pewność detektora, losowe 250 (0,6319).
- L (`PLAN.md`, stan wyjściowy zmierzony na teście): AUC chybień: wektor pod obrysem 0,76, Koopman 0,58, pewność 0,88.
- S: tylko plan.

**21. fastgan_narzedzie** (23–24.09)
- P: czy FastGAN z maską da się wytrenować i sterować tak, by ujawnić słabość detektora.
- Z: inne (operator „rozjaśnienia otoczenia", „nie operator czasu filmu").
- O: 1044 wycinki 256×256; próba na 12 wycinkach.
- M: gradient po wejściu zamrożonego generatora; afiniczne EDMD.
- K: oryginał, losowe wejścia, sterowanie samą detekcją, potasowane pary.
- L (`WYNIKI_PROBY_TRUDNYCH.md`): trafienia z 12: oryginał 8, detekcja 4, detekcja + Koopman 2. Błąd operatora 0,2555 wobec BZ 0,2519. R² = 0,029.
- W: „Przewaga Koopmana ani poprawa uczenia nie są wykazane."
- S: zmierzone (technicznie).

**22. fastgan_odwrotnosc_mlp** (28–29.09) — dotyczy małego ViT uczonego od zera, nie RF-DETR
- P: czemu FastGAN nie trafia w przewidziany latent i czy pomoże sieć odwrotna MLP.
- Z: czas uczenia.
- O: 768 aktywacji ostatniego bloku, 12 kotwic.
- M: prognoza Koopmana na 500 kroków; sterowanie wejściem z, głębszymi warstwami, pikselami.
- L (`PLAN.md`): postęp: samo z 0,04, głębsze warstwy 0,36, piksele 0,61. Sieć MLP z dokręceniem 0,24 wobec 0,25 bez niej. Wzdłuż wektora tożsamości: błąd Langevina 0,35 wobec 1,11 dla „bez zmiany".
- W: „Sieć nie pomaga."
- S: zmierzone, jedno ziarno.

## B. Co zadziałało (dodatnie i powtórzone)

1. **Stan narzędzia powstaje w latencie z uczeniem.** R2 położenia z `box_pool` rośnie z −0,10 do 0,86 na osi budżetu i wynosi 0,87 w epoce 2 na osi epok. Sześć wolnych modów Koopmana daje 0,73–0,87 wobec 0,04–0,45 dla PCA. Oba pomiary na ziarnie 42 (DOC24 §1.3, §1.7).
2. **Moment końca uczenia.** W trzech ziarnach dołek straty wal. wypada w epokach 6, 6, 7, a szczyt AP w 6, 8, 7. Sygnał bez etykiet kosztuje średnio 0,004 AP. Test potwierdza: 0,681 wobec 0,654, jeden bieg (DOC24 §1.25, §2.16).
3. **Jeden kierunek „tu jest narzędzie".** AUC 0,976 i 0,975 na puli; zapalony w 99,1 procent klatek testu. Niezależnie w zegarze uczenia: AUC 0,995–0,998.
4. **Pamięć w zegarze uczenia.** Błąd jednego kroku 0,0103 → 0,0037. L4: 0,171 wobec OP 0,617 w oknie wczesnym i 0,454 wobec 0,996 w późnym. Bez pamięci 0,482 wobec 0,151.
5. **Dryf tożsamości jest wczesny.** Kończy się około kroku 500 (bieg gęsty), w dwóch pierwszych obrotach pilota Nano i po około dwóch obrotach Large.
6. **Kod z operatora translacyjnego czyta wydłużenie ramki.** 85,8 procent; operator czasowy 88,1 i 85,7; kosinusy główne 0,95, 0,90, 0,87, 0,83. Dwie sesje dostały identyczne liczby (DOC24 §1.17, §1.19).
7. **GLE na tokenach bije persistence** w trzech foldach: 0,333–0,340 wobec 0,410–0,414.
8. **Chybienia w szybkim ruchu.** 13,9 / 13,3 / 14,5 procent w najszybszych 5 procentach klatek przy trzech budżetach; około 32 procent w każdym z dziewięciu ramion.

Dodatnie, ale niepowtórzone: K4 (+0,0068 AP; chybienia −1,42 punktu, przedział [−2,73; −0,26]) i pilot Koopman–GLE (+0,01770 w ostatniej epoce, jedno ziarno).

## C. Co nie zadziałało i dlaczego

- **Reszta Langevina jako selektor (K1).** Placebo dorównuje sygnałowi. Tylko 23 procent chybień L0 znika po pięciokrotnie większej liczbie etykiet; najlepszy sygnał ma 7 procent usuwalnych klatek w górnym decylu. Według dokumentu 39 (za DOC24 §1.9) wąskim gardłem jest „brak usuwalnego błędu we wskazanych klatkach".
- **Dobór D-optymalny.** Ślad kowariancji 0,116 wobec 0,210, ale umiejętność 4,0 wobec 7,3 procent.
- **Deep Koopman.** Od −45,5 do −61,4 procent względem „nic się nie zmienia".
- **Kod tożsamości Koopmana.** Żadna z trzech dróg nie bije ośmiu składowych głównych (wynik łączny 100,0 wobec 104,9).
- **Kierunki niezmiennicze.** Przy obrocie i zmianie tła: 0 z 32 w czterech detektorach.
- **Zaskoczenie Koopmana.** AUC chybień 0,585, z Langevinem 0,574, zwykła zmiana klatki 0,602 (test).
- **Dobory K6, kierunki stałe, cechy.** Przegrywają lub remisują z losowaniem: „na tym zbiorze żadna metoda wybierania trudnych albo „brakujących" klatek nie pokonała losowania" (`koopman_kierunki_stale\PLAN.md`).
- **Koopman na położeniu w zegarze uczenia.** `DIAGNOZA_PRZYCZYNY.md`: „Trajektoria zwalnia, a stała macierz tego nie widzi"; „Następny krok zależy od prędkości, nie od położenia"; „Stan nie jest domknięty". Prognoza jest o 62 procent za długa.
- **Pętla FastGAN na Nano.** `TEORIA_LANGEVIN.md`: „po kroku 245 przewidywana zmiana jest za mała, by zmienić wybór klatek". Wspólnych klatek 54/60.
- **Pętla na Large** (`DIAGNOZA_PORAZKI.md`):
  - „Prognoza Koopmana jest gorsza niż „latent się nie ruszy"" (0,196 wobec 0,078; 12 z 12).
  - „Latent siedzi w części sieci, która prawie się nie uczy" (kręgosłup 0,026 %, dekoder 1,3 %).
  - „Latent nie wiąże się z wynikiem detekcji" (ρ = 0,07, p = 0,61, n = 53).
  - „Wybór skupia klatki" (cos 0,66–0,68 wobec 0,62).
  - „Dźwignia jest słaba, a szum duży".
- **Trudne klatki.** +0,009 daleko od kotwic, −0,018 przy kotwicach.
- **Równanie Langevina.** Jeden krok gorszy o 2,2 procent (dopuszczalne 2); od kroku 485 rollout 1,23 × L4.

## D. Tożsamość narzędzia

- **Co jest mierzone.** Odróżnienie tokenu narzędzia od tła i odczyt kształtu ramki. Zbiór ma jedną klasę (cystotom), więc „które to narzędzie" jest niemierzalne (DOC24 §1.21). Wektor zapala się też na innych cienkich narzędziach (§2.13).
- **Wolne mody całej klatki to odcisk operacji.** 0,72 wariancji między filmami; stan narzędzia −0,01 (§1.10).
- **Kod translacyjny (8 modów).** Wydłużenie 85,8 %, pole 29,0 %, zwrot końcówki 0,1 i 0,3 %, położenie 10,0 i 18,5 %. 58 % zmienności leży między filmami. Dekoder pikseli: R2 −0,69 na tokenach narzędzia.
- **Liczba kierunków.**
  - Widmo bez symetryzacji: 16 modów trwałych, 6 przemiennych, 10 obrotowych; AUC 0,834 / 0,743 / 0,593, razem 0,893.
  - 15 części rzeczywistych daje 0,887 wobec 0,902 dla 20 liczb.
  - Przestrzeń tożsamości z operatora czasowego ma wymiar 5 (§2.7).
  - W tokenach wystarcza jeden kierunek LDA (0,976); drugi, prostopadły, daje 0,924.
  - W zegarze uczenia podprzestrzenie S_1…S_32 nie są bardziej przewidywalne niż losowe.
- **Stabilność między punktami kontrolnymi.**
  - Mod o największej Re: przeżycie 0,77 / 0,78 i wydłużenie 58,6 / 58,4 (500 wobec 2500 klatek).
  - Najlepszy mod z L0 (AUC 0,772) po douczeniu znika (0,667).
  - Kierunek LDA obraca się: cosinus z krokiem 5 wynosi 0,73 (krok 365) i 0,64 (krok 800), przy AUC stale 0,995–0,998. Tożsamość jest „dziedziczona z wstępnie wyuczonego backbone".
- **Stabilność między ziarnami.** Na Large R ≈ 0,78 we wszystkich pętlach (różnice ≤ 0,02). Cosinus kierunków między ziarnami: nie znaleziono.
- **Niezmienniczość.** „Kierunki tożsamości" Koopmana zmieniają się przy obrocie o 1,31 (losowe 1,58) i okazały się kątem ułożenia. Wektor LDA zapala się w 97–98 % klatek po obrocie narzędzia o 30° i w 81 % na obcym tle.
- **Związek z mAP.**
  - Skalar kręgosłupa wobec mAP: ρ = 0,07.
  - R wobec mAP: ρ = −0,20.
  - Jakość dekodera w czasie: ρ = 0,86; między ramionami Nano −0,07 i −0,51.
  - Trwałość modów wobec jakości wal.: 0,74–0,89.
- **Dane nietknięte (test).** Kontrast pod obrysem z adnotacji: AUC chybień 0,761. Wersja bez etykiet: 0,601. Obecność narzędzia: 0,777. Pewność detektora: 0,886.

## E. Sprzeczności i zastrzeżenia

1. **K4.** Raport ma 10 współrzędnych, nie 16. „Pokrycie" było k-centrum kosinusowym na nieznormalizowanych współrzędnych. Dwie wartości kontrastu (+0,0068 i +0,0072) pochodzą z różnych źródeł AP (DOC24 §1.10, §1.16).
2. **Kryteria macierzy.** Zapisano AP ostatniej epoki, a werdykty podano na najlepszym punkcie. Przedziałów bootstrapowych i korekty Holma dla AP w źródłach nie ma (§1.9).
3. **K6.** Dwie z trzech osi były tą samą liczbą (korelacja 1). Ramię równomierne jest obciążone (155 z 250 klatek brzegowych). Latent przypisywano detektorowi z 1500 klatek, a pochodził z `ut_b0500`.
4. **Artefakt naklejki.** Spadek pewności do 0,26 na obcym tle był artefaktem prostokątnej naklejki. Wniosek o rosnącej zależności od tła jest niepowtórzony, a wyników zadania 1809823 nie przeanalizowano.
5. **Dokumenty.** Istnieją dwa dokumenty nr 44. Równoważność „brak pamięci ⇔ domknięcie" z dokumentu 43 wycofano (kontrprzykład C = 0).
6. **Zbiór testowy.** W Etapie 1 „nie dotykano", ale pomiary torów z dokumentu 36 czytały wszystkie 167 grup. 23.09 użyto testu do diagnozy.
7. **Sondy „holdout".** TR_0134 i TR_0082 były wśród 1500 klatek uczących.
8. **Pilot Koopman–GLE.** Dodatni przy jednym ziarnie i w ostatniej epoce (+0,0034 w najlepszej). Stoi w napięciu z macierzą etapu 1 i z kampanią, w której ramię Koopman–GLE było najsłabsze.
9. **Prognoza Koopmana.** Na Nano bije BZ (0,171), na Large jest gorsza w 12 z 12 przypadków.
10. **Najlepszy operator L4** łamie zakazy planu `koopman_gesta_historia` (uśrednianie pod maską, rzut na jeden kierunek, jedna macierz dla wszystkich obrazów).
11. **Niewyjaśnione liczby.** 10 637 klatek puli wobec 9683 w nagraniu; 117 grup wobec 118 filmów; 3128 klatek w `PLAN.md` wobec 8600 w raporcie kierunków stałych.
12. **Nieusunięte usterki.** Strażnik determinizmu rejestratora (0,131). Skalowanie i PCA liczone na całej puli przed podziałem (plan 37).
13. **Nieaktualny opis.** `dobor_cech_adaptacyjny\README.md` twierdzi, że treningów nie uruchomiono.
14. **Założenia twierdzeń T5–T11** nie zostały zmierzone dla RF-DETR.

## F. Związek z wcześniejszą pracą na szeregach czasowych

- Projekt grantowy „był tylko źródłem aparatu matematycznego" (DOC24, wstęp Etapu 1).
- **Przeniesiono:**
  - cząstki latentu i rozszerzone równanie Langevina;
  - twierdzenie Lean o rozkładzie błędu (`DiagnostykaInnowacji.lean`);
  - schemat `schemat-diagnostyki-latentu-koopman-langevin-2026-09-18.md` (części A3 i C);
  - kod członów D, JP, W, N, Z i skali własnej z `tft_langevin_latent_56`, `tft_langevin_czlon_zdarzen`, `tft_langevin_optymalizacja_95`.
- **Jawne adaptacje** (`dobor_klatek_stan_sieci\README.md`): „24 zamiast 192 wymiarów, 8 zamiast 56 kroków, metadane obrazu zamiast sprzedaży i kalendarza, tryb przyczynowy zamiast znajomości wejść jutra"; „wynik TS nie przenosi się automatycznie na obrazy".
- **Co zachowało się inaczej:**
  - film „nie ma wejścia znanego z góry", a wymuszeniem jest nieobserwowana ręka chirurga, nie sterowany ruch kamery (§1.2, §1.13);
  - K1 nie miał członów W, N, Z;
  - baseny cząstek: IoU 0,52–0,56 wobec 0,81 dla niedźwiedzia z DAVIS.
- **Co się powtórzyło:**
  - pułapka bezwładu („jak wcześniej w projekcie grantowym");
  - ślad tokenu symetryczny w czasie, zgodny z ustaleniem U14 (jądro pamięci głębokości jednego kroku, 5–6 procent przyrostu).
- **Wątki poboczne.** Kalman z EM dla ukrytych zmiennych pamięci pochodzi z `rnn20_gle_emr_kalmannet` (§2.14). Dowody doboru klatek trafiły do `ViTFrameSelection.lean` w repozytorium grantowym.

## Kluczowe źródła

- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\docs\24_OPIS_EKSPERYMENTU_KOOPMAN_LANGEVIN_OD_POCZATKU_DO_KONCA.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\krystalizacja_latentu\README.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_tozsamosc_kierunki\WYNIKI.md` (oraz `TEORIA_LANGEVIN.md` i `NOWOSC_I_PUBLIKACJA.md` w tym folderze)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\koopman_osobno_obraz_token\DIAGNOZA_PRZYCZYNY.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\petla_od_poczatku_large\DIAGNOZA_PORAZKI.md` (oraz `WYNIKI.md`)
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\sterowanie_langevin_kalman\PLAN.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_klatek_stan_sieci\WYNIKI_PILOT.md`
- `F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\SmallViTAblation\experiments\dobor_cech_adaptacyjny\WYNIKI_KAMPANII_2026-09-22.md`
