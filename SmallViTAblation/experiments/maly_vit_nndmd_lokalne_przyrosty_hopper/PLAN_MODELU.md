# NNDMD: późne okno i zakotwiczona prognoza przyrostów

Osobny wariant autoryzowany przez użytkownika 2026-10-02. Nie nadpisuje poprzednika; job1819574 został anulowany przez root. Pytanie: czy wspólne uczenie lokalnej dynamiki przyrostów poprawia surowe RMSE względem persistence na tych samych1500 obrazach źródłowego ViT? Próg85procent znaków jest celem, nie gwarancją. Dane mają wszystkie600 cząsteczek i768 oryginalnych współrzędnych na obraz. Bbox nie oznacza maski segmentacji; trening waży równo wszystkie cząsteczki.

## Podział i wybór

Checkpointy0..100 odpowiadają aktualizacjom0..1000 co10. Inner gradient-fit: sliding windows origins50..70, po10 przejść, cele maksymalnie80. To21 okien obejmujących30 unikalnych przejść cp50..80. Walidacja wszystkich1500 obrazów: origin80, h1..10, prawdziwe cele81..90 wyłącznie do metryk i wyboru. Dwa z góry określone lrG:1e-4 i1e-3, network lr1e-3. Każdy kandydat świeży z tym samym seed20261002. Max_epochs10, patience2, min_delta raw RMSE1e-5; reguła taka sama jak poprzednio. Wybór minimalnego agregowanego raw RMSE full768 na walidacji80→90; remis: niższy lrG, następnie wcześniejsza epoka. J128 nie służy wyborowi.

Final refit świeży seed na cp50..90:31 origins50..80×1500 obrazów, po10 przejść, przez best_epoch zwycięzcy. Freeze modelu/normalizatora/źródeł PRZED odczytem cp91..100. Test origin90→100 na wszystkich1500 to eksploracyjna odłożona przyszłość tych samych obrazów; trajektoria była oglądana wcześniejszymi wariantami. Żadne przyszłe wartości nie uczestniczą w normalizacji, gradientach ani wyborze.

Native scalar kalibracji pochodzi z cp0 wszystkich1500, bez centrowania, nie z testu; auxiliary mean/std wyłącznie prefix<=80 dla inner i<=90 dla final. Historyczne wejście cp(o−2),cp(o−1),cp(o),59 rzeczywistych podsumowań wag/BN/ostatniego dostępnego AdamW, maski i wiek; input2544. Dane chwilowe m/v tylko w pełnych snapshotach co100 aktualizacji; causal hold-last bez interpolacji przyszłych stanów.

## Model i loss

Encoder2544→768→768 Swish; wspólny learnedstate768, A=exp(G)768×768; nieliniowy decoder768→768→827 (full768+aux59). Są to komponenty PyKoopman1.2.1 NNDMD, z jawnym własnym image-aware fit/loss adapterem. Nie używa się niezmodyfikowanego NNDMD.fit ani PCA. Learned axes nie są oryginalnymi osiami latentu.

Encode origin raz: phi0=E(x0). Rollout phi_h=A^h phi0 bez teacher forcing. Native prognoza zhat_h=z0+s·(D_native(phi_h)−D_native(phi0)). Różnica obliczana w nawiasie; decoded_origin wyliczony raz i NIE detach w loss. Aux prognoza D_aux(phi_h) bez kotwiczenia. Offset z0−sD_native(phi0) jest dodatkową stałą pamięcią originu poza learnedstate768. To anchored forecast, nie twierdzenie o zamkniętej globalnej dynamice native768. H0 exactz0 przez kotwicę nie oznacza idealnej rekonstrukcji.

Loss w przestrzeni z/s: mean_h MSE_full768(anchored_h,true_h) +rec_weight1·[MSE(D_native(phi0),z0/s)+mean_h MSE(D_native(E(x_true_h)),z_true_h/s)] +aux_weight0.1·[maskedmean_h MSE_auxprediction +rec_weight1·(maskedMSE_auxreconstruction_origin+maskedmean_h MSE_auxreconstruction_future)] +identity_weight0.001·||A−I||F²/768. Rekonstrukcje przyszłych stanów są wyłącznie dodatkowymi celami w legalnym FIT oknie, nie wejściem do phi_h. Aux bloki mają własne mianowniki liczby ważnych kanałów i tokenów. Kara identity proximity nie zapewnia rho(A)<=1. Adam betas.9/.999 eps1e-8; clip norm1 po agregacji wszystkich gradientów. RMSE i MSE przy tym samym ważeniu mają ten sam minimizer; nie gwarantujemy globalnego minimum.

## Równoległość i telemetria

Globalny logical batch32 OBRAZY, ostatni28; wszystkie600 tokenów każdego obrazu. SUM gradientów lokalnych strat z globalnym mianownikiem B·600 i średnią po kanałach/horyzontach. Penalty wyłącznie rank0. Exp(G) localVJP dla a_leaf, potem SUM gradientów wszystkich parametrów, clipping i jedna wspólna aktualizacja Adam. Nie średnia lokalnych mean. Przy world11 split32=3×10+2, split28=3×6+2×5; przy world4 odpowiednio8/rank i7/rank. Infrastruktura11 urządzeń Hopper-2 obejmuje pełneH200 i MIG i wymaga osobnego pozytywnego NCCL guard; żadnego automatycznego fallbacku bez decyzji root.

Gdy NCCL nie wspiera wspólnego fizycznego GPU dla odrębnych MIG, dopuszczony jawny transport EXPANDED_DISTRIBUTED_BACKEND=gloo przenosi każdy tensor image_count/gradient/statystyki na CPU, wykonuje ten sam SUM i kopiuje z powrotem na własne GPU PRZED clippingiem. Koszt transferów jest raportowany, metody i gradientu nie zmienia się. Backenddomyślny NCCL; każdy wariant transportu musi przejść pełny GPU1vsworld proof. Proces z jednym widocznym UUID używa cuda:0; globalrank nie oznacza lokalnego indeksuCUDA.

Każda epoka obejmuje wszystkie okna i1500 obrazów: inner987 logicalsteps; final1457. Rank0 zapisuje atomowe status/checkpoint, TRAINING.jsonl, actualtotalsteps, gradientG/network preclip, deltaG, deltaA/A, delta_network/network, ||A−I||F/||I||F i lrG. Per walidacja rho(A), ||A||2, ||A^10||2. Modelselection wyłącznie full768 RMSE, bez strojenia stabilności wynikiem testu. Zachowane originowe checkpointy umożliwiają wznowienie z identycznym config i phase, guard patience również po wznowieniu.

## Kontrole i wyniki

CPU całe gradienty vs microchunk; origin-only A^h i futurepoison; H0 anchor exact przy A=I; paired absolute/residual rollout tego samego modelu (diagnostyka, nie dodatkowa siatka); CPU Gloo1vs4/11 global32/tail28 loss/gradient/Adam/telemetry/checkpoint. GPU guard o faktycznie przyznanym world size, pełny model768 i600 cząsteczek; actualdataset preflight również przed resetseed i właściwym fittingiem. Infrastructure/model guard muszą PASS przed pełnym treningiem.

Raport raw full768/J128 RMSE h10..100 aktualizacji, sqrt wspólnego MSE po równych obrazach/tokenach/kanałach; J128 po decoderze. Persistence, historical mean trzech dostępnych stanów, continuation ostatniego przyrostu; sign_all wraz baselines i sign(0)=0 bez odsiewu. Osobno H0 reconstruction RMSE. Zapisy completeNN/A/G, perimage metryki,12 przykładówJ128 i4 full768matchedprobes. Wynik poniżej persistence falsyfikuje poprawę w tym protokole; sam H0exact lub duży gradientG jej nie dowodzi.
