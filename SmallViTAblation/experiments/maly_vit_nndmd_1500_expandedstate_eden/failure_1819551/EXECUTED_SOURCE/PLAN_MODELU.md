# Rozszerzony neural Koopman1500 na Edenie

Protokół ustalony przed treningiem i testem. Zachowujemy wcześniejsze eksperymenty. W tym przebiegu są dokładnie1500 obrazów źródłowego ViT, każde z600 cząsteczkami i pełną reprezentacją768. Adnotacje zawierają bbox, bez prawdziwych masek segmentacji; loss ma równe wagi wszystkich obrazów, cząsteczek, kanałów i horyzontów.

## Podział i budżet

Checkpointy0..100 oznaczają aktualizacje źródłowe0..1000 co10. Inner gradientowe uczenie0..80 obejmuje71 okien10-przejściowych (origins0..70). Walidacja to wyłącznie origin80→90 na wszystkich1500 obrazach. Maksymalnie10 pełnych epok, early stopping patience2, min_delta raw aggregateRMSE=1e−5. Pełna epoka obejmuje wszystkie71×1500 image-windows; żadnego losowania podzbioru. Wybór best_epoch jest minimum full768 walidacyjnego sqrt(globalMSE po wszystkich10horyzontach), przy remisie wcześniejsza epoka. Min_delta dotyczy zatrzymania, a nie rankingu minimum.

Final refit zaczyna od identycznego nowego seedu i normalizacji tylko0..90. Uczy81 origins0..80 przez best_epoch epok, bez dodatkowego wyboru. Freeze poprzedza odczyt testowych latentów91..100 do oceny. Test90→100 obejmuje100 źródłowych aktualizacji, a nie500. Dotyczy przyszłości tych samych1500 obrazów, a nie uogólniania na nowe zdjęcia. Early stopping nie wykorzystuje testu. Walltime48h, atomiccheckpoint po każdym originie umożliwia wznowienie przez EXPANDED_RESUME_CHECKPOINT; niedokończony origin jest liczony ponownie od ostatniego zapisanego optimizerstate.

## Architektura i przyczynowość

Wejście2544: pełne z768, dwie ostatnie różnice z (1536), aktualne i poprzednie summaryaux59 oraz ich maski, dwie flagi historii i dwa age. Zero padding i maski na początku. Auxiliary obejmują rzeczywiste delta wag/BN oraz ostatnie dostępne podsumowania momentów AdamW ze snapshotówco100aktualizacji. Brak interpolacji z przyszłego snapshotu. Encoder2544→768→768 ze swish, learnedstate768, A768×768=exp(G), decoder768→768→827 ze swish. Learned768 kompresuje wejście2544, nie jest oryginalną bazą768. Nie używamy PCA. Encoder i decoder są nieliniowe.

Encoder wywołany jeden raz dla originu. Dalszy rollout to wyłącznie A^h oraz decoder pełnych768 i59 auxiliary. Żaden prawdziwy przyszły latent, gradient, waga ani moment AdamW nie trafia do prognozy. Predykcje auxiliary są token-local proxy wspólnych globalnych summary; raportujemy ich agregat i rozrzut, nie pełny przyszły optimizerstate. Nie są wstrzykiwane ponownie do stanu ani encoderu. Historyczne różnice przyszłości można mechanicznie odtworzyć z decodedlatent. Age obserwowanych momentów rośnie dalej, bez resetu w chwili przyszłego nieobserwowanego snapshotu. Wspólny batch obrazów sam nie ustanawia interakcji między cząsteczkami.

## Dokładny loss i optimizer

L = mean_h MSE_full768_prediction + rec_weight·(MSE_full768_reconstruction_origin + mean_h MSE_full768_reconstruction_future) + aux_weight·(mean_h maskedMSE_aux_prediction + rec_weight·[maskedMSE_aux_reconstruction_origin + mean_h maskedMSE_aux_reconstruction_future]) + identity_weight·||A−I||F²/768.

rec_weight1, aux_weight0.1, identity_weight0.001, globalnative scale z wszystkich1500 stanówcp0. Native nie centrujemy; wspólny scalar to standard deviationcp0. Aux input/target mają oddzielne scalarperfeature z wyłącznie dozwolonego prefixu. Primarytarget pełne768, J128 tylko ewaluacja. Auxiliary block średni po dostępnych rzeczywistych etykietach; momenty mają targetmask tylko przy rzeczywistych snapshotach. Native reconstruction odtwarza z, auxiliary branch odtwarza dostępne summary; nie odtwarza całego wejścia2544 ani flag/lagów. Truefutureinput w reconstruction jest legalnym treningowym przykładem do fitboundary, nie teacher forcing ścieżki prediction. Pole reconstruction_native_mse_raw w trainlog jest sumą dwóch składników, nie jedną średnią po11stanach; aux_prediction_normalized_SSE to SSE.

Adam β=(0.9,0.999), epsilon1e−8; lr encoder/decoder0.001, lrG0.00001. Jeden krok Adam na32 obrazy; ostatni batch28. Pełne600 cząsteczek rozbijamy na microchunk1024, każdy ma wagę według rzeczywistej liczby tokenów/(B×600). Gradienty encoder/decoder i G sumowane przed jednorazowym clippingnorm1 i optimizerstep. Chainrule przez exp(G) stosowane jeden raz po akumulacji gradientA. Zapisujemy rzeczywisty counter, normę gradientu, czas i rawpredictionMSE perhorizon. Kara A−I to bliskość identyczności, nie gwarancja rho(A)≤1. RMSE i MSE tego samego ważenia mają to samo minimum; Adam nie gwarantuje minimum globalnego ani85procent poprawnych znaków.

## Pakiet, dane i wyniki

Rzeczywiście wykorzystujemy PyKoopman1.2.1: klasę NNDMD, jej FFNN i propagator Koopmana/exp(G). Własny jawny adapter zastępuje nativefit, które materializuje trajektorie i zakłada liniowy decoderC. Nie raportujemy iloczynu wag nieliniowego decoderu jako C. Runtime izolowany, Torch2.9.1cu128 bez zmian sourcevenv. Reader EDENfull101cache jeśli scratchpozwala, inaczej rolling13; jeden proces naprzemiennie ekstrakcja i NN, bez lokalnego fulltraining.

Pełna ewaluacja wszystkich1500/all600: rawRMSE/MSE full768 i decodedJ128, h1..10 (10..100aktualizacji), agregat sqrt(meanMSE), H0rekonstrukcja, znak wszystkich współrzędnych sign(0)=0, bez odsiewu nearzero, motioncos/norm z jawną fractionincluded. Baselines persistence, causallinearcontinuation z ostatniej różnicy oraz średnia trzech dostępnych stanóworigin−2..origin. PerimageJSON obejmuje wszystkie1500; predykcje do figur tylko indeksy0..11 ustalone przed testem. Fullfutureforecasts nie są materializowane. Kompletny model/optimizer/A/G, normalizatory, source/confighashes, freeze, telemetry i metryki są zachowane. Diagnosticsrho(A), operatornorm i ||A^10|| nie służą retuningowi po teście.

CPUtests: microbatchfullgradient z sparseaux, tail28, originonlyencode, futurepoison, brak przyszłych etykiet w fit i normalizacji, driverindices/weighting/selection/freezesequence. Niezależny Astraaudit przed sbatch. Pierwsze rzeczywiste kroki GPU w zadaniu zapisują benchmark i warunkowy koszt epoki; nie utożsamiamy sourceekstrakcji ze startem Adam.
