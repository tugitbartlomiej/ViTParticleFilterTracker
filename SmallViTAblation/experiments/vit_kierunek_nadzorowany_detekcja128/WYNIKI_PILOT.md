# Wspólny kierunek nadzorowany detekcją: pilot

Jedna delta[128] dodawana do wszystkich 600 tokenów każdego obrazu, tylko w channels0. Punkt interwencji postGELU768 ostatniego MLP; nie pełny stan modelu. Oryginalny k1000, zamrożone wagi i BN, niezmieniony residual.

Wybrano seed 202, LR 0.01, krok 100, skala 1.0, norma uczonej delty 5, alpha 5; v ma normę jeden. Sześć konfiguracji po 200 kroków, batch 8. Focal objectness jednej klasy + 2 GIoU, regularizator 0.001 mean(delta²), limit normy 5. Skale dwustronne [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0].

|Split|Wariant|AP50|AP50:95|IoU ramki o najwyższej confidence|Confidence|Loss|
|---|---|---|---|---|---|---|
|val|baseline|0.402075|0.112166|0.438647|0.675874|2.528891|
|val|shared|0.446314|0.121289|0.437526|0.676431|2.483642|
|val|random11|0.413415|0.117030|0.438997|0.680431|2.529623|
|val|random22|0.415488|0.115478|0.436576|0.653828|2.510374|
|val|random33|0.401537|0.114711|0.438400|0.674982|2.527911|
|test|baseline|0.457589|0.114275|0.465221|0.710464|2.454718|
|test|shared|0.458098|0.115268|0.466429|0.710645|2.409879|
|test|random11|0.459138|0.114011|0.466594|0.714804|2.455732|
|test|random22|0.454396|0.113456|0.463576|0.691448|2.440761|
|test|random33|0.457802|0.112785|0.464879|0.709550|2.452754|

AP to jednoclasowa interpolacja 101-punktowa po 10 lokalnych szczytach na obraz; AP50:95 to średnia 10 progów. To jawny protokół tej głowy, nie pełny COCO evaluator. IoU i confidence dotyczą najwyżej ocenionej ramki wybranej bez GT.

Kontrole: trzy izotropowe losowe kierunki o tej samej normie startowej co wybrana korekta; ich skale również dobierane tylko val. Finalne normy mogą różnić się po doborze skali i są zapisane w FREEZE.json. Bootstrap sparowany po całych filmach, warunkowy względem już wybranej korekty.
val: delta AP50 0.044238, delta AP50:95 0.009123, CI95 kolumn [AP50, AP50:95]: [[0.0045001846078456115, -0.00204629010571928], [0.10224958700382182, 0.02000763908119263]].
test: delta AP50 0.000509, delta AP50:95 0.000993, CI95 kolumn [AP50, AP50:95]: [[-0.00805742732580614, -0.006155583198893881], [0.013511102007188282, 0.0061486269891084825]].

Podział historyczny 240/75/85 obrazów i 69/23/24 filmów, bez overlap. Obrazy były używane rozwojowo i mogły uczyć ViT, a kanały wybrano historycznie. To pilot z rozdzieleniem filmów w tym zbiorze, bez niezależnego dowodu generalizacji. GT tylko loss i ocena, bez maski/ROI w inferencji.

Wspólny kierunek działa jednakowo dla wszystkich obrazów. Oracle per-image optymalizuje osobno z GT danego obrazu; tutaj go nie wykonano. CAV to kierunek sondy rozróżniającej koncept; TCAV to analiza znaku pochodnej odpowiedzi wzdłuż CAV. Tutaj kierunek uczony jest bezpośrednio stratą detekcji, nie CAV ani TCAV. Jest to poprawa przez jawną interwencję, nie wyuczenie naturalnego latentu w wagach. Brak poprawy AP albo wynik słabszy od losowych kontroli jest pełnoprawnym wynikiem.

Świeży reload, pełny ViT kontra szybka końcówka: PASS. Wagi i BN niezmienione. Czas 12.7 s, PID 32272, RTX 3070. Proces zakończył obliczenia.

Reprodukcja: F:/Python/python.exe -B experiments/vit_kierunek_nadzorowany_detekcja128/run.py z SmallViTAblation. RUN.lock zapobiega dwóm kopiom. Wyczyścić zakończone własne wyniki przed nowym biegiem lub zachować je osobno. Wszystkie źródła i indeksy: MANIFEST.json; RNG i parametry: CONFIG.json; korekta: DIRECTION.npz; dobór przed testem: FREEZE.json; krzywe: HISTORY.json i CURVES.png.


## Interpretacja końcowa i audyt

Walidacja: AP50 0.402075 → 0.446314. Test: AP50 0.457589 → 0.458098, AP50:95 0.114275 → 0.115268. Różnice na teście wynoszą 0.0509 i 0.0993 punktu procentowego.

95-procentowy przedział bootstrap różnicy AP50 to [-0.8057, 1.3511] punktu procentowego; AP50:95 to [-0.6156, 0.6149]. Oba obejmują zero. Pilot nie potwierdza poprawy AP na teście. Kontrola random11 osiąga wyższe testowe AP50 (0.459138) niż wspólny kierunek. Wspólny kierunek ma najwyższy punktowy AP50:95 spośród zbadanych wariantów, ale bez rozstrzygającego przedziału względem baseline. Walidacja służyła selekcji, więc jej przedział nie jest niezależną oceną skuteczności.

Audyt źródłowy SOURCE_AUDIT.json: 400/400 JPEG zgodnych byte-for-byte z archiwami, ramki i poligony zgodne z właściwymi rekordami frame_number źródłowych annotation.csv; 116 filmów. Nie używano poligonów do wyboru tokenów w modelu.

Trening wykonał PID 32168; ocenę i poprawny świeży reload zakończył PID 32272. Dwa błędy techniczne naprawiono bez ponownej optymalizacji i bez zmiany FREEZE.json lub DIRECTION.npz: serializacja logu oraz dtype korekty podczas reloadu. Zachowano FAILED_FIRST.json, FAILED_RELOAD.json i logi. Łączny czas runnerów bez importów: 67.46 s; optymalizacja i walidacyjna selekcja: 31.91 s; czas od pierwszego uruchomienia do zakończenia z importami i naprawami: 284.74 s. Koszt i peak GPU w COST.json.
