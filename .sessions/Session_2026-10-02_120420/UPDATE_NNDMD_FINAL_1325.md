---
date: 2026-10-02
type: mixed
tags: [pykoopman, nndmd, pelne-768, prognoza, audyt, no-pca]
status: complete
importance: high
---
# Uzupełnienie 13:25: rzeczywisty NNDMD 768 i końcowy wynik

Zapis z 2026-10-02, 13:25, Europe/Warsaw. Uzupełnia [główną historię sesji](SESSION_SUMMARY.md) i [wcześniejszy postscript](UPDATE_NNDMD.md), bez przepisywania dawnych wyników. Nie jest dosłownym transkryptem.

**Ostatecznie wykonano pełne wejście 768, learned state 768, decoder 768 i A 768×768.** Wcześniejsza propozycja learned state 128 została zastąpiona przed tym treningiem. J128 pozostało podzbiorem oryginalnych kanałów do dodatkowej straty i oceny po dekodowaniu; nie było wymiarem uczonego stanu ani wejścia NN. Osie uczonego stanu są wyuczone i nie są automatycznie oryginalnymi makrowektorami e_j.

Pełna ekstrakcja `[101,34,600,768]` float32 została ukończona. [Weryfikacja danych](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/DANE_768_WERYFIKACJA.json) potwierdza 6 329 548 928 bajtów, skończoność, zachowanie dawnych sond i nienaruszony cache J128. SHA256 pełnego pliku: `91a64e28612e2806cc203f8fc194b55cac4e7bbc8df7f5bbfee0ef1a377da140`.

Wykonano rzeczywisty PyKoopman 1.2.1 NNDMD z encoderem SiLU 768→768→768, liniowym decoderem 768→768, generatorem G i A=exp(G). Jeden krok oznacza checkpoint co 10 aktualizacji ViT. Bez PCA i redukcji wymiaru. Jawne adaptery zapewniają sampler coverage/równe obrazy oraz stratę full768 + beta×decodedJ128, obejmującą prognozę i rekonstrukcję. Nie jest to niezmieniona domyślna funkcja straty pakietu.

Wewnętrzny trening 24 obrazów używał prefiksu do 300. Walidacja 6 innych obrazów od 300 do 500 wybrała LF10, beta 1, epokę 5 według średniej pięciu J128 RMSE: 0,2379735282. Wszystkie 4 ustalone konfiguracje ukończyły 20 epok. Następnie wykonano refit 30 obrazów do 500 od nowej inicjalizacji przez 5 epok. Zamrożenie nastąpiło o 13:17:21, test ukończono o 13:18:22. Prognoza 50 kroków od 500 do 1000 koduje origin raz; później używa tylko A i decoderu, bez przyszłych pomiarów.

**Cel nie został osiągnięty.** Na H500, w czterech starych holdoutach, znak wszystkich J128 wynosi 54,4975 procent, RMSE 0,448238 wobec 0,233894 dla braku zmiany. MSE jest 3,6727 razy większy niż dla persystencji, czyli MSEskill=−267,2655 procent. Żaden z 50 horyzontów H10..500 nie osiąga 85 procent znaku ani dodatniego MSEskill w J128 lub full768. Rozpoznanie ROI 92,83 procent z poprzedniego eksperymentu jest inną metryką i pozostaje odrębnym wynikiem.

Źródła ukończonego wyniku: [raport NNDMD](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/RAPORT_NNDMD_FULL768.md), [metryki](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/NNDMD_METRYKI_HOLDOUT.json), [zamrożenie](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/ZAMROZENIE_NNDMD.json).

[Niezależny audyt Astry](../../SmallViTAblation/experiments/maly_vit_pykoopman_30_obrazow_128/AUDYT_NNDMD_ASTRA.md) potwierdził wybór z historii walidacji, hashe freeze, 2400 wartości metryk (maksymalna różnica 8,88×10⁻¹⁶) oraz bitową zgodność J128 z pełnym decoded wyjściem. Ręczny replay CPU wszystkich 60 tokenów tool przez 50 kroków różnił się od GPU float32 maksymalnie o 1,05×10⁻⁵. Wersję faktycznie uruchomionego treningu odzyskano jako dokładny snapshot zgodny ze startowym SHA; późniejsza zmiana dotyczyła dodatkowego zapisu J128 i docstringów, nie modelu lub wyboru. Bez nowych treningów i bez GPU w audycie.

Równy wymiar encodera i decoderu nie dowodzi odwracalności ani poprawnej nieliniowej linearyzacji. Przy odwracalnym liniowym C decoded rollout ma postać `(C A C^-1)^h R(z0)`; przy idealnej rekonstrukcji R=id jest liniowy w oryginalnych kanałach. Błąd rekonstrukcji H0 w J128, RMSE 0,058715, pozostaje osobną diagnostyką. To jeden eksploracyjny bieg, znane wcześniej holdouty i ograniczona siatka; wynik nie obala wszystkich wariantów Koopmana. Nie wykonano strojenia po obejrzeniu końcowego testu.

Wcześniejsze zapisy „ekstrakcja768 zlecona”, „NN wynik nieznany” i „learned128 propozycja” opisują chwile poprzedzające to uzupełnienie. Aktualny stan rozstrzygają powyższe ukończone artefakty.
