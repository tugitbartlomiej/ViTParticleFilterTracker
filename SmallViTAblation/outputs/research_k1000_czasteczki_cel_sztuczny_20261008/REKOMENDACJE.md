# Rekomendacje przed kosztownymi obliczeniami

Najpierw oddzielić cel oracle od celów osiągalnych przez obraz i wspólne wagi. Research wykazał nieosiągalność konkretnego historycznego pełnego celu K1000. Każdy nowy cel wymaga osobnego audytu.

1. **Sprawdzić cel i zapis.** Konieczne granice norm2/fc1, rozkład ruchu na rowspace/nullspace, zgodność FP32 z writerem i wznowieniem. Końcowy odczyt: writer już float32, panel 128 globalnie, budżet tensorów około 14,611 GB. Wciąż sprawdzić miejsce i kontrakt przed uruchomieniem.
2. **Mały pilotaż oracle.** Na 32–64 reprezentatywnych obrazach porównać pełne 768D, ograniczenia naturalnej mapy i minimalną korektę widoczną dla głowy. Zachować budget_exhausted w mianowniku.
3. **Transfer bez GT.** Średnia per token jako kontrola; regresja z kontekstem bez klastrów, reżimy, zero oraz adapter uczony stratą detekcji. Podział całymi filmami. Końcowe filmy testowe niezależne od treningu K1000.
4. **Cel zachowania.** Mierzyć W2 delta, residual/stan po bloku i użyteczność głowy. Naturalne z zachować jako diagnostykę. Rzut rowspace nie zapewnia naturalności.
5. **Osiągalność przez wagi.** Zgodne wagi, AdamW, BN i liczniki; asset net/krok nie wystarcza. JVP/VJP na małym panelu i rzeczywiste kroki na kopiach pełnego stanu.
6. **Ranking paczek.** Random, DINO i gradient z uwzględnieniem optimizera; ranking przed pomiarem. Sparowane kontynuacje 1/5/25/50, wspólne przyszłe paczki, osobno przeliczane gradienty.
7. **SD po wcześniejszych bramkach.** Porównać zapytanie z SD z naturalnym/original query i selekcją gradientową. Dopasowanie z nie zapewnia zgodności residual ani głowy. Zwykły pełny forward jest obowiązkowy.
8. **Duży bieg ze wspólnym kontraktem metryk i kosztu.** Kilka seedów/pul, nowe filmy, AP/FP/recall i strata detekcji. Czas wszystkich etapów, w tym selekcji i generatora, do tego samego docelowego AP.

Alternatywy do pilotażu: nadzór pośredni inspirowany LabelEnc, mały korektor z kontekstem, selector użyteczności walidacyjnej i ważone dopasowanie gradientów. Literatura pokazuje sukcesy tych klas; wynik dla SmallViT pozostaje pytaniem empirycznym.

Wykonano odczyty, pełnotekstowy przegląd i audyt CPU. Proponowane treningi, kontynuacje i generacja SD nie zostały uruchomione.

Raport: [RAPORT.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/RAPORT.md). Tezy i testy: [CLAIMS_EVIDENCE_FAILTEST.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/CLAIMS_EVIDENCE_FAILTEST.md). Źródła: [BIBLIOGRAFIA.md](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/outputs/research_k1000_czasteczki_cel_sztuczny_20261008/BIBLIOGRAFIA.md).
