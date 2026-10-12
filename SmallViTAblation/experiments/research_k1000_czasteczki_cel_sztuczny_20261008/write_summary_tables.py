from pathlib import Path
OUT = Path(__file__).resolve().parents[2] / 'outputs' / Path(__file__).parent.name
claims = '''# Tezy, dowody i testy obalenia

Data: 2026-10-08. Oznaczenia CPU, REPO, LIT, WNIOSEK i HIPOTEZA zgodne z raportem. R1–R24 odsyłają do bibliografii. Testy poza opisanymi pomiarami CPU są propozycjami.

| ID | Teza do oceny | Dowód za lub przeciw | Status | Minimalny test obalenia lub decyzja |
|---|---|---|---|---|
| C01 | Historyczny pełny cel K1000 da się uzyskać zwykłym obrazem | CPU: granice norm2/fc1 7,017–10,237; maksimum after 51,353; 15 661 naruszeń; konieczny RMSE ≥2,50349 | Obalona dla tego celu i zamrożonych wag | Forward poniżej wyprowadzonej granicy wymagałby błędu założeń, hashów lub dowodu. Sprawdzić aparat zamiast zwiększać budżet SD |
| C02 | Każdy nowy oracle pełnych 768D będzie nieosiągalny | Brak ukończonego wyniku nowej rodziny 1500 w migawce; ograniczenia mogą zmienić cele | Nieudowodniona | Jeden nowy osiągalny cel obala uniwersalne twierdzenie negatywne |
| C03 | Ruch wszystkich 768 współrzędnych identyfikuje użyteczną korektę | CPU: ranga fc2 192, kernel 576; 74,647 procent energii historycznej delty w kernelu; lokalna ranga Jacobianu głowy ≤5 | Surowa odległość jest nieidentyfikowalnym wskaźnikiem zastępczym | Rozkład row/null, perturbacje o dopasowanej normie, wyjścia głowy i użyteczność |
| C04 | Adam nie rusza kernelu, bo gradient detekcji należy do rowspace | CPU: syntetyczny pierwszy kierunek Adam ma 26,884 procent energii null; skalowanie współrzędnych zmienia podprzestrzeń | Obalona jako ogólna reguła | Oddzielić gradient, aktualizację i clamp. Kontrola SGD zachowującym rowspace |
| C05 | Wysokie confidence i IoU oracle dowodzą semantyki | REPO: końcówka działa osobno na tokenach. LIT: R2, R11–R13 | Nie wynika z kryteriów | Geometria stałej komórki kontra ranking; FP/AP na nowych obrazach i specyficzność perturbacji |
| C06 | Średnia delta per pozycja generalizuje | Możliwy wspólny bias, ale pozycja nie zapewnia zgodności semantycznej; przeciwne ruchy się znoszą | Hipoteza | Podział po filmach; świeże filmy dla detektora; kontrole zero, stały adapter i korektor z kontekstem |
| C07 | Reżim oracle da się przypisać z naturalnego z0 i kontekstu | Plan rozdziela opisowy klaster oracle od przypisania bez GT | Hipoteza | Klasteryzacja tylko train; przypisanie holdout bez GT/delty; regresja bez klastrów |
| C08 | Klastry opisują semantykę | Fazy LR, przełączenie celu i clamp mogą tworzyć podobne tory | Nieustalona | Zmiana schedule/optimizera; niezależne etykiety; kontrola przypisania z samego k/LR/fazy |
| C09 | Trajektoria oracle opisuje trajektorię uczenia | Optymalizacja z przy stałych wagach i aktualizacja wag AdamW mają różne operatory przejścia | Transfer nieuzasadniony | Porównanie rzeczywistych trajektorii uczenia ze zgodnego pełnego stanu |
| C10 | DMDc daje samodzielną prognozę H10 ze znanego schedule | REPO: zapisane przyszłe siły. LIT: R3, R15 | Nie | H10 bez przyszłych pomiarów albo gradienty ponownie liczone na własnych przewidywanych stanach |
| C11 | 95–97 procent oznacza skuteczność detekcji | REPO: skill = 1−SSE/SSE_no-change; pooled representation lub kilka sond | Błędna interpretacja | SSE i baseline oraz RMSE, błąd widocznego ruchu, FP/AP |
| C12 | 98,5 procent H10 przewiduje przyszłość ze znanych paczek | REPO: siły policzono w prawdziwych przyszłych stanach | Błędna interpretacja | Prognoza z bieżącego stanu/schedule bez przyszłych gradientów |
| C13 | Około 99 procent wariancji POD zapewnia prognozę ruchu | REPO: 98,98 procent wariancji stanu kontra skill około 0,5003 dla projekcji przyrostu H1 | Obalona lokalnie | Rekonstrukcja odłożonych przyrostów zamiast wariancji stanów |
| C14 | Dopasowanie z przez SD odtwarza głowę | WNIOSEK: nowy obraz zmienia residual r; oracle tail używał starego r | Nie | z, stan po bloku i wyjście zwykłego pełnego forwardu bez cache |
| C15 | DINO cosine przewiduje użyteczność treningową | REPO: osobny K8000, 8 podmian, cosine efektu do celu 0,0214 i guard AP −0,0052. LIT: R18, R21 | Brak gwarancji; lokalny kontrprzykład | Podobieństwo kontra przewidywana/rzeczywista użyteczność AdamW; random/DINO/gradient |
| C16 | Niezależne delty oracle są wspólnie osiągalne przez wagi | Jedna Delta_theta musi spełnić ograniczenia wszystkich sond | Hipoteza | Reszta projekcji JVP/VJP, zgodność sond i rzeczywiste krótkie aktualizacje |
| C17 | Aktualizacje AdamW per przykład można sumować | Mianownik zależy od pełnej paczki i momentów; clipping oraz BN | Nie w ogólności | Aktualizacja paczki kontra suma wirtualnych aktualizacji per przykład |
| C18 | Użyteczny ruch H1 ma trwały efekt | REPO: efekty przejściowe i feedback. LIT: R15, R16 | Nieustalona | Sparowane kontynuacje 1/5/10/25/50, wspólne przyszłe paczki/RNG, własne gradienty |
| C19 | Cele z etykiet i inwersja są z zasady bezużyteczne | LIT: R8 LabelEnc, R9 DeepInversion, R7/R23 distillation | Fałszywe uogólnienie | Strata pomocnicza i transfer do studenta na nowych danych |
| C20 | Porażka w 500 krokach dowodzi braku rozwiązania | Budżet optymalizacji nie jest dowodem nieistnienia | Nie | Niezależne granice analityczne lub inna optymalizacja; etykieta budget_exhausted |
| C21 | Wszystkie 900 000 cząsteczek mają pełną trajektorię | Plan: pełne stany 0/25/100/500 oraz gęsty panel 256 | Nie dla gęstej dynamiki całości | Zakres twierdzenia zgodny z próbkowaniem; rzadkie milestone nie identyfikują częstotliwości |
| C22 | Implementacja zachowuje kontrakt FP32 milestone | REPO, migawka: writer i test float16, PLAN FP32 | Niespójność odczytanej wersji | Dtype względem planu, hash kodu i błąd kwantyzacji |
| C23 | GroupKFold 1500 daje niezależny test detektora | K1000 trenowano na tych 1500; holdout korektora nie cofa historii detektora | Nie | Nowe filmy nieużyte przez K1000, generator i selector |
| C24 | SD→DINO poprawia koszt uczenia | Brak porównania całego łańcucha z kosztem selekcji. LIT: R18, R24 | Hipoteza | Czas do tego samego AP przy równym budżecie kontra retrieval, gradient i random; koszt wszystkich etapów |

Migawki i hashe: evidence/CPU_AUDIT.json. Bibliografia: BIBLIOGRAFIA.md. Porządek decyzji: bramki G0–G7 w RAPORT.md.
'''
summary = '''# Rekomendacje przed kosztownymi obliczeniami

Najpierw oddzielić cel oracle od celów osiągalnych przez obraz i wspólne wagi. Research wykazał nieosiągalność konkretnego historycznego pełnego celu K1000. Każdy nowy cel wymaga osobnego audytu.

1. **Sprawdzić cel i zapis.** Konieczne granice norm2/fc1, rozkład ruchu na rowspace/nullspace, zgodność FP32 z writerem. Panel 256 oznacza 256 cząsteczek łącznie.
2. **Mały pilotaż oracle.** Na 32–64 reprezentatywnych obrazach porównać pełne 768D, ograniczenia naturalnej mapy i minimalną korektę widoczną dla głowy. Zachować budget_exhausted w mianowniku.
3. **Transfer bez GT.** Średnia per token jako kontrola; regresja z kontekstem bez klastrów, reżimy, zero oraz adapter uczony stratą detekcji. Podział całymi filmami. Końcowe filmy testowe niezależne od treningu K1000.
4. **Cel zachowania.** Mierzyć W2 delta, residual/stan po bloku i użyteczność głowy. Naturalne z zachować jako diagnostykę. Rzut rowspace nie zapewnia naturalności.
5. **Osiągalność przez wagi.** Zgodne wagi, AdamW, BN i liczniki; asset net/krok nie wystarcza. JVP/VJP na małym panelu i rzeczywiste kroki na kopiach pełnego stanu.
6. **Ranking paczek.** Random, DINO i gradient z uwzględnieniem optimizera; ranking przed pomiarem. Sparowane kontynuacje 1/5/25/50, wspólne przyszłe paczki, osobno przeliczane gradienty.
7. **SD po wcześniejszych bramkach.** Porównać zapytanie z SD z naturalnym/original query i selekcją gradientową. Dopasowanie z nie zapewnia zgodności residual ani głowy. Zwykły pełny forward jest obowiązkowy.
8. **Duży bieg ze wspólnym kontraktem metryk i kosztu.** Kilka seedów/pul, nowe filmy, AP/FP/recall i strata detekcji. Czas wszystkich etapów, w tym selekcji i generatora, do tego samego docelowego AP.

Alternatywy do pilotażu: nadzór pośredni inspirowany LabelEnc, mały korektor z kontekstem, selector użyteczności walidacyjnej i ważone dopasowanie gradientów. Literatura pokazuje sukcesy tych klas; wynik dla SmallViT pozostaje pytaniem empirycznym.

Wykonano odczyty, pełnotekstowy przegląd i audyt CPU. Proponowane treningi, kontynuacje i generacja SD nie zostały uruchomione.

Raport: RAPORT.md. Tezy i testy: CLAIMS_EVIDENCE_FAILTEST.md. Źródła: BIBLIOGRAFIA.md.
'''
for name, content in [('CLAIMS_EVIDENCE_FAILTEST.md',claims),('REKOMENDACJE.md',summary)]:
    (OUT/name).write_text(content, encoding='utf-8')
print('Zapisano zredagowane tezy i rekomendacje.')
