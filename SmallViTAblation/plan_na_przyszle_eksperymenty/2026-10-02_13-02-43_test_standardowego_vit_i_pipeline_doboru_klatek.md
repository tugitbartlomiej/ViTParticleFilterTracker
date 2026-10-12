# Test standardowego ViT i pipeline doboru prawdziwych klatek

Utworzono: 2026-10-02T13:02:43+02:00 (Europe/Warsaw). Status: plan przyszłego eksperymentu, bez uruchomienia nowego treningu.

### 9.3. Test zwykłej sieci i pipeline: konkretne ramiona

Celem kroku 7 jest kontrolowany test, czy metoda poprawia uczenie detektora, a nie sama korelacja wysokiego score retrieval z wynikiem. Telemetria jest obowiązująca już w kroku 6. Każde ramię ma oddzielny run i checkpointy; nie wykonujemy kolejnych metod na tych samych już zmienionych wagach jako rzekomego porównania.

| Ramię | Dobór danych i znaczenie |
|---|---|
| A — standardowy ViT | Z góry ustalony losowy podzbiór train, dostępny od początku; zwykły shuffle i mini-batche, bez adaptacyjnego doboru. Docelowa liczba unikalnych obrazów dopasowana do budżetu F lub jawnie oznaczona jako kontekst przy innym budżecie. |
| B — progressive uniform random | W każdej rundzie losujemy bez zwracania z eligible train dokładnie n_r nowych klatek, gdzie n_r jest rzeczywistą zaakceptowaną liczbą F w sparowanym seedzie, 0–100. Nie dobieramy n_r do wyników B. |
| C — sequential | Dokładnie n_r kolejnych klatek według wcześniej ustalonego porządku filmów i frame-ID; przestrzegamy eligibility, deduplikacji i wykluczeń. Nie zmieniamy filmu/kolejności po wynikach. |
| D — direct real retrieval | Pełnoobrazowy DINO retrieval z rzeczywistego zapytania, bez SD i bez prognozy. Kotwice wybieramy regułą niezależną od prognozy, zamrożoną przed runem; n_r i budżet dopasowane do F. To brakująca wcześniej kontrola korzyści samego real retrieval. |
| E — SD current target, no forecast | Czysty szum SD i cel z obecnego stanu, bez prognozy. Dobór kotwic również niezależny od prognozowanego ruchu. Reszta prowadzenia/retrieval i n_r jak F. Historyczne „SD bez prognozy” z kotwicami wybieranymi za pomocą prognozy nie jest tą czystą ablacją. |
| F — pełny pipeline | NNDMD pełne 768 → prognozowany stan → SD z czystego szumu → pełnoobrazowy DINO ranking eligible train. Cosine>0,85, najwyżej 100, bez dopełniania. Wybrane prawdziwe obrazy uczą ViT. |

**Etap minimalny: A/B/F.** Najpierw odtwarzalny pilot porównawczy z uzgodnionym budżetem, później planowane 10 sparowanych seedów. Liczba 10 jest planem, nie liczbą wykonanych replik. Etap następny, gdy zasoby pozwolą: D i E jako kluczowe ablacje, następnie C. Nie uruchamiamy teraz wszystkich ramion ani nowego treningu ViT; zapisujemy protokół do wykonania.

Kontekstowo można dołączyć checkpoint bez dalszego uczenia/no-add, standardowe pretrained initialization oraz zwykłe uczenie na całym dostępnym train. Uczenie na całym train ma większy budżet etykiet/danych, a pretrained może korzystać z innej inicjalizacji; **nie są głównym równobudżetowym dowodem przewagi**. Nie mieszać takich liczb z A/B/F bez oznaczenia.

### 9.4. Warunki porównywalności i zamrożenie

W głównych ramionach zachowujemy tę samą architekturę, stan inicjalny i wspólny warmup checkpoint, sparowane seedy, AdamW/lr/scheduler, batch/gradient accumulation, augmentacje, kryteria detekcji, train-only pulę i niezmienny podział filmów. Te same seedy nie gwarantują identycznych augmentacji przy różnej kolejności ID; runtime zapisuje rzeczywiste parametry. Validation jest stałe, a final test odłożony po filmach.

Budżet optimizer-step jest wspólny. B/C/D/E dopasowują rzeczywistą liczebność nowych klatek F per runda, a logi pokazują prezentacje old/new oraz unikalne obrazy. Gdy F dodaje 0, kontrola nie dodaje nowych obrazów; wszystkie ramiona przestrzegają tej samej wcześniej ustalonej reguły aktualizacji rundy. Arm A ma stały zestaw od początku i inny harmonogram dostępności: to standardowa kontrola konwencjonalna, a B jest główną kontrolą efektu adaptacyjnej selekcji przy dopasowanej dostępności/liczebności.

Czas mierzymy end-to-end: trening ViT, NNDMD/prognoza, generacja SD, DINO/index/retrieval, walidacja i I/O. Oprócz optimizer-step raportujemy GPU-hours/wall-clock i nakład danych. Dla ustalonego wcześniej celu mAP mierzymy aktualizacje i czas do jego osiągnięcia; jeśli nieosiągnięty, zapisujemy censored/budget_exhausted, nie interpolujemy dowolnego sukcesu.

Przed runami zamrażamy budżet, liczbę rund/aktualizacji, harmonogram oceny, próg retrieval i SD acceptance, metryki, primary contrast F−B, reguły A i kotwic D/E, seedy oraz zasady awarii/zero-add. Nie usuwamy nieudanych runów. Final test odczytujemy raz po zamrożeniu protokołu i checkpoint-selection; nie używamy go do dalszego doboru metody.

### 9.5. Metryki, learning curves i niepewność

Główna miara to **sparowana różnica znormalizowanego pola pod krzywą mAP względem optimizer-step**, F−B, na wspólnym zakresie kroków i ustalonych punktach walidacji. AUC odnosi się tu do krzywej uczenia mAP, nie do AUROC klasyfikatora obecności. Konkretna definicja mAP, np. AP50 oraz dodatkowo AP50:95, decode/NMS i harmonogram punktów muszą być ustalone przed runem. Oprócz primary contrast pokazujemy F−A oraz późniejsze D/E/C bez wybierania najkorzystniejszego kontrastu po teście.

Jednostką replikacji jest **cały seed-run**. CI liczymy bootstrapem kompletnych sparowanych seedów, zachowując wspólne różnice ramion; tokeny, obrazy i rundy nie zastępują niezależnych treningów. Przy pilocie z małą liczbą seedów wynik jest opisowy. 10 seedów to cel planu; nie przedstawiamy niewykonanych replik ani przedziałów jako danych. Ewentualny bootstrap filmów oceny jest dodatkową niepewnością zbioru, nie zamiennikiem zmienności treningu.

Raportujemy final mAP, validation loss i training loss, precision/recall, IoU oraz FP/FN. Krzywe pokazujemy względem optimizer-step, liczby unikalnych obrazów, **łącznej liczby prezentacji** i runtime. Zmiana loss na zmieniającej się puli nie wystarcza do dowodu poprawy generalizacji; służy diagnostyce obok stałego validation.

Przed pomiarem ustalamy warstwy trudności validation: kąty/orientacje narzędzia, rozmycie, wielkość/położenie i sceny tła. Raportujemy wyniki, liczebności oraz FP/FN na tle; bez tworzenia warstw według tego, gdzie metoda akurat wygrywa. Monitorujemy rozkład wybranych klas/filmów/kątów i ekspozycje old/new: similarity sampling może zubożyć różnorodność. Nie gwarantujemy przewagi pipeline ani uzyskania zakładanego mAP.

### 9.6. Pierwotne źródła uzasadniające kontrole

Yoo i Kweon porównują dobór aktywny z random sampling, utrzymując te same hiperparametry modeli i raportując powtórzenia. To uzasadnia potrzebę silnej losowej kontroli, a nie przewagę naszego pipeline. Odczyt: artykuł HTML i fragmenty sekcji 4 porównań/obiektów detekcji; CVF PDF zwrócił 403, więc użyto autorskiego PDF i arXiv. [Learning Loss for Active Learning, CVPR 2019](https://arxiv.org/html/1905.03677v1), [autorski PDF](https://dgyoo.github.io/papers/cvpr19.pdf).

Emam i współautorzy pokazują, że dobór aktywny może nie przewyższać random, a imbalance wybranych klas jest istotnym problemem przy ImageNet. Uzasadnia to kontrolę rozkładu próbek i jakości na stałych warstwach; nie dowodzi takiego samego efektu dla naszej detekcji narzędzia. Odczyt: abstract, opis protokołu i sekcje 5.1.1–5.1.2. [Active Learning at the ImageNet Scale](https://arxiv.org/html/2111.12880v1).


## Telemetria i powiązane dokumenty

Dokładne frame-ID, moment added vs actual sampled, augment/RNG/checkpoint oraz walidacja before/after rundy są wymagane już podczas uczenia.

- [Pełny pipeline](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/PIPELINE_TOZSAMOSCI_NARZEDZIA.md).
- [Plan telemetrii i walidacji](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-59-31_telemetria_uzytych_klatek_uczenie_i_walidacja_rund.md).
- [Aktualny SD i retrieval](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/plan_na_przyszle_eksperymenty/2026-10-02_12-49-00_sd_z_czystego_szumu_prowadzenie_prognozowanym_latentem_768.md).
