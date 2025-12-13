# IEEE/IEEE Access writing audit for `access.tex`

## Źródła (PDF-y)
- `F:\Studia\Articles\Zbiory\HowToWrite\Immersive_Teaching_Enhancing_Ability_of_Scientific_Paper_Writing_A_Case_Study_in_the_Fields_of_Instrumentation_and_Measurement.pdf`
- `F:\Studia\Articles\Zbiory\HowToWrite\Jak pisać artykuły naukowe IEEE.pdf`
- `F:\Studia\Articles\Zbiory\HowToWrite\How_to_write_your_first_scientific_paper.pdf`

## Przeanalizowany manuskrypt
- `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`

## Executive summary (co jest OK / co blokuje submission)

**Mocne strony (zgodne z zaleceniami z PDF-ów):**
- Masz pełną architekturę IMRaD (Introduction/Methods/Results/Discussion) + Conclusion i bibliografię.
- Wstęp ma logikę „lejka” i kończy się RQ + listą wkładów (to jest dokładnie rekomendowane).
- Sekcja metod zawiera równania i pseudokod (plus dla reprodukowalności).
- Wyniki są bogate w tabele/ablacje, a dyskusja omawia trade-offy i ograniczenia.

**Największe blokery (priorytet P0/P1):**
1. **Niespójności w opisie eksperymentów** (np. YOLO: 100 epok/batch 16 vs podpis figury: 200 epok/batch 95 + EMA) — to podważa wiarygodność i łamie zasadę „precyzji metodyki”.
2. **Niespójna liczebność datasetu** („20,000” vs tabela „20,900”) — trzeba ujednolicić i wyjaśnić.
3. **Abstract prawdopodobnie przekracza typowy limit 150–250 słów** (u mnie wychodzi ~257) — do skrócenia.
4. **Keywords/Index terms**: jest ich dużo (11), nie są alfabetycznie i nie odnoszą się jawnie do IEEE Taxonomy (to jest wprost zalecane).
5. **Reproducibility gaps**: brakuje kilku kluczowych parametrów (np. konkretnego `k` w selekcji danych, detali proxy-modelu dla EL2N, progów/ustawień dla „YOLO–DETR consensus” i FastSAM).
6. **Metadane IEEE Access są placeholderami** (autorzy, afiliacje, funding, daty, DOI) — do wypełnienia przed submission.

---

## Checklist wymagań (zmapowane na `access.tex`)

Legenda: ✅ spełnione | 🟡 częściowo | ❌ do poprawy

### 1) Tytuł
**Z PDF-ów:** tytuł ma być zwięzły, deskryptywny, mówić o polu i głównym wkładzie; unikać „A study…/Analiza…” i marketingowych przymiotników.
- `access.tex:21` ✅ Jest deskryptywny i mówi o wkładzie (dataset selection + query specialization + DETR + cataract surgery).
- Sugestia (opcjonalna) 🟡: jest dość długi; rozważ skrócenie, jeśli da się bez utraty informacji.

### 2) Abstrakt
**Z PDF-ów:** abstrakt ma być samowystarczalny, bez cytowań/przypisów/równań; ma zawierać cel→metodę→wyniki (liczbowe)→implikacje; typowo 150–250 słów; nie może być „cliffhangerem”.
- `access.tex:39` 🟡 Bardzo dobry merytorycznie (problem + metoda + liczby), ale **prawdopodobnie jest zbyt długi** (~257 słów) i jest bardzo „gęsty” (wiele tez naraz).
- Do poprawy (P1): skrócić do ≤250 słów, zachowując najważniejsze liczby i 3 wkłady.
- Do poprawy (P2): oddzielić w abstrakcie „wynik” od „hipotezy/interpretacji” (np. linkowanie query specialization → generalization opisać ostrożniej jako możliwe wyjaśnienie).

### 3) Keywords / Index Terms
**Z PDF-ów:** słowa kluczowe sterują discoverability; zalecane użycie IEEE Taxonomy; układ alfabetyczny; nie przesadzać z liczbą.
- `access.tex:43` ❌ Jest 11 haseł i nie są alfabetycznie.
- Do poprawy (P1): zejść do ~6–8 haseł, ułożyć alfabetycznie i dopasować do IEEE Taxonomy (tam, gdzie to możliwe).

### 4) Introduction
**Z PDF-ów:** konstrukcja „lejka”/4 ruchy: (1) kontekst i ważność, (2) SOTA + ograniczenia, (3) luka + wkład (najczęściej w punktach), (4) organizacja pracy; bez wyników.
- `access.tex:51` ✅ Masz RQ + wkłady + organizację; to jest bardzo zgodne z zaleceniami.
- Sugestia (P2): dopilnować, żeby „SOTA” było krytyczną syntezą, a nie listą (masz to częściowo już w Related Work).

### 5) Related Work
**Z PDF-ów:** ma to być krytyczna analiza prowadząca do luki; trzeba cytować kluczowe prace w obszarze (brak cytowań budzi podejrzenia recenzenta).
- `access.tex:91` ✅ Struktura jest dobra (DETR variants, medical imaging, tool detection, dataset curation).
- Do poprawy (P2): doprecyzować, jakie *konkretnie* ograniczenia dotychczasowej literatury adresujesz (1–2 akapity „gap summary”).

### 6) Metodyka / reprodukowalność
**Z PDF-ów:** celem metodyki jest reproducibility; trzeba zdefiniować zmienne/symbole, podać wersje narzędzi, szczegóły sprzętu, parametry; pseudokod jest mile widziany.
- `access.tex:126` 🟡 Mocne: równania (Fourier), algorytm (K-center), jasna idea pipeline.
- Braki (P1/P0):
  - **Brak jawnego `k` / frakcji danych** dla pipeline (opisujesz „top-k”, ale nie mówisz ile to jest).
  - **EL2N**: brakuje szczegółów proxy-modelu (jaki model, ile epok, jaki split, jakie hiperparametry).
  - **Background-aware**: brakuje progów/ustawień dla „YOLO–DETR consensus” i FastSAM (jakie confidence thresholds, jakie kryteria odrzucenia, wersje modeli).
  - Brak wersji software (PyTorch/CUDA) i seedów.

### 7) Experimental Setup
**Z PDF-ów:** eksperymenty muszą być jasno zaplanowane i opisane; wyniki muszą wspierać wkład; porównania mają mieć referencję/baseline.
- `access.tex:294` 🟡 Masz dataset, konfiguracje modeli i metryki.
- Krytyczne niespójności (P0):
  - `access.tex:417` (YOLO config) vs `access.tex:521` (podpis figury: 200 epok/batch 95/EMA) — trzeba ujednolicić.
  - „20,000” vs „20,900” (`access.tex:40` vs `access.tex:391`) — wyjaśnić (np. czy 20,000 to subset bez background/aug?).

### 8) Results
**Z PDF-ów:** results = dane; discussion = interpretacja; nie opisywać każdej liczby z tabel; każdy eksperyment ma mieć cel i porównanie; wyniki muszą dowodzić wkładu.
- `access.tex:362` ✅ Jest dużo ablation i benchmarków.
- Do poprawy (P1): dopilnować, by każda tabela była jednoznacznie „przypięta” do jednego wkładu i miała jasny protokół (split/seed/threshold).
- Do poprawy (P1): sekcja generalizacji (`access.tex:751`) wymaga doprecyzowania protokołu i spójności epok.

### 9) Discussion
**Z PDF-ów:** interpretuj wyniki, skonfrontuj z literaturą, unikaj overclaim/underclaim.
- `access.tex:814` ✅ Dobra struktura (dlaczego DETR przegrywa na strict IoU, clinical implications, recommendations, limitations).
- Do poprawy (P2): ostrożniej oddzielać „dowód” od „hipotezy” (szczególnie w łączeniu query specialization z generalizacją).

### 10) Conclusion
**Z PDF-ów:** nie kopiować abstraktu; synteza wyższego poziomu; praktyczne implikacje i future work; bez nowych danych.
- `access.tex:943` ✅ Jest synteza + future work.
- Sugestia (P2): upewnić się, że nie pojawiają się nowe liczby/tezy tylko w conclusion.

### 11) Figury / tabele / jakość grafiki
**Z PDF-ów:** rysunki min. 300 dpi (zdjęcia) / 600 dpi (wykresy), czytelne fonty, poprawne podpisy.
- `access.tex` 🟡 Nie mogę ocenić DPI bez otwierania plików graficznych, ale podpisy są bardzo rozbudowane (to plus).
- Do zrobienia (P1): checklist graficzny przed submission (DPI, font size, kontrast, spójność stylu).

### 12) Bibliografia
**Z PDF-ów:** ma być kompletna, wszystko cytowane w tekście, numeracja, jakość źródeł.
- `access.tex` ✅ Jest duża bibliografia i klasyczny styl IEEE.
- Do poprawy (P1): szybki sanity-check: czy każde `\bibitem{...}` ma `\cite{...}` i odwrotnie; czy wszystkie pozycje są realne i poprawnie opisane.

### 13) Metadane / gotowość do IEEE Access
**Z PDF-ów:** sprawdzić metadane: tytuł, autorzy, afiliacje, ORCID, funding; użyć szablonu; proofreading; etyka (oryginalność, zgody, disclosure AI/conflicts).
- `access.tex:18`–`access.tex:37` ❌ Są placeholdery (autorzy, afiliacje, funding, daty).
- Do poprawy (P0/P1): wypełnić metadane, dodać (jeśli wymagane) sekcje dot. danych/kodu/konfliktu interesów.

---

## Plan poprawek (priorytety)

### P0 (blokery wiarygodności)
- [ ] Ujednolicić konfigurację YOLO (epoki, batch, EMA) w `access.tex:417` i w podpisach/wykresach (np. `access.tex:521`).
- [ ] Ujednolicić rozmiar datasetu: „20,000” vs „20,900” (i nazewnictwo CADTD/CaDTD).
- [ ] Wyjaśnić protokół i metryki w cross-patient generalization (`access.tex:751`) oraz ujednolicić epoki (e160/e170).

### P1 (submission readiness)
- [ ] Skrócić abstrakt do ≤250 słów (`access.tex:39`).
- [ ] Zredukować keywords do ~6–8, ułożyć alfabetycznie i dopasować do IEEE Taxonomy (`access.tex:43`).
- [ ] Dopisać brakujące parametry reprodukowalności: `k`, proxy-model dla EL2N, progi dla consensus/FastSAM, seedy, wersje software.
- [ ] Checklist grafiki: DPI, czytelność, spójność stylu.
- [ ] Wypełnić metadane IEEE Access (autorzy, afiliacje, funding, daty, DOI placeholdery).

### P2 (wzmocnienie narracji zgodnie z poradami „clarity/flow”)
- [ ] Dodać 1–2 akapity „gap summary” w Related Work (krytyczna synteza, nie lista).
- [ ] Oznaczyć wprost, co jest wynikiem, a co hipotezą (np. query specialization → generalization).
- [ ] Proofreading: skracanie zdań, unikanie wielo-wątkowych fraz; utrzymać „seamless flow”.

---

## Mini-checklista przed submission (z PDF „Jak pisać…”)
- [ ] Metadane kompletne (autorzy/afiliacje/ORCID/funding).
- [ ] Abstrakt samowystarczalny i zawiera liczby.
- [ ] Wstęp jasno definiuje lukę i wkład.
- [ ] Wnioski nie kopiują abstraktu i nie wprowadzają nowych danych.
- [ ] Figury: 300/600 dpi, czytelne, podpisy poprawne.
- [ ] Bibliografia: wszystko cytowane, format IEEE, brak brakujących kluczy.
- [ ] Proofreading + spójny styl językowy.
- [ ] Etyka: oryginalność, zgody, disclosure AI/conflicts (jeśli dotyczy).
