# Co poprawić w artykule IEEE - notatki

Data: 2026-01-11

---

## Najważniejsze problemy (recenzent na pewno się przyczepi)

### 1. Liczby w pipeline się nie zgadzają

W artykule piszę że wybieram 5000 obrazów, potem oversampling 2000, a na końcu mam 20 000. To się nie spina matematycznie. Recenzent od razu to zauważy.

**Co zrobić:**
- Narysować prosty schemat: ile obrazów wchodzi → ile wychodzi na każdym etapie
- Np. 90k → Fourier (80k) → DINO+K-Center (25k) → EL2N (20k)
- Musi się zgadzać!

---

### 2. Porównanie DETR vs YOLO jest nieuczciwe

Trenuję DETR na innych GPU, z innym batch size, innymi augmentacjami niż YOLO. Recenzent powie: "to nie jest porównanie architektury, tylko całego pipeline'u treningowego".

**Co zrobić:**
- Dodać sekcję "Fair Comparison" gdzie wyjaśniam:
  - Ten sam split danych
  - Te same definicje (IoU threshold, klasa)
  - Porównanie po liczbie kroków optymalizacji, nie epokach
- Albo przyznać wprost: "porównujemy best-practice dla każdej architektury"
- Złagodzić wnioski - zamiast "YOLO jest gorsze" napisać "w naszej konfiguracji YOLO wykazuje problemy"

---

### 3. Brak powtórzeń eksperymentów

Wszystko robione z jednym seedem (42). W medycynie recenzenci często wymagają żeby pokazać że wyniki są stabilne.

**Co zrobić:**
- Uruchomić główne eksperymenty 3-5 razy z różnymi seedami
- Pokazać średnią ± odchylenie standardowe
- Przynajmniej dla: DETR_20k, YOLO, cross-dataset

---

### 4. Metryki cross-dataset są mylące

W jednym miejscu piszę F1=15% dla DETR, w innym mAP=81%. Wygląda jak sprzeczność, choć to pewnie różne modele/etapy.

**Co zrobić:**
- Wprowadzić jasne nazwy: DETR_full, DETR_20k, DETR_20k_finetune
- Trzymać się ich w całym artykule
- Dla każdego modelu raportować te same metryki (mAP I F1)

---

### 5. YOLO 0% na zewnętrznym zbiorze - to wygląda podejrzanie

Wynik 0% mAP jest tak ekstremalny, że recenzent może pomyśleć że coś jest nie tak z konfiguracją.

**Co zrobić:**
- Sprawdzić jeszcze raz czy wszystko OK
- Wyjaśnić DLACZEGO tak się dzieje (YOLO nauczył się cech specyficznych dla jednego szpitala?)
- Dodać przykładowe obrazki pokazujące co YOLO "widzi" vs czego nie wykrywa
- Zmienić język z "totalna porażka" na "brak transferu w naszym protokole"

---

### 6. Query 81 - liczby się nie zgadzają

Raz piszę że Query 81 robi 96.3% detekcji, a gdzie indziej pojawia się 42-48%. To może być kwestia definicji, ale trzeba to wyjaśnić.

**Co zrobić:**
- Jasno zdefiniować: co to znaczy "detekcja"?
  - Czy to po thresholdzie confidence?
  - Czy to po dopasowaniu do ground truth?
- Zrobić tabelkę: hit-rate dla różnych thresholdów (0.1, 0.3, 0.5, 0.7, 0.9)

---

## Średnio ważne (warto dodać)

### 7. Ile czasu zajmuje selekcja danych?

Chwalimy się że pipeline wybiera lepsze dane, ale nie mówimy ile to trwa. Czy zysk z mniejszego zbioru rekompensuje czas na selekcję?

**Co zrobić:**
- Zmierzyć czas każdego etapu
- Pokazać bilans: "selekcja zajęła X godzin, ale zaoszczędziliśmy Y godzin treningu"

---

### 8. Dlaczego nie testowaliśmy RT-DETR?

Wspominamy o RT-DETR (szybsza wersja) ale go nie testujemy. Recenzent może zapytać czemu.

**Co zrobić:**
- Napisać wprost: celem było zbadanie mechanizmu atencji na standardowym DETR
- Optymalizacja prędkości to przyszła praca

---

### 9. Czy wybrane 20k to nie są te same obrazy po augmentacji?

Mamy 4670 oryginalnych → 90k po augmentacji → wybieramy 20k. Ale czy te 20k to różne oryginalne obrazy, czy może 5 wersji tego samego?

**Co zrobić:**
- Sprawdzić ile unikalnych oryginalnych obrazów jest w wybranych 20k
- Jeśli pipeline działa dobrze, powinno być dużo różnych

---

### 10. Brak info o etyce/zgodach

To są nagrania z operacji. Recenzent może pytać: czy macie zgodę? czy dane są anonimowe?

**Co zrobić:**
- Dodać 3-5 zdań:
  - Dane są anonimowe
  - Zgoda IRB (lub wyjaśnienie że nie dotyczy bo dataset publiczny)
  - Czy dane będą udostępnione

---

### 11. Dlaczego używamy MAX w EL2N?

Adaptujemy EL2N do detekcji biorąc MAX po query. Recenzent może zapytać: czemu nie średnia?

**Co zrobić:**
- Dodać krótką ablację: max vs mean vs coś innego
- Pokazać że max działa najlepiej (lub wyjaśnić intuicję)

---

## Drobiazgi (łatwe do naprawienia)

### 12. Formatowanie

- Usunąć "xxxx 00, 0000" i "VOLUME 4, 2016" - to placeholdery szablonu
- Poprawić literówki w nazwiskach
- Tabela 16 jest ucięta

### 13. Język

- Abstrakt za długi - skrócić do ~200 słów
- Zamienić "winner-take-all" na coś bardziej formalnego
- Wybrać jedno: "Q81" albo "Query 81" i trzymać się tego

### 14. Definicja zadania

- Dodać obrazek pokazujący co to jest "tooltip" - czy to sam czubek narzędzia czy całe?
- Jak wygląda bounding box w adnotacjach?

### 15. Podkreślić główny wynik

- 40× poprawa na zewnętrznym zbiorze to jest MEGA wynik
- Powinien być na początku abstraktu, nie ukryty gdzieś w środku

---

## Podsumowanie

Artykuł jest dobry, ale ma dziury które recenzent na pewno znajdzie. Najważniejsze:

1. **Spójność liczb** - musi się zgadzać matematycznie
2. **Fair comparison** - albo wyrównać warunki, albo przyznać że porównujemy "best practice"
3. **Multi-seed** - bez tego w medycynie ciężko
4. **Jasne nazewnictwo** - DETR_full vs DETR_20k vs DETR_20k_ft

Reszta to polish, ale te 4 rzeczy są krytyczne.

---

*Szacuję ~20-30h pracy żeby to wszystko ogarnąć*
