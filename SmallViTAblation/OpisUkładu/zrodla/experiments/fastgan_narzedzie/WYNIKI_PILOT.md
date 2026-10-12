# Lokalna próba FastGAN, 2026-09-23

Status: zakończona poprawnie. Proces zakończony; karta zwolniona. RF-DETR i jego środowisko pozostały nietknięte.

## Pytanie

Czy FastGAN z maską zmieści się na lokalnym RTX 3070 i ile trwa krok treningu?

## Aparat

1044 wycinki z 116 operacji treningowych, rzeczywiste maski `cystotome_seg`, oryginalne piksele zgodne z manifestem. Losowa inicjalizacja, szerokość 64, rozdzielczość 256 × 256, batch 2, FP32. Wagi modelu polipów nie były używane. Osobne środowisko `F:\Studia\PhD_projekt\VIT\venvs\fastgan_local`, torch 2.6.0+cu124, LPIPS 0.1.4.

Kod adaptuje funkcję straty: LPIPS na RGB i L1 maski w samorekonstrukcji dyskryminatora. Wspólne odbicie przestrzenne maski i obrazu, bez augmentacji kolorów maski. To wariant techniczny FastGAN, nie dokładna reprodukcja pracy.

## Wynik

- 200 kroków zakończonych, wszystkie zapisane straty skończone.
- Czas pętli wraz z zapisem końcowym: 58,52 s; bez przygotowania danych, pobierania VGG i inicjalizacji.
- Mediana po pominięciu rozgrzewki: 0,2483 s/krok; percentyl 90: 0,2745 s/krok.
- Szczyt pamięci przydzielonej przez PyTorch: 1318,35 MiB.
- Szczyt pamięci zarezerwowanej przez PyTorch: 1598 MiB. To nie jest całkowita pamięć wszystkich procesów karty ani narzut sterownika.
- Ekstrapolacja do 50 tysięcy kroków: 3,45 h samej pętli przy porównywalnym obciążeniu. Nie uwzględnia częstszego zapisu, oceny i rywalizacji z RF-DETR.

## Kontrola i granice

Sprawdzono obraz z nałożoną maską, zapis wszystkich kroków, skończoność strat i istnienie checkpointu. Sprawdzono podgląd generacji: obrazy są rozmyte i nie przedstawiają jeszcze wiarygodnie narzędzia. To oczekiwany brak wyuczenia po krótkiej próbie, nie dowód zbieżności.

Nie ma ramienia kontrolnego jakości, ponieważ celem jest wyłącznie test wykonania i czasu. Nie wykazano poprawnej segmentacji generowanych obrazów, zachowania tożsamości, odwracania latentu ani poprawy treningu detektora. Koszt pełnej metody z optymalizacją obrazu pod latent będzie dodatkowy.

## Artefakty

Katalog `wyniki/pilot_20260923_b2`: `REPORT.json`, `KROKI.jsonl`, `CONFIG.json`, `KONTROLA.json`, `pilot.pth`, podglądy RGB i maski. `KONTROLA.json` zawiera sumy kontrolne kodu oraz manifestu wycinków. Surowe wyniki i dane są pomijane przez Git.

Dłuższego treningu ani zadań Eden nie uruchomiono.
