# Cztery niezależne szumy

Pytanie: czy ograniczone prowadzenie przez oryginalny sztuczny latent poprawia dopasowanie naturalnych obrazów SD do celu dla czterech niezależnych szumów?

Ziarna ustalone przed biegiem: 2026100511, 2026100523, 2026100537, 2026100549. Wszystkie starty to niezależny standardowy szum Gaussa 1×4×48×72, zapisany przed generacją. Bez kodowania fotografii. Ten sam cel `after` 600×768, channels0, zamrożony checkpoint, lokalny UNet EMA 020000, VAE, pusty tekst i trzy zerowe maski.

Podłoga: dla każdego ziarna osobny DDIM bez prowadzenia. Zachowuje dokładnie ten sam szum, model SD, sampler, 60 kroków, tekst i maski. Tylko prowadzone ramię otrzymuje sztuczny cel; to badana interwencja, a nie równa informacja przy przypisaniu etykiet. Brak treningu parametrów. Cztery ziarna są jednostkami oceny; nie ma niezależnego zbioru zdjęć ani twierdzenia o generalizacji.

Prowadzenie inspirowane nowym img2img: dwie korekty clean VAE na krok, backtracking .006/.002/.000667, akceptacja tylko spadku MSE768 z karą za odchylenie od bieżącej propozycji SD. RMS odchylenia ograniczony do .012 na krok. Kara .005 razy kwadrat względnego odchylenia. Prowadzenie tylko indeksy 30–56; ostatnie trzy kroki bez prowadzenia. Początek denoisingu zachowuje prior SD. Żadnych 120 kroków optymalizacji końcowego VAE ani pikseli. Ograniczenie nie gwarantuje realizmu.

Obalenie: MSE768 nie maleje względem dopasowanej kontroli. Raportujemy także pogorszenia, MSE128, MSE640, cosinus i pewność. MSE naturalnego latentu mierzymy po zapisie/odczycie PNG i przeskalowaniu jak we wcześniejszych próbach. Różnorodność: wszystkie sześć par w każdym ramieniu, RMS pikseli, MSE i cosinus naturalnych latentów. Nie selekcjonujemy najlepszego ziarna.

Cel dokładnie nieosiągalny dla checkpointu według `DIAGNOSTYKA_OSIAGALNOSCI.json`. Każdy wynik jest przybliżeniem. Pewność detektora nie jest mAP. Wagi sprawdzane hashem przed i po.

Przebieg szeregowy na RTX 3070. Modele ładowane raz na CPU, przenoszone między CPU/GPU, attention slicing i checkpointing VAE. Start czeka w tle na wolne 3500 MiB; nie kończy cudzych procesów. Atomowy PROGRESS.json, oddzielne logi, pełne szumy, obrazy, VAE i naturalne latenty, provenance oraz sumy SHA256. Znane brudne drzewo: lokalny helper `_brudne_drzewo=lambda:True`, bez kosztownego skanu danych. Bez commitów.

