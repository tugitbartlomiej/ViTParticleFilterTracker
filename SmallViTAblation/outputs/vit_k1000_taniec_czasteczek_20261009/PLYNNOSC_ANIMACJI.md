# Płynna animacja i tempo

Zmiana z 2026-10-09 na polecenie użytkownika. Podgląd: http://127.0.0.1:8774/.

- Suwak tempa: 5–250 kroków optymalizacji na sekundę, domyślnie 50.
- Osobny wybór odstępu zapisów: 5, 10 lub 20 kroków dla całej chmury; tempo nie jest tym samym co pomijanie stanów.
- Płynne przejścia są domyślnie włączone. Shader WebGL2 miesza dwa rzeczywiście zapisane stany każdego z 900000 punktów, w tej samej bazie PCA3. Nie powstają dodatkowe naukowe pomiary ani pliki interpolowanych latentów.
- Liniowo interpolowany kolor jest mieszaniem norm końców. Nie oznacza obliczenia rzeczywistej normy nowego wektora w 768D.
- Metryki confidence, IoU, liczności i mapa norm pozostają pomiarami oznaczonego zapisanego kroku. Podczas interpolacji może on być wcześniejszy od bieżącej klatki podglądu.
- Odtwarzanie przez requestAnimationFrame, odczyt kolejnego stanu z wyprzedzeniem i maksymalnie sześć buforów GPU, około 86,4 MB. Wykresy boczne odświeżane co najwyżej około cztery razy na sekundę, zamiast w każdej klatce renderowania.
- Pauza zachowuje położenie między stanami. Wznowienie kontynuuje parę; suwak tempa działa również podczas odtwarzania. Ukrycie karty nie powoduje skoku czasu po powrocie.
- Wyłączenie płynnych przejść pokazuje tylko rzeczywiste stany. Panel kontrolny i widok jednego obrazu nie otrzymują dodatkowych interpolowanych danych.

W browserze rzeczywiście sprawdzono 900000 rysowanych punktów, brak błędów GL, przejścia z ułamkowym krokiem, pauzę w środku przejścia i wznowienie, zmianę tempa z 5 do 250 w trakcie, odstęp 20, wyłączenie interpolacji i działanie panelu kontrolnego. Na RTX3070 po rozgrzaniu obserwowano około 59 klatek/s; pierwsze wczytanie i najszybsze tempo mogą ograniczać faktyczną szybkość. To pomiar tej maszyny, nie gwarancja FPS na dowolnym GPU.

Manifest rzeczywistych danych pozostał niezmieniony: SHA256 dcf79fdf5cfa648088fc07b1023dc9e239f580d4788daace200757e6144a35fd. Nie uruchamiano ponownie optymalizacji na Edenie ani analizy PyDMD/deeptime. Zmieniono wyłącznie renderer i stronę podglądu; serwer odczytuje je przy każdym żądaniu.
