# Audyt szkła: pełne wyniki liczbowe

Data: 2026-10-10. Wszystkie miary obejmują 768 współrzędnych. MSD jest średnią kwadratu normy przemieszczenia po tokenach i obrazach, bez dzielenia przez 768. Przedziały bootstrapowe resamplują 7 filmów; opisują tę próbę obrazów, nie zmienność między treningami.

## Gęsty bieg 21 epok

### MSD w równych oknach pomiarowych

| Układ | Okno kroków | Opóźnienia do dopasowania | Wykładnik MSD | Przedział po filmach |
|---|---:|---:|---:|---:|
| raw | 300–1094 | 1–5 | 1.530 | 1.506–1.558 |
| raw | 300–1094 | 10–100 | 0.259 | 0.243–0.275 |
| raw | 300–1094 | 35–80 | 0.184 | 0.162–0.202 |
| raw | 1000–6000 | 100–1000 | 0.310 | 0.291–0.322 |
| raw | 10000–15000 | 100–1000 | 0.257 | 0.241–0.268 |
| raw | 17500–22500 | 100–1000 | 0.253 | 0.232–0.271 |
| raw | 1000–22950 | 100–1000 | 0.257 | 0.241–0.269 |
| raw | 1000–22950 | 500–5000 | 0.311 | 0.286–0.327 |
| image_relative | 300–1094 | 1–5 | 1.580 | 1.563–1.602 |
| image_relative | 300–1094 | 10–100 | 0.281 | 0.269–0.292 |
| image_relative | 300–1094 | 35–80 | 0.206 | 0.195–0.216 |
| image_relative | 1000–6000 | 100–1000 | 0.270 | 0.261–0.279 |
| image_relative | 10000–15000 | 100–1000 | 0.234 | 0.225–0.241 |
| image_relative | 17500–22500 | 100–1000 | 0.237 | 0.222–0.250 |
| image_relative | 1000–22950 | 100–1000 | 0.236 | 0.227–0.244 |
| image_relative | 1000–22950 | 500–5000 | 0.283 | 0.264–0.296 |
| cage | 300–1094 | 1–5 | 1.486 | 1.398–1.502 |
| cage | 300–1094 | 10–100 | 0.325 | 0.311–0.334 |
| cage | 300–1094 | 35–80 | 0.242 | 0.225–0.251 |
| cage | 1000–6000 | 100–1000 | 0.223 | 0.214–0.228 |
| cage | 10000–15000 | 100–1000 | 0.204 | 0.195–0.209 |
| cage | 17500–22500 | 100–1000 | 0.213 | 0.199–0.222 |
| cage | 1000–22950 | 100–1000 | 0.206 | 0.195–0.211 |
| cage | 1000–22950 | 500–5000 | 0.222 | 0.206–0.229 |

### Starzenie: te same opóźnienia i szerokości okien początkowych

Każdy wiek reprezentuje początki w oknie szerokości 500 kroków. Między punktami nie interpolowano wektorów. Okna przekraczające wznowienie w kroku 22974 są pomijane.

| Układ | Wiek początku | MSD po 1000 kroków | Czas do połowy MSD po 2000 kroków |
|---|---:|---:|---:|
| raw | 1000 | 45.859 | 899.3 |
| raw | 3000 | 41.092 | 261.8 |
| raw | 6000 | 38.521 | 149.5 |
| raw | 10000 | 43.536 | 210.4 |
| raw | 15000 | 51.163 | 108.4 |
| raw | 19000 | 50.659 | 134.5 |
| image_relative | 1000 | 21.993 | 853.8 |
| image_relative | 3000 | 23.999 | 280.8 |
| image_relative | 6000 | 24.271 | 130.1 |
| image_relative | 10000 | 28.547 | 115.6 |
| image_relative | 15000 | 32.789 | 82.0 |
| image_relative | 19000 | 32.233 | 100.8 |
| cage | 1000 | 2.760 | 633.7 |
| cage | 3000 | 3.292 | 185.1 |
| cage | 6000 | 3.984 | 121.8 |
| cage | 10000 | 5.148 | 88.7 |
| cage | 15000 | 6.292 | 61.6 |
| cage | 19000 | 6.606 | 92.3 |

Ten czas opisuje kształt w **ustalonym horyzoncie 2000 kroków**, nie zakłada istnienia asymptotycznego plateau.

### Czasy do wspólnego progu MSD

Progi 0,5×, 1×, 2× wartości MSD pierwszego wieku przy opóźnieniu 1000. Są umowne; brak osiągnięcia nie oznacza nieskończonego czasu.

| Układ | Mnożnik progu | Próg | Wiek | Czas do progu |
|---|---:|---:|---:|---:|
| raw | 0.5 | 22.929 | 1000 | 378.3 |
| raw | 0.5 | 22.929 | 3000 | 120.3 |
| raw | 0.5 | 22.929 | 6000 | 120.3 |
| raw | 0.5 | 22.929 | 10000 | 94.1 |
| raw | 0.5 | 22.929 | 15000 | ≤50.0 |
| raw | 0.5 | 22.929 | 19000 | ≤50.0 |
| raw | 1 | 45.859 | 1000 | 1000.0 |
| raw | 1 | 45.859 | 3000 | 1284.2 |
| raw | 1 | 45.859 | 6000 | 1523.2 |
| raw | 1 | 45.859 | 10000 | 1139.7 |
| raw | 1 | 45.859 | 15000 | 579.9 |
| raw | 1 | 45.859 | 19000 | 624.9 |
| raw | 2 | 91.718 | 1000 | 2270.7 |
| raw | 2 | 91.718 | 3000 | 4464.7 |
| raw | 2 | 91.718 | 6000 | 6668.1 |
| raw | 2 | 91.718 | 10000 | 7430.4 |
| raw | 2 | 91.718 | 15000 | >5000 (cenzurowany) |
| raw | 2 | 91.718 | 19000 | >3000 (cenzurowany) |
| image_relative | 0.5 | 10.996 | 1000 | 276.3 |
| image_relative | 0.5 | 10.996 | 3000 | 55.3 |
| image_relative | 0.5 | 10.996 | 6000 | ≤50.0 |
| image_relative | 0.5 | 10.996 | 10000 | ≤50.0 |
| image_relative | 0.5 | 10.996 | 15000 | ≤50.0 |
| image_relative | 0.5 | 10.996 | 19000 | ≤50.0 |
| image_relative | 1 | 21.993 | 1000 | 1000.0 |
| image_relative | 1 | 21.993 | 3000 | 776.8 |
| image_relative | 1 | 21.993 | 6000 | 651.8 |
| image_relative | 1 | 21.993 | 10000 | 327.1 |
| image_relative | 1 | 21.993 | 15000 | 138.0 |
| image_relative | 1 | 21.993 | 19000 | 175.7 |
| image_relative | 2 | 43.986 | 1000 | 2344.7 |
| image_relative | 2 | 43.986 | 3000 | 3409.7 |
| image_relative | 2 | 43.986 | 6000 | 4481.6 |
| image_relative | 2 | 43.986 | 10000 | 3854.0 |
| image_relative | 2 | 43.986 | 15000 | 3207.5 |
| image_relative | 2 | 43.986 | 19000 | >3000 (cenzurowany) |
| cage | 0.5 | 1.380 | 1000 | 131.6 |
| cage | 0.5 | 1.380 | 3000 | ≤50.0 |
| cage | 0.5 | 1.380 | 6000 | ≤50.0 |
| cage | 0.5 | 1.380 | 10000 | ≤50.0 |
| cage | 0.5 | 1.380 | 15000 | ≤50.0 |
| cage | 0.5 | 1.380 | 19000 | ≤50.0 |
| cage | 1 | 2.760 | 1000 | 1000.0 |
| cage | 1 | 2.760 | 3000 | 432.2 |
| cage | 1 | 2.760 | 6000 | 179.8 |
| cage | 1 | 2.760 | 10000 | 70.5 |
| cage | 1 | 2.760 | 15000 | ≤50.0 |
| cage | 1 | 2.760 | 19000 | ≤50.0 |
| cage | 2 | 5.520 | 1000 | 2898.7 |
| cage | 2 | 5.520 | 3000 | 3164.0 |
| cage | 2 | 5.520 | 6000 | 2780.4 |
| cage | 2 | 5.520 | 10000 | 1384.4 |
| cage | 2 | 5.520 | 15000 | 374.9 |
| cage | 2 | 5.520 | 19000 | 385.1 |

### Kontrola prostym modelem dynamiki

Dopasowanie na naprzemiennych opóźnieniach i sprawdzenie na pozostałych opóźnieniach **tej samej krzywej**. Nie jest to walidacja na niezależnym treningu. Błąd jest RMSE logarytmu MSD. Model colored OU + diffusion opisuje liniowe ograniczenie ruchu z pamięcią szumu i wolną składową dyfuzyjną; nie ma oddziaływań wielociałowych.

| Układ | Okno | Zwykły OU | OU + dyfuzja | OU z pamięcią + dyfuzja |
|---|---|---:|---:|---:|
| raw | early | 0.164 | 0.113 | 0.018 |
| raw | late1 | 0.212 | 0.017 | 0.017 |
| raw | late2 | 0.108 | 0.049 | 0.049 |
| raw | late3 | 0.103 | 0.039 | 0.039 |
| raw | late_all | 0.310 | 0.050 | 0.050 |
| image_relative | early | 0.163 | 0.128 | 0.022 |
| image_relative | late1 | 0.194 | 0.017 | 0.017 |
| image_relative | late2 | 0.102 | 0.042 | 0.042 |
| image_relative | late3 | 0.099 | 0.039 | 0.039 |
| image_relative | late_all | 0.290 | 0.050 | 0.050 |
| cage | early | 0.148 | 0.112 | 0.021 |
| cage | late1 | 0.157 | 0.022 | 0.022 |
| cage | late2 | 0.091 | 0.040 | 0.040 |
| cage | late3 | 0.093 | 0.039 | 0.039 |
| cage | late_all | 0.243 | 0.050 | 0.050 |

![Diagnostyka Gęsty bieg 21 epok](dense_diagnostics.png)

## Osobny bieg 60 epok

### MSD w równych oknach pomiarowych

| Układ | Okno kroków | Opóźnienia do dopasowania | Wykładnik MSD | Przedział po filmach |
|---|---:|---:|---:|---:|
| raw | 300–1000 | 10–100 | 0.270 | 0.249–0.289 |
| raw | 300–1000 | 35–80 | 0.238 | 0.219–0.259 |
| raw | 1000–6000 | 100–1000 | 0.297 | 0.278–0.310 |
| raw | 10000–15000 | 100–1000 | 0.265 | 0.251–0.276 |
| raw | 17500–22500 | 100–1000 | 0.263 | 0.248–0.275 |
| raw | 25000–40000 | 100–1000 | 0.318 | 0.301–0.334 |
| raw | 25000–40000 | 500–5000 | 0.254 | 0.234–0.270 |
| raw | 50000–65000 | 100–1000 | 0.368 | 0.338–0.394 |
| raw | 50000–65000 | 500–5000 | 0.329 | 0.294–0.357 |
| raw | 24000–65600 | 100–1000 | 0.339 | 0.319–0.357 |
| raw | 24000–65600 | 500–5000 | 0.295 | 0.266–0.317 |
| image_relative | 300–1000 | 10–100 | 0.292 | 0.282–0.302 |
| image_relative | 300–1000 | 35–80 | 0.240 | 0.225–0.251 |
| image_relative | 1000–6000 | 100–1000 | 0.260 | 0.252–0.271 |
| image_relative | 10000–15000 | 100–1000 | 0.241 | 0.231–0.248 |
| image_relative | 17500–22500 | 100–1000 | 0.248 | 0.238–0.255 |
| image_relative | 25000–40000 | 100–1000 | 0.285 | 0.274–0.296 |
| image_relative | 25000–40000 | 500–5000 | 0.224 | 0.211–0.233 |
| image_relative | 50000–65000 | 100–1000 | 0.328 | 0.306–0.346 |
| image_relative | 50000–65000 | 500–5000 | 0.248 | 0.231–0.264 |
| image_relative | 24000–65600 | 100–1000 | 0.303 | 0.289–0.316 |
| image_relative | 24000–65600 | 500–5000 | 0.239 | 0.223–0.251 |
| cage | 300–1000 | 10–100 | 0.337 | 0.325–0.346 |
| cage | 300–1000 | 35–80 | 0.267 | 0.239–0.276 |
| cage | 1000–6000 | 100–1000 | 0.214 | 0.207–0.223 |
| cage | 10000–15000 | 100–1000 | 0.208 | 0.196–0.214 |
| cage | 17500–22500 | 100–1000 | 0.218 | 0.205–0.223 |
| cage | 25000–40000 | 100–1000 | 0.251 | 0.232–0.259 |
| cage | 25000–40000 | 500–5000 | 0.179 | 0.161–0.185 |
| cage | 50000–65000 | 100–1000 | 0.285 | 0.259–0.297 |
| cage | 50000–65000 | 500–5000 | 0.197 | 0.176–0.207 |
| cage | 24000–65600 | 100–1000 | 0.266 | 0.246–0.276 |
| cage | 24000–65600 | 500–5000 | 0.189 | 0.170–0.196 |

### Starzenie: te same opóźnienia i szerokości okien początkowych

Każdy wiek reprezentuje początki w oknie szerokości 500 kroków. Między punktami nie interpolowano wektorów. Okna przekraczające wznowienie w kroku 22974 są pomijane.

| Układ | Wiek początku | MSD po 1000 kroków | Czas do połowy MSD po 2000 kroków |
|---|---:|---:|---:|
| raw | 1000 | 45.451 | 894.6 |
| raw | 3000 | 36.339 | 336.4 |
| raw | 6000 | 39.195 | 127.5 |
| raw | 10000 | 45.433 | ≤100.0 |
| raw | 15000 | 52.564 | 147.6 |
| raw | 19000 | 53.246 | 113.6 |
| raw | 24000 | 60.217 | 171.9 |
| raw | 30000 | 54.389 | 176.1 |
| raw | 40000 | 49.426 | 167.1 |
| raw | 50000 | 55.140 | 176.8 |
| raw | 60000 | 44.160 | 295.2 |
| image_relative | 1000 | 21.251 | 858.8 |
| image_relative | 3000 | 22.301 | 308.3 |
| image_relative | 6000 | 24.129 | 120.7 |
| image_relative | 10000 | 28.467 | 107.4 |
| image_relative | 15000 | 33.859 | 105.5 |
| image_relative | 19000 | 33.693 | 106.7 |
| image_relative | 24000 | 36.965 | 159.1 |
| image_relative | 30000 | 32.641 | 152.3 |
| image_relative | 40000 | 30.874 | 150.6 |
| image_relative | 50000 | 32.714 | 149.9 |
| image_relative | 60000 | 26.193 | 236.7 |
| cage | 1000 | 2.681 | 638.0 |
| cage | 3000 | 3.155 | 213.5 |
| cage | 6000 | 3.945 | ≤100.0 |
| cage | 10000 | 5.158 | ≤100.0 |
| cage | 15000 | 6.574 | ≤100.0 |
| cage | 19000 | 6.631 | ≤100.0 |
| cage | 24000 | 7.396 | 128.8 |
| cage | 30000 | 7.480 | 130.1 |
| cage | 40000 | 7.108 | 128.6 |
| cage | 50000 | 7.544 | 113.4 |
| cage | 60000 | 6.631 | 181.2 |

Ten czas opisuje kształt w **ustalonym horyzoncie 2000 kroków**, nie zakłada istnienia asymptotycznego plateau.

### Czasy do wspólnego progu MSD

Progi 0,5×, 1×, 2× wartości MSD pierwszego wieku przy opóźnieniu 1000. Są umowne; brak osiągnięcia nie oznacza nieskończonego czasu.

| Układ | Mnożnik progu | Próg | Wiek | Czas do progu |
|---|---:|---:|---:|---:|
| raw | 0.5 | 22.726 | 1000 | 337.7 |
| raw | 0.5 | 22.726 | 3000 | 162.5 |
| raw | 0.5 | 22.726 | 6000 | ≤100.0 |
| raw | 0.5 | 22.726 | 10000 | ≤100.0 |
| raw | 0.5 | 22.726 | 15000 | ≤100.0 |
| raw | 0.5 | 22.726 | 19000 | ≤100.0 |
| raw | 0.5 | 22.726 | 24000 | ≤100.0 |
| raw | 0.5 | 22.726 | 30000 | ≤100.0 |
| raw | 0.5 | 22.726 | 40000 | ≤100.0 |
| raw | 0.5 | 22.726 | 50000 | ≤100.0 |
| raw | 0.5 | 22.726 | 60000 | 135.9 |
| raw | 1 | 45.451 | 1000 | 1000.0 |
| raw | 1 | 45.451 | 3000 | 1468.4 |
| raw | 1 | 45.451 | 6000 | 1379.3 |
| raw | 1 | 45.451 | 10000 | 1001.5 |
| raw | 1 | 45.451 | 15000 | 606.1 |
| raw | 1 | 45.451 | 19000 | 398.6 |
| raw | 1 | 45.451 | 24000 | 405.6 |
| raw | 1 | 45.451 | 30000 | 473.1 |
| raw | 1 | 45.451 | 40000 | 787.1 |
| raw | 1 | 45.451 | 50000 | 447.0 |
| raw | 1 | 45.451 | 60000 | 1114.7 |
| raw | 2 | 90.902 | 1000 | 2335.5 |
| raw | 2 | 90.902 | 3000 | 4997.8 |
| raw | 2 | 90.902 | 6000 | 5553.2 |
| raw | 2 | 90.902 | 10000 | 6295.2 |
| raw | 2 | 90.902 | 15000 | >5000 (cenzurowany) |
| raw | 2 | 90.902 | 19000 | >3000 (cenzurowany) |
| raw | 2 | 90.902 | 24000 | 6872.2 |
| raw | 2 | 90.902 | 30000 | 5710.1 |
| raw | 2 | 90.902 | 40000 | 5899.5 |
| raw | 2 | 90.902 | 50000 | 5476.3 |
| raw | 2 | 90.902 | 60000 | >5000 (cenzurowany) |
| image_relative | 0.5 | 10.626 | 1000 | 190.6 |
| image_relative | 0.5 | 10.626 | 3000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 6000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 10000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 15000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 19000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 24000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 30000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 40000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 50000 | ≤100.0 |
| image_relative | 0.5 | 10.626 | 60000 | ≤100.0 |
| image_relative | 1 | 21.251 | 1000 | 1000.0 |
| image_relative | 1 | 21.251 | 3000 | 902.3 |
| image_relative | 1 | 21.251 | 6000 | 416.2 |
| image_relative | 1 | 21.251 | 10000 | 342.5 |
| image_relative | 1 | 21.251 | 15000 | 142.8 |
| image_relative | 1 | 21.251 | 19000 | 189.4 |
| image_relative | 1 | 21.251 | 24000 | 149.5 |
| image_relative | 1 | 21.251 | 30000 | 174.8 |
| image_relative | 1 | 21.251 | 40000 | 185.1 |
| image_relative | 1 | 21.251 | 50000 | 206.2 |
| image_relative | 1 | 21.251 | 60000 | 436.6 |
| image_relative | 2 | 42.503 | 1000 | 2365.8 |
| image_relative | 2 | 42.503 | 3000 | 3480.6 |
| image_relative | 2 | 42.503 | 6000 | 4093.2 |
| image_relative | 2 | 42.503 | 10000 | 3484.1 |
| image_relative | 2 | 42.503 | 15000 | 2971.6 |
| image_relative | 2 | 42.503 | 19000 | >3000 (cenzurowany) |
| image_relative | 2 | 42.503 | 24000 | 1931.8 |
| image_relative | 2 | 42.503 | 30000 | 2600.1 |
| image_relative | 2 | 42.503 | 40000 | 3048.4 |
| image_relative | 2 | 42.503 | 50000 | 3341.2 |
| image_relative | 2 | 42.503 | 60000 | >5000 (cenzurowany) |
| cage | 0.5 | 1.341 | 1000 | 123.4 |
| cage | 0.5 | 1.341 | 3000 | ≤100.0 |
| cage | 0.5 | 1.341 | 6000 | ≤100.0 |
| cage | 0.5 | 1.341 | 10000 | ≤100.0 |
| cage | 0.5 | 1.341 | 15000 | ≤100.0 |
| cage | 0.5 | 1.341 | 19000 | ≤100.0 |
| cage | 0.5 | 1.341 | 24000 | ≤100.0 |
| cage | 0.5 | 1.341 | 30000 | ≤100.0 |
| cage | 0.5 | 1.341 | 40000 | ≤100.0 |
| cage | 0.5 | 1.341 | 50000 | ≤100.0 |
| cage | 0.5 | 1.341 | 60000 | ≤100.0 |
| cage | 1 | 2.681 | 1000 | 1000.0 |
| cage | 1 | 2.681 | 3000 | 591.4 |
| cage | 1 | 2.681 | 6000 | 129.2 |
| cage | 1 | 2.681 | 10000 | ≤100.0 |
| cage | 1 | 2.681 | 15000 | ≤100.0 |
| cage | 1 | 2.681 | 19000 | ≤100.0 |
| cage | 1 | 2.681 | 24000 | ≤100.0 |
| cage | 1 | 2.681 | 30000 | ≤100.0 |
| cage | 1 | 2.681 | 40000 | ≤100.0 |
| cage | 1 | 2.681 | 50000 | ≤100.0 |
| cage | 1 | 2.681 | 60000 | ≤100.0 |
| cage | 2 | 5.362 | 1000 | 2956.4 |
| cage | 2 | 5.362 | 3000 | 3199.5 |
| cage | 2 | 5.362 | 6000 | 2988.1 |
| cage | 2 | 5.362 | 10000 | 1212.1 |
| cage | 2 | 5.362 | 15000 | 392.9 |
| cage | 2 | 5.362 | 19000 | 360.5 |
| cage | 2 | 5.362 | 24000 | 238.0 |
| cage | 2 | 5.362 | 30000 | 213.1 |
| cage | 2 | 5.362 | 40000 | 213.2 |
| cage | 2 | 5.362 | 50000 | 282.0 |
| cage | 2 | 5.362 | 60000 | 414.7 |

### Kontrola prostym modelem dynamiki

Dopasowanie na naprzemiennych opóźnieniach i sprawdzenie na pozostałych opóźnieniach **tej samej krzywej**. Nie jest to walidacja na niezależnym treningu. Błąd jest RMSE logarytmu MSD. Model colored OU + diffusion opisuje liniowe ograniczenie ruchu z pamięcią szumu i wolną składową dyfuzyjną; nie ma oddziaływań wielociałowych.

| Układ | Okno | Zwykły OU | OU + dyfuzja | OU z pamięcią + dyfuzja |
|---|---|---:|---:|---:|
| raw | early | 0.159 | 0.009 | 0.009 |
| raw | late1 | 0.197 | 0.013 | 0.014 |
| raw | late2 | 0.087 | 0.046 | 0.046 |
| raw | late3 | 0.082 | 0.029 | 0.029 |
| raw | postresume | 0.137 | 0.051 | 0.051 |
| raw | end | 0.161 | 0.047 | 0.047 |
| raw | postresume_all | 0.240 | 0.083 | 0.083 |
| image_relative | early | 0.145 | 0.010 | 0.010 |
| image_relative | late1 | 0.187 | 0.011 | 0.012 |
| image_relative | late2 | 0.084 | 0.037 | 0.037 |
| image_relative | late3 | 0.079 | 0.028 | 0.028 |
| image_relative | postresume | 0.130 | 0.048 | 0.048 |
| image_relative | end | 0.136 | 0.047 | 0.047 |
| image_relative | postresume_all | 0.220 | 0.074 | 0.074 |
| cage | early | 0.130 | 0.014 | 0.014 |
| cage | late1 | 0.150 | 0.012 | 0.012 |
| cage | late2 | 0.079 | 0.032 | 0.032 |
| cage | late3 | 0.070 | 0.027 | 0.027 |
| cage | postresume | 0.113 | 0.044 | 0.044 |
| cage | end | 0.117 | 0.047 | 0.047 |
| cage | postresume_all | 0.183 | 0.067 | 0.067 |

![Diagnostyka Osobny bieg 60 epok](long_diagnostics.png)

## Zawracanie: długość przyrostu to osobna zmienna

Okno 300–1094, sąsiednie niepokrywające się przyrosty. VACF = średni iloczyn skalarny przyrostów / średnia ich kwadratowej normy. Ujemna wartość oznacza przeciwną składową ruchu, nie pełny obrót każdego tokenu.

| Układ | Przyrost [kroki] | Korelacja kolejnych przyrostów | Przedział po filmach |
|---|---:|---:|---:|
| raw | 1 | 0.678 | 0.663–0.692 |
| raw | 2 | 0.357 | 0.323–0.387 |
| raw | 5 | -0.110 | -0.141–-0.078 |
| raw | 10 | -0.299 | -0.317–-0.282 |
| raw | 20 | -0.419 | -0.432–-0.409 |
| raw | 50 | -0.401 | -0.420–-0.382 |
| image_relative | 1 | 0.691 | 0.681–0.700 |
| image_relative | 2 | 0.423 | 0.395–0.446 |
| image_relative | 5 | -0.060 | -0.092–-0.029 |
| image_relative | 10 | -0.289 | -0.304–-0.276 |
| image_relative | 20 | -0.403 | -0.414–-0.394 |
| image_relative | 50 | -0.398 | -0.407–-0.386 |
| cage | 1 | 0.477 | 0.366–0.535 |
| cage | 2 | 0.358 | 0.324–0.383 |
| cage | 5 | -0.057 | -0.080–-0.034 |
| cage | 10 | -0.264 | -0.274–-0.254 |
| cage | 20 | -0.387 | -0.396–-0.380 |
| cage | 50 | -0.398 | -0.410–-0.389 |

## Sąsiedztwo odświeżane na każdym początku pomiaru

Sześciu najbliższych sąsiadów według pełnej odległości 768D, wyłącznie wewnątrz obrazu. Pięć początków dla każdego wieku; w głównym pomiarze sąsiedzi byli zamrożeni w kroku 300.

| Bieg | Wiek | MSD względem odświeżanych sąsiadów po 1000 kroków |
|---|---:|---:|
| dense | 1000 | 2.249 |
| dense | 3000 | 2.631 |
| dense | 6000 | 2.886 |
| dense | 10000 | 3.906 |
| dense | 15000 | 4.615 |
| dense | 19000 | 4.944 |
| long | 300 | 3.412 |
| long | 1000 | 2.212 |
| long | 6000 | 3.060 |
| long | 10000 | 3.923 |
| long | 19000 | 4.904 |
| long | 24000 | 5.416 |
| long | 30000 | 5.640 |
| long | 40000 | 5.355 |
| long | 50000 | 5.936 |
| long | 60000 | 5.049 |

## FP32, BF16 i zamrożone statystyki BatchNorm

Ta kontrola dotyczy osobnego biegu 60 epok. BatchNorm zamrożony na statystykach kroku 300 jest kontrolą wrażliwości; nie jest alternatywnym treningiem.

| Początek–koniec | MSD FP32 | MSD BF16 / FP32 | MSD zamrożone BN | MSD FP32 po normowaniu tokenu do długości 1 |
|---|---:|---:|---:|---:|
| 300–310 | 8.333 | 0.999 | 8.512 | 0.10148 |
| 310–320 | 9.696 | 1.002 | 10.173 | 0.12025 |
| 900–910 | 5.948 | 0.996 | 6.765 | 0.08055 |
| 910–920 | 5.013 | 1.003 | 7.317 | 0.06684 |
| 990–1000 | 5.567 | 0.999 | 6.804 | 0.07510 |
| 300–1000 | 61.066 | 1.000 | 59.149 | 0.72441 |
| 10000–20000 | 110.535 | 0.999 | 96.112 | 0.54830 |
| 30000–40000 | 150.542 | 1.000 | 86.862 | 0.55253 |
| 50000–60000 | 111.372 | 1.000 | 66.930 | 0.55963 |
| 60000–65640 | 80.272 | 0.999 | 51.807 | 0.48741 |

## Stałe opóźnienie 5 epok po kontroli długości wektorów

Początki w trzech sąsiednich epokach. Przejścia przez wznowienie są wykluczone. Normalizacja każdego tokenu do długości 1 zachowuje informację o kierunku, usuwa informację o długości; to kontrola wrażliwości, a nie jedyna właściwa metryka.

| Wariant | Wiek [epoki] | MSD po 5 epokach |
|---|---:|---:|
| fp32_raw | 5 | 86.28602 |
| fp32_raw | 10 | 82.45450 |
| fp32_raw | 25 | 81.32132 |
| fp32_raw | 35 | 97.67532 |
| fp32_raw | 45 | 83.28959 |
| fp32_raw | 50 | 92.57601 |
| fp32_unit | 5 | 0.46733 |
| fp32_unit | 10 | 0.41013 |
| fp32_unit | 25 | 0.34808 |
| fp32_unit | 35 | 0.40979 |
| fp32_unit | 45 | 0.45282 |
| fp32_unit | 50 | 0.51043 |
| frozen_bn_raw | 5 | 71.04997 |
| frozen_bn_raw | 10 | 64.17162 |
| frozen_bn_raw | 25 | 53.31436 |
| frozen_bn_raw | 35 | 58.82657 |
| frozen_bn_raw | 45 | 52.07614 |
| frozen_bn_raw | 50 | 50.41427 |
| frozen_bn_unit | 5 | 0.70469 |
| frozen_bn_unit | 10 | 0.57805 |
| frozen_bn_unit | 25 | 0.39501 |
| frozen_bn_unit | 35 | 0.40685 |
| frozen_bn_unit | 45 | 0.40239 |
| frozen_bn_unit | 50 | 0.40478 |

![Kontrola skali](epoch_scale_controls.png)


## Odtworzenie

Kod: `experiments/audyt_szkla_768d_20261010/`. Kolejność:

```powershell
python compute_grams.py dense
python analyze.py dense
python multiscale.py
python direct_controls.py
python compute_grams.py long
python analyze.py long
python summarize_controls.py
python epoch_controls.py
python validate_math.py
python final_figure.py
python write_report.py
```

Dane historyczne pozostały bez zmian. Pliki macierzy Grama zachowują informację wystarczającą do odtworzenia średnich MSD i VACF w dowolnych dostępnych parach czasu, ale nie całych wektorów. Tymczasowy zapis latentów długiego biegu jest usuwany po utworzeniu wyników.
