# Paczka 32 a oryginalny sztuczny cel

Pytanie: czy 32 unikalne obrazy train najbliższe naturalnemu latentowi starego kolorowego SD kierują reprezentację oryginalnej sondy 29 ku jej sztucznemu celowi? Obalenie: brak poprawy względem niezmieniania wag lub losowej paczki32. To eksploracja jednej sondy, nie ocena generalizacji detekcji.

Wybór: sortowanie całego zakończonego rankingu po mse768 i ordinal, deduplikacja archive/member, tylko train. Kontrola: losowe 32 unikalne train, seed 20261005, oraz brak zmiany. Obrazy z ZIP i wszystkie ramki narzędzi z annotation.csv, bez rogówki/źrenicy; zapis surowych adnotacji, skali i SHA256. RGB 480x320 INTER_AREA. Oryginalna sonda: zachowane ZDJECIE_ZRODLOWE.jpg, indeks29 TR_0154 klatka1926. Cele after/before/channels0 i kolejność tokenów wyłącznie z oryginalnego LATENT_PELNY.npz.

Oryginalne równanie: prognoza_krok.py i rownanie.py z maly_vit_sd_prowadzenie_tozsamoscia/kod. S_nowa = z(theta+d_nowa)-z(theta); S_ped analogicznie. Główna wersja oryginalna: delta=S_nowa+S_ped. Osobno historycznie dopasowane współczynniki nowa/ped z TOZSAMOSC_JEDEN_KROK.json, bez ponownego dopasowania. Rozszerzenie ze średnich tokenów narzędzia do pełnej siatki600x768 jest niezweryfikowanym transferem; trzeba ocenić pierwszy krok. Nie ingerujemy w latent.

Pełny stan AdamW: oryginalny Eden /home2/faculty/bpiotrowski/smallvit_runs/2026-10-01_maly_vit_ruch_czastek_gesta_historia/wyniki/pelny/pelny/k1000.pt. Pobrać tylko ten plik do nowej rodziny; sprawdzić SHA256 zdalne oraz identyczność każdego tensora net względem modelu celu. Brak opt lub różne wagi = BLOCKER, bez udawania autentycznej kontynuacji.

Przed treningiem obliczyć gradient całej paczki32 przy aktualnych wagach, oryginalna strata CenterNet+GIoU, bf16, train, kopie buforów BN, obcięcie normy1. Siły wyliczyć przy oryginalnym AdamW. Prognoza na 1..50 kroków: stała początkowa siła, jawne założenie; żadnych przyszłych gradientów. MSE128, MSE768, cosinus128/768 wobec braku zmiany. Próg trafienia: pierwsze MSE <= (1-r)*MSE0 dla r=0.10,0.25,0.50, oddzielnie dla128/768. Brak trafienia do50 zapisujemy jako no_hit, bez wymyślonych epok. Epoka puli32 = jedna aktualizacja na całych32; to nie epoka pełnego zbioru.

Weryfikacja na niezależnych kopiach tego samego autentycznego stanu: powtarzana wybrana32 i stała losowa32, 50 prawdziwych kroków, zapis1/5/10/25/50. Brak augmentacji jest jawnie określoną zmianą protokołu wobec historycznego treningu; nie zmieniamy architektury ani nie zamrażamy głowy. Wszystkie wagi i bufory BN aktualizowane normalnie. Pierwszy krok oceniony oddzielnie z nowymi wagami i starymi buforami oraz pełnym rzeczywistym BN. Prognozy zapisane przed krokiem1.

GPU: najpierw przygotowanie CPU i pobranie checkpointu; kolejka do statusu complete rodziny vit_sd_cztery_szumy i >=6500MiB wolnej pamięci. Nie kończyć cudzych procesów. Wszystkie nowe pliki w tej rodzinie eksperymentów/wyników; rejestracja przez istniejące narzędzie. Bez commitów.

Poprawka przed treningiem, na prośbę użytkownika: pełny smoke gradientu32 i wirtualnych sił, bez mikrobatche i ze zwykłym train BN, przeszedł. Peak allocated1383.12MiB, reserved1660MiB, wolne po pomiarze1837MiB. Próg z pomiaru =2816MiB, czyli1660+1024 zapasu zaokrąglone w górę do256. GPU_MEMORY_SMOKE.json zawiera pomiar i PID17468. Worker używa tego progu tylko przy statusie passed, inaczej6500. Progi trafienia, liczba kroków, architektura i BN pozostają jak zadeklarowano. Własna kolejka PID44748 zostaje wznowiona; cudze procesy nietknięte.

Ograniczenie osiągalności zamrożonego checkpointu nie dowodzi niemożliwości po zmianie wag. Pomiar zawsze dotyczy identycznych pikseli oryginalnej sondy i tej samej kolejności patchy; nowa fotografia jest wejściem treningowym, nie substytutem sondy.
