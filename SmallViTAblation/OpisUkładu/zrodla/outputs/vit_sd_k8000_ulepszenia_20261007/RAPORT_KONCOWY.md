# SD K8000: wynik porównania i weryfikacji lokalnej

Wszystkie cztery warianty ukończyły budżet 4000 nowych aktualizacji. Job 1823544 zakończył się COMPLETED 0:0 po 12:21:32. Nie uruchamiano ponownego treningu.

Najmniejszy pełny względny MSE uzyskał **A_control**, w checkpointcie po 3000 nowych aktualizacji, czyli 10500 łącznie. Błąd spadł z 0.623735 do 0.519988, o **16.63%**. Żaden z trzech badanych pakietów zmian nie pokonał kontroli według tej miary.

## Porównanie na tych samych odłożonych filmach

| Wariant | Nowe kroki najlepszego modelu | Względny MSE VAL | Cosine VAL | Względny MSE ROI | Względny MSE TRAIN |
|---|---:|---:|---:|---:|---:|
| A_control | 3000 | 0.519988 | 0.713380 | 0.976266 | 0.538693 |
| B_mixed | 4000 | 0.542292 | 0.698251 | 1.057675 | 0.493293 |
| C_optim | 3000 | 0.536456 | 0.707469 | 1.036664 | 0.461633 |
| D_roi | 4000 | 0.544533 | 0.702820 | 1.037688 | 0.474931 |

A: sekwencyjne próbkowanie co 600 klatek. B: losowanie klatki w oknach 600 i mieszanie filmów. C: dane B oraz mniejszy LR z EMA. D: C z dodatkową stratą ROI. C/B mierzy pakiet LR+EMA, nie oddzielny wpływ każdego składnika.

Walidacja: 25 filmów, 2 klatki na film, 2 seedy, 100 generacji na wariant i ocenę. Te same ustawienia odszumiania: DDIM 50 leading. Średnie kolejno po seedach, klatkach i filmach. Stała próbka 3 filmów TRAIN jest osobną diagnostyką, nie źródłem wyboru checkpointa. Pokrycie TRAIN 116/118; dwa znane filmy odrzucono z powodu niezgodnych długości MP4/CSV. Nie wyciągamy wniosku o istotności statystycznej ani o przeuczeniu z samej różnicy tych średnich.

Pełny względny MSE = mean((z_obraz-z_cel)^2) / mean(z_cel^2), mianownik ograniczony od dołu przez 1e-6. Cosine jest osobną miarą kierunku i nie jest procentem zgodności wartości. ROI obejmuje tokeny nakładające się na ramkę narzędzia, nie wybrane 128 kanałów.

## Warunkowanie i jakość

A_control: MSE właściwego warunku 0.5200, innego filmu 0.7114. Średnia indywidualnych względnych korzyści wynosi 22.84%. To dowód mierzalnej reakcji na warunek w tym porównaniu, nie gwarancja odtworzenia szczegółów.

Błąd ROI A wynosi 0.9763; IoU względem ramki źródłowej 0.1718. To diagnostyka zachowania układu, nie trafność detekcji względem niezależnej etykiety obrazu wygenerowanego.

Rekonstrukcja VAE na tych samych filmach: PSNR 30.45 dB, SSIM 0.9444, względny błąd pełnego latentu 0.0262. Sam generator ma SSIM 0.2391 i PSNR 10.07 dB. Nie osiągnął wcześniejszego celu względnego błędu <=0,10 ani cosine >=0,90. Dobre wyniki rekonstrukcji VAE kierują dalszą diagnostykę na generowanie i warunkowanie, ale nie stanowią pełnego dowodu przyczynowego.

Oględziny czterech lokalnych generacji: powstają rozpoznawalne obrazy pola operacyjnego oka, a nie jednolite pola. Rekonstrukcje VAE są wizualnie bliskie źródłom. Generacje SD mają widoczne zniekształcenia i zmienioną teksturę; położenie, kształt i obecność narzędzia oraz przesłonięcia obrazu nie są wiernie zachowane. Seed silnie zmienia wygląd; zmiana latentu również wpływa na wynik. Nie uznajemy tych przykładów za wierne odtworzenie naturalnego latentu ani za gotowe dane treningowe z etykietą odziedziczoną ze źródła.

## Świeży test lokalny

Potwierdzono SHA256 `1a3fbac4234c7a16c1da036dce208971b579b81d012513c96f2938f697492e41`, strict load sieci i VAE oraz zgodność teachera K8000 i manifestu. Generowanie na cuda, z odciążaniem pamięci przez przenoszenie nieużywanych modułów na CPU. Pełny latent 600x768; początek z losowego szumu. Dwa filmy VAL, po jednej klatce, dwa seedy. Naturalny forward ViT bez podmiany aktywacji. Wszystkie wartości skończone.

| Film | Seed | Względny MSE | Cosine | Względny MSE ROI | MSE innego warunku | Confidence | IoU układu źródłowego |
|---|---:|---:|---:|---:|---:|---:|---:|
| TR_0127_S1_P03 | 20261007 | 0.5119 | 0.6989 | 1.0208 | 0.7679 | 0.6799 | 0.0598 |
| TR_0127_S1_P03 | 20261008 | 0.2946 | 0.8403 | 0.8432 | 0.5856 | 0.8709 | 0.0000 |
| TR_0129_S1_P03 | 20261007 | 0.5575 | 0.6882 | 1.0394 | 0.9455 | 0.6243 | 0.2156 |
| TR_0129_S1_P03 | 20261008 | 0.6156 | 0.7153 | 1.4611 | 1.0250 | 0.8989 | 0.1259 |

Kontrola zamienia latenty filmów przy identycznym szumie. Ten mały test służy kontroli działania kopii i oględzinom, nie jest niezależnym testem generalizacji: obrazy pochodzą z już używanego zbioru VAL. Nie zapisano obrazów ani latentów na dysku.

Nie wykonano generacji ze sztucznego celu: brak potwierdzonego celu utworzonego przez K8000. Nie zastąpiono go niezgodnym celem K1000/K6402. To osobne ograniczenie, nie błąd naturalnego testu.

Podgląd obrazów w RAM: http://127.0.0.1:62614/ — działa, dopóki proces lokalnego podglądu pozostaje uruchomiony.

## Zachowane wyniki i sprzątanie

Model lokalny: [best_sd.pt](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500/best_sd.pt). Na polecenie użytkownika 8 października przeniesiono go do `assets/SD_ViT_K8000_A_control_K10500`, potwierdzając identyczny SHA256. [Opis modelu](F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/SmallViTAblation/assets/SD_ViT_K8000_A_control_K10500/README.md) zawiera pochodzenie, wyniki, ograniczenia i sposób wczytania. Kod, logi, manifest, metryki, historyczne provenance i sumy kontrolne pozostają w `eden_kopia`. Zweryfikowano 105 małych plików.

Z nowej rodziny na Edenie usunięto 14 zbędnych checkpointów smoke i recovery, odzyskując 17.82 GB (16.60 GiB). Zachowano najlepszy model każdego z czterech wariantów oraz wszystkie małe artefakty. Nie dotykano poprzednich eksperymentów, danych ani teachera. Szczegółowa lista: RAPORT_SPRZATANIA.json.

Sprawdzono /dev/shm na hopper przez krótki, uwierzytelniony job SLURM 1823778: brak katalogów scratch wszystkich trzech zadań tej rodziny. Nie zmieniono konfiguracji ani weryfikacji kluczy SSH.

## Wniosek

Kontynuacja bazowego przepisu pomogła liczbowo, ale badane ulepszenia nie poprawiły głównej miary względem kontroli. Najlepszy model nadal nie odtwarza wiernie narzędzia. Nie uruchomiono kolejnego treningu. Kolejny eksperyment powinien najpierw sprawdzić, czy generator potrafi wiernie zapamiętać niewielki zbiór naturalnych par, i porównać globalne cross-attention z warunkiem zachowującym jawnie siatkę przestrzenną. Są to propozycje dalszego testu, nie potwierdzone rozwiązania.
