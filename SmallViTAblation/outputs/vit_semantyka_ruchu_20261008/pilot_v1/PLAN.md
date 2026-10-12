# Semantyka ruchu tokenów: eksperyment pilotażowy

Jedna rodzina checkpointów gęstej historii, kroki 0–1000 co 50. Dwanaście stałych obrazów: osiem do dopasowania sond, cztery do ich niezależnego sprawdzenia. Wszystkie obrazy pochodzą ze zbioru treningowego sieci. Nie jest to test generalizacji detektora.

Przestrzeń pomiaru: 600 tokenów × 768 współrzędnych, wyjście GELU ostatniego MLP. Nie zmieniamy GUI, wag ani danych źródłowych.

Operacyjne czynniki: przesunięcie całego obrazu ±16 px w poziomie/pionie; zmiana chromatyczności narzędzia w masce (Lab a/b ±12); deformacja proporcji całego kadru względem środka ramki narzędzia, skala 1.15 i odwrotność, zachowana powierzchnia. Ostatnia sonda dotyczy geometrii/proporcji i zmienia także tło: nie dowodzi izolacji pojęcia kształtu.

Dla każdego checkpointu i czynnika SVD odpowiedzi tokenów na perturbację wyznacza podprzestrzeń o ośmiu kierunkach. Dopasowanie tylko na ośmiu obrazach, na tokenach w rozszerzonej masce narzędzia. Zapisujemy odpowiedzi i udział energii wyjaśniany na czterech odłożonych obrazach oraz nakładanie się podprzestrzeni. Udziały nie sumują się do jedności.

Ruch między checkpointami mierzymy dla niezmienionych obrazów, w pełnych 768 wymiarach. Rzutujemy go na podprzestrzeń z wcześniejszego checkpointu. Mierzymy związek udziału ruchu z równoczesną zmianą IoU ramki o najwyższej pewności. To analiza retrospektywna, nie prognoza ani dowód przyczynowości.

Interwencja funkcjonalna: na późniejszym checkpointcie cofamy w ostatniej gałęzi MLP tylko projekcję zmiany tokenów na wcześniejszą podprzestrzeń; reszta sieci i połączenie rezydualne pozostają rzeczywistą częścią aktualnego forward pass. Porównujemy IoU z niezmienionym latentem oraz pięcioma losowymi podprzestrzeniami o tym samym wymiarze. Dodatkowa losowa kontrola ma taką samą normę ingerencji. To lokalna interwencja w gałęzi, nie odtworzenie poprzedniej sieci. Edytowany latent może być poza rozkładem i zawierać wartości ujemne.

Raport obejmuje selektywność sond, ruch, interwencje, kontrolę losową, korelacje z przyrostami IoU, pliki CSV/NPZ i wykres. Brak podziału latentów na rozłączne moduły jest możliwym wynikiem. Jeden przebieg i cztery obrazy testujące sondy nie wystarczają do potwierdzenia hipotezy publikacyjnej.
