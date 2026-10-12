# Kontrolowane obrazy i semantyka ruchu

Zatwierdzony przez użytkownika kolejny eksperyment. Cel: wyznaczyć bardziej selektywne sondy położenia, koloru i proporcji na obrazach zmieniających jeden czynnik, sprawdzić ich przeniesienie na naturalne klatki i związek z ruchem podczas uczenia.

Jednolite RGB, szarości i gradienty są kontrolą odpowiedzi na kolor/jasność i fałszywych detekcji, nie przykładami narzędzia. Dwanaście instrumentów wycinamy z istniejących masek adnotacji, skalujemy do stałego obszaru i umieszczamy na stałym szarym tle. Osobno przesuwamy obiekt, zmieniamy chromatyczność w Lab lub deformujemy jego proporcje z zachowaniem środka. Nie zmieniamy tła. Zmiana proporcji jest geometrią 2D, nie nową kategorią narzędzia ani nowym widokiem 3D.

FastGAN: cztery próbki zamrożonego EMA checkpoint_050000.pth, stałe ziarna. Kanał maski jest maską generatora, nie prawdziwą adnotacją; obrazy nie są używane do uczenia detektora. Kontrolowane warianty tworzymy dopiero po wygenerowaniu, aby generator nie zmieniał kilku czynników naraz. SD może służyć dodatkowej wizualizacji, lecz nie jest precyzyjną miarką czynników. Nie ładujemy go przy zajętej pamięci urządzenia.

Checkpointy tej samej rodziny: 0, 250, 500, 750, 1000. Osiem rzeczywistych wycinków dopasowuje rank-8 SVD sond kanałowych. Cztery pozostałe i FastGAN sprawdzają selektywność. Mierzymy także selektywne kierunki kontrastujące odpowiedź na jeden czynnik z pozostałymi (regularizowana uogólniona dekompozycja własna), bez dopasowywania do wyników detekcji.

Na niezmienionych naturalnych obrazach: IoU ramki o najwyższej pewności, udział energii ruchu w podprzestrzeniach wcześniejszego checkpointu oraz interwencja cofająca tę część ruchu w ostatniej gałęzi MLP. Losowa kontrola ma ten sam wymiar i normę ingerencji. Nakładające się udziały nie są sumowalnym podziałem ruchu.

Wszystkie rzeczywiste kotwice pochodzą z treningu sieci. Podział fit/test dotyczy sondy, nie generalizacji detektora. Jeden przebieg i rzadkie checkpointy: wynik pilotażowy, korelacje wyłącznie opisowe. Pomiary w 768D, bez rzutu 3D; 600 tokenów na obraz.
