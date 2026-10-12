# Pilot finite-lag EDMD, learned lifting i obserwatora reszt

Protokół zapisany przed estymacją 2026-10-01. CPU, jeden wątek Torch, maksymalnie18 epok małego autoenkodera na skalarnych parach. Bez uczenia ViT i bez GPU/Edenu. Stan to nadal zagregowany s, nie trajektorie600 cząsteczek.

## Podział i zakres

H=1,3,5,10,20,50 aktualizacji. Tylko lokalne istniejące s47/s48. Walidacja s47 starty1800..2000 co5, końcowy refit do2500, test eksploracyjny s48 starty2100..2950 co5 oraz s47 starty2500..2950. S48 był wcześniej oglądany; potrzebny świeży przebieg przed potwierdzeniem.

Wewnętrzne chronologiczne rozdzielenie: lifting/PCA/dryf dopasować na s47 z końcem etykiety<=1400; pamięć i wejścia na późniejszych parach z początkiem>=1400 i końcem<=1800. W reficie odpowiednio2000 i2500. To umowna kolejność projekcji. Druga część uczy się reszt dryfu poza jego dopasowaniem, a nie identyfikuje unikalnego mechanizmu fizycznego.

## Modele i wejścia

Fixed EDMD: pełny lifting psi=[1,x,x²,x³], gdzie x=s−m_t, m_t=średnia ostatnich500 stanów. Learned lifting: encoder1->16->4 i decoder4->16->1 z rekonstrukcją, stratą dynamiki obserwacji i latentną stratą liniowej aktualizacji. Encoder uczy się na dojrzałych parach jednokrokowych tylko w części dryfu. Następnie K_H fituje cały lifting celu s_(t+H)−m_t względem liftingu s_t−m_t. Referencja m_t jest zamrożona w prognozie. Każdy H ma osobny operator; bez gwarancji semigrupy, globalnej linearyzacji lub domknięcia. Ridge wokół operatora tożsamości.

Osobne ramiona: sam dryf; dryf+pamięć20 przyrostów; dryf+pamięć+cechy obrazów historycznych paczek; dodatkowo historyczne pchnięcia paczki/AdamW. Cechy obrazowe pochodzą z paczka_lat rzeczywistych obrazów przed aktualizacją: średnia768 cech narzędzia w obecnych przykładach i średnia768 cech całego obrazu. To jawny pooling obrazów paczki, a nie zastępstwo600 trajektorii tokenów. PCA8 dopasowana tylko na historycznej części dryfu s47; kolejne cechy transformowane zamrożoną bazą. Historia wejść to ostatnia paczka i średnie5/20 ukończonych paczek. Wiersz źródłowy k powoduje stan k+1; q[t] odpowiada wyłącznie k=t−1.

## Obserwator

Augmentacja eta_(j+1)=rho eta_j+w_j, pomiar dojrzałej reszty e_H(j−H)=eta_j+nu_j. Parametry rho/Q/R z treningowych reszt; względny udział R wybrać na walidacji z0.1/0.5/0.9. Filtr przyczynowy aktualizuje tylko z zakończonych etykiet. Korekta prognozy w t to rho^H eta_t, nie przyszła reszta e_H(t). Proces/pomiar Gaussian to roboczy model drugich momentów; nie dowód rozkładu. Nakładanie H-celów może dawać kolorowy szum pomiarowy i naruszać założenie diagonalnegoR, dlatego wynik filtrowania jest hipotezą do falsyfikacji.

EKF/UKF pozostają kolejną opcją po sprawdzeniu konkretnego modelu nieliniowego i obserwowalności, nie są zastępowane tym skalarnym KF. Dla rozszerzonej realizacji AR1 pomiar eta jest bezpośredni; to nie dowodzi obserwowalności pełnego ViT. Testować rangi/kondycję macierzy regresji, błąd domknięcia pełnego liftingu, reszty, kalibrację i stabilność operatora. Identyfikacja z historycznych danych jest oddzielona od marginalnej prognozy bez przyszłych paczek; nie wyznaczamy odpowiedzi na dowolną rzeczywistą przyszłą paczkę.

## Ocena i falsyfikacja

Wszystkie warianty na tych samych oknach. Wybór przez MSE walidacji. Zachować warianty bez obserwatora i wszystkie pogorszenia. MAE/RMSE/redukcjaMSE, znak wszystkich sond, balanced accuracy, kontrola większości z przeszłości, przedziały walidacyjne i maksymalny zmierzony H z progiem80 dla każdej osobnej miary. Kontrole starego Langevina i wcześniejszego direct lifting+GLE+wpływów z odrębnego pilota porównać wyłącznie na zgodnych zapisanych oknach. To bezpośrednie rodziny, nie EDMD. Test prefix/suffix musi obejmować operator, obrazowe wejście i opóźnione reszty.

Brak przewagi, zysk wyłącznie przy użyciu e_H(t) z przyszłym końcem etykiety lub niestabilny wynik między s47/s48 obala daną poprawkę. Nowy pilot nie nadpisuje wcześniejszych metryk, prognoz ani raportów.
