# Wynik zakończonego pilota lokalnego

Data: 2026-09-24. Status: zakończony zgodnie z regułą zatrzymania.

Pretrening RF-DETR na Edenie zakończył cztery epoki. Walidacja: około 0,6574 mAP.
Model, historia latentu i stała sonda zostały pobrane oraz zweryfikowane SHA256.
Indeks obrazów i analiza Koopmana wykonane lokalnie. Lokalny przebieg po pobraniu,
wraz z końcową obsługą oceny, trwał około 135 sekund; nie obejmuje generowania.

Prognoza na horyzoncie całej epoki, na odłożonych operacjach, nie przeszła
ustalonego kryterium przewagi nad obiema prostymi metodami.
Średni błąd wektora: Koopman 0,218159; brak zmiany 0,137424; trend liniowy 0,185775.
Niższy wynik jest lepszy. Są to błędy prognozy latentu, nie błędy wykrywania.

Nie uruchomiono FastGAN-a ani gałęzi control/prim/random. Nie wybrano nowych
klatek i nie ma wyniku poprawy detektora po doborze. Nie zmieniono progów.
Pusta partycja test została jawnie oznaczona jako unavailable.
Wynik dotyczy tego operatora, reprezentacji i historii; nie obala wszystkich
możliwych sposobów użycia Koopmana. Automatyczne monitorowanie zakończono.

Źródła: wyniki/local_1810448/run/REPORT.json,
wyniki/local_1810448/run/round_01/FORECAST.json,
wyniki/local_1810448/run/pretrain/test/REPORT.json.
