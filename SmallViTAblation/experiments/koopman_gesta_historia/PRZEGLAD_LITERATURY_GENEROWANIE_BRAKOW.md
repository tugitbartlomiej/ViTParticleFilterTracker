# Co pokazuje obraz wygenerowany według prognozy latentu?

Data: 2026-09-24. Synteza literatury i interpretacja proponowanego eksperymentu. Nie jest to raport z wykonanej generacji ani potwierdzenie skuteczności doboru danych.

## Znaczenie eksperymentu

Obraz ma pokazać, jaki wygląd wejścia wywołuje zadaną odpowiedź wewnątrz obecnego detektora. Pozwala przełożyć liczby na wizualną hipotezę: być może zmieni się kąt narzędzia, światło, tło albo inna cecha. Nie przypisujemy takiego znaczenia przed obejrzeniem i kontrolą wyników.

Kluczowe rozróżnienie: Koopman przewiduje odpowiedź tego samego obrazu po zmianie wag podczas uczenia. Generator szuka innego obrazu przy niezmienionych wagach. To dwa różne działania. Ich powiązanie jest hipotezą naszego eksperymentu.

Nie ma ogólnego wynikania: przewidywany przyszły latent → brak wiedzy → najlepszy obraz do treningu. Nawet trafna prognoza może prowadzić do obrazu łatwego dla obecnego detektora. Nietrafna prognoza nadal może służyć jako cel wizualizacji, zgodnie z żądaniem użytkownika.

## Zapis matematyczny

Niech f_theta(x) oznacza latent detektora, a P jego ustalony odczyt w wybranych kierunkach narzędzia. Współrzędne muszą być porównywalne między checkpointami. Oznaczmy c_t(x) = P(f_theta_t(x)).

Koopman dopasowany do historii uczenia daje prognozę:

    c_hat_(t+h)(x) = K_h(c_t(x)).
    delta_hat = c_hat_(t+h)(x) - c_t(x).

FastGAN G otrzymuje własny wektor wejściowy u. Przy zamrożonych wagach generatora i detektora szukamy:

    u_star = argmin_u [ ||W(c_t(G(u)) - c_hat_(t+h)(x))||² + lambda R(u,G(u)) ].

W skaluje współrzędne według statystyk ustalonych na danych dopasowania. R ogranicza nierealistyczne rozwiązania lub utratę narzędzia; wymaga osobnego określenia i walidacji. Gradient przechodzi przez obie sieci, ale aktualizuje tylko u. Cel to obecny kod plus przyrost, a nie sam przyrost.

Mała wartość błędu oznacza zgodność wybranych współrzędnych. Nie gwarantuje zgodności całego latentu, poprawnej maski, błędu detekcji ani korzyści z uczenia. Rozwiązanie może nie istnieć w zakresie obrazów dostępnych generatorowi. Może też istnieć wiele różnych rozwiązań.

## Najbliższe prace

| Praca | Autorzy, rok | Co daje dla naszego schematu | Źródło |
|---|---|---|---|
| Distilling Model Failures as Directions in Latent Space | Saachi Jain, Hannah Lawrence, Ankur Moitra, Aleksander Madry; 2022, ICLR 2023 | Powtarzalne błędy opisane kierunkami cech; połączenie z dyfuzją do generowania trudnych obrazów. Punktem startowym są zmierzone błędy, nie prognoza przyszłego latentu. | [arXiv](https://arxiv.org/abs/2206.14754) |
| Adversarial Sampling for Active Learning | Christoph Mayer, Radu Timofte; preprint 2018, WACV 2020 | GAN tworzy przykłady o dużej niepewności; wyszukiwane są podobne prawdziwe obrazy z puli. Bardzo bliski odpowiednik etapu generacja → wyszukiwanie → uczenie. | [arXiv](https://arxiv.org/abs/1808.06671) |
| Show Me What You Don't Know: Efficient Sampling from Invariant Sets for Model Validation | Armand Rousselot, Joran Wendebourg, Ullrich Köthe; 2026 | Różne naturalne obrazy przy ustalonej reprezentacji ujawniają, jakie różnice model ignoruje. Niezmienność oznacza tu ten sam opis różnych obrazów, nie niezmiennik dynamiki Koopmana. | [arXiv](https://arxiv.org/abs/2603.21782) |
| DiG-IN: Diffusion Guidance for Investigating Networks | Maximilian Augustin, Yannic Neuhaus, Matthias Hein; CVPR 2024 | Sterowanie generacją celami wynikającymi z analizowanej sieci: wizualizacje neuronów, różnice klasyfikatorów i wyjaśnienia kontrfaktyczne. | [CVF, PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Augustin_DiG-IN_Diffusion_Guidance_for_Investigating_Networks_-_Uncovering_Classifier_Differences_CVPR_2024_paper.pdf) |
| Discovering Bugs in Vision Models using Off-the-shelf Image Generation and Captioning | Olivia Wiles, Isabela Albuquerque, Sven Gowal; 2022 | Generowanie, grupowanie błędnie rozpoznanych przykładów i opisywanie powtarzalnych problemów. | [arXiv](https://arxiv.org/abs/2208.08831) |
| Synthesizing the preferred inputs for neurons in neural networks via deep generator networks | Anh Nguyen, Alexey Dosovitskiy, Jason Yosinski, Thomas Brox, Jeff Clune; 2016 | Generator jako ograniczenie do wizualizacji aktywacji. Uzasadnia techniczny most odpowiedź sieci → optymalizacja wejścia generatora. | [arXiv](https://arxiv.org/abs/1605.09304) |
| Optimizing Neural Networks via Koopman Operator Theory | Akshunna S. Dogra, William T. Redman; 2020 | Prognozowanie wag podczas uczenia. Nie dowodzi, że zmiana kodu obrazu oznacza brak wiedzy lub określa wartościową próbkę treningową. | [arXiv](https://arxiv.org/abs/2006.02361) |

Zakres sprawdzenia: metadane i abstrakty źródeł pierwotnych, a dla wybranych prac również tekst artykułu lub opis autorów. To przegląd celowany, nie systematyczny ani dowód pierwszeństwa pomysłu. Uwaga bibliograficzna: ASAL ma adnotację WACV 2020, nie ICCV 2019.

## Co możemy uzyskać

1. Wizualizację tego, do jakiej odpowiedzi latentu prowadzi prognoza. Badamy, czy obraz jest realistyczny i czy cel został osiągnięty.
2. Test interpretacji kierunków. Powtarzające się zmiany wyglądu mogą podpowiedzieć ich znaczenie, ale trzeba wykluczyć artefakty i wpływ tła.
3. Kandydatów do badania słabości. Dopiero błędna detekcja przy wiarygodnej adnotacji pokazuje trudność danego obrazu.
4. Zapytanie do wyszukiwarki prawdziwych klatek. Osobny zamrożony opis obrazu i indeks FAISS mogą znaleźć wizualnie podobne przypadki. Sam FAISS nie określa znaczenia podobieństwa; zależy ono od wybranych cech.

ASAL jest ważnym precedensem dla punktu czwartego. Praca Jain i współautorów jest ważniejsza dla nazywania czegoś brakiem wiedzy: kierunek jest tam związany ze zmierzonymi błędami. Nasz sygnał z historii uczenia wymaga dopiero sprawdzenia takiego związku.

## Kontrole do rozważenia po podstawowej demonstracji

Przy tych samych startach generatora porównać cel obecny, cel przewidziany przez Koopmana i losowy przyrost o takiej samej wielkości. Mierzyć błąd dopasowania kodu, wiarygodność narzędzia i detekcję oddzielnie. To propozycja przyszłej diagnostyki, nie uruchomiona dodatkowa kampania.

Zgodnie z ustaleniem użytkownika demonstracja FastGAN ma powstać również po negatywnym teście prognozy. Zachowujemy rzeczywistą prognozę; nie zastępujemy jej prawdziwym przyszłym latentem. Wynik oznaczamy jako wizualizację niepotwierdzonego celu. Nie zmieniamy tym samym progów akceptacji Koopmana ani nie uznajemy obrazu za dowód braku wiedzy.

W ewentualnym późniejszym doborze danych trzeba porównać dalszy trening z dobranymi klatkami, bez nich oraz z losowym doborem przy jednakowym budżecie. Ocena na osobnych filmach rozstrzyga korzyść dla detekcji. Obecny etap nie daje jeszcze takiego wyniku.
