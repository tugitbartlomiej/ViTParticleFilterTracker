# Następne sprawdzenie: kiedy Koopman i Langevin zaczynają przewidywać tożsamość

Data: 2026-09-28. Status: **plan użytkownika do następnego sprawdzenia**. Nic nie uruchomiono. Przed zgłoszeniem
opisać szczegóły i czekać na zgodę.

## Słowa użytkownika (2026-09-28, sedno)

> na początku to sieć nic nie wie i ona się bardzo powoli uczy, więc na początku [Koopman/Langevin] nie będzie w stanie
> nic przewidywać […] przy pierwszych epokach […]. Jak zacznie się uczyć, no to po prostu zaczyna się zawsze
> przewidywanie i dopiero wtedy możemy obserwować po prostu to, co wychodzi z tożsamości.
>
> Trzeba to jakoś ugryźć dobrze, przy pomocy Koopmana, Langevin'a i zobaczyć jak to tam się zaczyna kreować na
> początku […] pierwsza, druga epoka, no to są takie epoki, w których Koopman i Langevin mają prawo się pomylić, więc
> te pomiary być może nie są zbyt dobre, natomiast w kolejnych pomiarach […] Koopman i Langevin powinny zyskiwać […]
> powinni wskazywać przebieg, w którą stronę latent będzie się uczył i co jest tożsamością w latencie dla narzędzia,
> pełną i rdzeniową […] sukcesywnie zwiększamy po prostu zbiór uczący, no i musimy to porównywać do losowego
> zwiększania różnymi klatkami, albo nawet […] dokładania kolejnych z rzędu klatek do uczenia.

## Co z tego wynika do zmierzenia

1. **Jakość prognozy w funkcji czasu uczenia.** Trzeba zmierzyć, od którego obrotu Koopman i Langevin zaczynają
   przewidywać lepiej niż założenie „bez zmian”.
   - Liczymy test wsteczny w każdym obrocie od kroku 0, także w rozgrzewce.
   - Oczekiwanie użytkownika: w pierwszych 1–2 epokach prognoza może się mylić, a potem stosunek do „braku zmiany”
     powinien spadać poniżej 1.
   - Pierwsze epoki raportujemy osobno i nie mieszamy ich z oceną metody.
2. **Narodziny tożsamości.** Obserwujemy, jak od początku uczenia powstaje pełny latent i rdzeń tożsamości
   narzędzia.
   - Śledzimy, kiedy pojawiają się pierwsze neurony, które zapalają się tylko na narzędziu.
   - Śledzimy, które warianty sond domykają się najpierw: obrót, tło, długość, jasność.
   - Sprawdzamy, czy Koopman i Langevin przewidują kierunek tego procesu, czyli które neurony urosną, a które
     zmaleją.
3. **Sterowanie dopiero po starcie przewidywania.** Klatki wybierane przez Koopmana i FastGAN liczą się dopiero
   wtedy, gdy prognoza zaczyna bić „brak zmiany”. Wcześniej można dokładać klatki tak samo jak w ramieniu losowym.
4. **Punkty odniesienia przy stopniowym powiększaniu zbioru uczącego:**
   - `losowe`: te same liczby losowych klatek z puli;
   - `kolejne z rzędu` (nowe ramię): kolejne klatki tych samych filmów, na przykład następne po klatkach bazy
     w czasie nagrania;
   - `bez`: nauka bez dodawania klatek.

## Na czym się oprzeć

Pętla `petla.py` już zapisuje latent sond co 10 kroków od kroku 0, więc punkty 1–2 da się częściowo policzyć bez
nowego GPU na istniejących `LATENT.npz`.

Dotyczy to:
- ziarna 42 lokalnie;
- ziaren 47–56 z Edenu (zadanie 1817935, `~/smallvit_runs/2026-09-28_maly_vit_od_zera/wyniki`), po ich pobraniu.

Nowe ramię `kolejne z rzędu` i próg „sterowanie dopiero po starcie przewidywania” wymagają zmiany kodu i nowej
kampanii, więc potrzebują zgody.
