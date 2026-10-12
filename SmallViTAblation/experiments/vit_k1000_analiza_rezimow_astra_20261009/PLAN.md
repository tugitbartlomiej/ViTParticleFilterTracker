# Audyt istniejącego oracle1500 i korekt A/B

Pytanie: co rzeczywiście wykonano, dlaczego transfer nie poprawił detekcji i jakie dane pozwalają analizować wspólne ruchy co krok?

To diagnostyka CPU istniejących artefaktów, bez treningu, ponownego inference obrazów i nowych jobów. Zapis wyłącznie w tej rodzinie experiments/outputs. Na jawne polecenie użytkownika bez modyfikacji wspólnego rejestru rodzin, AU i pracy Hubble'a.

1. Rozliczyć kod względem zachowanych inventory, wdrożenia Eden i logów.
2. Policzyć z lokalnych zapisów udział kasowania ruchu, rozdział faz, rozkład celu per patch, trafność B względem majority oraz skalę uśrednienia. To opis istniejących danych, nie nowy fit.
3. Dla 16 istniejących plików panelu policzyć kroki w pełnych 768D, zgodność odtworzenia zapisanej aktualizacji Adam z clamp, reprezentatywność tokenów, energię nullspace fc2 dla endpointów oraz błędy istniejącego DMD/DMDc według faz. Bez ponownego fit operatorów. Punkty odniesienia: persistence i analityczna aktualizacja Adam ze znanym gradientem/pamięcią. Oba widzą jawnie podane informacje; Adam jest kontrolą osi czasu, nie korektorem inference.
4. Hipoteza tłumienia tła pozostaje nierozstrzygnięta przyczynowo bez ablacji interwencji. Liczby wyjaśniają ograniczenia i mogą obalić zbyt mocne poprzednie twierdzenia.
5. Podać projekt dalszego testu na odłożonych filmach. Niczego z tego projektu teraz nie trenować.

Źródła surowych danych są tylko odczytywane. Skrypty nie importują runtime eksperymentów (unikamy pycache poza ownership). Przy odczycie zdalnym włączone PYTHONDONTWRITEBYTECODE=1 i jeden wątek CPU. Diagnostyka zdalna drukuje JSON na stdout, zapis wykonuje lokalny skrypt.
