# Koopman: prognoza wag i ruch cząsteczek

Przegląd z 2026-10-01. Najmocniejszy kandydat na zapamiętany artykuł to **Dogra i Redman, Optimizing Neural Networks via Koopman Operator Theory, NeurIPS 2020**. Jeśli zapamiętanym elementem był uczony autoenkoder, mocnym kandydatem jest **Koopman Message Passing, Yeh i in., 2023**. Sam opis nie pozwala wskazać tytułu z pewnością.

Wspólny związek jest ścisły: token stałego obrazu jest obserwablą pełnego stanu uczenia. Nie wynika z tego zamknięta dynamika samego tokenu; momenty AdamW i bufory BN mogą zmieniać przyszłość mimo podobnej obecnej reprezentacji. Potrzebny jest test prognoza wag+BN→forward wobec bezpośredniej prognozy pełnych tokenów.

- [Raport i wyprowadzenie](RAPORT_2026-10-01.md).
- [Źródła, zakres odczytu i ograniczenia](ZRODLA_2026-10-01.md).
- [Protokół falsyfikowalnego porównania](PROTOKOL_POROWNANIA_2026-10-01.md).
- [Surowe ekstrakcje pełnych tekstów — część 1](zrodla/undermind_odczyt_pdf_1_2026-10-01.md) i [część 2](zrodla/undermind_odczyt_pdf_2_2026-10-01.md).

Przesiano 64 rekordy z jednego wyszukiwania Undermind; osiem prac odczytano pełnotekstowo przez osobne zadania, a najważniejsze definicje dodatkowo sprawdzono w pierwotnych tekstach. Finalny PDF PDT pozostał zablokowany; jego opis opiera się jawnie na oficjalnym repo autorów. Nie pobrano dużych trajektorii, nie uruchomiono treningów ani zadań Eden, nie wykonano porównania numerycznego ani commitu.

Status protokołu: **propozycja przyszłego eksperymentu**. Istniejące wagi są zapisane co 10 aktualizacji, dlatego główne porównanie korzysta z horyzontów 10, 20, 50 i 100. Obecne wyniki skalarnego rzutu LDA nie są wynikiem prognozy 600 tokenów po 768 współrzędnych.
