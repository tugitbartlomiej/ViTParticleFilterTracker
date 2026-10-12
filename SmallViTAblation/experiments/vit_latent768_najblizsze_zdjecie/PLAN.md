# Najbliższe rzeczywiste zdjęcie

Pytanie: które z400 dostępnych zdjęć ma naturalny latent600×768 najbliższy sztucznemu `after`? Zamrożony checkpoint1000, identyczna warstwa poGELU i preprocessing480×320. Bez zmian wag, aktywacji, obrazów i bez używania etykiet do rankingu.

Główne kryterium wyboru: surowy MSE wszystkich768 współrzędnych, po wszystkich600 tokenach, bez uśredniania reprezentacji. Dodatkowo MSE128, MSE640, cosinus pełnego wektora oraz ważona energia jak w próbieSD. Rankingujemy również128 samodzielnie i sprawdzamy, czy zwycięzca jest ten sam.

Kontrole: naturalny latent zdjęcia źródłowego29 oraz końcowy obrazSD. Źródłowe zdjęcie jest pełnoprawnym kandydatem, nie ukrywamy jego zwycięstwa. Kadr i pozycja tokenów nie są wyrównywane. Cały ranking400 jest zapisywany. Najbliższy obraz nie musi być bliski bezwzględnie. Odległość nie jest detekcją ani dowodem możliwości odtworzenia sztucznego celu.

Po wyborze powtarzamy forward pojedynczego zdjęcia i sprawdzamy zgodność z pomiarem paczkowym. Zapisujemy oryginalnyJPG bez zmian, pełny latent, cel, różnicę, wybrane128 kanałów i wynik naturalnego detektora. Wagi muszą pozostać niezmienione. Brak przewagi nad oryginalnym zdjęciem i duży pozostały dystans są raportowane jako wynik negatywny poszukiwania nowej realizacji celu.
