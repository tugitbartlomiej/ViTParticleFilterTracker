# Pełne wyszukiwanie w lokalnym zbiorzeTracking

Przeszukujemy wszystkie pliki JPG/JPEG/PNG we wszystkich170 lokalnych archiwach adnotacji Cataract-LMM Tracking, bez próbkowania czasowego i bez filtrów etykiet. Inwentaryzacja wykazała461547 plików: train310257, val64167, test82370, unknown4753. Nie utożsamiamy tego z pełnym katalogiem innych zadań Cataract-LMM ani z brakującymi klatkami, których nie ma w archiwach.

Zamrożony checkpoint1000, naturalny forward poGELU,600 tokenów i768 współrzędnych, wejście480×320RGB, bez augmentacji i podmiany aktywacji. Główne kryterium: surowy MSE całego600×768. Osobno rejestrujemy minimum dla train i MSE128/MSE640/cosinus. Etykiety nie biorą udziału w rankingu. Podział służy wyłącznie nazwaniu zakresu znalezionego przykładu; nie raportujemy wyniku generalizacji ani mAP.

Kontrole: wcześniejsze minimum400, oryginał29, wygenerowanySD. Wszystkie mają ten sam cel i checkpoint. Większa pula nie gwarantuje bliskiej realizacji celu. Kryterium negatywne: utrzymujący się duży błąd i znikoma poprawa nad400. Raportujemy dokładną liczbę przetworzonych zdjęć, błędy odczytu i wszystkie surowe wynikiCSV. Nie pomijamy błędów: bieg kończy się failure zamiast fałszywej deklaracji pełnego pokrycia.

Strumieniowy odczytZIP, równoległe dekodowanieCPU, paczki16GPU, pełna precyzjaFloat32. Finalne64 najlepsze kandydatury globalne i64 treningowe przeliczamy pojedynczo dla kontroli błędu paczkowego. Zapis oryginalnego zdjęcia, pełnego latentu i J128 w outputs/vit_latent768_pelny_zbior. Log postępu umożliwia obserwację długiego przebiegu. Źródła tylko do odczytu.

## Diagnostyka koniecznych granic aktywacji

Równolegle `diagnostyka_granic.py` bada ograniczenie wynikające z LayerNorm192 przed FC1. Dla wektora standaryzowanegoq: suma współrzędnych0 i norma≤√192. Jeśli a=W_jγ oraz c=W_jβ+b_j, to h_j≤c+√192‖a-mean(a)‖. Ponieważ GELU(h)≤max(0,h), uzyskujemy górną granicę każdego kanału. Dodano0.01 zapasu numerycznego. Oryginalny latent musi spełniać wszystkie granice, inaczej odrzucamy diagnostykę. Przekroczenie granicy przez sztuczny cel wyklucza dokładne odtworzenie; brak przekroczeń sam nie dowodzi osiągalności. Diagnostyka nie zmienia celu ani rankingu zdjęć.
