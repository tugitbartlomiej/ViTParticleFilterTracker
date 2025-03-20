import os

import cv2

# 1) Folder, z którego wczytujemy obrazy do przeglądania
PATH_INPUT = r"../DeepSortYolo/ProcessedVideos/sort_id_changes"

# 2) Folder, z którego usuwamy TYLKO pliki JPG
PATH_REMOVE_JPG = r"../DeepSortYolo/ProcessedVideos/RAW"

# 3) Folder, z którego usuwamy pliki JPG ORAZ odpowiadające im pliki TXT
PATH_REMOVE_JPG_TXT = r"../DeepSortYolo/ProcessedVideos/RAW"
 
# Ustal maksymalne wymiary okna/obrazu (w pikselach)
MAX_WIDTH = 1280
MAX_HEIGHT = 720


def remove_file_if_exists(filepath: str) -> None:
    """
    Usuwa plik, jeśli istnieje, w przeciwnym razie nic nie robi.
    """
    if os.path.isfile(filepath):
        try:
            os.remove(filepath)
            print(f"Usunięto: {filepath}")
        except Exception as e:
            print(f"Błąd przy usuwaniu {filepath}: {e}")


def show_image(image_path: str) -> bool:
    """
    Wczytuje i wyświetla obraz w oknie 'Podglad' (z możliwością ręcznej regulacji rozmiaru).
    Skaluje obraz do MAX_WIDTH x MAX_HEIGHT (z zachowaniem proporcji), jeśli jest za duży.
    Zwraca True, jeśli obraz został poprawnie wczytany, w przeciwnym razie False.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie można wczytać pliku: {image_path}")
        return False

    # Sprawdź aktualne wymiary obrazu
    h, w = image.shape[:2]

    # Wylicz współczynnik skalowania, aby nie przekroczyć maksymalnych wymiarów
    scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
    if scale < 1:
        # Zmniejsz obraz, aby zmieścił się w ustalonym rozmiarze
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imshow("Podglad", image)
    return True


def main() -> None:
    # Ustawiamy okno w tryb WINDOW_NORMAL, aby można je było ręcznie regulować
    cv2.namedWindow("Podglad", cv2.WINDOW_NORMAL)
    # Opcjonalnie można ustawić początkowy rozmiar okna
    cv2.resizeWindow("Podglad", MAX_WIDTH, MAX_HEIGHT)

    # Zbierz listę plików JPG z folderu wejściowego
    files = [f for f in os.listdir(PATH_INPUT) if f.lower().endswith(".jpg")]
    files.sort()  # Posortuj listę wg nazwy (opcjonalnie)

    if not files:
        print("Brak plików JPG w folderze wejściowym.")
        return

    print("Sterowanie klawiszami:")
    print("  - [A] => Poprzedni obraz")
    print("  - [D] => Następny obraz")
    print("  - [SPACJA] => Usuń powiązane pliki w folderach 2 i 3 (JPG i TXT)")
    print("  - [ESC] lub [q] => Zakończ program")

    idx = 0  # indeks aktualnie wyświetlanego obrazu

    # Wyświetl pierwszy obraz
    current_path = os.path.join(PATH_INPUT, files[idx])
    if not show_image(current_path):
        return

    while True:
        key = cv2.waitKey(0)

        # ESC (27) lub 'q' => zakończ
        if key == 27 or key == ord('q'):
            break

        # A => poprzedni obraz
        elif key == ord('a') or key == ord('A'):
            idx = max(0, idx - 1)
            current_path = os.path.join(PATH_INPUT, files[idx])
            show_image(current_path)

        # D => następny obraz
        elif key == ord('d') or key == ord('D'):
            idx = min(len(files) - 1, idx + 1)
            current_path = os.path.join(PATH_INPUT, files[idx])
            show_image(current_path)

        # Spacja (32) => usuń powiązane pliki
        elif key == 32:
            filename = files[idx]
            # 1) Usuń plik JPG z folderu PATH_REMOVE_JPG
            remove_file_if_exists(os.path.join(PATH_REMOVE_JPG, filename))

            # 2) Usuń plik JPG z folderu PATH_REMOVE_JPG_TXT
            remove_file_if_exists(os.path.join(PATH_REMOVE_JPG_TXT, filename))

            # 3) Usuń plik TXT (o tej samej nazwie, tylko z rozszerzeniem .txt) z folderu PATH_REMOVE_JPG_TXT
            base_name, _ = os.path.splitext(filename)
            txt_path = os.path.join(PATH_REMOVE_JPG_TXT, base_name + ".txt")
            remove_file_if_exists(txt_path)

            print(f"Usunięto powiązane pliki dla: {filename}")

    cv2.destroyAllWindows()
    print("Zakończono działanie skryptu.")


if __name__ == "__main__":
    main()
