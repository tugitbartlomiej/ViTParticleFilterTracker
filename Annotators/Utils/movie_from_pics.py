import cv2
import os
import argparse
import re # Do sortowania numerycznego
from pathlib import Path
from tqdm import tqdm # Opcjonalny pasek postępu

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    """ Klucz sortujący dla naturalnego porządku (np. 'img1', 'img2', 'img10'). """
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(str(s))]

def create_video_from_frames(input_dir, output_file, fps=30, image_extension='png'):
    """
    Tworzy plik wideo z klatek obrazów w danym katalogu.

    Args:
        input_dir (str): Ścieżka do katalogu zawierającego klatki obrazów.
        output_file (str): Ścieżka do wyjściowego pliku wideo (np. 'output.mp4').
        fps (int): Liczba klatek na sekundę dla wyjściowego wideo.
        image_extension (str): Rozszerzenie plików obrazów do wyszukania (np. 'png', 'jpg').
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        print(f"Błąd: Katalog wejściowy '{input_dir}' nie istnieje.")
        # Sugestia dla użytkownika, jeśli używa domyślnej ścieżki
        if str(input_path) == './frames':
            print("Upewnij się, że istnieje podkatalog 'frames' w bieżącym katalogu lub podaj inną ścieżkę za pomocą -i.")
        return

    # Utwórz katalog wyjściowy, jeśli nie istnieje
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Wyszukiwanie klatek z rozszerzeniem '.{image_extension}' w '{input_dir}'...")
    image_files = list(input_path.glob(f'*.{image_extension.lower()}'))

    if not image_files:
        print(f"Błąd: Nie znaleziono plików '.{image_extension}' w '{input_dir}'.")
        return

    try:
        image_files.sort(key=natural_sort_key)
        print(f"Znaleziono {len(image_files)} klatek. Posortowano.")
    except Exception as e:
        print(f"Ostrzeżenie: Wystąpił problem podczas sortowania plików ({e}). Używam sortowania alfabetycznego.")
        image_files.sort()

    try:
        first_frame = cv2.imread(str(image_files[0]))
        if first_frame is None:
             print(f"Błąd: Nie można odczytać pierwszej klatki: {image_files[0]}")
             return
        height, width, layers = first_frame.shape
        size = (width, height)
        print(f"Wymiary klatek: {width}x{height}")
    except Exception as e:
        print(f"Błąd podczas odczytu wymiarów pierwszej klatki: {e}")
        return

    fourcc = None
    output_suffix = output_path.suffix.lower()
    if output_suffix == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Wybrano kodek: mp4v (dla .mp4)")
    elif output_suffix == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        print("Wybrano kodek: XVID (dla .avi)")
    elif output_suffix == '.mov':
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         print("Wybrano kodek: mp4v (dla .mov)")
    else:
        print(f"Ostrzeżenie: Nieznane rozszerzenie pliku wyjściowego '{output_suffix}'. Używam domyślnego kodeka XVID i rozszerzenia .avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.with_suffix('.avi')

    try:
        out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
        if not out.isOpened():
             if output_suffix == '.mp4':
                 print("Kodek 'mp4v' nie zadziałał, próba z 'avc1'...")
                 fourcc = cv2.VideoWriter_fourcc(*'avc1')
                 out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
                 if not out.isOpened():
                     print("Kodek 'avc1' również nie zadziałał. Sprawdź instalację OpenCV/ffmpeg.")
                     print("Próba z kodekiem 'XVID' i rozszerzeniem .avi...")
                     fourcc = cv2.VideoWriter_fourcc(*'XVID')
                     output_path = output_path.with_suffix('.avi')
                     out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
             if not out.isOpened():
                 raise IOError(f"Nie można otworzyć pliku wideo do zapisu: {output_path}. Sprawdź kodek ({fourcc:#04x}) i uprawnienia.")
    except Exception as e:
        print(f"Błąd inicjalizacji VideoWriter: {e}")
        return

    print(f"Tworzenie wideo: '{output_path}' (FPS: {fps})")

    try:
        for i, filename in enumerate(tqdm(image_files, desc="Dodawanie klatek")):
            img = cv2.imread(str(filename))
            if img is None:
                print(f"\nOstrzeżenie: Pominięto klatkę - nie można odczytać: {filename}")
                continue
            h, w, _ = img.shape
            if (w, h) != size:
                print(f"\nOstrzeżenie: Pominięto klatkę - niezgodny rozmiar ({w}x{h} vs {size[0]}x{size[1]}): {filename}")
                continue
            out.write(img)
    except Exception as e:
        print(f"\nWystąpił błąd podczas dodawania klatki: {e}")
    finally:
        out.release()
        print(f"\nZakończono tworzenie wideo. Plik zapisano jako: '{output_path}'")

# --- Obsługa argumentów linii poleceń ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tworzy wideo z sekwencji klatek obrazów.")

    # --- Argumenty z dodanymi wartościami domyślnymi ---
    parser.add_argument("-i", "--input_dir", type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/detr_predictions_with_heatmaps', # Domyślny katalog wejściowy
                        help="Katalog zawierający klatki obrazów (domyślnie: ./frames).")
    parser.add_argument("-o", "--output_file", type=str,
                        default='F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/detr_heatmap_predictions.mp4', # Domyślny plik wyjściowy
                        help="Ścieżka do wyjściowego pliku wideo (np. output.mp4, output.avi; domyślnie: ./output_video.mp4).")
    # --- Pozostałe argumenty bez zmian ---
    parser.add_argument("-r", "--fps", type=int, default=30,
                        help="Liczba klatek na sekundę (domyślnie: 30).")
    parser.add_argument("-e", "--extension", type=str, default='png',
                        help="Rozszerzenie plików obrazów (bez kropki, np. 'png', 'jpg', domyślnie: 'png').")

    args = parser.parse_args()

    create_video_from_frames(args.input_dir, args.output_file, args.fps, args.extension)