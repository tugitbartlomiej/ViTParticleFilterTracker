import csv
import os
from datetime import datetime

import cv2

# Parametry
video_path = 'E:/Cataract/videos/micro/train01.mp4'  # Ścieżka do pliku wideo
save_folder = 'zapisane_sekwencje'  # Główny folder do zapisywania sekwencji
sequence_length = 8  # Liczba klatek przed aktualną klatką do zapisania

# Upewnij się, że główny folder do zapisywania istnieje
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Inicjalizacja wideo
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# Lista do przechowywania informacji o zapisanych sekwencjach
saved_sequences_info = []

# Licznik sekwencji
sequence_counter = 0

# Funkcja zwrotna dla trackbara
def on_trackbar(val):
    global current_frame
    current_frame = val

# Tworzenie okna i trackbara
cv2.namedWindow('Frame Annotator')
cv2.createTrackbar('Pozycja', 'Frame Annotator', 0, total_frames - 1, on_trackbar)

# Instrukcje
instructions = [
    "Sterowanie:",
    "',' (przecinek) - poprzednia klatka",
    "'.' (kropka) - następna klatka",
    "'a'/'d' - skok o 20 klatek",
    "'+'/'-' - skok o 8 klatek",
    "Klawisze 1-4 - zapis sekwencji z odpowiednią etykietą",
    "'q' - wyjście",
    "Użyj paska do przewijania filmu"
]

# Definicja etykiet
labels = {
    1: "brak_zagrozenia",
    2: "zbliżanie_sie",
    3: "bezposrednie_zagrozenie",
    4: "przebicie"
}

def show_info(frame):
    """Wyświetlenie informacji na klatce."""
    # Wyświetl numer klatki
    cv2.putText(frame, f"Klatka: {current_frame}/{total_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Wyświetl instrukcje
    y = 70
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25

    return frame

# Zmienna do śledzenia ostatniej zapisanej klatki
last_saved_frame = -1

while True:
    # Ustawienie pozycji klatki
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    if not ret:
        print("Nie można odczytać klatki.")
        break

    # Aktualizacja pozycji trackbara
    cv2.setTrackbarPos('Pozycja', 'Frame Annotator', current_frame)

    # Dodaj informacje na klatkę
    display_frame = show_info(frame.copy())

    # Wyświetlenie klatki
    cv2.imshow('Frame Annotator', display_frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('.'):  # Kropka - następna klatka
        if current_frame < total_frames - 1:
            current_frame += 1
    elif key == ord(','):  # Przecinek - poprzednia klatka
        if current_frame > 0:
            current_frame -= 1
    elif key == ord('d'):  # 'd' - skok o 40 klatek do przodu
        current_frame = min(current_frame + 20, total_frames - 1)
    elif key == ord('a'):  # 'a' - skok o 40 klatek do tyłu
        current_frame = max(current_frame - 20, 0)
    elif key == ord('+') or key == ord('='):  # '+' - skok o 20 klatek do przodu
        current_frame = min(current_frame + 8, total_frames - 1)
    elif key == ord('-'):  # '-' - skok o 20 klatek do tyłu
        current_frame = max(current_frame - 8, 0)
    elif key >= ord('1') and key <= ord('4'):  # Klawisze 1-4 - zapis sekwencji
        label_number = key - ord('0')
        label_description = labels.get(label_number, f"nieznana_etykieta_{label_number}")

        # Określ zakres klatek do zapisania
        if label_number == 4:
            # Dla etykiety 4 (przebicie) zapisujemy sekwencję kończącą się na aktualnej klatce
            start_frame = max(last_saved_frame + 1, current_frame - sequence_length)
            end_frame = current_frame
        else:
            # Dla etykiet 1-3 zapisujemy sekwencję kończącą się tuż przed aktualną klatką
            start_frame = max(last_saved_frame + 1, current_frame - sequence_length)
            end_frame = current_frame - 1

        # Sprawdź, czy zakres klatek jest poprawny
        if start_frame > end_frame:
            print("Nie można utworzyć sekwencji: niewłaściwy zakres klatek.")
            continue

        # Aktualizuj ostatnio zapisaną klatkę
        last_saved_frame = end_frame

        # Inkrementacja licznika sekwencji
        sequence_counter += 1
        # Tworzenie unikalnej nazwy folderu dla sekwencji
        sequence_folder_name = f"sequence_{sequence_counter:04d}"
        sequence_folder_path = os.path.join(save_folder, sequence_folder_name)

        # Tworzenie folderu dla sekwencji
        os.makedirs(sequence_folder_path, exist_ok=True)

        # Zapisz description.txt z odpowiednią etykietą
        description_file = os.path.join(sequence_folder_path, 'description.txt')
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write(label_description)

        # Lista do przechowywania nazw plików klatek w tej sekwencji
        frames_in_sequence = []

        for i in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, seq_frame = cap.read()
            if ret:
                frame_filename = f"frame_{i:06d}.png"
                frame_path = os.path.join(sequence_folder_path, frame_filename)
                cv2.imwrite(frame_path, seq_frame)
                frames_in_sequence.append(frame_filename)
            else:
                print(f"Nie można odczytać klatki {i}.")
                continue

        print(f"Zapisano sekwencję {sequence_folder_name} od klatki {start_frame} do {end_frame} z etykietą '{label_description}'.")

        # Dodanie informacji o sekwencji do listy
        saved_sequences_info.append({
            'sequence_id': sequence_folder_name,
            'label': label_description,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames': ';'.join(frames_in_sequence),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # **Nowość:** Przesunięcie current_frame o liczbę zapisanych klatek
        current_frame = end_frame + 1
        if current_frame >= total_frames:
            current_frame = total_frames - 1

    elif key == ord('q'):  # 'q' - wyjście
        break
    else:
        print("Sterowanie:")
        for instruction in instructions:
            print(instruction)

cap.release()
cv2.destroyAllWindows()

# Zapisz informacje o zapisanych sekwencjach do pliku CSV
csv_file = os.path.join(save_folder, 'informacje_sekwencji.csv')
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['sequence_id', 'label', 'start_frame', 'end_frame', 'frames', 'timestamp']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for info in saved_sequences_info:
        writer.writerow(info)

print(f"Zapisano informacje o sekwencjach do pliku {csv_file}.")
