import csv
import os
from datetime import datetime

import cv2

# Parametry
video_path = 'E:/Cataract/videos/micro/train01.mp4'  # Ścieżka do pliku wideo
save_folder = 'zapisane_sekwencje'  # Główny folder do zapisywania sekwencji
sequence_length = 5  # Liczba klatek przed i po aktualnej klatce do zapisania

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
cv2.namedWindow('Frame')
cv2.createTrackbar('Pozycja', 'Frame', 0, total_frames - 1, on_trackbar)

while True:
    # Ustawienie pozycji klatki
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    if not ret:
        print("Nie można odczytać klatki.")
        break

    # Aktualizacja pozycji trackbara
    cv2.setTrackbarPos('Pozycja', 'Frame', current_frame)

    # Wyświetlenie klatki
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(0)

    if key == ord('.'):  # Kropka - następna klatka
        if current_frame < total_frames - 1:
            current_frame += 1
    elif key == ord(','):  # Przecinek - poprzednia klatka
        if current_frame > 0:
            current_frame -= 1
    elif key == ord('2'):  # '2' - przeskocz 20 klatek do przodu
        current_frame = min(current_frame + 20, total_frames - 1)
    elif key == ord('1'):  # '1' - przeskocz 20 klatek do tyłu
        current_frame = max(current_frame - 20, 0)
    elif key == ord('s'):  # 's' - zapisz sekwencję klatek
        start_frame = max(0, current_frame - sequence_length)
        end_frame = min(total_frames - 1, current_frame + sequence_length)

        # Inkrementacja licznika sekwencji
        sequence_counter += 1
        # Tworzenie unikalnej nazwy folderu dla sekwencji
        sequence_folder_name = f"sequence_{sequence_counter:04d}"
        sequence_folder_path = os.path.join(save_folder, sequence_folder_name)

        # Tworzenie folderu dla sekwencji
        os.makedirs(sequence_folder_path, exist_ok=True)

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

        print(f"Zapisano sekwencję {sequence_folder_name} od klatki {start_frame} do {end_frame}.")

        # Dodanie informacji o sekwencji do listy
        saved_sequences_info.append({
            'sequence_id': sequence_folder_name,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frames': ';'.join(frames_in_sequence),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    elif key == ord('q'):  # 'q' - wyjście
        break
    else:
        print("Sterowanie:")
        print("'.' - następna klatka")
        print("',' - poprzednia klatka")
        print("'2' - przeskocz 20 klatek do przodu")
        print("'1' - przeskocz 20 klatek do tyłu")
        print("'s' - zapisz sekwencję klatek")
        print("'q' - wyjście")

cap.release()
cv2.destroyAllWindows()

# Zapisz informacje o zapisanych sekwencjach do pliku CSV
csv_file = os.path.join(save_folder, 'informacje_sekwencji.csv')
with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['sequence_id', 'start_frame', 'end_frame', 'frames', 'timestamp']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for info in saved_sequences_info:
        writer.writerow(info)

print(f"Zapisano informacje o sekwencjach do pliku {csv_file}.")
