import os
import json
from datetime import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class TimesformerAutoAnnotator:
    def __init__(self, model_path):
        """
        Inicjalizacja auto-anotatora wykorzystującego model TimeSformer
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Używane urządzenie: {self.device}")

        # Wczytaj mapowanie klas
        with open('./ClassMapping/class_mapping.json', 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
            self.id_to_class = {int(k): v for k, v in self.class_mapping.items()}

        # Inicjalizacja modelu i procesora
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_frames=8
        )
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=len(self.class_mapping),
            ignore_mismatched_sizes=True,
            num_frames=8
        ).to(self.device)

        # Wczytaj wagi modelu
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Stan aplikacji
        self.mode = 'step'  # 'step', 'continuous' lub 'semi-continuous'
        self.current_frame = 0
        self.frames_buffer = []
        self.last_saved_frame = -1
        self.sequence_counter = 0
        self.all_frames = []  # Bufor wszystkich klatek do przechowywania historii
        self.last_state = None  # Śledzenie poprzedniego stanu

    def process_frame_batch(self, frames):
        """
        Przetwarzanie batcha klatek przez model
        """
        # Upewnij się, że mamy dokładnie 8 klatek
        if len(frames) > 8:
            frames = frames[-8:]
        elif len(frames) < 8:
            # Jeśli mamy mniej niż 8 klatek, powiel ostatnią klatkę
            while len(frames) < 8:
                frames.append(frames[-1].copy())

        # Konwersja klatek na format PIL
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        try:
            inputs = self.processor(
                images=pil_frames,
                return_tensors="pt",
                do_resize=True,
                size={"height": 224, "width": 224},
            )

            pixel_values = inputs.pixel_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0][predicted_class].item()

            return predicted_class, confidence

        except Exception as e:
            print(f"Error during frame processing: {str(e)}")
            return None, 0.0

    def save_sequence(self, frames, label_name, confidence):
        """
        Zapisywanie sekwencji klatek
        """
        if self.current_frame <= self.last_saved_frame:
            return

        # Określ folder na podstawie pewności predykcji
        if 0 <= confidence <= 0.25:
            confidence_folder = "0_25"
        elif 0.25 < confidence <= 0.50:
            confidence_folder = "25_50"
        elif 0.50 < confidence <= 0.75:
            confidence_folder = "50_75"
        else:
            confidence_folder = "75_100"

        # Utwórz strukturę folderów
        base_folder = os.path.join("SavedSequencesAuto", confidence_folder)
        os.makedirs(base_folder, exist_ok=True)

        # Utwórz folder dla sekwencji
        self.sequence_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sequence_folder = os.path.join(base_folder, f"sequence_{self.sequence_counter:04d}_{timestamp}")
        os.makedirs(sequence_folder, exist_ok=True)

        # Zapisz opis
        with open(os.path.join(sequence_folder, "description.txt"), 'w', encoding='utf-8') as f:
            f.write(f"{label_name}")

        # Zapisz klatki
        for i, frame in enumerate(frames):
            frame_path = os.path.join(sequence_folder, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)

        self.last_saved_frame = self.current_frame

        print(f"Zapisano sekwencję {self.sequence_counter} w {sequence_folder}")
        print(f"Klasa: {label_name}, Pewność: {confidence:.2f}")

    def run(self, video_path):
        """
        Główna pętla programu
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Nie można otworzyć pliku wideo")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_buffer = []
        key = -1  # Inicjalizacja zmiennej key

        while True:
            # Wczytaj klatkę
            ret, frame = cap.read()
            if not ret:
                break

            # Dodaj klatkę do buforów
            self.frames_buffer.append(frame)
            self.all_frames.append(frame)
            self.current_frame += 1

            # Przetwórz sekwencję gdy mamy wystarczająco klatek
            if len(self.frames_buffer) >= 8:
                predicted_class, confidence = self.process_frame_batch(self.frames_buffer)

                if predicted_class is not None:
                    class_name = self.id_to_class[predicted_class]

                    # Automatyczne przetwarzanie dla continuous i semi-continuous
                    if ((self.mode == 'continuous' or self.mode == 'semi-continuous') and
                            class_name == "przebicie" and
                            self.last_state is not None and
                            self.last_state != "przebicie" and
                            len(self.all_frames) >= 16):

                        if self.mode == 'semi-continuous':
                            # Zatrzymaj się i czekaj na decyzję użytkownika
                            while True:
                                display_frame = frame.copy()
                                cv2.putText(display_frame, f"Class: {class_name}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(display_frame, f"Frame: {self.current_frame}/{total_frames}", (10, 110),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(display_frame, f"Mode: {self.mode}", (10, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.putText(display_frame, "Wykryto przebicie!", (10, 190),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(display_frame, "S - zapisz, N - odrzuc", (10, 230),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.imshow('TimeSformer Auto Annotator', display_frame)

                                key = cv2.waitKey(0) & 0xFF
                                if key == ord('s'):
                                    # Zapisz sekwencję
                                    previous_frames = self.all_frames[-16:-8]
                                    self.save_sequence(previous_frames, "bezposrednie_zagrozenie", confidence)
                                    break
                                elif key == ord('n'):
                                    # Odrzuć i kontynuuj
                                    break
                                elif key == ord('q'):
                                    return
                        else:  # continuous mode
                            previous_frames = self.all_frames[-16:-8]
                            self.save_sequence(previous_frames, "bezposrednie_zagrozenie", confidence)

                        # Przesunięcie o 8 klatek do przodu
                        for _ in range(8):
                            if self.current_frame < total_frames:
                                ret, _ = cap.read()
                                if ret:
                                    self.current_frame += 1
                                else:
                                    break
                        self.frames_buffer = []
                        continue

                    # Aktualizuj poprzedni stan
                    self.last_state = class_name

                    # Przygotuj klatkę do wyświetlenia
                    display_frame = frame.copy()

                    # Dodaj informacje na klatkę
                    cv2.putText(display_frame, f"Class: {class_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Frame: {self.current_frame}/{total_frames}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Mode: {self.mode}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Wyświetl klatkę
                    cv2.imshow('TimeSformer Auto Annotator', display_frame)

                    # Obsługa klawiszy
                    if self.mode == 'step':
                        key = cv2.waitKey(0)
                    else:
                        key = cv2.waitKey(30)

                    if key & 0xFF == ord('q'):
                        break
                    elif key & 0xFF == ord('p'):
                        self.mode = 'step'
                        # W trybie krokowym, przesuwamy się o 8 klatek
                        if len(self.frames_buffer) >= 8:
                            for _ in range(7):
                                if self.current_frame < total_frames:
                                    ret, _ = cap.read()
                                    if ret:
                                        self.current_frame += 1
                    elif key & 0xFF == ord('c'):
                        self.mode = 'continuous'
                    elif key & 0xFF == ord('m'):
                        self.mode = 'semi-continuous'
                    elif key & 0xFF == ord('s') and self.mode == 'step':
                        self.save_sequence(self.frames_buffer, class_name, confidence)
                    elif key & 0xFF == ord(','):  # Poprzednia klatka
                        if self.current_frame > 8:  # Upewnij się, że jest wystarczająco klatek do tyłu
                            # Przesunięcie o jedną klatkę do tyłu
                            self.current_frame -= 9  # -9 bo zostanie zwiększone o 1 w następnej iteracji
                            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                            self.frames_buffer = []  # Wyczyść bufor
                            self.all_frames = []  # Wyczyść historię
                            continue
                    elif key & 0xFF == ord('.'):  # Następna klatka
                        if self.current_frame < total_frames:
                            continue

                # Usuń najstarszą klatkę z bufora jeśli nie jest pusty
                if self.frames_buffer:
                    self.frames_buffer.pop(0)

                # Zachowaj tylko ostatnie 16 klatek w historii (dla trybu ciągłego)
                if len(self.all_frames) > 16:
                    self.all_frames.pop(0)

            # Wyświetl postęp
            if self.current_frame % 100 == 0:
                print(f"Przetworzono {self.current_frame}/{total_frames} klatek")

        cap.release()
        cv2.destroyAllWindows()


def main():
    # Ścieżki
    model_path = 'Models/best_surgical_timesformer.pth'
    video_path = 'E:/Cataract/videos/micro/train01.mp4'

    # Instrukcje
    print("Sterowanie:")
    print("'P' - tryb krokowy (Step-by-Step, przesuwa o 8 klatek)")
    print("'C' - tryb ciągły (Continuous)")
    print("'M' - tryb półautomatyczny (Semi-continuous)")
    print("',' - poprzednia klatka")
    print("'.' - następna klatka")
    print("'S' - zapisz aktualną sekwencję")
    print("'Q' - wyjście")

    # Uruchom auto-anotator
    annotator = TimesformerAutoAnnotator(model_path)
    annotator.run(video_path)


if __name__ == "__main__":
    main()
