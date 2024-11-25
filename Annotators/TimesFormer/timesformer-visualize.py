import json
import os
from datetime import datetime

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class SurgicalVideoPredictor:
    def __init__(self, model_path='./Models/best_surgical_timesformer.pth'):
        """
        Inicjalizacja predyktora dla sekwencji chirurgicznych.

        Args:
            model_path: Ścieżka do zapisanego modelu
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Wczytaj mapowanie klas
        self.class_mapping = self._load_class_mapping()
        print(f"Załadowano mapowanie klas: {self.class_mapping}")

        # Utwórz odwrotne mapowanie z id klasy na nazwę klasy
        self.id_to_class = {v: k for k, v in self.class_mapping.items()}

        # Inicjalizacja procesora i modelu
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=len(self.class_mapping),
            ignore_mismatched_sizes=True
        )

        # Wczytanie wag modelu
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Wczytano model z {model_path}")
        else:
            print(f"Nie znaleziono pliku modelu: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def _load_class_mapping(self):
        """
        Wczytuje mapowanie klas z pliku JSON w formacie {"nazwa_klasy": id_klasy}.
        """
        mapping_file = "SavedSequences/class_mapping.json"

        # Sprawdź, czy plik mapowania istnieje
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
                # Upewnij się, że identyfikatory klas są liczbami całkowitymi
                class_mapping = {k: int(v) for k, v in class_mapping.items()}
                return class_mapping
        else:
            raise FileNotFoundError(f"Nie znaleziono pliku mapowania klas: {mapping_file}")

    def add_text_with_background(self, image, text, position, font_scale=0.7, thickness=2,
                                 text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        """
        Dodaje tekst z tłem do obrazu.

        Args:
            image: Obraz OpenCV
            text: Tekst do wyświetlenia
            position: Pozycja (x, y)
            font_scale: Skala czcionki
            thickness: Grubość czcionki
            text_color: Kolor tekstu (BGR)
            bg_color: Kolor tła (BGR)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        padding = 5

        # Narysuj prostokątne tło
        p1 = (position[0], position[1] - text_height - padding)
        p2 = (position[0] + text_width + padding * 2, position[1] + padding)
        cv2.rectangle(image, p1, p2, bg_color, -1)

        # Dodaj tekst
        cv2.putText(image, text,
                    (position[0] + padding, position[1] - padding),
                    font, font_scale, text_color, thickness)

    def predict_video_with_visualization(self, video_path, output_path=None):
        """
        Przewidywanie z wizualizacją na wideo.

        Args:
            video_path: Ścieżka do pliku wideo
            output_path: Opcjonalna ścieżka do zapisania wyjściowego wideo
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Nie można otworzyć wideo")

        # Przygotuj writer do zapisu wideo
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_buffer = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Rozpoczynam przetwarzanie wideo ({total_frames} klatek)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Konwersja i dodanie do bufora
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames_buffer.append(frame_pil)

            if len(frames_buffer) >= 8:
                # Predykcja
                inputs = self.processor(
                    images=frames_buffer,
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": 224, "width": 224},
                )
                pixel_values = inputs.pixel_values.to(self.device)

                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    probs = F.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][predicted_class].item()

                # Sprawdź pewność predykcji
                if confidence < 0.15:  # próg 15%
                    prediction_text = "Brak wykrycia czynności. Pewność poniżej 0.15"
                else:
                    # Pobierz nazwę klasy z odwrotnego mapowania
                    class_name = self.id_to_class.get(predicted_class, f"Nieznana klasa {predicted_class}")
                    prediction_text = f"{class_name} ({confidence:.2f})"

                # Dodaj numer klatki w lewym górnym rogu z tłem
                frame_text = f"Frame: {frame_count}/{total_frames}"
                self.add_text_with_background(frame, frame_text, (10, 30))

                # Dodaj predykcję poniżej numeru klatki
                self.add_text_with_background(frame, prediction_text, (10, 70))

                frames_buffer.pop(0)
            else:
                frame_text = f"Frame: {frame_count}/{total_frames}"
                self.add_text_with_background(frame, frame_text, (10, 30))

            # Pokaż i/lub zapisz klatkę
            cv2.imshow('Prediction', frame)
            if output_path:
                out.write(frame)

            # Obsługa klawisza 'q' do wyjścia
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            # Wyświetl postęp co 100 klatek
            if frame_count % 100 == 0:
                print(f"Przetworzono {frame_count}/{total_frames} klatek")

        print(f"Zakończono przetwarzanie. Przetworzono {frame_count} klatek.")
        cap.release()
        if output_path:
            out.release()
            print(f"Zapisano wideo do: {output_path}")
        cv2.destroyAllWindows()


def main():
    # Utwórz folder Video jeśli nie istnieje
    os.makedirs('Video', exist_ok=True)

    # Generuj nazwę pliku ze znacznikiem czasowym
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'Video/predictions_{timestamp}.mp4'

    predictor = SurgicalVideoPredictor()
    predictor.predict_video_with_visualization(
        video_path='E:/Cataract/videos/micro/train01.mp4',
        output_path=output_filename
    )

    print(f"Zapisano wideo z predykcjami do: {output_filename}")


if __name__ == "__main__":
    main()
