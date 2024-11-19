import os

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class SurgicalVideoPredictor:
    def __init__(self, model_path='best_surgical_timesformer.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Inicjalizacja procesora i modelu
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=2,  # Dla dwóch klas
            ignore_mismatched_sizes=True
        )

        # Wczytanie wag modelu
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Wczytano model z {model_path}")
        else:
            print(f"Nie znaleziono pliku modelu: {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def add_text_with_background(self, image, text, position, font_scale=0.7, thickness=2, text_color=(255, 255, 255),
                                 bg_color=(0, 0, 0)):
        """Dodaje tekst z tłem"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Oblicz rozmiar tekstu
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Dodaj padding
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
        class_mapping = {0: "przebicie przez galke", 1: "wyjscie z galki"}

        # Licznik klatek
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

                # Dodaj tekst z predykcją i numerem klatki
                class_name = class_mapping[predicted_class]

                # Dodaj numer klatki w lewym górnym rogu z tłem
                frame_text = f"Frame: {frame_count}/{total_frames}"
                self.add_text_with_background(frame, frame_text, (10, 30))

                # Dodaj predykcję poniżej numeru klatki
                prediction_text = f"{class_name} ({confidence:.2f})"
                self.add_text_with_background(frame, prediction_text, (10, 70))

                frames_buffer.pop(0)
            else:
                # Jeśli bufor nie jest jeszcze pełny, tylko wyświetl numer klatki
                frame_text = f"Frame: {frame_count}/{total_frames}"
                self.add_text_with_background(frame, frame_text, (10, 30))

            # Pokaż i/lub zapisz klatkę
            cv2.imshow('Prediction', frame)
            if output_path:
                out.write(frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        print(f"Processed {frame_count} frames")
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()


def main():
    # Przykład użycia
    predictor = SurgicalVideoPredictor()
    predictor.predict_video_with_visualization(
        video_path='E:/Cataract/videos/micro/train01.mp4',
        output_path='output_predictions.mp4'
    )


if __name__ == "__main__":
    main()