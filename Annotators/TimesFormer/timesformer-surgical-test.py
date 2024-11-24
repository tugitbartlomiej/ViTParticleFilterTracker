import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class SurgicalVideoPredictor:
    def __init__(self, model_path, processor=None):
        """
        Inicjalizacja predyktora dla sekwencji chirurgicznych.

        Args:
            model_path: Ścieżka do zapisanego modelu
            processor: Procesor obrazu (opcjonalnie)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Załaduj mapowanie klas
        with open('ClassMapping/class_mapping.json', 'r') as f:
            self.class_mapping = json.load(f)

        # Inicjalizacja modelu
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=len(self.class_mapping),
            ignore_mismatched_sizes=True
        ).to(self.device)

        # Załaduj wagi modelu
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Inicjalizacja procesora
        self.processor = processor or AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

    def extract_frames(self, video_path, num_frames=8):
        """
        Ekstrahuje klatki z wideo.
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)

        cap.release()
        return frames

    def predict_video(self, video_path):
        """
        Wykonuje predykcję na wideo.
        """
        # Ekstrakcja klatek
        frames = self.extract_frames(video_path)
        if not frames:
            raise ValueError("Nie udało się wczytać klatek z wideo")

        # Przygotowanie danych wejściowych
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            padding=True,
            do_resize=True,
            size={"height": 224, "width": 224},
        )

        pixel_values = inputs.pixel_values.to(self.device)

        # Predykcja
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()

        # Mapowanie przewidywanej klasy na etykietę
        predicted_label = {v: k for k, v in self.class_mapping.items()}.get(predicted_class, "Nieznana klasa")

        return predicted_label, confidence

    def process_video_with_visualization(self, video_path, output_path=None, show_preview=True):
        """
        Przetwarza wideo z wizualizacją predykcji.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Nie można otworzyć wideo")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_buffer = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frames_buffer.append(frame)
            if len(frames_buffer) >= 8:  # Gdy mamy wystarczająco klatek
                # Konwersja klatek do formatu PIL
                pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_buffer]

                # Predykcja
                inputs = self.processor(
                    images=pil_frames,
                    return_tensors="pt",
                    padding=True,
                    do_resize=True,
                    size={"height": 224, "width": 224},
                )

                pixel_values = inputs.pixel_values.to(self.device)

                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][predicted_class].item()

                predicted_label = {v: k for k, v in self.class_mapping.items()}.get(predicted_class, "Nieznana klasa")

                # Dodaj adnotacje do klatki
                current_frame = frames_buffer[-1].copy()
                text = f"{predicted_label} ({confidence:.2f})"
                cv2.putText(current_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if output_path:
                    out.write(current_frame)

                if show_preview:
                    cv2.imshow('Prediction', current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frames_buffer.pop(0)  # Usuń najstarszą klatkę

        cap.release()
        if output_path:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()

def main():
    # Ścieżki
    model_path = 'Models/best_surgical_timesformer.pth'
    video_path = 'E:/Cataract/videos/micro/train01.mp4'
    output_path = 'predictions_output.mp4'

    try:
        # Sprawdź czy model istnieje
        if not os.path.exists(model_path):
            print(f"Błąd: Nie znaleziono modelu w {model_path}")
            return

        # Sprawdź czy plik wideo istnieje
        if not os.path.exists(video_path):
            print(f"Błąd: Nie znaleziono pliku wideo w {video_path}")
            return

        # Inicjalizacja predyktora
        print("Inicjalizacja predyktora...")
        predictor = SurgicalVideoPredictor(model_path=model_path)

        # Opcja 1: Prosty test predykcji
        print("\nWykonywanie pojedynczej predykcji...")
        label, confidence = predictor.predict_video(video_path)
        print(f"Przewidywana klasa: {label}")
        print(f"Pewność: {confidence:.2f}")

        # Opcja 2: Pełne przetwarzanie wideo z wizualizacją
        print("\nRozpoczęcie przetwarzania wideo z wizualizacją...")
        predictor.process_video_with_visualization(
            video_path=video_path,
            output_path=output_path,
            show_preview=True
        )

        print("\nPrzetwarzanie zakończone!")
        print(f"Zapisano przetworzone wideo do: {output_path}")

    except Exception as e:
        print(f"Wystąpił błąd podczas przetwarzania: {str(e)}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()