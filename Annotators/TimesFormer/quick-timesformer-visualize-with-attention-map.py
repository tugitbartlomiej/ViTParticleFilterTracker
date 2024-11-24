import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class SurgicalVideoPredictor:
    def __init__(self, model_path='best_surgical_timesformer.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()

        # Wczytanie mapowania klas
        with open('class_mapping.json', 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)

        # Inicjalizacja modelu
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        self.model = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            num_labels=len(self.class_mapping),
            ignore_mismatched_sizes=True,
            output_attentions=True,
            torch_dtype=torch.float16
        ).to(self.device)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()

    def process_frame_batch(self, frames):
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        with autocast(device_type='cuda', dtype=torch.float16):
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

            # Przetwarzanie mapy attention
            attention = outputs.attentions[-1]  # Ostatnia warstwa attention
            batch_size, num_heads, seq_len, _ = attention.shape

            # Usuwamy token CLS i przekształcamy na format przestrzenny
            attention = attention.mean(dim=1)  # Uśredniamy po głowicach
            attention = attention[0, 1:, 1:]  # Usuwamy token CLS

            # Obliczamy wymiary siatki patchy
            num_patches = int(np.sqrt(seq_len - 1))  # -1 dla tokena CLS
            patches_per_frame = num_patches * num_patches

            # Przekształcamy mapę attention dla ostatniej klatki
            attention_map = attention.view(-1, patches_per_frame)[-1]
            attention_map = attention_map.view(num_patches, num_patches)
            attention_map = attention_map.float().cpu().numpy()

            # Skalujemy do rozmiaru oryginalnej klatki
            attention_map = cv2.resize(
                attention_map,
                (frames[0].shape[1], frames[0].shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Normalizacja
            attention_map = (attention_map - attention_map.min()) / (
                    attention_map.max() - attention_map.min() + 1e-8
            )

        torch.cuda.empty_cache()

        return predicted_class, confidence, attention_map

    def process_video(self, video_path, output_path=None, show_preview=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Nie można otworzyć wideo")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_buffer = []
        frame_count = 0
        buffer_size = 8

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                frames_buffer.append(frame)

                if len(frames_buffer) >= buffer_size:
                    try:
                        class_id, confidence, attention_map = self.process_frame_batch(frames_buffer)
                        class_name = self.class_mapping.get(str(class_id), "unknown")

                        # Tworzenie wizualizacji
                        attention_heat = cv2.applyColorMap(
                            (attention_map * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )

                        output_frame = cv2.addWeighted(
                            frames_buffer[-1],
                            0.7,
                            attention_heat,
                            0.3,
                            0
                        )

                        # Dodawanie informacji tekstowych
                        cv2.putText(
                            output_frame,
                            f"{class_name} ({confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )

                        cv2.putText(
                            output_frame,
                            f"Klatka: {frame_count}/{total_frames}",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )

                        if output_path:
                            out.write(output_frame)

                        if show_preview:
                            cv2.imshow('TimeSformer Visualization', output_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break

                        frames_buffer.pop(0)

                    except Exception as e:
                        print(f"Błąd podczas przetwarzania klatki {frame_count}: {e}")
                        frames_buffer.pop(0)
                        continue

                if frame_count % 100 == 0:
                    print(f"Przetworzono {frame_count}/{total_frames} klatek")

        except Exception as e:
            print(f"Wystąpił błąd podczas przetwarzania wideo: {e}")
            raise e

        finally:
            cap.release()
            if output_path:
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            torch.cuda.empty_cache()

def main():
    predictor = SurgicalVideoPredictor()
    predictor.process_video(
        video_path='E:/Cataract/videos/micro/train01.mp4',
        output_path='output_predictions.mp4',
        show_preview=True
    )


if __name__ == "__main__":
    main()