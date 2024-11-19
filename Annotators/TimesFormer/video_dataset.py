import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, seq_dir, processor, num_frames=8):  # Zmieniono domyślną liczbę klatek na 8
        self.seq_dir = seq_dir
        self.processor = processor
        self.num_frames = num_frames
        self.samples = self._load_samples()
        self.class_mapping = self._create_class_mapping()

    def _load_samples(self):
        samples = []
        for sequence_name in sorted(os.listdir(self.seq_dir)):
            sequence_dir = os.path.join(self.seq_dir, sequence_name)
            if not os.path.isdir(sequence_dir):
                continue

            # Odczytaj opis klasy z pliku description.txt
            description_file = os.path.join(sequence_dir, 'description.txt')
            if not os.path.exists(description_file):
                continue

            with open(description_file, 'r', encoding='utf-8') as f:
                label = f.read().strip()

            # Zbierz ścieżki do klatek
            frame_files = sorted([
                os.path.join(sequence_dir, f)
                for f in os.listdir(sequence_dir)
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])

            # Jeśli mamy wystarczającą liczbę klatek
            if len(frame_files) >= self.num_frames:
                # Wybierz równomiernie rozłożone klatki
                step = len(frame_files) // self.num_frames
                selected_frames = frame_files[::step][:self.num_frames]

                samples.append({
                    'frame_paths': selected_frames,
                    'label': label
                })

        return samples

    def _create_class_mapping(self):
        unique_classes = sorted(list(set(sample['label'] for sample in self.samples)))
        return {class_name: idx for idx, class_name in enumerate(unique_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        # Wczytaj wszystkie klatki
        for frame_path in sample['frame_paths']:
            image = Image.open(frame_path).convert('RGB')
            frames.append(image)

        # Przetwórz klatki używając procesora TimeSformera
        inputs = self.processor(
            images=frames,
            return_tensors="pt",
            do_resize=True,
            size={"height": 224, "width": 224},
            do_normalize=True,
        )

        # Przygotuj tensor wejściowy
        pixel_values = inputs.pixel_values.squeeze(0)  # Usuń wymiar batch

        # Przekształć etykietę na tensor
        label = torch.tensor(self.class_mapping[sample['label']], dtype=torch.long)

        return pixel_values, label