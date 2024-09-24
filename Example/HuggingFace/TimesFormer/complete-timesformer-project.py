# Importowanie niezbędnych bibliotek
import torch
import torch.nn as nn
from transformers import TimeSformerModel, TimeSformerConfig, TimeSformerFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd

# Definicja głowy regresji
class RegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super(RegressionHead, self).__init__()
        self.regressor = nn.Linear(hidden_size, 2)  # Przewidujemy x i y

    def forward(self, x):
        # x: [batch_size, sequence_length, hidden_size]
        x = x.mean(dim=1)  # Średnia po sekwencji
        return self.regressor(x)  # [batch_size, 2]

# Definicja modelu z TimeSformerem i głową regresji
class TimeSformerForRegression(nn.Module):
    def __init__(self, model_name):
        super(TimeSformerForRegression, self).__init__()
        self.timesformer = TimeSformerModel.from_pretrained(model_name)
        hidden_size = self.timesformer.config.hidden_size
        self.regression_head = RegressionHead(hidden_size)

    def forward(self, pixel_values):
        outputs = self.timesformer(pixel_values=pixel_values)
        last_hidden_state = outputs.last_hidden_state
        tip_position = self.regression_head(last_hidden_state)
        return tip_position

# Definicja klasy Dataset
class SurgicalToolDataset(Dataset):
    def __init__(self, annotations_file, video_dir, feature_extractor, sequence_length=8, transform=None):
        self.video_dir = video_dir
        self.annotations = self.load_annotations(annotations_file)
        self.feature_extractor = feature_extractor
        self.sequence_length = sequence_length
        self.transform = transform
        self.video_list = list(self.annotations.keys())

    def load_annotations(self, annotations_file):
        df = pd.read_csv(annotations_file)
        annotations = {}
        for idx, row in df.iterrows():
            video_name = row['video_name']
            frame_number = int(row['frame_number'])
            x = float(row['x'])
            y = float(row['y'])
            if video_name not in annotations:
                annotations[video_name] = {}
            annotations[video_name][frame_number] = (x, y)
        return annotations

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]
        frames_dir = os.path.join(self.video_dir, video_name)
        frame_files = sorted(os.listdir(frames_dir))
        num_frames = len(frame_files)
        frames = []
        labels = []

        for i in range(0, num_frames, self.sequence_length):
            frames_seq = []
            labels_seq = []
            for j in range(i, min(i + self.sequence_length, num_frames)):
                frame_file = frame_files[j]
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames_seq.append(frame)
                frame_number = int(frame_file.split('_')[1].split('.')[0])
                x, y = self.annotations[video_name].get(frame_number, (0.0, 0.0))
                labels_seq.append([x, y])

            # Uzupełnienie sekwencji do pełnej długości
            if len(frames_seq) < self.sequence_length:
                padding_frames = [frames_seq[-1]] * (self.sequence_length - len(frames_seq))
                frames_seq.extend(padding_frames)
                padding_labels = [labels_seq[-1]] * (self.sequence_length - len(labels_seq))
                labels_seq.extend(padding_labels)

            # Przetwarzanie za pomocą feature_extractora
            inputs = self.feature_extractor(frames_seq, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # [num_frames, 3, height, width]
            label = torch.tensor(labels_seq[-1], dtype=torch.float32)  # Używamy ostatniej etykiety w sekwencji

            return {'pixel_values': pixel_values, 'labels': label}

# Prosta implementacja filtra cząsteczkowego
class ParticleFilter:
    def __init__(self, num_particles, initial_state, state_std, obs_std):
        self.num_particles = num_particles
        self.particles = np.ones((num_particles, 2)) * initial_state  # [num_particles, 2]
        self.weights = np.ones(num_particles) / num_particles
        self.state_std = state_std
        self.obs_std = obs_std

    def predict(self):
        # Dodaj szum ruchu do cząstek
        self.particles += np.random.normal(0, self.state_std, size=self.particles.shape)

    def update(self, observation):
        # Oblicz prawdopodobieństwo na podstawie obserwacji
        distances = np.linalg.norm(self.particles - observation, axis=1)
        self.weights = self.gaussian(distances, self.obs_std)
        self.weights += 1e-300  # Uniknięcie zerowych wag
        self.weights /= np.sum(self.weights)

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Upewnij się, że suma wynosi 1
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, self.num_particles))
        self.particles = self.particles[indexes]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        # Estymacja stanu na podstawie ważonych cząstek
        return np.average(self.particles, weights=self.weights, axis=0)

    def gaussian(self, x, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(- (x ** 2) / (2 * std ** 2))

# Główna funkcja trenująca model
def train_model(model, train_loader, val_loader, num_epochs, device, criterion, optimizer):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoka {epoch + 1}/{num_epochs}, Strata treningowa: {avg_loss:.4f}")

        # Walidacja po każdej epoce
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(pixel_values)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoka {epoch + 1}/{num_epochs}, Strata walidacyjna: {avg_val_loss:.4f}")

    return model

# Funkcja do przetwarzania wideo z użyciem wytrenowanego modelu i filtra cząsteczkowego
def process_video(model, feature_extractor, test_video_dir, test_video_name, device, particle_filter):
    test_frames_dir = os.path.join(test_video_dir, test_video_name)
    frame_files = sorted(os.listdir(test_frames_dir))
    num_frames = len(frame_files)
    sequence_length = 8

    model.eval()
    with torch.no_grad():
        for i in range(0, num_frames, sequence_length):
            frames_seq = []
            for j in range(i, min(i + sequence_length, num_frames)):
                frame_file = frame_files[j]
                frame_path = os.path.join(test_frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frames_seq.append(frame_resized)

            # Uzupełnienie sekwencji do pełnej długości
            if len(frames_seq) < sequence_length:
                padding_frames = [frames_seq[-1]] * (sequence_length - len(frames_seq))
                frames_seq.extend(padding_frames)

            # Przetwarzanie klatek
            inputs = feature_extractor(frames_seq, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0).to(device)

            # Predykcja modelu
            predicted_position = model(pixel_values).cpu().numpy()[0]

            # Aktualizacja filtra cząsteczkowego
            particle_filter.predict()
            particle_filter.update(predicted_position)
            particle_filter.resample()
            estimated_position = particle_filter.estimate()

            # Wizualizacja
            frame_to_show = frames_seq[-1]
            x_pred, y_pred = predicted_position
            x_est, y_est = estimated_position

            # Skalowanie pozycji do oryginalnego rozmiaru obrazu
            h, w, _ = frame_rgb.shape
            x_scale = w / 224
            y_scale = h / 224
            x_pred_scaled = int(x_pred * x_scale)
            y_pred_scaled = int(y_pred * y_scale)
            x_est_scaled = int(x_est * x_scale)
            y_est_scaled = int(y_est * y_scale)

            # Rysowanie predykcji modelu
            cv2.circle(frame_rgb, (x_pred_scaled, y_pred_scaled), 5, (0, 255, 0), -1)  # Zielony punkt
            # Rysowanie estymacji filtra
            cv2.circle(frame_rgb, (x_est_scaled, y_est_scaled), 5, (0, 0, 255), -1)  # Czerwony punkt

            # Wyświetlanie obrazu
            cv2.imshow('Tracking', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Główna funkcja
def main():
    # Konfiguracja ścieżek
    train_annotations_file = 'data/train_labels.csv'
    val_annotations_file = 'data/val_labels.csv'
    test_annotations_file = 'data/test_labels.csv'
    train_video_dir = 'data/train/'
    val_video_dir = 'data/val/'
    test_video_dir = 'data/test/'

    # Konfiguracja modelu i treningu
    model_name = 'facebook/timesformer-base-finetuned-k400'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 5
    learning_rate = 5e-5
    sequence_length = 8
    batch_size = 2

    # Inicjalizacja feature extractora i transformacji
    feature_extractor = TimeSformerFeatureExtractor.from_pretrained(model_name)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Tworzenie datasetów i dataloaderów
    train_dataset = SurgicalToolDataset(train_annotations_file, train_video_dir, feature_extractor, sequence_length, transform)
    val_dataset = SurgicalToolDataset(val_annotations_file, val_video_dir, feature_extractor, sequence_length, transform)
    test_dataset = SurgicalToolDataset(test_annotations_file, test_video_dir, feature_extractor, sequence_length, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Inicjalizacja modelu
    model = TimeSformerForRegression(model_name)
    model.to(device)

    # Funkcja kosztu i optymalizator
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Trening modelu
    trained_model = train_model(model, train_loader, val_loader, num_epochs, device, criterion, optimizer)

    # Zapisanie wytrenowanego modelu
    torch.save(trained_model.state_dict(), 'timesformer_regression.pth')

    # Ewaluacja na zbiorze testowym
    test_loss = 0
    trained_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = trained_model(pixel_values)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Strata na zbiorze testowym: {avg_test_loss:.4f}")

    # Inicjalizacja filtra cząsteczkowego
    initial_state = np.array([0, 0])
    state_std = 5.0
    obs_std = 10.0
    num_particles = 1000
    particle_filter = ParticleFilter(num_particles, initial_state, state_std, obs_std)

    # Przetwarzanie wideo testowego
    test_