"""
Przykład strojenia hiperparametrów dla prostej sieci neuronowej
===============================================================

Ten skrypt demonstruje jak używać biblioteki Optuna do automatycznego
strojenia hiperparametrów sieci neuronowej w PyTorch.

Hiperparametry które strojmy:
- Liczba warstw ukrytych
- Liczba neuronów w warstwach
- Learning rate (współczynnik uczenia)
- Dropout rate
- Rozmiar batcha
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import Trial
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SimpleNeuralNetwork(nn.Module):
    """
    Prosta sieć neuronowa z konfigurowalną architekturą.

    Args:
        input_size: Wymiar wejścia
        hidden_sizes: Lista z liczbą neuronów w każdej warstwie ukrytej
        output_size: Wymiar wyjścia (liczba klas)
        dropout_rate: Współczynnik dropout
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int, dropout_rate: float):
        super(SimpleNeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # Tworzenie warstw ukrytych
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Warstwa wyjściowa
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_synthetic_data(n_samples: int = 1000, n_features: int = 20, n_classes: int = 3):
    """
    Tworzy syntetyczny zbiór danych do klasyfikacji.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=n_classes,
        random_state=42
    )

    # Normalizacja danych
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y


def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs: int, device: str):
    """
    Trenuje model i zwraca najlepszą dokładność walidacyjną.
    """
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        # Trening
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Walidacja
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        val_accuracy = correct / total
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    return best_val_accuracy


def objective(trial: Trial, X_train, y_train, X_val, y_val, input_size: int, output_size: int, device: str):
    """
    Funkcja celu dla Optuna - definiuje przestrzeń hiperparametrów do przeszukania.

    To jest najważniejsza część - tutaj definiujemy jakie hiperparametry chcemy strojić
    i w jakich zakresach.
    """

    # === HIPERPARAMETRY DO STROJENIA ===

    # 1. Liczba warstw ukrytych (1-4)
    n_layers = trial.suggest_int("n_layers", 1, 4)

    # 2. Liczba neuronów w każdej warstwie (32-256)
    hidden_sizes = []
    for i in range(n_layers):
        hidden_size = trial.suggest_int(f"hidden_size_layer_{i}", 32, 256)
        hidden_sizes.append(hidden_size)

    # 3. Dropout rate (0.1-0.5)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # 4. Learning rate (1e-5 do 1e-1, skala logarytmiczna)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # 5. Rozmiar batcha (16, 32, 64, 128)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    # 6. Optymalizator (Adam lub SGD)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # === TWORZENIE MODELU I DANYCH ===

    # Przygotowanie DataLoaderów
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Tworzenie modelu
    model = SimpleNeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate
    ).to(device)

    # Wybór optymalizatora
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # === TRENING I EWALUACJA ===
    n_epochs = 30
    val_accuracy = train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, device)

    return val_accuracy


def run_hyperparameter_tuning(n_trials: int = 50):
    """
    Główna funkcja uruchamiająca strojenie hiperparametrów.

    Args:
        n_trials: Liczba prób (kombinacji hiperparametrów do przetestowania)
    """
    print("=" * 60)
    print("STROJENIE HIPERPARAMETRÓW SIECI NEURONOWEJ")
    print("=" * 60)

    # Konfiguracja urządzenia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUrządzenie: {device}")

    # Tworzenie danych
    print("\nTworzenie syntetycznego zbioru danych...")
    X, y = create_synthetic_data(n_samples=2000, n_features=20, n_classes=5)

    # Podział na zbiór treningowy i walidacyjny
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_size = X.shape[1]
    output_size = len(np.unique(y))

    print(f"Zbiór treningowy: {len(X_train)} próbek")
    print(f"Zbiór walidacyjny: {len(X_val)} próbek")
    print(f"Wymiar wejścia: {input_size}")
    print(f"Liczba klas: {output_size}")

    # Tworzenie studium Optuna
    print(f"\nRozpoczynam strojenie hiperparametrów ({n_trials} prób)...")
    print("-" * 60)

    study = optuna.create_study(
        direction="maximize",  # Maksymalizujemy dokładność
        study_name="neural_network_tuning"
    )

    # Uruchomienie optymalizacji
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_size, output_size, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Wyniki
    print("\n" + "=" * 60)
    print("WYNIKI STROJENIA")
    print("=" * 60)

    print(f"\nNajlepsza dokładność walidacyjna: {study.best_value:.4f}")
    print("\nNajlepsze hiperparametry:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Podsumowanie wszystkich prób
    print(f"\nStatystyki z {len(study.trials)} prób:")
    accuracies = [trial.value for trial in study.trials if trial.value is not None]
    print(f"  Średnia dokładność: {np.mean(accuracies):.4f}")
    print(f"  Odchylenie std: {np.std(accuracies):.4f}")
    print(f"  Min dokładność: {np.min(accuracies):.4f}")
    print(f"  Max dokładność: {np.max(accuracies):.4f}")

    return study


def main():
    """
    Główny punkt wejścia programu.
    """
    # Ustawienie ziarna losowości dla powtarzalności
    torch.manual_seed(42)
    np.random.seed(42)

    # Uruchomienie strojenia z 30 próbami (możesz zwiększyć dla lepszych wyników)
    study = run_hyperparameter_tuning(n_trials=30)

    print("\n" + "=" * 60)
    print("PODSUMOWANIE")
    print("=" * 60)
    print("""
Wyjaśnienie metod strojenia hiperparametrów:

1. GRID SEARCH (przeszukiwanie siatki)
   - Testuje wszystkie kombinacje hiperparametrów
   - Zalety: Wyczerpujące przeszukiwanie
   - Wady: Bardzo wolne dla wielu hiperparametrów

2. RANDOM SEARCH (losowe przeszukiwanie)
   - Losowo wybiera kombinacje hiperparametrów
   - Zalety: Szybsze niż grid search
   - Wady: Może pominąć optymalne kombinacje

3. BAYESIAN OPTIMIZATION (użyte w Optuna)
   - Buduje probabilistyczny model funkcji celu
   - Inteligentnie wybiera kolejne kombinacje do testowania
   - Zalety: Efektywne, znajduje dobre rozwiązania szybciej
   - Wady: Bardziej skomplikowane w implementacji

4. HYPERBAND / SUCCESSIVE HALVING
   - Wcześnie odrzuca słabe konfiguracje
   - Zalety: Bardzo szybkie dla dużych przestrzeni
   - Wady: Może przedwcześnie odrzucić dobre konfiguracje

W tym przykładzie używamy Optuna z TPE (Tree-structured Parzen Estimator),
który jest formą optymalizacji bayesowskiej.
""")


if __name__ == "__main__":
    main()
