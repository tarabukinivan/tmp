import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from pathlib import Path
import random

# Ustawienie ziarna dla reprodukowalności wyników
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# Konfiguracja parametrów
class Config:
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Mniejszy batch size dla mniejszego zbioru danych
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "efficientnet_b1"  # Mniejszy model dla małego zbioru danych
    BETA = 2  # dla F-beta score
    TRAIN_VAL_SPLIT = 0.8  # 80% trening, 20% walidacja
    NUM_WORKERS = 4  # Liczba wątków do ładowania danych
    SAVE_DIR = "models"
    
    # Funkcja drukująca informacje o konfiguracji
    @classmethod
    def print_config(cls):
        print("\n=== Konfiguracja treningu ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Model: {cls.MODEL_TYPE}")
        print(f"Image Size: {cls.IMG_SIZE}x{cls.IMG_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"Beta for F-score: {cls.BETA}")
        print(f"Train/Val Split: {cls.TRAIN_VAL_SPLIT}")
        print("============================\n")

# Dataset dla obrazów skórnych
class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir=None, transform=None):
        """
        Args:
            csv_file (string): Ścieżka do pliku CSV z etykietami.
            img_dir (string, optional): Katalog bazowy ze zdjęciami.
            transform (callable, optional): Transformacje do zastosowania na obrazach.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Jeśli img_dir nie jest podany, zakładamy że ścieżki w CSV są pełne
        # W przeciwnym razie łączymy img_dir ze ścieżkami w CSV
        if self.img_dir:
            self.data['full_path'] = self.data['path'].apply(
                lambda x: os.path.join(img_dir, x)
            )
        else:
            self.data['full_path'] = self.data['path']
        
        # Weryfikacja istnienia plików
        valid_rows = []
        for idx, row in self.data.iterrows():
            if os.path.exists(row['full_path']):
                valid_rows.append(idx)
            else:
                print(f"Ostrzeżenie: Plik {row['full_path']} nie istnieje!")
        
        self.data = self.data.iloc[valid_rows].reset_index(drop=True)
        print(f"Załadowano {len(self.data)} prawidłowych obrazów.")
        
        # Statystyki zbioru
        melanoma_count = self.data['is_Melanoma'].sum()
        print(f"Obrazy z czerniakiem: {melanoma_count} ({melanoma_count/len(self.data)*100:.1f}%)")
        print(f"Obrazy bez czerniaka: {len(self.data) - melanoma_count} ({(len(self.data) - melanoma_count)/len(self.data)*100:.1f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['full_path']
        
        # Wczytanie obrazu za pomocą PIL
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = float(self.data.iloc[idx]['is_Melanoma'])
            return image, label
            
        except Exception as e:
            print(f"Błąd wczytywania obrazu {img_path}: {e}")
            # W przypadku błędu zwracamy pierwszy poprawny obraz
            return self[0]

# Augmentacja danych i transformacje
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        # Augmentacje specyficzne dla obrazów dermatologicznych
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

# Funkcja straty z większą wagą dla pozytywnych przykładów (true)
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=3.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        loss = nn.BCELoss(reduction='none')(inputs, targets)
        # Dodaj większą wagę dla pozytywnych próbek (melanoma=1)
        weights = torch.ones_like(targets) + self.pos_weight * targets
        weighted_loss = weights * loss
        return weighted_loss.mean()

# Funkcja do obliczania F-beta score
def fbeta_score(precision, recall, beta=2):
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-7)

# Funkcja do obliczania metryk
def calculate_metrics(y_true, y_pred, y_prob, beta=2):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    f_beta = fbeta_score(precision, recall, beta=beta)
    
    # Obliczenie AUC tylko jeśli mamy przykłady z obu klas
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta,
        'auc': auc
    }

# Funkcja do obliczania finalnego wyniku modelu wg wag
def calculate_final_score(metrics):
    return (
        metrics['f_beta'] * 0.6 +
        metrics['accuracy'] * 0.3 +
        metrics['auc'] * 0.1
    )

# Funkcja do przygotowania modelu
def get_model(model_type, pretrained=True):
    if model_type == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        )
    elif model_type == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        )
    elif model_type == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        )
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        )
    else:
        raise ValueError(f"Nieobsługiwany typ modelu: {model_type}")
    
    return model

# Funkcja do wizualizacji wyników treningu
def plot_training_results(train_losses, val_losses, metrics_history, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 12))
    
    # Plot strat
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Strata treningowa')
    plt.plot(epochs, val_losses, 'r-', label='Strata walidacyjna')
    plt.title('Straty podczas treningu')
    plt.xlabel('Epoki')
    plt.ylabel('Strata')
    plt.legend()
    
    # Plot metryk
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [m['accuracy'] for m in metrics_history], 'g-', label='Accuracy')
    plt.plot(epochs, [m['precision'] for m in metrics_history], 'm-', label='Precision')
    plt.plot(epochs, [m['recall'] for m in metrics_history], 'c-', label='Recall')
    plt.title('Metryki podczas treningu')
    plt.xlabel('Epoki')
    plt.ylabel('Wartość')
    plt.legend()
    
    # Plot F-beta i AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [m['f_beta'] for m in metrics_history], 'y-', label=f'F-beta (β={Config.BETA})')
    plt.plot(epochs, [m['auc'] for m in metrics_history], 'k-', label='AUC')
    plt.title('F-beta i AUC podczas treningu')
    plt.xlabel('Epoki')
    plt.ylabel('Wartość')
    plt.legend()
    
    # Plot wyniku finalnego
    plt.subplot(2, 2, 4)
    final_scores = [calculate_final_score(m) for m in metrics_history]
    plt.plot(epochs, final_scores, 'r-', label='Wynik finalny')
    plt.title('Wynik finalny podczas treningu')
    plt.xlabel('Epoki')
    plt.ylabel('Wartość')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Zapisano wykres w {save_path}")

# Trenowanie modelu
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=10):
    best_score = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    metrics_history = []
    
    # Utwórz katalog dla modeli, jeśli nie istnieje
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    for epoch in range(epochs):
        # Tryb treningu
        model.train()
        train_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images = images.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Tryb ewaluacji
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images = images.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).float().view(-1, 1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = outputs.cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_preds.extend(preds.flatten())
                all_probs.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Obliczanie metryk
        metrics = calculate_metrics(all_labels, all_preds, all_probs, beta=Config.BETA)
        metrics_history.append(metrics)
        final_score = calculate_final_score(metrics)
        
        # Aktualizacja schedulera
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(final_score)
            else:
                scheduler.step()
        
        # Zapisanie najlepszego modelu
        if final_score > best_score:
            best_score = final_score
            best_model_state = model.state_dict().copy()
            # Zapisz model
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, f"best_model_epoch_{epoch+1}.pth"))
            print(f"✅ Zapisano najlepszy model z wynikiem {final_score:.4f}")
        
        # Wypisanie wyników
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"F-beta: {metrics['f_beta']:.4f}, AUC: {metrics['auc']:.4f}")
        print(f"Final Score: {final_score:.4f}\n")
        
        # Zapisywanie modelu co kilka epok
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'metrics': metrics,
                'final_score': final_score
            }, os.path.join(Config.SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
    
    # Załadowanie najlepszego stanu modelu
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Wizualizacja wyników treningu
    plot_path = os.path.join(Config.SAVE_DIR, "training_results.png")
    plot_training_results(train_losses, val_losses, metrics_history, plot_path)
    
    return model, best_score, train_losses, val_losses, metrics_history

# Konwersja modelu do ONNX
def convert_to_onnx(model, file_path):
    model.eval()
    dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE, device=Config.DEVICE)
    
    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model zapisany w formacie ONNX: {file_path}")

# Funcja główna
def main(args):
    # Wyświetlenie konfiguracji
    Config.print_config()
    
    # Przygotowanie transformacji
    train_transform, val_transform = get_transforms()
    
    # Wczytanie danych
    print(f"Wczytywanie danych z pliku: {args.csv_file}")
    full_dataset = MelanomaDataset(csv_file=args.csv_file, img_dir=args.img_dir, transform=None)
    
    # Podział na zbiór treningowy i walidacyjny
    train_size = int(Config.TRAIN_VAL_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"Podział danych: {train_size} obrazów treningowych, {val_size} obrazów walidacyjnych")
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Przypisanie odpowiednich transformacji
    train_dataset = torch.utils.data.Subset(full_dataset, train_dataset.indices)
    train_dataset.dataset.transform = train_transform
    
    val_dataset = torch.utils.data.Subset(full_dataset, val_dataset.indices)
    val_dataset.dataset.transform = val_transform
    
    # Tworzenie dataloaderów
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Tworzenie modelu
    print(f"Tworzenie modelu: {Config.MODEL_TYPE}")
    model = get_model(Config.MODEL_TYPE)
    model = model.to(Config.DEVICE)
    
    # Definiowanie funkcji straty i optymalizatora
    # Używamy straty z wyższą wagą dla pozytywnych próbek (czerniaka)
    criterion = WeightedBCELoss(pos_weight=3.0)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    # Learning rate scheduler - reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maksymalizujemy wynik
        factor=0.5,  # Zmniejszanie LR o połowę
        patience=3,  # Czekamy 3 epoki bez poprawy
        verbose=True
    )
    
    # Trenowanie modelu
    print("Rozpoczynanie treningu...")
    model, best_score, train_losses, val_losses, metrics_history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        scheduler=scheduler, epochs=Config.EPOCHS
    )
    
    # Zapisywanie finalnego modelu
    final_model_path = os.path.join(Config.SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Zapisano finalny model w: {final_model_path}")
    
    # Konwersja do ONNX
    onnx_path = os.path.join(Config.SAVE_DIR, args.output_model)
    convert_to_onnx(model, onnx_path)
    
    print(f"Trening zakończony! Najlepszy wynik: {best_score:.4f}")
    print(f"Model ONNX zapisany w: {onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skrypt trenujący model wykrywania czerniaka")
    parser.add_argument("--csv_file", type=str, required=True, help="Ścieżka do pliku CSV z etykietami")
    parser.add_argument("--img_dir", type=str, default=None, help="Katalog z obrazami (opcjonalny, jeśli ścieżki w CSV są pełne)")
    parser.add_argument("--output_model", type=str, default="melanoma_model.onnx", help="Nazwa pliku wyjściowego modelu ONNX")
    
    args = parser.parse_args()
    main(args)
