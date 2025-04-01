import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Callable, Set, List, Dict
import os
import wandb

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from functools import lru_cache

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_score, recall_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from datasets import Dataset as Dataset_hf
from transformers import AutoImageProcessor, AutoModelForImageClassification


### LOADING DATASET
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
import os
import numpy as np

# Load the CSV file
csv_file_path = '/tmp/datasets/melanoma-1/labels.csv'  #@param {type: 'string'}
seed = 42 #@param {type: 'number'}

dataset_id2label = {0: '0', 1: '1'}  # Map to labels
dataset_cancer_class = ['1']  # Specify which classes you're interested in
df = pd.read_csv(csv_file_path)

# Create a Dataset from the DataFrame
dataset = Dataset.from_pandas(df)
# Function to load images
def load_image(row):
    image_path = os.path.join('/tmp/datasets/melanoma-1', row['path'])
    image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    image = image.resize((250, 250))
    # Map labels: 1 for melanoma, 0 for non-melanoma
    label = 1 if row['is_Melanoma'] is True else 0

    return {"image": image, "label": label}

# Apply the load_image function to the dataset in batches
dataset = dataset.map(load_image, remove_columns=['path', 'is_Melanoma'], num_proc=16)
# Split the dataset into train and test sets (80% train, 20% test)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

# Create DatasetDict
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

print(dataset)


### INIT TRAINING PIPELINE 

def calculate_fbeta(precision, recall, beta=2):
    """
    Calculate F-beta score with beta=2 to emphasize recall
    """
    beta_squared = beta ** 2
    fbeta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    return fbeta if (beta_squared * precision + recall) != 0 else 0

def calculate_weighted_score(fbeta, accuracy, auc):
    """
    Calculate weighted score based on the specified weights
    """
    return (0.60 * fbeta) + (0.30 * accuracy) + (0.10 * auc)

class MetricsCalculator:
    def __init__(self, beta=2):
        self.beta = beta

    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Calculate all metrics and weighted score
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        fbeta = calculate_fbeta(precision, recall, self.beta)
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        weighted_score = calculate_weighted_score(fbeta, accuracy, auc)
        
        return {
            'f_beta': fbeta,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'weighted_score': weighted_score
        }

class SkinCancerDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.labels = torch.tensor([item['label'] for item in dataset])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        image = np.array(image)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32).unsqueeze(-1)
        }

    def get_labels(self):
        return self.labels

def create_weighted_sampler(dataset):
    """
    Create a weighted random sampler using PyTorch's implementation
    """
    labels = dataset.get_labels()
    
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    
    samples_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    return sampler

class SkinCancerDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self._init_model()
        
    def _init_model(self):
        model = AutoModelForImageClassification.from_pretrained(
            self.config['model_repo'],
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        return model

    def forward(self, x):
        return self.model(x).logits

class TrainingConfig:
    def __init__(self):
        self.num_epochs = 80
        self.batch_size = 32
        self.base_lr = 1e-4
        self.max_lr = 3e-4
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        self.mixup_alpha = 0.2
        self.label_smoothing = 0.1
        self.mixed_precision = True
        self.use_weighted_sampling = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = './checkpoints'
        self.onnx_input_size = (1, 3, 224, 224)  # (batch_size, channels, height, width)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

class SkinCancerTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.device = config.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.metrics_calculator = MetricsCalculator(beta=2)

        if config.use_weighted_sampling:
            labels = train_loader.dataset.get_labels()
            class_counts = torch.bincount(labels)
            class_distribution = {
                f"class_{i}_count": count.item() 
                for i, count in enumerate(class_counts)
            }
            
            self.run = wandb.init(
                project="skin-cancer-detection",
                config={
                    "dataset_size": len(train_loader.dataset),
                    "batch_size": config.batch_size,
                    "learning_rate": config.base_lr,
                    "epochs": config.num_epochs,
                    "use_weighted_sampling": config.use_weighted_sampling,
                    "class_distribution": class_distribution
                }
            )
        else:
            self.run = wandb.init(
                project="skin-cancer-detection",
                config={
                    "dataset_size": len(train_loader.dataset),
                    "batch_size": config.batch_size,
                    "learning_rate": config.base_lr,
                    "epochs": config.num_epochs,
                }
            )

        self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.base_lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.max_lr,
            epochs=config.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )

        self.scaler = GradScaler() if config.mixed_precision else None

    def export_to_onnx(self, path, dummy_input_size=None):
        """
        Export the model to ONNX format
        """
        if dummy_input_size is None:
            dummy_input_size = self.config.onnx_input_size

        dummy_input = torch.randn(dummy_input_size, device=self.device)
        self.model.eval()
        
        torch.onnx.export(self.model,
                        dummy_input,
                        path,
                        export_params=True,
                        opset_version=14,  # Changed from 11 to 14
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})
        
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_preds_proba = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            if self.config.mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_data(
                    images, labels, self.config.mixup_alpha
                )

            self.optimizer.zero_grad()

            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(images)
                if self.config.mixup_alpha > 0:
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)

            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                self.optimizer.step()

            self.scheduler.step()

            proba = torch.sigmoid(outputs).detach()
            preds = (proba > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_preds_proba.extend(proba.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_preds_proba)
        )
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_preds_proba = []
        all_targets = []

        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            proba = torch.sigmoid(outputs)
            preds = (proba > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_preds_proba.extend(proba.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            total_loss += loss.item()

        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_preds_proba)
        )
        metrics['val_loss'] = total_loss / len(self.val_loader)

        return metrics

    def train(self):
        best_weighted_score = float('-inf')
        
        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate()
            
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            metrics = {**train_metrics, **val_metrics}
            
            wandb.log(metrics)
            
            if val_metrics['val_weighted_score'] > best_weighted_score:
                best_weighted_score = val_metrics['val_weighted_score']
                self.save_checkpoint('best_model.pt', metrics)
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', metrics)

    def save_checkpoint(self, filename, metrics):
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, path)
        
        # If this is the best model, also save in ONNX format
        if filename == 'best_model.pt':
            onnx_path = os.path.join(self.config.checkpoint_dir, 'best_model.onnx')
            self.export_to_onnx(onnx_path)

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def main():
    # Load your dataset
    # Assuming you have your dataset loaded as 'dataset'
    config = TrainingConfig()
    
    # Create datasets
    train_dataset = SkinCancerDataset(dataset['train'], transform=get_transforms(train=True))
    val_dataset = SkinCancerDataset(dataset['test'], transform=get_transforms(train=False))

    if config.use_weighted_sampling:
        train_sampler = create_weighted_sampler(train_dataset)
    else:
        train_sampler = None

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model_config = {
        'model_repo': 'Anwarkh1/Skin_Cancer-Image_Classification',
    }
    model = SkinCancerDetector(model_config)

    # Initialize trainer
    trainer = SkinCancerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Start training
    trainer.train()


### RUNNING TRAINING
if __name__ == "__main__":
    main()
