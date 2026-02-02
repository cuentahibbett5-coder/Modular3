"""
Pipeline de entrenamiento para MCDNet

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import yaml

from models.mcdnet import MCDNet3D, create_mcdnet


class DoseDataset(Dataset):
    """Dataset de pares (dosis ruidosa, dosis limpia) para entrenamiento."""
    
    def __init__(self, data_dir, noise_level='low', transform=None):
        """
        Args:
            data_dir: Directorio con datos de entrenamiento
            noise_level: 'low', 'medium', 'high'
            transform: Transformaciones opcionales
        """
        self.data_dir = Path(data_dir)
        self.noise_level = noise_level
        self.transform = transform
        
        # Buscar pares de archivos
        self.noisy_files = sorted(list(self.data_dir.glob(f'*{noise_level}_dose.mhd')))
        self.clean_files = sorted(list(self.data_dir.glob('*clean_dose.mhd')))
        
        assert len(self.noisy_files) == len(self.clean_files), \
            f"Número de archivos no coincide: {len(self.noisy_files)} vs {len(self.clean_files)}"
        
        print(f"Dataset inicializado: {len(self.noisy_files)} pares de muestras")
    
    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        # Cargar imágenes
        noisy_dose = sitk.GetArrayFromImage(sitk.ReadImage(str(self.noisy_files[idx])))
        clean_dose = sitk.GetArrayFromImage(sitk.ReadImage(str(self.clean_files[idx])))
        
        # Normalizar a [0, 1]
        max_val = np.max(clean_dose)
        if max_val > 0:
            noisy_dose = noisy_dose / max_val
            clean_dose = clean_dose / max_val
        
        # Convertir a tensores [1, D, H, W]
        noisy_dose = torch.from_numpy(noisy_dose).unsqueeze(0).float()
        clean_dose = torch.from_numpy(clean_dose).unsqueeze(0).float()
        
        if self.transform:
            noisy_dose, clean_dose = self.transform(noisy_dose, clean_dose)
        
        return noisy_dose, clean_dose


class MCDNetTrainer:
    """Entrenador para modelos MCDNet."""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 learning_rate=1e-4, checkpoint_dir='models/checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizador y criterio
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # o nn.L1Loss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Historial
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Entrena una época."""
        self.model.train()
        epoch_loss = 0.0
        
        for noisy, clean in tqdm(self.train_loader, desc="Training"):
            noisy, clean = noisy.to(self.device), clean.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            output = self.model(noisy)
            loss = self.criterion(output, clean)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for noisy, clean in self.val_loader:
                noisy, clean = noisy.to(self.device), clean.to(self.device)
                output = self.model(noisy)
                loss = self.criterion(output, clean)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, epochs=100):
        """Entrena el modelo."""
        print(f"\nIniciando entrenamiento por {epochs} épocas...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best')
            
            # Guardar checkpoint periódicamente
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
            
            # Ajustar learning rate
            self.scheduler.step(val_loss)
    
    def save_checkpoint(self, epoch, name):
        """Guarda un checkpoint del modelo."""
        checkpoint_path = self.checkpoint_dir / f'mcdnet_{name}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path}")


def main():
    """Script principal de entrenamiento."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar MCDNet')
    parser.add_argument('--data-dir', type=str, required=True, help='Directorio de datos')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Crear datasets
    train_dataset = DoseDataset(Path(args.data_dir) / 'train')
    val_dataset = DoseDataset(Path(args.data_dir) / 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Crear modelo
    model = create_mcdnet('standard')
    
    # Entrenar
    trainer = MCDNetTrainer(model, train_loader, val_loader, 
                           device=args.device, learning_rate=args.lr)
    trainer.train(epochs=args.epochs)
    
    print("\n✓ Entrenamiento completado")


if __name__ == '__main__':
    main()
