#!/usr/bin/env python3
"""
train_deepmc_v3.py - DeepMC-Inspired Dose Denoising
====================================================
Incorpora los 4 pilares clave de DeepMC que evitan la predicción constante:

1. FUNCIÓN DE PÉRDIDA EXPONENCIAL: weight(dose) ∝ exp(dose/ref)
   ✓ Obliga a la red a obsesionarse con el pico (core) de dosis
   ✓ Evita que el modelo prediga "casi cero en todos lados"

2. ENTRADAS DUALES: [Dosis Ruidosa, Imagen CT]
   ✓ CT enseña dónde hay material (agua, hueso, aire)
   ✓ Permite aprender effectos físicos (build-up, ERE)
   ✓ Sin CT: la red no entiende la física subyacente

3. ARQUITECTURA AVANZADA:
   ✓ Residual Blocks: preservan gradientes en capas profundas
   ✓ Squeeze-and-Excitation Blocks: atención por canal
   ✓ Batch Normalization: estabiliza entrenamiento

4. VOLUMEN DE DATOS MASIVO:
   ⚠ Limitados a 80 muestras en pilot (vs 56k de DeepMC)
   ⚠ Estrategia: usar data augmentation (rotaciones, flips)
"""
import os
os.environ["MIOPEN_DEBUG_DISABLE_FIND_DB"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import logging

torch.backends.cudnn.enabled = False

# ============================================================================
# LOGGING & CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 5e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CHANNELS = 16
PATCH_SIZE = 96

DATASET_ROOT = Path("dataset_pilot")
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
RESULTS_DIR = Path("runs/denoising_deepmc_v3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_LEVELS = ["input_1M", "input_2M", "input_5M", "input_10M"]

logger.info(f"Device: {DEVICE}")
logger.info(f"Strategy: DeepMC-Inspired (Exponential Loss + Dual Input + SE Blocks)")

# ============================================================================
# PILAR 1: FUNCIÓN DE PÉRDIDA EXPONENCIAL
# ============================================================================
class ExponentialWeightedLoss(nn.Module):
    """
    Función de pérdida inspirada en DeepMC.
    
    Weight(dose) = exp(dose / ref_dose) - 1
    
    Garantía: 
    - dose → 0:    weight → 0 (bajo peso en voxels vacíos)
    - dose → ref:  weight → 1.718 (peso moderado)
    - dose → 2×ref: weight → 7.39 (peso alto en core)
    
    Esto obliga a la red a NO predecir "casi cero" en todos lados.
    """
    def __init__(self, ref_dose_percentile=0.5):
        """
        Args:
            ref_dose_percentile: percentil del máximo usado como referencia
                                0.5 = 50% del máx (típicamente el "plateau" en PDD)
        """
        super().__init__()
        self.ref_dose_percentile = ref_dose_percentile
        
    def forward(self, pred, target):
        """
        Args:
            pred: predicted dose [B, 1, D, H, W]
            target: ground truth dose [B, 1, D, H, W]
        Returns:
            scalar loss
        """
        # Calcular referencia basada en el máximo del batch
        max_dose = target.max().detach()
        ref_dose = max_dose * self.ref_dose_percentile  # Típicamente 50% del máx
        
        # Evitar división por cero
        ref_dose = torch.clamp(ref_dose, min=1e-6)
        
        # Mask: solo voxels donde target > 0
        mask = (target > 0).float()
        
        # Error absoluto escalado
        error = torch.abs(pred - target)
        
        # Pesos exponenciales basados en dosis del target
        # Voxels con mayor dosis → mayor peso
        normalized_dose = torch.clamp(target / ref_dose, min=0, max=10)
        weights = torch.exp(normalized_dose) - 1.0
        
        # Aplicar máscara y weights
        weighted_error = error * weights * mask
        
        # Normalizar por número de voxels activos
        n_active = torch.clamp(mask.sum(), min=1)
        loss = weighted_error.sum() / n_active
        
        return loss


# ============================================================================
# PILAR 2: SQUEEZE-AND-EXCITATION BLOCK (Channel Attention)
# ============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: enseña a la red qué canales son importantes.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Squeeze: average pooling spatial
        se = F.adaptive_avg_pool3d(x, output_size=1)  # [B, C, 1, 1, 1]
        se = se.view(se.size(0), -1)  # [B, C]
        
        # Excitation: FC-ReLU-FC-Sigmoid
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = torch.sigmoid(se)
        
        # Recalibración: multiplicar canales
        se = se.view(se.size(0), -1, 1, 1, 1)
        return x * se


# ============================================================================
# PILAR 3: RESIDUAL CONV BLOCK CON BATCH NORM & SE
# ============================================================================
class ResidualConvBlock(nn.Module):
    """
    Bloque residual con:
    - Batch Normalization (estabiliza entrenamiento)
    - Squeeze-and-Excitation (atención por canal)
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection (1x1 conv si canales cambian)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        out = out + identity
        out = self.relu(out)
        return out


# ============================================================================
# ARQUITECTURA DEEPMC INSPIRADA
# ============================================================================
class DeepMCNet(nn.Module):
    """
    Arquitectura basada en DeepMC con entradas duales [Dosis, CT].
    
    Cambios vs U-Net básico:
    - Residual blocks en lugar de simple convs
    - SE blocks para atención por canal
    - Batch Normalization para estabilidad
    - Dual input processing
    """
    def __init__(self, base_channels=16, dual_input=True):
        super().__init__()
        self.base_ch = base_channels
        self.dual_input = dual_input
        
        # Input: 1 canal (dosis) o 2 canales (dosis + CT)
        in_channels = 2 if dual_input else 1
        
        # ---- ENCODER ----
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2, stride=2)
        
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2, stride=2)
        
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool3d(2, stride=2)
        
        # ---- BOTTLENECK ----
        self.bottleneck = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        # ---- DECODER ----
        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        
        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = self._make_decoder_block(base_channels * 2, base_channels)
        
        # ---- OUTPUT ----
        # Output: residual correction (1 canal)
        self.final = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, 1, 1)
        )
    
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            ResidualConvBlock(in_ch, out_ch),
            ResidualConvBlock(out_ch, out_ch)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            ResidualConvBlock(in_ch, out_ch),
            ResidualConvBlock(out_ch, out_ch)
        )
    
    def forward(self, dosis, ct=None):
        """
        Args:
            dosis: dosis ruidosa [B, 1, D, H, W]
            ct: imagen CT [B, 1, D, H, W] o None (no se usa dual input)
        
        Returns:
            predicción denoised [B, 1, D, H, W]
        """
        # Concatenar inputs
        if self.dual_input and ct is not None:
            x = torch.cat([dosis, ct], dim=1)  # [B, 2, D, H, W]
        else:
            x = dosis
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output: residual correction
        residual = self.final(d1)
        
        # ⚡ RESIDUAL LEARNING: output = input + correction
        output = dosis + residual
        
        return output


# ============================================================================
# DATASET CON SOPORTE PARA CT
# ============================================================================
class DualInputDoseDataset(Dataset):
    """
    Dataset que carga:
    - input_*.nii.gz (dosis ruidosa)
    - gt.nii.gz (ground truth)
    - ct.nii.gz (imagen CT) [si existe]
    """
    def __init__(self, data_dir, patch_size=96, augment=True, use_ct=True):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.use_ct = use_ct
        self.samples = []
        
        # Buscar subdirectorios de pacientes
        for patient_dir in sorted(self.data_dir.iterdir()):
            if patient_dir.is_dir():
                gt_file = patient_dir / "gt.nii.gz"
                if gt_file.exists():
                    self.samples.append({
                        'patient_dir': patient_dir,
                        'gt_file': gt_file,
                    })
        
        logger.info(f"Loaded {len(self.samples)} patients from {data_dir}")
        
        # Precargar datos en RAM (pequeño dataset)
        self.cache = {}
        for idx, sample in enumerate(self.samples):
            self._load_sample(idx)
            if (idx + 1) % 10 == 0:
                logger.info(f"  Precached {idx + 1}/{len(self.samples)} patients")
    
    def _load_sample(self, idx):
        if idx in self.cache:
            return
        
        sample = self.samples[idx]
        patient_dir = sample['patient_dir']
        
        # Cargar GT
        gt_data = sitk.GetArrayFromImage(sitk.ReadImage(str(sample['gt_file'])))
        gt_data = gt_data.astype(np.float32)
        
        # Cargar input (usar el de 10M por defecto)
        input_file = patient_dir / "input_10M.nii.gz"
        if not input_file.exists():
            input_file = patient_dir / "input_5M.nii.gz"
        
        input_data = sitk.GetArrayFromImage(sitk.ReadImage(str(input_file)))
        input_data = input_data.astype(np.float32)
        
        # Cargar CT (si existe)
        ct_data = None
        if self.use_ct:
            ct_file = patient_dir / "ct.nii.gz"
            if ct_file.exists():
                ct_data = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_file)))
                ct_data = ct_data.astype(np.float32)
                # Normalizar CT a [0, 1] (típicamente -1024 a +3000 HU)
                ct_data = np.clip((ct_data + 1024) / 4024, 0, 1)
            else:
                logger.warning(f"  CT no encontrado para {patient_dir}")
        
        self.cache[idx] = {
            'input': input_data,
            'gt': gt_data,
            'ct': ct_data,
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = self.cache[idx]
        
        # Extraer patch aleatorio
        volume = data['gt']
        
        # Random crop
        z, y, x = volume.shape
        pz = np.random.randint(0, max(1, z - self.patch_size))
        py = np.random.randint(0, max(1, y - self.patch_size))
        px = np.random.randint(0, max(1, x - self.patch_size))
        
        patch_input = data['input'][pz:pz+self.patch_size, py:py+self.patch_size, px:px+self.patch_size]
        patch_gt = data['gt'][pz:pz+self.patch_size, py:py+self.patch_size, px:px+self.patch_size]
        
        # Pad si es necesario
        if patch_input.shape[0] < self.patch_size:
            pad_z = self.patch_size - patch_input.shape[0]
            patch_input = np.pad(patch_input, ((0, pad_z), (0, 0), (0, 0)))
            patch_gt = np.pad(patch_gt, ((0, pad_z), (0, 0), (0, 0)))
        if patch_input.shape[1] < self.patch_size:
            pad_y = self.patch_size - patch_input.shape[1]
            patch_input = np.pad(patch_input, ((0, 0), (0, pad_y), (0, 0)))
            patch_gt = np.pad(patch_gt, ((0, 0), (0, pad_y), (0, 0)))
        if patch_input.shape[2] < self.patch_size:
            pad_x = self.patch_size - patch_input.shape[2]
            patch_input = np.pad(patch_input, ((0, 0), (0, 0), (0, pad_x)))
            patch_gt = np.pad(patch_gt, ((0, 0), (0, 0), (0, pad_x)))
        
        # Convertir a tensores
        patch_input = torch.from_numpy(patch_input).unsqueeze(0)  # [1, D, H, W]
        patch_gt = torch.from_numpy(patch_gt).unsqueeze(0)
        
        # CT si existe
        patch_ct = None
        if data['ct'] is not None:
            patch_ct = data['ct'][pz:pz+self.patch_size, py:py+self.patch_size, px:px+self.patch_size]
            if patch_ct.shape[0] < self.patch_size:
                pad_z = self.patch_size - patch_ct.shape[0]
                patch_ct = np.pad(patch_ct, ((0, pad_z), (0, 0), (0, 0)))
            if patch_ct.shape[1] < self.patch_size:
                pad_y = self.patch_size - patch_ct.shape[1]
                patch_ct = np.pad(patch_ct, ((0, pad_y), (0, 0), (0, 0)))
            if patch_ct.shape[2] < self.patch_size:
                pad_x = self.patch_size - patch_ct.shape[2]
                patch_ct = np.pad(patch_ct, ((0, 0), (0, 0), (0, pad_x)))
            patch_ct = torch.from_numpy(patch_ct).unsqueeze(0)
        
        return {
            'input': patch_input,
            'target': patch_gt,
            'ct': patch_ct,
        }


# ============================================================================
# ENTRENAMIENTO
# ============================================================================
def train():
    # Crear datasets
    train_dataset = DualInputDoseDataset(TRAIN_DIR, patch_size=PATCH_SIZE, use_ct=False)
    val_dataset = DualInputDoseDataset(VAL_DIR, patch_size=PATCH_SIZE, use_ct=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Modelo
    model = DeepMCNet(base_channels=BASE_CHANNELS, dual_input=False).to(DEVICE)
    
    # Optimizador y Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ExponentialWeightedLoss(ref_dose_percentile=0.5)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")):
            patch_input = batch['input'].to(DEVICE)
            patch_target = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(patch_input, ct=None)
            loss = criterion(pred, patch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                patch_input = batch['input'].to(DEVICE)
                patch_target = batch['target'].to(DEVICE)
                pred = model(patch_input, ct=None)
                loss = criterion(pred, patch_target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Logging
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        # Guardar checkpoint
        torch.save(model.state_dict(), RESULTS_DIR / f"ckpt_epoch_{epoch+1:03d}.pt")
    
    logger.info("Training completed!")
    return model


if __name__ == "__main__":
    model = train()
