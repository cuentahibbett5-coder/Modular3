#!/usr/bin/env python3
"""
Evaluar y comparar modelos: simple vs weighted
"""
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

# Definir modelo inline (igual en ambos casos)
class UNet3D(torch.nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        import torch.nn as nn
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            )
        
        self.enc1 = conv_block(1, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = conv_block(base_ch*4, base_ch*8)
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)
        self.out = nn.Conv3d(base_ch, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


def load_model(model_path, device):
    """Cargar modelo"""
    model = UNet3D(base_ch=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Comparar modelos simple vs weighted')
    parser.add_argument('--simple', type=str, default='runs/denoising_v2/best.pt', help='Modelo simple')
    parser.add_argument('--weighted', type=str, default='runs/denoising_weighted/best.pt', help='Modelo weighted')
    parser.add_argument('--output', type=str, default='model_comparison.png', help='Output figure')
    
    args = parser.parse_args()
    
    simple_path = Path(args.simple)
    weighted_path = Path(args.weighted)
    
    if not simple_path.exists():
        print(f"❌ Modelo simple no encontrado: {simple_path}")
        return
    
    if not weighted_path.exists():
        print(f"❌ Modelo weighted no encontrado: {weighted_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Cargando modelos...")
    model_simple = load_model(simple_path, device)
    model_weighted = load_model(weighted_path, device)
    
    # Obtener info de checkpoints
    ckpt_simple = torch.load(simple_path, map_location='cpu')
    ckpt_weighted = torch.load(weighted_path, map_location='cpu')
    
    epoch_simple = ckpt_simple.get('epoch', 'N/A')
    val_loss_simple = ckpt_simple.get('val_loss', 'N/A')
    
    epoch_weighted = ckpt_weighted.get('epoch', 'N/A')
    val_loss_weighted = ckpt_weighted.get('val_loss', 'N/A')
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE MODELOS")
    print("="*70)
    print("\nMODELO SIMPLE (baseline)")
    print(f"  Path:     {simple_path}")
    print(f"  Epoch:    {epoch_simple}")
    print(f"  Val Loss: {val_loss_simple}")
    
    print("\nMODELO WEIGHTED (con máscara ponderada + PDD filtrado)")
    print(f"  Path:     {weighted_path}")
    print(f"  Epoch:    {epoch_weighted}")
    print(f"  Val Loss: {val_loss_weighted}")
    
    print("\n" + "="*70)
    
    if isinstance(val_loss_simple, float) and isinstance(val_loss_weighted, float):
        improvement = (val_loss_simple - val_loss_weighted) / val_loss_simple * 100
        if improvement > 0:
            print(f"✅ Weighted es MEJOR: {improvement:.1f}% reducción en val_loss")
        else:
            print(f"⚠️  Simple es MEJOR: {-improvement:.1f}% reducción")
    
    print("="*70 + "\n")
    
    # Crear figura de comparación
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Comparación de Modelos', fontsize=14, fontweight='bold')
    
    models = ['Simple\n(baseline)', 'Weighted\n(ponderado)']
    epochs = [epoch_simple if isinstance(epoch_simple, (int, float)) else 0,
              epoch_weighted if isinstance(epoch_weighted, (int, float)) else 0]
    losses = [val_loss_simple if isinstance(val_loss_simple, float) else 0,
              val_loss_weighted if isinstance(val_loss_weighted, float) else 0]
    colors = ['steelblue', 'coral']
    
    # Epochs
    axes[0].bar(models, epochs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Epoch en Best.pt')
    axes[0].set_title('Época del Mejor Checkpoint')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (m, e) in enumerate(zip(models, epochs)):
        axes[0].text(i, e + 1, f'{int(e)}', ha='center', fontweight='bold')
    
    # Validation Loss
    axes[1].bar(models, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Mejor Validation Loss')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, (m, l) in enumerate(zip(models, losses)):
        axes[1].text(i, l + 0.0001, f'{l:.6f}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Figura guardada: {args.output}\n")


if __name__ == '__main__':
    main()
