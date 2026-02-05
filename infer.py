#!/usr/bin/env python3
"""
Script de inferencia: Carga el mejor modelo y aplica denoising a samples.
Genera visualizaciones de input ruidoso, predicción y ground truth.
"""

from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from training.dataset import DosePairDataset
from training.model import UNet3D


def load_model(checkpoint_path: Path, device: torch.device) -> UNet3D:
    """Carga el modelo desde checkpoint."""
    model = UNet3D(in_ch=1, out_ch=1, base_ch=32).to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def infer_batch(model, inp: torch.Tensor, device: torch.device) -> np.ndarray:
    """Realiza inferencia en un batch."""
    with torch.no_grad():
        inp = inp.to(device)
        pred = model(inp)
        pred = pred.cpu().squeeze(0).numpy()  # (1, Z, Y, X) -> (Z, Y, X)
    return pred


def plot_slices(input_vol, pred_vol, target_vol, output_path: Path, title: str):
    """Grafica 3 slices axiales: input, predicción, target."""
    
    z_idx = input_vol.shape[0] // 2  # Slice central
    
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    slices = [
        (input_vol[z_idx], "Input (Ruidoso)"),
        (pred_vol[z_idx], "Predicción"),
        (target_vol[z_idx], "Ground Truth"),
    ]
    
    vmin = min(v[v > 0].min() for v, _ in slices if v[v > 0].size > 0)
    vmax = max(v.max() for v, _ in slices)
    
    for idx, (data, label) in enumerate(slices):
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(data, cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Dosis (a.u.)")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"✅ Guardado: {output_path}")


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Calcula métricas de calidad."""
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    psnr = 20 * np.log10(target.max() / np.sqrt(mse)) if mse > 0 else float('inf')
    
    # Correlación
    corr = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
    
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "Corr": corr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results_inference"))
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo
    model = load_model(args.checkpoint, device)
    print(f"✅ Modelo cargado desde {args.checkpoint}")
    
    # Dataset de validación
    val_ds = DosePairDataset(
        root_dir=args.data_root,
        split="val",
        patch_size=64,  # Usar mismo patch size que en entrenamiento
        cache_size=0,
        normalize=True,
        seed=4321,
    )
    print(f"✅ Dataset validation: {len(val_ds)} samples")
    
    # Inferencia
    metrics_list = []
    for sample_idx in range(min(args.num_samples, len(val_ds))):
        print(f"\n📊 Sample {sample_idx + 1}/{args.num_samples}")
        batch = val_ds[sample_idx]
        
        inp = batch["input"].unsqueeze(0)  # (1, 1, Z, Y, X)
        tgt = batch["target"].squeeze(0).numpy()  # (Z, Y, X)
        
        pred = infer_batch(model, inp, device)
        
        # Métricas
        metrics = compute_metrics(pred, tgt)
        metrics_list.append(metrics)
        print(f"   MSE: {metrics['MSE']:.6f}")
        print(f"   MAE: {metrics['MAE']:.6f}")
        print(f"   PSNR: {metrics['PSNR']:.2f} dB")
        print(f"   Corr: {metrics['Corr']:.4f}")
        
        # Visualizar
        title = f"Sample {sample_idx + 1} | PSNR={metrics['PSNR']:.2f}dB | Corr={metrics['Corr']:.4f}"
        out_path = args.output_dir / f"sample_{sample_idx + 1:02d}.png"
        inp_vis = inp.squeeze(0).numpy()
        plot_slices(inp_vis, pred, tgt, out_path, title)
    
    # Resumen
    print("\n" + "="*50)
    print("📈 RESUMEN DE MÉTRICAS")
    print("="*50)
    avg_mse = np.mean([m["MSE"] for m in metrics_list])
    avg_mae = np.mean([m["MAE"] for m in metrics_list])
    avg_psnr = np.mean([m["PSNR"] for m in metrics_list])
    avg_corr = np.mean([m["Corr"] for m in metrics_list])
    
    print(f"MSE promedio:  {avg_mse:.6f}")
    print(f"MAE promedio:  {avg_mae:.6f}")
    print(f"PSNR promedio: {avg_psnr:.2f} dB")
    print(f"Corr promedio: {avg_corr:.4f}")
    print(f"\n✅ Resultados guardados en {args.output_dir}")


if __name__ == "__main__":
    main()
