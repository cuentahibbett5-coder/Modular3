#!/usr/bin/env python3
"""
Gráficas comparativas: Input vs Predicción IA
Visualiza la mejora en calidad lograda por el modelo
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_exported_data():
    """Carga datos exportados"""
    exports_dir = Path("exports")
    if not exports_dir.exists():
        print("❌ Error: ejecuta export_predictions.py primero")
        return None
    
    # Buscar archivos exportados
    files = list(exports_dir.glob("*_pred.npy"))
    if not files:
        print("❌ No se encontraron predicciones exportadas")
        return None
    
    data = {}
    for pred_file in files:
        # Extraer nombre base
        base_name = pred_file.stem.replace("_pred", "")
        
        # Cargar los tres archivos
        input_vol = np.load(exports_dir / f"{base_name}_input.npy")
        pred_vol = np.load(exports_dir / f"{base_name}_pred.npy")
        target_vol = np.load(exports_dir / f"{base_name}_target.npy")
        
        data[base_name] = {
            'input': input_vol,
            'pred': pred_vol,
            'target': target_vol
        }
    
    return data

def create_quality_comparison_plots(data):
    """Crea gráficas comparativas de calidad"""
    
    # Crear directorio para gráficas
    plots_dir = Path("quality_comparison_plots")
    plots_dir.mkdir(exist_ok=True)
    
    for case_name, volumes in data.items():
        input_vol = volumes['input']
        pred_vol = volumes['pred']
        target_vol = volumes['target']
        
        print(f"📊 Generando gráficas para {case_name}...")
        
        # 1. COMPARACIÓN SLICE A SLICE
        create_slice_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir)
        
        # 2. CURVAS PDD COMPARATIVAS
        create_pdd_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir)
        
        # 3. HISTOGRAMAS DE ERRORES
        create_error_histograms(input_vol, pred_vol, target_vol, case_name, plots_dir)
        
        # 4. SCATTER PLOTS DE CALIDAD
        create_scatter_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir)
        
        # 5. MÉTRICAS DE CALIDAD POR SLICE
        create_quality_metrics_plot(input_vol, pred_vol, target_vol, case_name, plots_dir)

def create_slice_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir):
    """Comparación slice por slice"""
    z_levels = [0, 5, 10, 15]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    vmax = target_vol.max()
    
    for idx, z in enumerate(z_levels):
        # Input
        im0 = axes[0, idx].imshow(input_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[0, idx].set_title(f'Input - z={z}', fontsize=12)
        axes[0, idx].axis('off')
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046)
        
        # Prediction
        im1 = axes[1, idx].imshow(pred_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[1, idx].set_title(f'Predicción IA - z={z}', fontsize=12)
        axes[1, idx].axis('off')
        plt.colorbar(im1, ax=axes[1, idx], fraction=0.046)
        
        # Target
        im2 = axes[2, idx].imshow(target_vol[z], cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[2, idx].set_title(f'Ground Truth - z={z}', fontsize=12)
        axes[2, idx].axis('off')
        plt.colorbar(im2, ax=axes[2, idx], fraction=0.046)
    
    fig.suptitle(f'Comparación Visual: Input vs IA vs GT\n{case_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / f"{case_name}_slice_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_pdd_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir):
    """Curvas PDD comparativas"""
    z_size = target_vol.shape[0]
    
    # Calcular PDD
    pdd_input = np.array([input_vol[z].max() for z in range(z_size)])
    pdd_pred = np.array([pred_vol[z].max() for z in range(z_size)])
    pdd_target = np.array([target_vol[z].max() for z in range(z_size)])
    
    # Errores relativos
    err_input = np.abs(pdd_input - pdd_target) / (pdd_target + 1e-10) * 100
    err_pred = np.abs(pdd_pred - pdd_target) / (pdd_target + 1e-10) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PDD curves
    axes[0].plot(pdd_target, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[0].plot(pdd_input, 'r--', linewidth=2, label='Input (ruidoso)', alpha=0.7)
    axes[0].plot(pdd_pred, 'b-', linewidth=2, label='Predicción IA', alpha=0.8)
    axes[0].set_xlabel('Profundidad Z')
    axes[0].set_ylabel('Dosis Máxima')
    axes[0].set_title('Curvas PDD (Percent Depth Dose)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error comparison
    axes[1].plot(err_input, 'r--', linewidth=2, label='Error Input', alpha=0.7)
    axes[1].plot(err_pred, 'b-', linewidth=2, label='Error Predicción IA', alpha=0.8)
    axes[1].set_xlabel('Profundidad Z')
    axes[1].set_ylabel('Error Relativo (%)')
    axes[1].set_title('Comparación de Errores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    # Estadísticas
    input_mae = np.mean(err_input)
    pred_mae = np.mean(err_pred)
    improvement = input_mae / pred_mae
    
    fig.suptitle(f'PDD Analysis - {case_name}\n'
                f'Error promedio: Input={input_mae:.1f}%, IA={pred_mae:.1f}% '
                f'(mejora {improvement:.1f}×)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{case_name}_pdd_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_error_histograms(input_vol, pred_vol, target_vol, case_name, plots_dir):
    """Histogramas de distribución de errores"""
    
    # Máscara para voxels significativos
    mask = target_vol > 0.01 * target_vol.max()
    
    # Errores absolutos
    err_input = np.abs(input_vol[mask] - target_vol[mask])
    err_pred = np.abs(pred_vol[mask] - target_vol[mask])
    
    # Errores relativos
    rel_err_input = err_input / (target_vol[mask] + 1e-10) * 100
    rel_err_pred = err_pred / (target_vol[mask] + 1e-10) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histograma errores absolutos
    bins = np.logspace(-2, 3, 50)
    axes[0,0].hist(err_input, bins=bins, alpha=0.6, label='Input', color='red', density=True)
    axes[0,0].hist(err_pred, bins=bins, alpha=0.6, label='Predicción IA', color='blue', density=True)
    axes[0,0].set_xscale('log')
    axes[0,0].set_xlabel('Error Absoluto')
    axes[0,0].set_ylabel('Densidad')
    axes[0,0].set_title('Distribución de Errores Absolutos')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Histograma errores relativos
    bins_rel = np.logspace(-1, 2, 50)
    axes[0,1].hist(rel_err_input, bins=bins_rel, alpha=0.6, label='Input', color='red', density=True)
    axes[0,1].hist(rel_err_pred, bins=bins_rel, alpha=0.6, label='Predicción IA', color='blue', density=True)
    axes[0,1].set_xscale('log')
    axes[0,1].set_xlabel('Error Relativo (%)')
    axes[0,1].set_ylabel('Densidad')
    axes[0,1].set_title('Distribución de Errores Relativos')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Box plots
    bp1 = axes[1,0].boxplot([err_input, err_pred], labels=['Input', 'Predicción IA'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('red')
    bp1['boxes'][1].set_facecolor('blue')
    axes[1,0].set_ylabel('Error Absoluto')
    axes[1,0].set_title('Distribución de Errores (Box Plot)')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    bp2 = axes[1,1].boxplot([rel_err_input, rel_err_pred], labels=['Input', 'Predicción IA'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('red')
    bp2['boxes'][1].set_facecolor('blue')
    axes[1,1].set_ylabel('Error Relativo (%)')
    axes[1,1].set_title('Distribución de Errores Relativos (Box Plot)')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    # Estadísticas
    mae_input = np.mean(err_input)
    mae_pred = np.mean(err_pred)
    improvement = mae_input / mae_pred
    
    fig.suptitle(f'Distribución de Errores - {case_name}\n'
                f'MAE: Input={mae_input:.2f}, IA={mae_pred:.2f} '
                f'(mejora {improvement:.1f}×)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{case_name}_error_histograms.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_scatter_comparison(input_vol, pred_vol, target_vol, case_name, plots_dir):
    """Scatter plots de correlación"""
    
    # Máscara para voxels significativos
    mask = target_vol > 0.05 * target_vol.max()
    
    target_flat = target_vol[mask]
    input_flat = input_vol[mask]
    pred_flat = pred_vol[mask]
    
    # Submuestreo para visualización
    n_sample = min(10000, len(target_flat))
    idx = np.random.choice(len(target_flat), n_sample, replace=False)
    
    target_sample = target_flat[idx]
    input_sample = input_flat[idx]
    pred_sample = pred_flat[idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Input vs Target
    axes[0].scatter(target_sample, input_sample, alpha=0.5, s=1, c='red', label='Input')
    axes[0].plot([0, target_vol.max()], [0, target_vol.max()], 'k--', alpha=0.8, label='Ideal')
    
    # Correlación
    corr_input = np.corrcoef(target_sample, input_sample)[0, 1]
    axes[0].set_xlabel('Ground Truth')
    axes[0].set_ylabel('Input')
    axes[0].set_title(f'Input vs GT (Correlación: {corr_input:.4f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # Prediction vs Target
    axes[1].scatter(target_sample, pred_sample, alpha=0.5, s=1, c='blue', label='Predicción IA')
    axes[1].plot([0, target_vol.max()], [0, target_vol.max()], 'k--', alpha=0.8, label='Ideal')
    
    corr_pred = np.corrcoef(target_sample, pred_sample)[0, 1]
    axes[1].set_xlabel('Ground Truth')
    axes[1].set_ylabel('Predicción IA')
    axes[1].set_title(f'Predicción vs GT (Correlación: {corr_pred:.4f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    fig.suptitle(f'Correlación con Ground Truth - {case_name}\n'
                f'Mejora en correlación: {corr_input:.4f} → {corr_pred:.4f}', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{case_name}_scatter_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_quality_metrics_plot(input_vol, pred_vol, target_vol, case_name, plots_dir):
    """Métricas de calidad por slice"""
    
    z_size = target_vol.shape[0]
    
    # Calcular métricas por slice
    mae_input_per_slice = []
    mae_pred_per_slice = []
    psnr_input_per_slice = []
    psnr_pred_per_slice = []
    
    for z in range(z_size):
        slice_target = target_vol[z]
        slice_input = input_vol[z]
        slice_pred = pred_vol[z]
        
        # MAE
        mae_input = np.mean(np.abs(slice_input - slice_target))
        mae_pred = np.mean(np.abs(slice_pred - slice_target))
        mae_input_per_slice.append(mae_input)
        mae_pred_per_slice.append(mae_pred)
        
        # PSNR
        mse_input = np.mean((slice_input - slice_target)**2)
        mse_pred = np.mean((slice_pred - slice_target)**2)
        
        max_val = slice_target.max()
        if max_val > 0 and mse_input > 0:
            psnr_input = 10 * np.log10(max_val**2 / mse_input)
        else:
            psnr_input = 100
            
        if max_val > 0 and mse_pred > 0:
            psnr_pred = 10 * np.log10(max_val**2 / mse_pred)
        else:
            psnr_pred = 100
            
        psnr_input_per_slice.append(psnr_input)
        psnr_pred_per_slice.append(psnr_pred)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # MAE por slice
    axes[0].plot(mae_input_per_slice, 'r--', linewidth=2, label='Input', alpha=0.7)
    axes[0].plot(mae_pred_per_slice, 'b-', linewidth=2, label='Predicción IA', alpha=0.8)
    axes[0].set_xlabel('Slice Z')
    axes[0].set_ylabel('Mean Absolute Error')
    axes[0].set_title('MAE por Slice')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # PSNR por slice
    axes[1].plot(psnr_input_per_slice, 'r--', linewidth=2, label='Input', alpha=0.7)
    axes[1].plot(psnr_pred_per_slice, 'b-', linewidth=2, label='Predicción IA', alpha=0.8)
    axes[1].set_xlabel('Slice Z')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR por Slice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Estadísticas
    avg_mae_improvement = np.mean(mae_input_per_slice) / np.mean(mae_pred_per_slice)
    avg_psnr_gain = np.mean(psnr_pred_per_slice) - np.mean(psnr_input_per_slice)
    
    fig.suptitle(f'Métricas por Slice - {case_name}\n'
                f'MAE mejora {avg_mae_improvement:.1f}×, PSNR ganancia +{avg_psnr_gain:.1f}dB', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f"{case_name}_quality_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_plot(data):
    """Gráfica resumen de todos los casos"""
    plots_dir = Path("quality_comparison_plots")
    
    cases = list(data.keys())
    n_cases = len(cases)
    
    improvements = []
    psnr_gains = []
    correlations_input = []
    correlations_pred = []
    
    for case_name, volumes in data.items():
        input_vol = volumes['input']
        pred_vol = volumes['pred']
        target_vol = volumes['target']
        
        # Máscara
        mask = target_vol > 0.05 * target_vol.max()
        
        # MAE improvement
        mae_input = np.mean(np.abs(input_vol[mask] - target_vol[mask]))
        mae_pred = np.mean(np.abs(pred_vol[mask] - target_vol[mask]))
        improvement = mae_input / mae_pred
        improvements.append(improvement)
        
        # PSNR gain
        mse_input = np.mean((input_vol - target_vol)**2)
        mse_pred = np.mean((pred_vol - target_vol)**2)
        psnr_input = 10 * np.log10(target_vol.max()**2 / mse_input)
        psnr_pred = 10 * np.log10(target_vol.max()**2 / mse_pred)
        psnr_gain = psnr_pred - psnr_input
        psnr_gains.append(psnr_gain)
        
        # Correlaciones
        corr_input = np.corrcoef(target_vol[mask], input_vol[mask])[0, 1]
        corr_pred = np.corrcoef(target_vol[mask], pred_vol[mask])[0, 1]
        correlations_input.append(corr_input)
        correlations_pred.append(corr_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Improvements
    x_pos = np.arange(n_cases)
    axes[0,0].bar(x_pos, improvements, color='skyblue', alpha=0.7)
    axes[0,0].set_xlabel('Caso')
    axes[0,0].set_ylabel('Factor de Mejora MAE')
    axes[0,0].set_title('Mejora en Mean Absolute Error')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels([c.replace('pair_021_', '') for c in cases], rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # PSNR gains
    axes[0,1].bar(x_pos, psnr_gains, color='lightcoral', alpha=0.7)
    axes[0,1].set_xlabel('Caso')
    axes[0,1].set_ylabel('Ganancia PSNR (dB)')
    axes[0,1].set_title('Ganancia en PSNR')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels([c.replace('pair_021_', '') for c in cases], rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Correlaciones Input
    axes[1,0].bar(x_pos, correlations_input, color='orange', alpha=0.7, label='Input')
    axes[1,0].set_xlabel('Caso')
    axes[1,0].set_ylabel('Correlación con GT')
    axes[1,0].set_title('Correlación Input vs GT')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels([c.replace('pair_021_', '') for c in cases], rotation=45)
    axes[1,0].set_ylim([0, 1])
    axes[1,0].grid(True, alpha=0.3)
    
    # Correlaciones Prediction
    axes[1,1].bar(x_pos, correlations_pred, color='green', alpha=0.7, label='Predicción IA')
    axes[1,1].set_xlabel('Caso')
    axes[1,1].set_ylabel('Correlación con GT')
    axes[1,1].set_title('Correlación Predicción vs GT')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([c.replace('pair_021_', '') for c in cases], rotation=45)
    axes[1,1].set_ylim([0, 1])
    axes[1,1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Resumen de Mejoras - Todos los Casos\n'
                f'Mejora MAE promedio: {np.mean(improvements):.1f}×, '
                f'PSNR ganancia promedio: +{np.mean(psnr_gains):.1f}dB', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "summary_all_cases.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("GENERANDO GRÁFICAS COMPARATIVAS: INPUT vs PREDICCIÓN IA")
    print("="*80)
    
    # Cargar datos
    data = load_exported_data()
    if data is None:
        return
    
    print(f"📊 Encontrados {len(data)} casos para analizar")
    
    # Generar gráficas individuales
    create_quality_comparison_plots(data)
    
    # Gráfica resumen
    print("📊 Generando gráfica resumen...")
    create_summary_plot(data)
    
    print(f"\n✅ Gráficas guardadas en: quality_comparison_plots/")
    print("📂 Archivos generados:")
    print("   • *_slice_comparison.png - Comparación visual por slices")
    print("   • *_pdd_comparison.png - Curvas PDD y errores")  
    print("   • *_error_histograms.png - Distribución de errores")
    print("   • *_scatter_comparison.png - Correlaciones")
    print("   • *_quality_metrics.png - Métricas por slice")
    print("   • summary_all_cases.png - Resumen de todos los casos")

if __name__ == "__main__":
    main()