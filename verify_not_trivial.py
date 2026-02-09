#!/usr/bin/env python3
"""
Verificación Anti-Multiplicador Trivial
¿El modelo solo multiplica por un factor constante o realmente denoisa?
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_test_data():
    """Carga datos exportados para verificación"""
    exports_dir = Path("exports")
    
    # Buscar un caso para análisis
    pred_files = list(exports_dir.glob("*1M*_pred.npy"))
    if not pred_files:
        print("❌ No se encontraron archivos de predicción con 1M")
        return None
    
    # Tomar el primer caso
    pred_file = pred_files[0]
    base_name = pred_file.stem.replace("_pred", "")
    
    input_vol = np.load(exports_dir / f"{base_name}_input.npy")
    pred_vol = np.load(exports_dir / f"{base_name}_pred.npy") 
    target_vol = np.load(exports_dir / f"{base_name}_target.npy")
    
    print(f"📊 Analizando: {base_name}")
    print(f"   Input shape: {input_vol.shape}")
    print(f"   Pred shape:  {pred_vol.shape}")
    print(f"   Target shape: {target_vol.shape}")
    
    return input_vol, pred_vol, target_vol, base_name

def test_constant_multiplier(input_vol, pred_vol, target_vol):
    """Prueba 1: ¿Es solo un multiplicador constante?"""
    print(f"\n{'='*60}")
    print("🧪 PRUEBA 1: ¿MULTIPLICADOR CONSTANTE?")
    print(f"{'='*60}")
    
    # Calcular ratio pred/input donde ambos > 0
    mask = (input_vol > 0.01 * input_vol.max()) & (target_vol > 0.01 * target_vol.max())
    
    if mask.sum() == 0:
        print("❌ No hay voxels significativos para analizar")
        return False
    
    ratios = pred_vol[mask] / (input_vol[mask] + 1e-10)
    
    # Estadísticas del ratio
    ratio_mean = np.mean(ratios)
    ratio_std = np.std(ratios)
    ratio_cv = ratio_std / ratio_mean  # Coeficiente de variación
    
    print(f"📈 Ratio Predicción/Input:")
    print(f"   Media:              {ratio_mean:.3f}")
    print(f"   Desviación estándar: {ratio_std:.3f}")
    print(f"   Coef. variación:    {ratio_cv:.3f}")
    print(f"   Min/Max:            {ratios.min():.3f} / {ratios.max():.3f}")
    
    # Factor esperado si fuera solo multiplicación
    target_factor = target_vol.max() / input_vol.max()
    print(f"   Factor esperado:    {target_factor:.3f} (si fuera multiplicación simple)")
    
    # Criterios para detectar multiplicador constante
    is_constant = ratio_cv < 0.1  # Variación < 10%
    is_close_to_expected = abs(ratio_mean - target_factor) / target_factor < 0.1
    
    if is_constant and is_close_to_expected:
        print(f"❌ POSIBLE MULTIPLICADOR TRIVIAL")
        print(f"   El ratio es muy constante (CV={ratio_cv:.3f}) y cerca del factor esperado")
        return True
    else:
        print(f"✅ NO ES MULTIPLICADOR TRIVIAL")
        print(f"   El ratio varía significativamente (CV={ratio_cv:.3f})")
        return False

def test_spatial_patterns(input_vol, pred_vol, target_vol):
    """Prueba 2: ¿Captura patrones espaciales complejos?"""
    print(f"\n{'='*60}")
    print("🧪 PRUEBA 2: ¿PATRONES ESPACIALES COMPLEJOS?")
    print(f"{'='*60}")
    
    # Calcular gradientes (cambios espaciales)
    def compute_gradients_3d(vol):
        gx = np.gradient(vol, axis=2)  # X
        gy = np.gradient(vol, axis=1)  # Y 
        gz = np.gradient(vol, axis=0)  # Z
        return np.sqrt(gx**2 + gy**2 + gz**2)
    
    grad_input = compute_gradients_3d(input_vol)
    grad_pred = compute_gradients_3d(pred_vol)
    grad_target = compute_gradients_3d(target_vol)
    
    # Máscara para región significativa
    mask = target_vol > 0.05 * target_vol.max()
    
    # Correlaciones de gradientes
    corr_input_target = np.corrcoef(grad_input[mask].flatten(), 
                                   grad_target[mask].flatten())[0,1]
    corr_pred_target = np.corrcoef(grad_pred[mask].flatten(), 
                                  grad_target[mask].flatten())[0,1]
    
    print(f"📐 Correlación de Gradientes (patrones espaciales):")
    print(f"   Input vs Target:  {corr_input_target:.4f}")
    print(f"   Pred vs Target:   {corr_pred_target:.4f}")
    print(f"   Mejora:           {corr_pred_target - corr_input_target:.4f}")
    
    # Suavidad (Laplaciano)
    def compute_laplacian_3d(vol):
        return (np.roll(vol, 1, axis=0) + np.roll(vol, -1, axis=0) +
                np.roll(vol, 1, axis=1) + np.roll(vol, -1, axis=1) + 
                np.roll(vol, 1, axis=2) + np.roll(vol, -1, axis=2) - 6*vol)
    
    lap_input = compute_laplacian_3d(input_vol)
    lap_pred = compute_laplacian_3d(pred_vol)
    lap_target = compute_laplacian_3d(target_vol)
    
    smoothness_input = np.std(lap_input[mask])
    smoothness_pred = np.std(lap_pred[mask])
    smoothness_target = np.std(lap_target[mask])
    
    print(f"📏 Suavidad (menor = más suave):")
    print(f"   Input:   {smoothness_input:.2f}")
    print(f"   Pred:    {smoothness_pred:.2f}")
    print(f"   Target:  {smoothness_target:.2f}")
    
    # ¿Mejora patrones espaciales?
    improves_correlation = corr_pred_target > corr_input_target + 0.01
    improves_smoothness = abs(smoothness_pred - smoothness_target) < abs(smoothness_input - smoothness_target)
    
    if improves_correlation or improves_smoothness:
        print(f"✅ SÍ MEJORA PATRONES ESPACIALES")
        return True
    else:
        print(f"❌ NO MEJORA PATRONES ESPACIALES SIGNIFICATIVAMENTE")
        return False

def test_noise_reduction(input_vol, pred_vol, target_vol):
    """Prueba 3: ¿Realmente reduce ruido estadístico?"""
    print(f"\n{'='*60}")
    print("🧪 PRUEBA 3: ¿REDUCCIÓN DE RUIDO ESTADÍSTICO?")
    print(f"{'='*60}")
    
    # Región de alta dosis (menos afectada por ruido estadístico)
    high_dose_mask = target_vol > 0.7 * target_vol.max()
    
    # Región de dosis media (más afectada por ruido)
    mid_dose_mask = ((target_vol > 0.2 * target_vol.max()) & 
                     (target_vol < 0.7 * target_vol.max()))
    
    if high_dose_mask.sum() == 0 or mid_dose_mask.sum() == 0:
        print("❌ No hay suficientes voxels para análisis")
        return False
    
    # Variabilidad local (ruido)
    def local_variability(vol, mask):
        """Calcula variabilidad local como medida de ruido"""
        vol_masked = vol * mask
        # Convolución simple para variabilidad local
        kernel = np.ones((3,3,3)) / 27
        mean_local = np.zeros_like(vol)
        
        # Aproximación de convolución
        for z in range(1, vol.shape[0]-1):
            for y in range(1, vol.shape[1]-1):
                for x in range(1, vol.shape[2]-1):
                    if mask[z,y,x]:
                        patch = vol_masked[z-1:z+2, y-1:y+2, x-1:x+2]
                        mean_local[z,y,x] = patch.mean()
        
        variance = ((vol_masked - mean_local)**2 * mask).sum() / mask.sum()
        return np.sqrt(variance)
    
    # Calcular ruido en cada región
    noise_input_high = local_variability(input_vol, high_dose_mask)
    noise_pred_high = local_variability(pred_vol, high_dose_mask)
    noise_target_high = local_variability(target_vol, high_dose_mask)
    
    noise_input_mid = local_variability(input_vol, mid_dose_mask)
    noise_pred_mid = local_variability(pred_vol, mid_dose_mask)
    noise_target_mid = local_variability(target_vol, mid_dose_mask)
    
    print(f"🔊 Ruido Local (menor = menos ruido):")
    print(f"   Región alta dosis:")
    print(f"     Input:   {noise_input_high:.3f}")
    print(f"     Pred:    {noise_pred_high:.3f}")
    print(f"     Target:  {noise_target_high:.3f}")
    
    print(f"   Región dosis media:")
    print(f"     Input:   {noise_input_mid:.3f}")
    print(f"     Pred:    {noise_pred_mid:.3f}")
    print(f"     Target:  {noise_target_mid:.3f}")
    
    # ¿Reduce ruido hacia el nivel del target?
    reduces_noise_high = noise_pred_high < noise_input_high * 0.9
    reduces_noise_mid = noise_pred_mid < noise_input_mid * 0.9
    
    closer_to_target_high = abs(noise_pred_high - noise_target_high) < abs(noise_input_high - noise_target_high)
    closer_to_target_mid = abs(noise_pred_mid - noise_target_mid) < abs(noise_input_mid - noise_target_mid)
    
    print(f"📊 Análisis de reducción de ruido:")
    print(f"   Reduce ruido alta dosis:  {reduces_noise_high}")
    print(f"   Reduce ruido dosis media: {reduces_noise_mid}")
    print(f"   Más cerca target (alta):  {closer_to_target_high}")  
    print(f"   Más cerca target (media): {closer_to_target_mid}")
    
    if (reduces_noise_high or reduces_noise_mid) and (closer_to_target_high or closer_to_target_mid):
        print(f"✅ SÍ REDUCE RUIDO ESTADÍSTICO")
        return True
    else:
        print(f"❌ NO REDUCE RUIDO SIGNIFICATIVAMENTE")
        return False

def create_verification_plots(input_vol, pred_vol, target_vol, case_name):
    """Genera gráficas de verificación"""
    
    plots_dir = Path("verification_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Calcular ratios
    mask = (input_vol > 0.01 * input_vol.max()) & (target_vol > 0.01 * target_vol.max())
    ratios = pred_vol[mask] / (input_vol[mask] + 1e-10)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Histograma de ratios
    axes[0,0].hist(ratios, bins=50, alpha=0.7, color='skyblue')
    axes[0,0].axvline(ratios.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {ratios.mean():.2f}')
    axes[0,0].set_xlabel('Ratio Predicción/Input')
    axes[0,0].set_ylabel('Frecuencia') 
    axes[0,0].set_title('Distribución de Ratios Pred/Input')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Scatter plot predicción vs input
    sample_idx = np.random.choice(mask.sum(), min(5000, mask.sum()), replace=False)
    input_sample = input_vol[mask][sample_idx]
    pred_sample = pred_vol[mask][sample_idx]
    
    axes[0,1].scatter(input_sample, pred_sample, alpha=0.5, s=1)
    # Línea de multiplicación perfecta
    x_range = np.array([input_sample.min(), input_sample.max()])
    axes[0,1].plot(x_range, x_range * ratios.mean(), 'r--', linewidth=2, label=f'y = {ratios.mean():.2f}x')
    axes[0,1].set_xlabel('Input')
    axes[0,1].set_ylabel('Predicción')
    axes[0,1].set_title('Predicción vs Input')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Slice central comparativo
    z_mid = input_vol.shape[0] // 2
    vmax = target_vol.max()
    
    im = axes[1,0].imshow(pred_vol[z_mid] - ratios.mean() * input_vol[z_mid], 
                         cmap='RdBu_r', vmin=-vmax*0.1, vmax=vmax*0.1, aspect='auto')
    axes[1,0].set_title(f'Diferencia: Pred - {ratios.mean():.2f}×Input')
    axes[1,0].axis('off')
    plt.colorbar(im, ax=axes[1,0], fraction=0.046)
    
    # 4. Correlación espacial
    z_levels = [5, 10, 15, 20]
    correlations = []
    
    for z in z_levels:
        if z < target_vol.shape[0]:
            mask_slice = target_vol[z] > 0.05 * target_vol.max()
            if mask_slice.sum() > 0:
                corr = np.corrcoef(pred_vol[z][mask_slice], target_vol[z][mask_slice])[0,1]
                correlations.append(corr)
            else:
                correlations.append(0)
    
    axes[1,1].plot(z_levels[:len(correlations)], correlations, 'bo-', linewidth=2)
    axes[1,1].set_xlabel('Slice Z')
    axes[1,1].set_ylabel('Correlación Pred-Target')
    axes[1,1].set_title('Correlación por Slice')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim([0, 1])
    
    fig.suptitle(f'Verificación Anti-Multiplicador: {case_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / f"verification_{case_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("="*80)
    print("🔍 VERIFICACIÓN ANTI-MULTIPLICADOR TRIVIAL")
    print("="*80)
    print("Verificando que el modelo no sea solo input × factor_constante")
    
    # Cargar datos
    data = load_test_data()
    if data is None:
        return
    
    input_vol, pred_vol, target_vol, case_name = data
    
    # Ejecutar pruebas
    test1 = test_constant_multiplier(input_vol, pred_vol, target_vol)
    test2 = test_spatial_patterns(input_vol, pred_vol, target_vol) 
    test3 = test_noise_reduction(input_vol, pred_vol, target_vol)
    
    # Generar gráficas
    print(f"\n📊 Generando gráficas de verificación...")
    create_verification_plots(input_vol, pred_vol, target_vol, case_name)
    
    # Veredicto final
    print(f"\n{'='*80}")
    print("🏆 VEREDICTO FINAL")
    print(f"{'='*80}")
    
    if test1:
        print("❌ PREOCUPANTE: El modelo parece ser un multiplicador casi constante")
        verdict = "MULTIPLICADOR TRIVIAL"
    elif test2 and test3:
        print("✅ EXCELENTE: El modelo mejora patrones espaciales Y reduce ruido")
        verdict = "DENOISING INTELIGENTE"
    elif test2 or test3:
        print("✅ BUENO: El modelo hace al menos mejoras espaciales O reduce ruido")
        verdict = "DENOISING PARCIAL"
    else:
        print("❌ PROBLEMÁTICO: No se detectan mejoras significativas")
        verdict = "FUNCIÓN INCIERTA"
    
    print(f"\n🎯 CLASIFICACIÓN: {verdict}")
    
    if verdict == "MULTIPLICADOR TRIVIAL":
        print(f"\n🔧 RECOMENDACIONES:")
        print(f"   • Revisar función de pérdida y normalización")
        print(f"   • Verificar que el modelo no colapse a solución trivial")
        print(f"   • Considerar arquitectura más compleja o regularización")
    else:
        print(f"\n🎉 ¡El modelo está funcionando correctamente!")
        print(f"   Gráficas guardadas en: verification_plots/")

if __name__ == "__main__":
    main()