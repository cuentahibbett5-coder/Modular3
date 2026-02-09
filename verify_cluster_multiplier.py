#!/usr/bin/env python3
"""
Verificación Cluster: ¿Es el modelo solo un multiplicador trivial?
Análisis usando datos reales del cluster
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Para cluster sin display
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path
import argparse
import glob

def find_model_file():
    """Busca automáticamente el mejor modelo disponible"""
    
    # Posibles ubicaciones de modelos
    model_patterns = [
        "runs/*/best_model.pt",
        "runs/*/best.pt", 
        "runs/denoising_v2_residual/best_model.pt",
        "runs/denoising/best.pt",
        "*.pt",
        "**/*.pt"
    ]
    
    for pattern in model_patterns:
        models = glob.glob(pattern, recursive=True)
        if models:
            # Preferir modelos con "best" en el nombre
            best_models = [m for m in models if 'best' in m.lower()]
            if best_models:
                return best_models[0]
            return models[0]
    
    return None

def find_export_data():
    """Busca datos exportados en el cluster"""
    
    exports_dir = Path("exports")
    if not exports_dir.exists():
        print("❌ Directorio 'exports' no encontrado")
        return {}
    
    # Buscar archivos .npy
    input_files = list(exports_dir.glob("*_input.npy"))
    
    if not input_files:
        print("❌ No se encontraron archivos *_input.npy")
        return {}
    
    cases = {}
    
    for input_file in input_files:
        case_name = input_file.stem.replace("_input", "")
        
        # Archivos asociados
        pred_file = exports_dir / f"{case_name}_pred.npy"
        target_file = exports_dir / f"{case_name}_target.npy"
        
        if pred_file.exists() and target_file.exists():
            cases[case_name] = {
                'input': str(input_file),
                'pred': str(pred_file),
                'target': str(target_file)
            }
            print(f"✓ Encontrado caso: {case_name}")
        else:
            print(f"⚠️ Caso incompleto: {case_name}")
    
    return cases

def load_case_data(case_files):
    """Carga datos de un caso específico"""
    
    try:
        input_vol = np.load(case_files['input'])
        pred_vol = np.load(case_files['pred'])
        target_vol = np.load(case_files['target'])
        
        print(f"   Input shape: {input_vol.shape}, range: [{input_vol.min():.3e}, {input_vol.max():.3e}]")
        print(f"   Pred shape: {pred_vol.shape}, range: [{pred_vol.min():.3e}, {pred_vol.max():.3e}]")
        print(f"   Target shape: {target_vol.shape}, range: [{target_vol.min():.3e}, {target_vol.max():.3e}]")
        
        return input_vol, pred_vol, target_vol
    
    except Exception as e:
        print(f"❌ Error cargando caso: {e}")
        return None, None, None

def analyze_trivial_multiplier(input_vol, pred_vol, target_vol, case_name):
    """Analiza si la predicción es solo input × factor_constante"""
    
    print(f"\n🔍 ANALIZANDO: {case_name}")
    print("="*60)
    
    # Crear máscara para voxels significativos (>1% del máximo)
    threshold = 0.01 * target_vol.max()
    mask = target_vol > threshold
    
    if mask.sum() < 1000:  # Muy pocos voxels
        print("   ⚠️ Muy pocos voxels significativos para análisis")
        return None
    
    print(f"   Voxels analizados: {mask.sum():,} / {target_vol.size:,} ({100*mask.sum()/target_vol.size:.1f}%)")
    
    # Extraer valores válidos
    input_masked = input_vol[mask]
    pred_masked = pred_vol[mask]
    target_masked = target_vol[mask]
    
    # Evitar división por cero
    valid_idx = input_masked > 1e-12
    input_valid = input_masked[valid_idx]
    pred_valid = pred_masked[valid_idx]
    target_valid = target_masked[valid_idx]
    
    if len(input_valid) < 100:
        print("   ⚠️ Insuficientes puntos válidos para análisis")
        return None
    
    print(f"   Puntos válidos: {len(input_valid):,}")
    
    # 1. CALCULAR FACTORES DE ESCALA
    # Factor teórico (target/input)
    theoretical_factors = target_valid / input_valid
    mean_theoretical = np.mean(theoretical_factors)
    std_theoretical = np.std(theoretical_factors)
    
    # Factor observado (pred/input)
    observed_factors = pred_valid / input_valid
    mean_observed = np.mean(observed_factors)
    std_observed = np.std(observed_factors)
    
    print(f"\n   📊 FACTORES DE ESCALA:")
    print(f"      Teórico (Target/Input):    μ={mean_theoretical:.2f}, σ={std_theoretical:.2f}")
    print(f"      Observado (Pred/Input):    μ={mean_observed:.2f}, σ={std_observed:.2f}")
    print(f"      Diferencia en medias:      {abs(mean_theoretical - mean_observed):.2f}")
    
    # 2. ¿ES LA PRED SOLO INPUT × FACTOR_CONSTANTE?
    # Crear predicción "ingenua" con factor constante
    naive_pred_vol = input_vol * mean_observed
    naive_pred_valid = naive_pred_vol[mask][valid_idx]
    
    # Comparar predicción real vs ingenua
    mae_pred_vs_naive = np.mean(np.abs(pred_valid - naive_pred_valid))
    correlation_pred_naive = np.corrcoef(pred_valid, naive_pred_valid)[0, 1]
    
    # Normalizar MAE por la magnitud típica
    typical_magnitude = np.mean(pred_valid)
    normalized_mae = mae_pred_vs_naive / typical_magnitude
    
    print(f"\n   🔍 ANÁLISIS DE TRIVIALIDAD:")
    print(f"      MAE(Pred vs Input×{mean_observed:.2f}): {mae_pred_vs_naive:.3e}")
    print(f"      MAE normalizado:                {normalized_mae:.4f}")
    print(f"      Correlación Pred vs Ingenua:    {correlation_pred_naive:.6f}")
    
    # 3. COMPARACIÓN CON TARGET
    mae_pred_target = np.mean(np.abs(pred_valid - target_valid))
    mae_naive_target = np.mean(np.abs(naive_pred_valid - target_valid))
    mae_input_target = np.mean(np.abs(input_valid - target_valid))
    
    improvement_vs_input = mae_input_target / mae_pred_target if mae_pred_target > 0 else 1
    improvement_vs_naive = mae_naive_target / mae_pred_target if mae_pred_target > 0 else 1
    
    print(f"\n   🎯 COMPARACIÓN DE CALIDAD:")
    print(f"      MAE Input→Target:     {mae_input_target:.3e}")
    print(f"      MAE Pred→Target:      {mae_pred_target:.3e}")
    print(f"      MAE Ingenua→Target:   {mae_naive_target:.3e}")
    print(f"      Mejora vs Input:      {improvement_vs_input:.2f}x")
    print(f"      Mejora vs Ingenua:    {improvement_vs_naive:.2f}x")
    
    # 4. CORRELACIONES
    corr_pred_target = np.corrcoef(pred_valid, target_valid)[0, 1]
    corr_input_target = np.corrcoef(input_valid, target_valid)[0, 1]
    corr_naive_target = np.corrcoef(naive_pred_valid, target_valid)[0, 1]
    
    print(f"\n   📈 CORRELACIONES CON TARGET:")
    print(f"      Input:     {corr_input_target:.4f}")
    print(f"      Pred:      {corr_pred_target:.4f}")
    print(f"      Ingenua:   {corr_naive_target:.4f}")
    
    # 5. ANÁLISIS ESPACIAL - ¿Varía el factor localmente?
    spatial_variation = analyze_spatial_factors(input_vol, pred_vol, target_vol, mask)
    
    # 6. VEREDICTO
    # Criterios para clasificar como multiplicador trivial:
    is_trivial = False
    reasons = []
    
    # Alta correlación con predicción ingenua
    if correlation_pred_naive > 0.98:
        is_trivial = True
        reasons.append(f"Alta correlación con ingenua ({correlation_pred_naive:.4f})")
    
    # Muy pequeña diferencia vs multiplicador simple
    if normalized_mae < 0.01:  # <1% de diferencia
        is_trivial = True
        reasons.append(f"Diferencia mínima vs ingenua ({normalized_mae:.3f})")
    
    # No mejora significativa vs multiplicador simple
    if improvement_vs_naive < 1.1:  # <10% de mejora
        is_trivial = True
        reasons.append(f"Sin mejora vs ingenua ({improvement_vs_naive:.2f}x)")
    
    # Factores de escala muy uniformes espacialmente
    if spatial_variation['cv'] < 0.05:  # Coeficiente variación <5%
        reasons.append(f"Factor espacialmente uniforme (CV={spatial_variation['cv']:.3f})")
    
    # Asignar veredicto
    if is_trivial:
        verdict = "⚠️ MULTIPLICADOR TRIVIAL"
        useful = False
        color = "red"
    elif improvement_vs_input > 3.0:
        verdict = "✅ MODELO MUY INTELIGENTE"
        useful = True
        color = "green"
    elif improvement_vs_input > 1.5:
        verdict = "🔶 MODELO MODERADAMENTE ÚTIL"
        useful = True
        color = "orange"
    else:
        verdict = "❌ MODELO POCO ÚTIL"
        useful = False
        color = "red"
    
    print(f"\n   🏆 VEREDICTO: {verdict}")
    if reasons:
        print("      Razones:")
        for reason in reasons:
            print(f"        • {reason}")
    
    return {
        'case_name': case_name,
        'is_trivial': is_trivial,
        'verdict': verdict,
        'useful': useful,
        'reasons': reasons,
        'metrics': {
            'mean_theoretical_factor': float(mean_theoretical),
            'mean_observed_factor': float(mean_observed),
            'correlation_pred_naive': float(correlation_pred_naive),
            'normalized_mae_vs_naive': float(normalized_mae),
            'improvement_vs_input': float(improvement_vs_input),
            'improvement_vs_naive': float(improvement_vs_naive),
            'corr_pred_target': float(corr_pred_target),
            'spatial_cv': float(spatial_variation['cv'])
        }
    }

def analyze_spatial_factors(input_vol, pred_vol, target_vol, mask):
    """Analiza la variación espacial de factores de escala"""
    
    # Dividir en regiones para análisis local
    shape = input_vol.shape
    
    # Crear grid de regiones 4x4x4
    regions = []
    z_splits = np.linspace(0, shape[0], 5, dtype=int)
    y_splits = np.linspace(0, shape[1], 5, dtype=int)  
    x_splits = np.linspace(0, shape[2], 5, dtype=int)
    
    local_factors = []
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Definir región
                z0, z1 = z_splits[i], z_splits[i+1]
                y0, y1 = y_splits[j], y_splits[j+1]
                x0, x1 = x_splits[k], x_splits[k+1]
                
                # Extraer datos de la región
                region_mask = mask[z0:z1, y0:y1, x0:x1]
                
                if region_mask.sum() < 10:  # Muy pocos puntos
                    continue
                
                input_region = input_vol[z0:z1, y0:y1, x0:x1][region_mask]
                pred_region = pred_vol[z0:z1, y0:y1, x0:x1][region_mask]
                
                # Calcular factor local
                valid_region = input_region > 1e-12
                if valid_region.sum() < 5:
                    continue
                    
                local_factor = np.mean(pred_region[valid_region] / input_region[valid_region])
                local_factors.append(local_factor)
    
    if len(local_factors) < 5:
        return {'cv': 0.0, 'n_regions': 0}
    
    # Coeficiente de variación de factores locales
    mean_factor = np.mean(local_factors)
    std_factor = np.std(local_factors)
    cv = std_factor / mean_factor if mean_factor > 0 else 0
    
    return {
        'cv': cv,
        'n_regions': len(local_factors),
        'mean_local_factor': mean_factor,
        'std_local_factor': std_factor
    }

def create_comparison_plot(input_vol, pred_vol, target_vol, naive_pred, case_name, output_dir):
    """Crea visualización comparativa"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Seleccionar slice representativo
    slice_sums = [target_vol[z].sum() for z in range(target_vol.shape[0])]
    best_z = np.argmax(slice_sums)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    vmax = target_vol.max()
    
    # Fila 1: Volúmenes
    titles = ['Input (ruidoso)', 'Target (limpio)', 'Predicción IA', 'Multiplicador Simple']
    volumes = [input_vol[best_z], target_vol[best_z], pred_vol[best_z], naive_pred[best_z]]
    
    for i, (vol, title) in enumerate(zip(volumes, titles)):
        im = axes[0,i].imshow(vol, cmap='hot', vmin=0, vmax=vmax, aspect='auto')
        axes[0,i].set_title(title, fontsize=14, fontweight='bold')
        axes[0,i].axis('off')
        plt.colorbar(im, ax=axes[0,i], fraction=0.046)
    
    # Fila 2: Mapas de error vs target
    error_titles = ['Error Input', 'Error Target', 'Error IA', 'Error Multiplicador']
    errors = [
        np.abs(input_vol[best_z] - target_vol[best_z]),
        np.zeros_like(target_vol[best_z]),  # Target vs target = 0
        np.abs(pred_vol[best_z] - target_vol[best_z]),
        np.abs(naive_pred[best_z] - target_vol[best_z])
    ]
    
    error_max = max(err.max() for err in errors if err.max() > 0)
    
    for i, (err, title) in enumerate(zip(errors, error_titles)):
        if i == 1:  # Skip target vs target
            axes[1,i].text(0.5, 0.5, 'Target\n(referencia)', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[1,i].set_xlim(0, 1)
            axes[1,i].set_ylim(0, 1)
        else:
            im = axes[1,i].imshow(err, cmap='viridis', vmin=0, vmax=error_max, aspect='auto')
            plt.colorbar(im, ax=axes[1,i], fraction=0.046)
        
        axes[1,i].set_title(title, fontsize=14)
        axes[1,i].axis('off')
    
    fig.suptitle(f'Análisis Multiplicador Trivial: {case_name} (z={best_z})', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_dir / f'{case_name}_trivial_analysis.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file

def main():
    parser = argparse.ArgumentParser(description='Verificar multiplicador trivial en cluster')
    parser.add_argument('--output-dir', type=str, default='trivial_analysis_results',
                        help='Directorio de salida')
    parser.add_argument('--max-cases', type=int, default=10,
                        help='Máximo número de casos a analizar')
    
    args = parser.parse_args()
    
    print("="*90)
    print("🔍 VERIFICACIÓN CLUSTER: ¿MULTIPLICADOR TRIVIAL?")
    print("="*90)
    print("Hipótesis: El modelo solo hace Predicción = Input × factor_constante")
    print("Si es cierto → El modelo es inútil vs simular más eventos\n")
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Buscar modelo
    model_file = find_model_file()
    if model_file:
        print(f"✓ Modelo encontrado: {model_file}")
    else:
        print("⚠️ No se encontró modelo, continuando con análisis de exports")
    
    # Buscar datos exportados
    cases = find_export_data()
    
    if not cases:
        print("❌ No se encontraron datos para analizar")
        print("\n💡 Para generar datos:")
        print("   python export_predictions.py")
        return
    
    print(f"\n📊 Casos encontrados: {len(cases)}")
    
    # Limitar número de casos
    case_names = list(cases.keys())[:args.max_cases]
    
    all_results = []
    
    # Procesar cada caso
    for case_name in case_names:
        print(f"\n{'='*90}")
        
        # Cargar datos
        input_vol, pred_vol, target_vol = load_case_data(cases[case_name])
        
        if input_vol is None:
            continue
        
        # Análisis principal
        result = analyze_trivial_multiplier(input_vol, pred_vol, target_vol, case_name)
        
        if result:
            all_results.append(result)
            
            # Crear predicción ingenua para visualización
            mask = target_vol > 0.01 * target_vol.max()
            if mask.sum() > 0:
                factor = result['metrics']['mean_observed_factor']
                naive_pred = input_vol * factor
                
                # Visualización
                plot_file = create_comparison_plot(
                    input_vol, pred_vol, target_vol, naive_pred, 
                    case_name, output_dir
                )
                print(f"   ✓ Gráfica guardada: {plot_file}")
    
    # Resumen global
    print(f"\n{'='*90}")
    print("📊 RESUMEN GLOBAL")
    print("="*90)
    
    if not all_results:
        print("❌ No se pudo analizar ningún caso")
        return
    
    # Estadísticas
    n_total = len(all_results)
    n_trivial = sum(1 for r in all_results if r['is_trivial'])
    n_useful = sum(1 for r in all_results if r['useful'])
    
    avg_improvement = np.mean([r['metrics']['improvement_vs_input'] for r in all_results])
    avg_corr_naive = np.mean([r['metrics']['correlation_pred_naive'] for r in all_results])
    
    print(f"Casos analizados:        {n_total}")
    print(f"Multiplicadores triviales: {n_trivial} ({100*n_trivial/n_total:.1f}%)")
    print(f"Modelos útiles:          {n_useful} ({100*n_useful/n_total:.1f}%)")
    print(f"Mejora promedio vs input:  {avg_improvement:.2f}x")
    print(f"Correlación promedio con ingenua: {avg_corr_naive:.4f}")
    
    # Veredicto final
    print(f"\n🎯 VEREDICTO FINAL:")
    if n_trivial > n_total // 2:
        print("⚠️ LA MAYORÍA SON MULTIPLICADORES TRIVIALES")
        print("   → El modelo no es mejor que simular más eventos")
        print("   → Revisar arquitectura y entrenamiento")
    elif n_useful > n_total * 0.7:
        print("✅ EL MODELO ES GENUINAMENTE INTELIGENTE")
        print("   → Va más allá del simple escalado")
        print("   → Útil para denoising de dosis")
    else:
        print("🔶 RESULTADOS MIXTOS")
        print("   → Algunos casos triviales, otros útiles")
        print("   → Considerar mejoras específicas")
    
    # Guardar resultados
    summary = {
        'analysis_summary': {
            'total_cases': n_total,
            'trivial_cases': n_trivial,
            'useful_cases': n_useful,
            'avg_improvement_vs_input': float(avg_improvement),
            'avg_correlation_with_naive': float(avg_corr_naive)
        },
        'detailed_results': all_results,
        'interpretation': {
            'is_mostly_trivial': n_trivial > n_total // 2,
            'is_mostly_useful': n_useful > n_total * 0.7
        }
    }
    
    results_file = output_dir / 'multiplier_analysis_summary.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Resultados detallados: {results_file}")
    print(f"✓ Gráficas en: {output_dir}/")
    
    print(f"\n💡 INTERPRETACIÓN TÉCNICA:")
    if avg_corr_naive > 0.95:
        print("   Correlación muy alta con multiplicador simple")
        print("   → El modelo aprendió principalmente escalado")
        print("   → No justifica la complejidad vs simular más eventos")
    else:
        print("   Correlación baja con multiplicador simple")
        print("   → El modelo aprendió patrones espaciales complejos")
        print("   → Genuinamente útil para denoising médico")

if __name__ == "__main__":
    main()