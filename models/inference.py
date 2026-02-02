"""
Inferencia y exportación de modelos MCDNet

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path

from models.mcdnet import MCDNet3D, create_mcdnet


class DoseDenoiser:
    """Clase para aplicar denoising a mapas de dosis."""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """Carga un modelo entrenado."""
        checkpoint = torch.load(model_path, map_location=self.device)
        model = create_mcdnet('standard')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        print(f"Modelo cargado desde: {model_path}")
        return model
    
    def denoise(self, noisy_dose):
        """
        Aplica denoising a una dosis ruidosa.
        
        Args:
            noisy_dose: Array numpy [D, H, W] o tensor
        
        Returns:
            Dosis limpia como numpy array
        """
        # Preparar input
        if isinstance(noisy_dose, np.ndarray):
            max_val = np.max(noisy_dose)
            dose_norm = noisy_dose / max_val if max_val > 0 else noisy_dose
            dose_tensor = torch.from_numpy(dose_norm).unsqueeze(0).unsqueeze(0).float()
        else:
            dose_tensor = noisy_dose
            max_val = 1.0
        
        dose_tensor = dose_tensor.to(self.device)
        
        # Inferencia
        with torch.no_grad():
            clean_dose = self.model(dose_tensor)
        
        # Convertir a numpy y desnormalizar
        clean_dose = clean_dose.cpu().squeeze().numpy() * max_val
        
        return clean_dose
    
    def denoise_file(self, input_path, output_path):
        """Aplica denoising a un archivo de imagen."""
        # Cargar
        image = sitk.ReadImage(str(input_path))
        dose_array = sitk.GetArrayFromImage(image)
        
        # Denoise
        clean_dose = self.denoise(dose_array)
        
        # Guardar
        clean_image = sitk.GetImageFromArray(clean_dose.astype(np.float32))
        clean_image.CopyInformation(image)
        sitk.WriteImage(clean_image, str(output_path))
        
        print(f"Dosis limpia guardada en: {output_path}")
    
    def export_to_onnx(self, output_path, input_shape=(1, 1, 80, 80, 80)):
        """Exporta el modelo a formato ONNX."""
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['noisy_dose'],
            output_names=['clean_dose'],
            dynamic_axes={
                'noisy_dose': {0: 'batch_size'},
                'clean_dose': {0: 'batch_size'}
            }
        )
        print(f"Modelo exportado a ONNX: {output_path}")


def main():
    """Script de inferencia."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inferencia con MCDNet')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo')
    parser.add_argument('--input', type=str, required=True, help='Dosis ruidosa')
    parser.add_argument('--output', type=str, required=True, help='Dosis limpia')
    parser.add_argument('--export-onnx', type=str, help='Exportar a ONNX')
    
    args = parser.parse_args()
    
    denoiser = DoseDenoiser(args.model)
    denoiser.denoise_file(args.input, args.output)
    
    if args.export_onnx:
        denoiser.export_to_onnx(args.export_onnx)


if __name__ == '__main__':
    main()
