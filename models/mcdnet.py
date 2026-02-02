"""
Arquitectura MCDNet 3D para Denoising de Dosis Monte Carlo

Implementación de una red neuronal convolucional profunda (dCNN) para
eliminar el ruido estadístico de distribuciones de dosis Monte Carlo
con baja estadística (~1e7 partículas) y predecir la distribución
equivalente de alta estadística (~1e9 partículas).

Características principales:
- Sin downsampling para preservar detalles espaciales
- Capas convolucionales 3D con activaciones ReLU
- Skip connections para facilitar el gradiente
- Normalización por batch para estabilidad

Referencia:
- "Deep convolutional neural network for denoising Monte Carlo dose 
   distributions in radiotherapy" (2020)

Autor: Proyecto Modular 3 - CUCEI
Fecha: Enero 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """
    Bloque convolucional 3D con BatchNorm y ReLU.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 padding=1, use_batch_norm=True):
        """
        Args:
            in_channels: Canales de entrada
            out_channels: Canales de salida
            kernel_size: Tamaño del kernel
            padding: Padding para mantener dimensiones
            use_batch_norm: Usar normalización por batch
        """
        super(ConvBlock3D, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            bias=not use_batch_norm  # No bias si usamos BatchNorm
        )
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class MCDNet3D(nn.Module):
    """
    Red neuronal MCDNet 3D para denoising de dosis Monte Carlo.
    
    Arquitectura:
    - 10-15 capas convolucionales 3D sin downsampling
    - Skip connections residuales
    - Sin pooling para preservar resolución espacial
    - Salida: dosis limpia predicha
    """
    
    def __init__(self, in_channels=1, out_channels=1, 
                 base_filters=32, num_layers=10,
                 use_residual=True):
        """
        Args:
            in_channels: Canales de entrada (1 para dosis)
            out_channels: Canales de salida (1 para dosis limpia)
            base_filters: Número de filtros en la primera capa
            num_layers: Número total de capas convolucionales
            use_residual: Usar conexiones residuales
        """
        super(MCDNet3D, self).__init__()
        
        self.use_residual = use_residual
        self.num_layers = num_layers
        
        # Primera capa: entrada -> features
        self.input_layer = ConvBlock3D(in_channels, base_filters, kernel_size=3)
        
        # Capas intermedias
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 2):
            # Aumentar gradualmente el número de filtros
            if i < num_layers // 2:
                out_filters = base_filters * (2 ** min(i // 2, 2))  # Max 4x
            else:
                out_filters = base_filters * (2 ** min((num_layers - i) // 2, 2))
            
            in_filters = base_filters if i == 0 else self.conv_layers[-1].conv.out_channels
            
            self.conv_layers.append(
                ConvBlock3D(in_filters, out_filters, kernel_size=3)
            )
        
        # Capa de salida: features -> dosis limpia
        final_in_channels = self.conv_layers[-1].conv.out_channels if len(self.conv_layers) > 0 else base_filters
        self.output_layer = nn.Conv3d(
            final_in_channels, out_channels,
            kernel_size=3, padding=1
        )
        
        # Skip connection directa de entrada a salida (residual learning)
        if use_residual:
            self.residual_conv = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, padding=0
            )
        
        # Inicialización de pesos
        self._initialize_weights()
        
        print(f"MCDNet3D inicializada:")
        print(f"  Capas: {num_layers}")
        print(f"  Filtros base: {base_filters}")
        print(f"  Residual: {use_residual}")
        print(f"  Parámetros totales: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Inicializa los pesos de la red."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada [batch, 1, depth, height, width]
        
        Returns:
            Dosis limpia predicha [batch, 1, depth, height, width]
        """
        # Guardar entrada para conexión residual
        identity = x
        
        # Primera capa
        x = self.input_layer(x)
        
        # Capas intermedias con posibles skip connections
        for i, layer in enumerate(self.conv_layers):
            # Skip connection cada N capas
            if i > 0 and i % 3 == 0 and self.use_residual:
                # Asegurar que las dimensiones coincidan
                if x.size(1) == layer.conv.in_channels:
                    x_skip = x
                    x = layer(x)
                    x = x + x_skip  # Skip connection
                else:
                    x = layer(x)
            else:
                x = layer(x)
        
        # Capa de salida
        x = self.output_layer(x)
        
        # Conexión residual global: aprender el residuo (ruido)
        # Output = Input - Ruido_Predicho
        if self.use_residual:
            residual = self.residual_conv(identity)
            x = residual + x  # O: x = identity - x si aprendemos el ruido directamente
        
        return x
    
    def count_parameters(self):
        """Cuenta el número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Calcula el tamaño del modelo en MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb


class MCDNetLightweight(nn.Module):
    """
    Versión ligera de MCDNet para inferencia rápida.
    Menos capas y filtros, ideal para aplicaciones en tiempo real.
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super(MCDNetLightweight, self).__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, base_filters),
            ConvBlock3D(base_filters, base_filters * 2),
            ConvBlock3D(base_filters * 2, base_filters * 4),
        )
        
        self.decoder = nn.Sequential(
            ConvBlock3D(base_filters * 4, base_filters * 2),
            ConvBlock3D(base_filters * 2, base_filters),
            nn.Conv3d(base_filters, out_channels, kernel_size=3, padding=1)
        )
        
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = x
        x = self.encoder(x)
        x = self.decoder(x)
        x = x + self.residual(identity)
        return x


def create_mcdnet(model_type='standard', **kwargs):
    """
    Factory function para crear modelos MCDNet.
    
    Args:
        model_type: 'standard', 'deep', 'lightweight'
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        Modelo MCDNet
    """
    if model_type == 'standard':
        return MCDNet3D(num_layers=10, base_filters=32, **kwargs)
    elif model_type == 'deep':
        return MCDNet3D(num_layers=15, base_filters=64, **kwargs)
    elif model_type == 'lightweight':
        return MCDNetLightweight(**kwargs)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")


def test_model():
    """Prueba básica del modelo."""
    print("\nProbando MCDNet3D...")
    
    # Crear modelo
    model = MCDNet3D(num_layers=10, base_filters=32)
    
    # Tensor de prueba: [batch=2, channels=1, depth=80, height=80, width=80]
    x = torch.randn(2, 1, 80, 80, 80)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Parámetros: {model.count_parameters():,}")
    print(f"Tamaño del modelo: {model.get_model_size_mb():.2f} MB")
    
    # Forward pass
    with torch.no_grad():
        y = model(x)
    
    print(f"Output shape: {y.shape}")
    print(f"\n✓ Modelo funciona correctamente")
    
    return model


if __name__ == '__main__':
    # Prueba del modelo
    model = test_model()
    
    # Probar versión lightweight
    print("\n" + "="*60)
    print("Probando MCDNetLightweight...")
    model_light = MCDNetLightweight()
    x = torch.randn(1, 1, 80, 80, 80)
    y = model_light(x)
    print(f"Parámetros lightweight: {sum(p.numel() for p in model_light.parameters()):,}")
    print(f"✓ Modelo lightweight funciona correctamente")
