# Contribuyendo a Proyecto Modular 3

¡Gracias por tu interés en contribuir! Este documento proporciona guías para contribuir al proyecto.

## Código de Conducta

Este proyecto sigue un código de conducta. Al participar, te comprometes a mantener un ambiente respetuoso y profesional.

## ¿Cómo Puedo Contribuir?

### Reportar Bugs

Antes de reportar un bug, verifica que:
1. No sea un duplicado (busca en issues existentes)
2. Sea reproducible con los pasos claros
3. Incluyas información del sistema (OS, Python version, GPU)

**Template de Bug Report:**
```markdown
**Descripción del bug**
Descripción clara del problema.

**Pasos para reproducir**
1. Ejecutar comando '...'
2. Ver error '...'

**Comportamiento esperado**
Lo que debería suceder.

**Screenshots/Logs**
Si aplica, añadir capturas o logs.

**Entorno:**
- OS: [e.g. Ubuntu 22.04]
- Python: [e.g. 3.10]
- GATE version: [e.g. 10.0]
- GPU: [e.g. NVIDIA RTX 3080]
```

### Sugerir Mejoras

Las sugerencias de mejoras son bienvenidas. Incluye:
- **Motivación**: ¿Por qué es útil?
- **Descripción**: ¿Qué propones?
- **Alternativas**: ¿Consideraste otras opciones?

### Pull Requests

#### Proceso

1. **Fork el repositorio**
```bash
git clone https://github.com/tu-usuario/Modular3.git
cd Modular3
```

2. **Crear una rama**
```bash
git checkout -b feature/nueva-caracteristica
# o
git checkout -b fix/correccion-bug
```

3. **Hacer cambios**
- Sigue el estilo de código existente
- Añade tests si es aplicable
- Actualiza documentación

4. **Commit**
```bash
git add .
git commit -m "feat: añadir nueva característica X"
```

Usa [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` nueva característica
- `fix:` corrección de bug
- `docs:` cambios en documentación
- `test:` añadir/modificar tests
- `refactor:` refactorización sin cambio funcional
- `perf:` mejoras de performance
- `style:` formateo, espacios, etc.

5. **Push y crear PR**
```bash
git push origin feature/nueva-caracteristica
```

Luego crea el Pull Request en GitHub con:
- Título descriptivo
- Descripción de cambios
- Referencias a issues relacionados
- Screenshots si es visual

#### Checklist para PR

- [ ] Código sigue estilo del proyecto
- [ ] Tests añadidos/actualizados
- [ ] Documentación actualizada
- [ ] Tests pasan localmente
- [ ] No hay conflictos con main
- [ ] Commit messages son descriptivos

## Guías de Estilo

### Python

Seguimos [PEP 8](https://pep8.org/) con algunas excepciones:

```python
# ✅ BUENO
def calculate_dose(phantom_size: tuple, 
                   n_particles: float = 1e8) -> np.ndarray:
    """
    Calcula distribución de dosis.
    
    Args:
        phantom_size: Dimensiones (x, y, z) en voxels
        n_particles: Número de partículas a simular
    
    Returns:
        Array 3D con dosis en Gy
    """
    pass

# ❌ MALO
def calc(ps, n=1e8):  # Sin tipos, nombres confusos
    pass
```

**Convenciones:**
- Snake_case para funciones y variables
- PascalCase para clases
- UPPER_CASE para constantes
- 4 espacios (no tabs)
- Max 100 caracteres por línea
- Docstrings en todas las funciones públicas

### Docstrings

Usa formato Google:

```python
def train_model(data_dir: Path, epochs: int = 100) -> dict:
    """
    Entrena modelo MCDNet.
    
    Args:
        data_dir: Directorio con dataset de entrenamiento
        epochs: Número de épocas a entrenar
    
    Returns:
        dict: Historial de entrenamiento con keys:
            - train_losses: List[float]
            - val_losses: List[float]
            - best_epoch: int
    
    Raises:
        ValueError: Si data_dir no existe
        RuntimeError: Si GPU no disponible en modo CUDA
    
    Example:
        >>> history = train_model(Path('data/train'), epochs=50)
        >>> print(f"Best val loss: {min(history['val_losses'])}")
    """
    pass
```

### Tests

```python
import unittest

class TestDoseCalculation(unittest.TestCase):
    """Tests para módulo dose_calculation."""
    
    def setUp(self):
        """Configuración antes de cada test."""
        self.phantom_size = (100, 100, 100)
    
    def test_dose_normalization(self):
        """Verifica normalización correcta de dosis."""
        dose = calculate_dose(self.phantom_size)
        
        self.assertTrue(np.all(dose >= 0))
        self.assertAlmostEqual(np.max(dose), 1.0, places=3)
    
    def tearDown(self):
        """Limpieza después de cada test."""
        pass
```

## Estructura de Commits

Ejemplo de commit bien estructurado:

```
feat(models): añadir arquitectura ResUNet para denoising

- Implementar ResUNet3D con skip connections
- Añadir tests unitarios
- Actualizar documentación con ejemplos de uso

Closes #42
```

## Testing

Antes de hacer PR, ejecuta:

```bash
# Tests unitarios
make run-tests

# O manualmente
python -m pytest tests/ -v

# Con coverage
pytest tests/ --cov=simulations --cov=models --cov=analysis
```

## Documentación

Al añadir nuevas características:

1. **Actualizar README.md** si cambia el uso básico
2. **Actualizar QUICKSTART.md** si hay nuevos comandos
3. **Añadir docstrings** a todas las funciones públicas
4. **Crear ejemplo** en `examples/` si es complejo

## Revisión de Código

Los PRs serán revisados por mantenedores. Criterios:

- ✅ Funcionalidad correcta
- ✅ Tests pasan
- ✅ Documentación clara
- ✅ Sin código duplicado
- ✅ Performance adecuado
- ✅ Seguridad (no exponer credenciales, etc.)

## Preguntas

Si tienes dudas:
1. Revisa documentación en `docs/`
2. Busca en issues cerrados
3. Abre un issue con la etiqueta `question`

## Reconocimiento

Los contribuidores serán añadidos a `CONTRIBUTORS.md` automáticamente.

¡Gracias por contribuir! 🚀
