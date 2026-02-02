# Arreglo Experimental - Elekta Versa 6MV

## 1. ACELERADOR LINEAL

**Modelo:** Elekta Versa HD (oficial OpenGate)  
**Modo:** Fotones 6 MV  
**Configuración:**
- Target: Tungsteno (W)
- Flattening filter: Incluido
- Jaws: 10×10 cm² (campo cuadrado)
- MLC: No configurado en esta simulación

## 2. FUENTE DE ELECTRONES

**Partícula primaria:** Electrones (e-)  
**Energía nominal:** 6 MeV  
**Distribución:** Gaussiana (3% FWHM)  
**Posición:** Cabeza del linac (z = 0 cm)  
**Spot size:** ~3 mm de radio  

## 3. PHASE SPACE PLANE

**Posición:** z = 65.35 cm (debajo del linac head)  
**Distancia SSD equivalente:** ~65 cm  
**Partículas capturadas:** 363,642 de 10,000,000 primarios (3.64%)

**Composición:**
- Fotones (γ): 361,531 (99.42%)
- Electrones (e-): 2,045 (0.56%)
- Positrones (e+): 66 (0.02%)

**Espectro energético:**
- Rango: 0.002 - 6.168 MeV
- Media: 1.302 MeV
- Distribución: Característica de haz 6 MV

**Dirección:**
- Promedio: (0, 0, -0.948) → principalmente -Z (hacia abajo)
- Divergencia: Angular limitada por colimadores

## 4. PHANTOM

**Material:** Agua líquida (G4_WATER, ρ = 1.0 g/cm³)  
**Geometría:** Caja cúbica  
**Dimensiones:** 30 × 30 × 30 cm³  
**Posición:** z = 60.35 cm  
**Distancia fuente-superficie (SSD):** ~60 cm  
**Distancia desde phase space:** 5 cm  

## 5. DOSE ACTOR

**Matriz de dosis:** 150 × 150 × 150 voxels  
**Resolución espacial:** 2.0 mm isotrópica  
**Volumen total:** 27,000 cm³  
**Rango de dosis:** 0 - 81.8 μGy  
**Dosis media:** 4.87 nGy  

**Outputs:**
- Dosis absorbida (Gy)
- Incertidumbre estadística (%)
- Energía depositada (MeV)

## 6. FÍSICA

**Lista de física:** QGSP_BIC_EMZ  
- **QGSP:** Quark-Gluon String Precompound model
- **BIC:** Binary Ion Cascade
- **EMZ:** Electromagnetic physics (Opción Z - precisión estándar)

**Production cuts:**
- World (aire): 1.0 mm
- Phantom (agua): 0.1 mm

**Procesos incluidos:**
- Dispersión Compton
- Efecto fotoeléctrico
- Producción de pares
- Bremsstrahlung
- Ionización
- Dispersión múltiple

## 7. CONFIGURACIÓN COMPUTACIONAL

### Phase Space Generation
- **Software:** OpenGate 10.0.3
- **Backend:** Geant4 11.x
- **Threads:** 1 (single-thread óptimo)
- **Tiempo de cómputo:** ~13 minutos (780 s)
- **Hardware:** AMD Ryzen 7 260 (8 cores)

### Dose Calculation
- **Threads:** 4 (multi-thread eficiente)
- **Tiempo de cómputo:** 17.7 segundos
- **Eventos secundarios:** 1,454,568
- **Tracks totales:** ~4.5M

## 8. FORMATO DE DATOS

**Phase Space:**
- ROOT: 36 MB (formato nativo OpenGate)
- NumPy: 7.6 MB (formato comprimido, .npz)

**Dosis:**
- MetaImage (.mhd/.raw): 78 MB total
- Formato: Float64 (double precision)
- Compatible con: ITK, SimpleITK, 3D Slicer, MITK

## 9. VALIDACIÓN

✅ Espectro energético consistente con haz 6 MV  
✅ Composición 99.4% fotones (modo photon beam)  
✅ Dirección predominante -Z (correcta geometría)  
✅ Distribución de dosis con buildup esperado  
✅ Eficiencia 3.64% (típica para linac head completo)  

## 10. REFERENCIAS

- **OpenGate:** https://opengate.readthedocs.io
- **Elekta Versa:** Linac pre-configurado oficial OpenGate
- **Geant4 Physics:** QGSP_BIC_EMZ physics list
- **Validation:** Comparar con datos IAEA phase space database
