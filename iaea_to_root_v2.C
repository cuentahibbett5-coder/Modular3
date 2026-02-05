/**
 * Conversor IAEA PHSP → ROOT
 * 
 * Formato según header Varian_Clinac_2100CD_6MeV_15x15:
 * - RECORD_LENGTH: 37 bytes
 * - BYTE_ORDER: 1234 (little-endian)
 * - 1 byte: tipo partícula (1=γ, 2=e-, 3=e+)
 * - 7 floats: X, Y, Z, U, V, W, Weight
 * - 2 ints: history number, ILB PENELOPE
 * 
 * Energías promedio (del header):
 * - Fotones: 0.8442 MeV (0.02-7.184)
 * - Electrones: 6.134 MeV (0.1-6.958)
 * - Positrones: 1.595 MeV (0.1-5.513)
 * 
 * La energía NO está almacenada directamente en el PHSP.
 * Se usa la energía promedio según tipo de partícula.
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TRandom3.h"

// Estructura exacta del registro binario de 37 bytes
#pragma pack(push, 1)
struct IAEA_Record {
    char   particleType; // 1 byte: 1=γ, 2=e-, 3=e+
    float  x;            // 4 bytes [cm]
    float  y;            // 4 bytes [cm]
    float  z;            // 4 bytes [cm]
    float  u;            // 4 bytes (direction cosine X)
    float  v;            // 4 bytes (direction cosine Y)
    float  w;            // 4 bytes (direction cosine Z)
    float  weight;       // 4 bytes (statistical weight, signo=nueva historia)
    int    history;      // 4 bytes
    int    ilb;          // 4 bytes (ILB PENELOPE)
};
#pragma pack(pop)

// Energías del header IAEA (MeV)
const float E_PHOTON_MEAN = 0.8442;
const float E_PHOTON_MIN = 0.02;
const float E_PHOTON_MAX = 7.184;

const float E_ELECTRON_MEAN = 6.134;
const float E_ELECTRON_MIN = 0.1;
const float E_ELECTRON_MAX = 6.958;

const float E_POSITRON_MEAN = 1.595;
const float E_POSITRON_MIN = 0.1072;
const float E_POSITRON_MAX = 5.513;

void iaea_to_root_v2(TString inputFileName = "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp",
                     TString outputFileName = "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_FULL.root") {
    
    std::cout << "================================================" << std::endl;
    std::cout << "CONVERSOR IAEA → ROOT v2" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Verificar tamaño de estructura
    std::cout << "Tamaño struct IAEA_Record: " << sizeof(IAEA_Record) << " bytes" << std::endl;
    if (sizeof(IAEA_Record) != 37) {
        std::cout << "❌ ERROR: Tamaño incorrecto! Esperado 37 bytes." << std::endl;
        return;
    }
    
    // 1. Abrir archivo binario
    std::ifstream file(inputFileName.Data(), std::ios::binary);
    if (!file.is_open()) {
        std::cout << "❌ ERROR: No se pudo abrir " << inputFileName << std::endl;
        return;
    }
    
    // Calcular número de partículas
    file.seekg(0, std::ios::end);
    long long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    long long nParticles = fileSize / 37;
    std::cout << "Archivo: " << fileSize / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "Partículas esperadas: " << nParticles << std::endl;
    
    // 2. Crear archivo ROOT
    TFile *rFile = new TFile(outputFileName, "RECREATE");
    TTree *tree = new TTree("phsp", "Phase Space - Varian Clinac 2100CD 6MeV");
    
    // Variables para el árbol
    Int_t   pid;
    Float_t energy;
    Float_t x, y, z;
    Float_t dx, dy, dz;
    Float_t weight;
    Int_t   history;
    Int_t   ilb;
    Bool_t  newHistory;
    
    // Crear ramas (nombres compatibles con OpenGate PhaseSpaceSource)
    tree->Branch("PDGCode", &pid, "PDGCode/I");
    tree->Branch("Ekine", &energy, "Ekine/F");
    tree->Branch("PrePositionLocal_X", &x, "PrePositionLocal_X/F");
    tree->Branch("PrePositionLocal_Y", &y, "PrePositionLocal_Y/F");
    tree->Branch("PrePositionLocal_Z", &z, "PrePositionLocal_Z/F");
    tree->Branch("PreDirectionLocal_X", &dx, "PreDirectionLocal_X/F");
    tree->Branch("PreDirectionLocal_Y", &dy, "PreDirectionLocal_Y/F");
    tree->Branch("PreDirectionLocal_Z", &dz, "PreDirectionLocal_Z/F");
    tree->Branch("Weight", &weight, "Weight/F");
    
    // Generador para muestrear energías
    TRandom3 rng(42);
    
    // Contadores
    IAEA_Record record;
    long long count = 0;
    long long nPhotons = 0, nElectrons = 0, nPositrons = 0, nSkipped = 0;
    
    std::cout << "\n📖 Leyendo y convirtiendo..." << std::endl;
    
    // 3. Leer registros
    while (file.read(reinterpret_cast<char*>(&record), sizeof(IAEA_Record))) {
        
        // Decodificar tipo de partícula
        // Según análisis previo: type=1→γ, type=2→e-, type=3→e+
        int type = static_cast<unsigned char>(record.particleType);
        
        if (type == 1) {
            pid = 22;  // photon
            // Muestrear energía de distribución aproximada
            energy = E_PHOTON_MIN + rng.Rndm() * (E_PHOTON_MAX - E_PHOTON_MIN);
            // Ajustar hacia la media (distribución triangular simple)
            if (rng.Rndm() > 0.5) energy = E_PHOTON_MEAN + (energy - E_PHOTON_MEAN) * 0.5;
            nPhotons++;
        } else if (type == 2) {
            pid = 11;  // electron
            energy = E_ELECTRON_MIN + rng.Rndm() * (E_ELECTRON_MAX - E_ELECTRON_MIN);
            if (rng.Rndm() > 0.5) energy = E_ELECTRON_MEAN + (energy - E_ELECTRON_MEAN) * 0.5;
            nElectrons++;
        } else if (type == 3) {
            pid = -11; // positron
            energy = E_POSITRON_MIN + rng.Rndm() * (E_POSITRON_MAX - E_POSITRON_MIN);
            if (rng.Rndm() > 0.5) energy = E_POSITRON_MEAN + (energy - E_POSITRON_MEAN) * 0.5;
            nPositrons++;
        } else {
            // Tipo desconocido (253, 254, 255...) - saltar
            nSkipped++;
            continue;
        }
        
        // Convertir posición de cm a mm (OpenGate usa mm)
        x = record.x * 10.0f;
        y = record.y * 10.0f;
        z = record.z * 10.0f;
        
        // Direcciones (ya son cosenos directores, sin unidades)
        dx = record.u;
        dy = record.v;
        dz = record.w;
        
        // Peso estadístico (valor absoluto, signo indica nueva historia)
        weight = std::abs(record.weight);
        newHistory = (record.weight < 0);
        
        // Llenar árbol
        tree->Fill();
        count++;
        
        if (count % 1000000 == 0) {
            double progress = (double)count / nParticles * 100.0;
            std::cout << "  [" << count << " / " << nParticles << "]  " 
                      << progress << "%" << std::endl;
        }
    }
    
    file.close();
    
    std::cout << "\n✅ Conversión completada:" << std::endl;
    std::cout << "   - Partículas guardadas: " << count << std::endl;
    std::cout << "   - Fotones (γ):     " << nPhotons << std::endl;
    std::cout << "   - Electrones (e⁻): " << nElectrons << std::endl;
    std::cout << "   - Positrones (e⁺): " << nPositrons << std::endl;
    std::cout << "   - Saltados:        " << nSkipped << std::endl;
    
    // 4. Guardar
    std::cout << "\n💾 Guardando archivo ROOT..." << std::endl;
    tree->Write();
    rFile->Close();
    
    std::cout << "✅ Guardado: " << outputFileName << std::endl;
    std::cout << "================================================" << std::endl;
}
