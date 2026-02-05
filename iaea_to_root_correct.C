#include <iostream>
#include <fstream>
#include <cmath>
#include <regex>
#include <random>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"

/**
 * Conversor IAEA PHSP → ROOT (ESTRUCTURA CORRECTA)
 * 
 * Basado en análisis real del binario:
 * - byte[0]: particle type (1=γ, 2=e-, 3=e+)
 * - float[0]: X [cm]
 * - float[1]: Y [cm]
 * - float[2]: Weight (signo indica nueva historia)
 * - float[3]: Z [cm]
 * - float[4]: U (direction cosine)
 * - float[5]: V (direction cosine)
 * - float[6]: W (direction cosine)
 * - int[0]: history number
 * - int[1]: ILB PENELOPE variable
 */

#pragma pack(push, 1)
struct IAEA_Record {
    char   particleType;  // 1 byte
    float  x;             // 4 bytes
    float  y;             // 4 bytes
    float  weight;        // 4 bytes (statistical weight)
    float  z;             // 4 bytes
    float  u;             // 4 bytes (direction cosine)
    float  v;             // 4 bytes (direction cosine)
    float  w;             // 4 bytes (direction cosine)
    int    history;       // 4 bytes
    int    ilb;           // 4 bytes
};
#pragma pack(pop)

struct EnergyStats {
    float mean;
    float min;
    float max;
};

// Estadísticas de energía del header IAEA (6MeV)
EnergyStats photon_energy = {0.8442f, 0.02f, 7.184f};
EnergyStats electron_energy = {6.134f, 0.1f, 6.958f};
EnergyStats positron_energy = {1.595f, 0.1072f, 5.513f};

/**
 * Generar energía usando distribución triangular
 * centrada en la media
 */
float sampleEnergy(const EnergyStats& stats, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    
    // Distribución triangular: más probable cerca de la media
    float range = stats.max - stats.min;
    float mid = (stats.mean - stats.min) / range;
    
    float e;
    if (u < mid) {
        // Rama izquierda
        e = stats.min + std::sqrt(u * mid) * range;
    } else {
        // Rama derecha
        e = stats.max - std::sqrt((1.0f - u) * (1.0f - mid)) * range;
    }
    
    return e;
}

void iaea_to_root_correct(
    TString headerFileName = "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAheader",
    TString inputFileName = "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15.IAEAphsp",
    TString outputFileName = "data/IAEA/Varian_Clinac_2100CD_6MeV_15x15_CORRECT.root",
    Long64_t maxParticles = 0) {
    
    std::cout << "================================================" << std::endl;
    std::cout << "CONVERSOR IAEA → ROOT (ESTRUCTURA CORRECTA)" << std::endl;
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
    if (maxParticles > 0 && maxParticles < nParticles) {
        nParticles = maxParticles;
    }
    
    std::cout << "Archivo: " << fileSize / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "Partículas a convertir: " << nParticles << std::endl;
    
    // Estadísticas del header
    std::cout << "\nEstadísticas de energía (del header):" << std::endl;
    std::cout << "  Fotones:    <E>=" << photon_energy.mean << " MeV, "
              << "rango [" << photon_energy.min << ", " << photon_energy.max << "]" << std::endl;
    std::cout << "  Electrones: <E>=" << electron_energy.mean << " MeV, "
              << "rango [" << electron_energy.min << ", " << electron_energy.max << "]" << std::endl;
    std::cout << "  Positrones: <E>=" << positron_energy.mean << " MeV, "
              << "rango [" << positron_energy.min << ", " << positron_energy.max << "]" << std::endl;
    
    // 2. Crear archivo ROOT
    TFile *rFile = new TFile(outputFileName, "RECREATE");
    TTree *tree = new TTree("phsp", "Phase Space - Varian Clinac 2100CD 6MeV");
    
    // Variables para el árbol
    Int_t   pid;
    Float_t E;
    Float_t x, y, z;
    Float_t dx, dy, dz;
    Float_t w;
    Int_t   history;
    Int_t   ilb;
    Bool_t  newHist;
    
    // Crear ramas
    tree->Branch("pid", &pid, "pid/I");
    tree->Branch("E", &E, "E/F");
    tree->Branch("x", &x, "x/F");
    tree->Branch("y", &y, "y/F");
    tree->Branch("z", &z, "z/F");
    tree->Branch("dx", &dx, "dx/F");
    tree->Branch("dy", &dy, "dy/F");
    tree->Branch("dz", &dz, "dz/F");
    tree->Branch("w", &w, "w/F");
    tree->Branch("history", &history, "history/I");
    tree->Branch("ilb", &ilb, "ilb/I");
    tree->Branch("newHist", &newHist, "newHist/O");
    
    // RNG para muestreo de energías
    std::mt19937 rng(42);
    
    // Buffer de lectura
    IAEA_Record record;
    long long count = 0;
    long long nPhotons = 0, nElectrons = 0, nPositrons = 0, nSkipped = 0;
    
    std::cout << "\n📖 Leyendo y convirtiendo..." << std::endl;
    
    // 3. Leer registros
    while (file.read(reinterpret_cast<char*>(&record), sizeof(IAEA_Record))) {
        
        if (count >= nParticles) break;
        
        // Tipo de partícula viene del byte[0] directamente
        int type_byte = static_cast<unsigned char>(record.particleType);
        
        int pdg_pid = 0;
        EnergyStats* energyStats = nullptr;
        
        if (type_byte == 1) {  // Fotón
            pdg_pid = 22;
            energyStats = &photon_energy;
            nPhotons++;
        } else if (type_byte == 2) {  // Electrón
            pdg_pid = 11;
            energyStats = &electron_energy;
            nElectrons++;
        } else if (type_byte == 3) {  // Positrón
            pdg_pid = -11;
            energyStats = &positron_energy;
            nPositrons++;
        } else {
            // Tipo desconocido
            nSkipped++;
            continue;
        }
        
        // Muestrear energía
        E = sampleEnergy(*energyStats, rng);
        
        // Asignar valores
        pid = pdg_pid;
        x = record.x;
        y = record.y;
        z = record.z;
        dx = record.u;
        dy = record.v;
        dz = record.w;
        w = std::abs(record.weight);
        history = record.history;
        ilb = record.ilb;
        
        // Nueva historia: signo negativo del weight
        newHist = (record.weight < 0);
        
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
    std::cout << "   Partículas guardadas: " << count << std::endl;
    std::cout << "   - Fotones (γ):     " << nPhotons << std::endl;
    std::cout << "   - Electrones (e⁻): " << nElectrons << std::endl;
    std::cout << "   - Positrones (e⁺): " << nPositrons << std::endl;
    if (nSkipped > 0) {
        std::cout << "   - Saltados:        " << nSkipped << std::endl;
    }
    
    // 4. Guardar
    std::cout << "\n💾 Guardando archivo ROOT..." << std::endl;
    tree->Write();
    rFile->Close();
    
    std::cout << "✅ Guardado: " << outputFileName << std::endl;
    std::cout << "================================================" << std::endl;
}
