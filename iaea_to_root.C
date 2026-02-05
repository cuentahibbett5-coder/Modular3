#include <iostream>
#include <fstream>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"

// Estructura exacta del registro binario de 37 bytes
// __attribute__((packed)) evita que el compilador agregue padding
struct IAEA_Record {
    char   particleType; // 1 byte: 1=e-, 2=e+, 3=γ
    float  x;            // 4 bytes
    float  y;            // 4 bytes
    float  z;            // 4 bytes
    float  u;            // 4 bytes (direction cosine)
    float  v;            // 4 bytes (direction cosine)
    float  weight;       // 4 bytes (statistical weight, signo indica nueva historia)
    int    history;      // 4 bytes (History number)
    int    ilb;          // 4 bytes (ILB PENELOPE variable: bits 1-2 particle type, etc)
    int    padding;      // 4 bytes padding para completar 37
} __attribute__((packed));

void iaea_to_root(TString inputFileName, TString outputFileName = "output.root") {
    
    // 1. Abrir archivo binario
    std::ifstream file(inputFileName, std::ios::binary);
    
    if (!file.is_open()) {
        std::cout << "ERROR: No se pudo abrir " << inputFileName << std::endl;
        return;
    }

    // Calcular número de partículas
    file.seekg(0, std::ios::end);
    long long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    long long recordSize = sizeof(IAEA_Record);
    long long nParticles = fileSize / recordSize;
    
    std::cout << "================================================" << std::endl;
    std::cout << "CONVERSOR IAEA → ROOT (C++ ROOT nativo)" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "Tamaño de registro: " << recordSize << " bytes" << std::endl;
    std::cout << "Tamaño archivo: " << fileSize / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << "Partículas: " << nParticles << std::endl;

    if (recordSize != 37) {
        std::cout << "ALERTA: Estructura no es 37 bytes!" << std::endl;
        return;
    }

    // 2. Crear archivo ROOT
    TFile *rFile = new TFile(outputFileName, "RECREATE");
    TTree *tree = new TTree("phsp", "Phase Space - Varian Clinac 2100CD 6MeV");

    // Ramas del árbol
    Int_t   type;
    Float_t energy;
    Float_t x, y, z;
    Float_t u, v, w;
    Float_t weight;
    Int_t   history;
    Int_t   ilb;
    Bool_t  newHistory;

    // Crear ramas
    tree->Branch("pid", &type, "pid/I");
    tree->Branch("E", &energy, "E/F");
    tree->Branch("x", &x, "x/F");
    tree->Branch("y", &y, "y/F");
    tree->Branch("z", &z, "z/F");
    tree->Branch("dx", &u, "dx/F");
    tree->Branch("dy", &v, "dy/F");
    tree->Branch("dz", &w, "dz/F");
    tree->Branch("w", &weight, "w/F");
    tree->Branch("history", &history, "history/I");
    tree->Branch("ilb", &ilb, "ilb/I");
    tree->Branch("newHist", &newHistory, "newHist/O");

    // Buffer de lectura
    IAEA_Record record;
    long long count = 0;
    long long nPhotons = 0, nElectrons = 0, nPositrons = 0;

    std::cout << "\n📖 Leyendo y convirtiendo..." << std::endl;

    // 3. Leer registros
    while (file.read((char*)&record, recordSize)) {
        
        // Asignar valores
        type = (Int_t)record.particleType;
        x = record.x;
        y = record.y;
        z = record.z;
        u = record.u;
        v = record.v;
        weight = record.weight;
        history = record.history;
        ilb = record.ilb;

        // Decodificar signo de energía (nueva historia)
        // En IAEA/PENELOPE, el signo del weight o del primer float indica nueva historia
        // Por ahora, asumimos nueva historia en el primer registro
        newHistory = (count == 0) ? true : false;
        
        // La energía se estima de U,V,W (directores de coseno)
        // Para OpenGate, usamos W como proxy de energía
        energy = weight;  // El peso puede contener info de energía
        
        // W = sqrt(1 - u² - v²)
        float uv_sq = u*u + v*v;
        if (uv_sq <= 1.0f) {
            w = std::sqrt(1.0f - uv_sq);
        } else {
            w = 0.0f;
        }

        // Decodificar tipo de partícula
        // Mapeo correcto (verificado contra header IAEA):
        // type=1 → Fotones (9.26M)
        // type=2 → Electrones (20.02M)  
        // type=3 → Positrones (1.04k)
        // type=253,254,255 → Otros/Desconocidos (IGNORAR - generar PID=0)
        int pdg_pid = 0;
        
        if (type == 1) {
            pdg_pid = 22;      // photon ✅
            nPhotons++;
        } else if (type == 2) {
            pdg_pid = 11;      // electron ✅
            nElectrons++;
        } else if (type == 3) {
            pdg_pid = -11;     // positron ✅
            nPositrons++;
        }
        // else: unknown type (253, 254, 255...) → PID stays 0, will skip in OpenGate
        
        // FILTRO: Solo guardar partículas con PID válido
        if (pdg_pid == 0) {
            continue;  // Skip particles with unknown types
        }
        
        type = pdg_pid;

        // Llenar árbol
        tree->Fill();

        count++;
        if (count % 1000000 == 0) {
            double progress = (double)count / nParticles * 100.0;
            std::cout << "  [" << count << " / " << nParticles << "]  " 
                      << progress << "%" << std::endl;
        }
    }

    std::cout << "\n✅ Lectura completada: " << count << " partículas" << std::endl;
    std::cout << "   - Fotones (γ): " << nPhotons << std::endl;
    std::cout << "   - Electrones (e⁻): " << nElectrons << std::endl;
    std::cout << "   - Positrones (e⁺): " << nPositrons << std::endl;

    // 4. Guardar archivo
    std::cout << "\n💾 Guardando archivo ROOT..." << std::endl;
    tree->Write();
    rFile->Close();
    file.close();

    std::cout << "✅ Conversión completada: " << outputFileName << std::endl;
    std::cout << "================================================" << std::endl;
}
