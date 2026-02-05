#include <iostream>
#include <fstream>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TString.h"

// Definimos la estructura BINARIA exacta de 37 bytes.
// __attribute__((packed)) obliga al compilador a no dejar espacios vacíos (padding).
struct IAEARecord {
    char   particleType; // 1 byte
    float  energy;       // 4 bytes (El signo negativo indica nueva historia)
    float  x;            // 4 bytes
    float  y;            // 4 bytes
    float  z;            // 4 bytes
    float  u;            // 4 bytes
    float  v;            // 4 bytes
    float  weight;       // 4 bytes
    // Nota: W no se lee del archivo, se calcula.
    int    extra_history; // 4 bytes (Extra long 1)
    int    extra_ilb;     // 4 bytes (Extra long 2: variable ILB PENELOPE)
} __attribute__((packed));

void iaea_converter(TString inputFileName, TString outputFileName = "output_phase_space.root") {
    
    // 1. Configuración de lectura
    std::ifstream file(inputFileName, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "ERROR: No se puede abrir el archivo " << inputFileName << std::endl;
        return;
    }

    // Calcular número de partículas para la barra de progreso
    file.seekg(0, std::ios::end);
    long long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Verificación de integridad
    if (sizeof(IAEARecord) != 37) {
        std::cout << "ERROR FATAL: La estructura no suma 37 bytes. Revisa el compilador." << std::endl;
        return;
    }

    long long nParticles = fileSize / sizeof(IAEARecord);
    std::cout << "Archivo: " << inputFileName << std::endl;
    std::cout << "Tamaño total: " << fileSize << " bytes" << std::endl;
    std::cout << "Partículas estimadas: " << nParticles << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // 2. Configuración de ROOT
    TFile *rFile = new TFile(outputFileName, "RECREATE");
    TTree *tree = new TTree("IAEA_PHSP", "Varian Clinac 2100CD Phase Space");

    // Variables de ROOT (OpenGate compatible)
    Int_t   b_pid;
    Float_t b_energy;
    Float_t b_x, b_y, b_z;
    Float_t b_dx, b_dy, b_dz; // W será calculado
    Float_t b_weight;
    Int_t   b_history;
    Int_t   b_ilb;
    Bool_t  b_newHistory; // Flag derivado del signo de la energía

    // Ramas (Branches) con nombres esperados por OpenGate
    tree->Branch("pid", &b_pid, "pid/I");
    tree->Branch("E", &b_energy, "E/F"); // MeV
    tree->Branch("x", &b_x, "x/F"); // mm
    tree->Branch("y", &b_y, "y/F"); // mm
    tree->Branch("z", &b_z, "z/F"); // mm
    tree->Branch("dx", &b_dx, "dx/F"); // Cosino director X
    tree->Branch("dy", &b_dy, "dy/F"); // Cosino director Y
    tree->Branch("dz", &b_dz, "dz/F"); // Cosino director Z (Calculado)
    tree->Branch("w", &b_weight, "w/F");
    tree->Branch("history", &b_history, "history/I");
    tree->Branch("ilb", &b_ilb, "ilb/I"); // Variable de PENELOPE
    tree->Branch("newHist", &b_newHistory, "newHist/O");

    // 3. Bucle de conversión
    IAEARecord record;
    long long count = 0;

    while (file.read((char*)&record, sizeof(IAEARecord))) {
        
        // Mapeo directo (convertir cm → mm)
        int type = (Int_t)record.particleType;
        b_x = record.x * 10.0f;
        b_y = record.y * 10.0f;
        b_z = record.z * 10.0f;
        b_dx = record.u;
        b_dy = record.v;
        b_weight = record.weight;
        b_history = record.extra_history;
        b_ilb = record.extra_ilb;

        // Lógica de Energía y Nueva Historia (Estándar IAEA)
        // Si Energy < 0, es el primer registro de una nueva historia primaria.
        if (record.energy < 0) {
            b_newHistory = true;
            b_energy = std::abs(record.energy);
        } else {
            b_newHistory = false;
            b_energy = record.energy;
        }

        // Reconstrucción de W
        // W = sqrt(1 - u^2 - v^2). Asumimos dirección +Z (downstream) como indica el header.
        double uv_sq = b_dx*b_dx + b_dy*b_dy;
        if (uv_sq <= 1.0) {
            b_dz = std::sqrt(1.0 - uv_sq);
        } else {
            // Corrección de error numérico pequeño
            b_dz = 0.0;
        }

        // Mapear tipo a PDGCode (1=γ, 2=e-, 3=e+)
        if (type == 1) {
            b_pid = 22;      // photon
        } else if (type == 2) {
            b_pid = 11;      // electron
        } else if (type == 3) {
            b_pid = -11;     // positron
        } else {
            continue; // skip unknown types
        }
        
        // Opcional: Si necesitas simular retrodispersión (backscatter),
        // necesitarías lógica adicional aquí, pero para un PHSP de Linac 
        // debajo de las mandíbulas, W siempre es positivo.

        tree->Fill();

        // Barra de progreso
        count++;
        if (count % 1000000 == 0) {
            std::cout << "Procesando: " << (int)((double)count/nParticles*100) << "% \r" << std::flush;
        }
    }

    // 4. Guardar
    tree->Write();
    rFile->Close();
    file.close();

    std::cout << "\nConversión finalizada. Archivo guardado como: " << outputFileName << std::endl;
}