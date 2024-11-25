#include <iostream>
#include <stdexcept>
#include <cstring> // For strcmp
#include <fstream>
#include <random>
#include <cmath>
#include <numeric>
#include <memory> // For unique_ptr
#include "interface_function.h"
#include "data_storage.h"
#include "neural_network.h"
#include "data_normalization.h"

#ifdef _WIN32  // For Windows
    #include <windows.h>
#else // For Linux/macOS
    #include <dlfcn.h>
#endif


// Interface function prototype (exposed to other applications)
extern "C" __declspec(dllexport) std::vector<double> processData(const std::vector<BarData>& barData, 
                                                            const std::map<std::string, std::vector<double>>& indicatorData, 
                                                            bool useIndicators, bool isTraining);

extern "C" __declspec(dllexport) bool setNetworkParameters(size_t numInputs, size_t numOutputs, const char* normalizationTypeStr, const char* modelVersion);

extern "C" __declspec(dllexport) bool addLayerToNetwork(size_t numOutputs, const char* activationTypeStr);

extern "C" __declspec(dllexport) bool saveNetworkModel(const char* filename);

extern "C" __declspec(dllexport) bool loadNetworkModel(const char* filename);

// Global variables
static std::unique_ptr<NeuralNetwork> g_neuralNetwork = nullptr;
static std::unique_ptr<DataNormalization> g_dataNormalization = nullptr;
static std::string g_modelVersion = "1.0";


bool initializeNeuralNetwork(size_t numInputs, size_t numOutputs, DataNormalization::NormalizationType normalizationType, const std::string& modelVersion) {
    try {
        g_neuralNetwork = std::make_unique<NeuralNetwork>(numInputs, numOutputs);
        g_dataNormalization = std::make_unique<DataNormalization>(normalizationType);
        g_modelVersion = modelVersion;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing: " << e.what() << std::endl;
        return false;
    }
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    return TRUE;
}

extern "C" __declspec(dllexport) std::vector<double> processData(const std::vector<BarData>& barData, 
                                                            const std::map<std::string, std::vector<double>>& indicatorData, 
                                                            bool useIndicators, bool isTraining) {
    try {
        if (!g_neuralNetwork || !g_dataNormalization) {
            throw std::runtime_error("Network not initialized.");
        }

        InterfaceFunction interface(*g_neuralNetwork, *g_dataNormalization);
        interface.setTrainingMode(isTraining);
        return interface.processData(barData, indicatorData, useIndicators);
    } catch (const std::exception& e) {
        std::cerr << "Error processing data: " << e.what() << std::endl;
        return {}; 
    }
}


extern "C" __declspec(dllexport) bool setNetworkParameters(size_t numInputs, size_t numOutputs, const char* normalizationTypeStr, const char* modelVersion) {
    try {
        DataNormalization::NormalizationType normalizationType;
        if (std::strcmp(normalizationTypeStr, "MinMax") == 0) {
            normalizationType = DataNormalization::NormalizationType::MinMax;
        } else if (std::strcmp(normalizationTypeStr, "ZScore") == 0) {
            normalizationType = DataNormalization::NormalizationType::ZScore;
        } else {
            throw std::invalid_argument("Invalid normalization type.");
        }

        return initializeNeuralNetwork(numInputs, numOutputs, normalizationType, modelVersion);

    } catch (const std::exception& e) {
        std::cerr << "Error setting parameters: " << e.what() << std::endl;
        return false;
    }
}


extern "C" __declspec(dllexport) bool addLayerToNetwork(size_t numOutputs, const char* activationTypeStr) {
     try {
         if (!g_neuralNetwork) {
            throw std::runtime_error("Network not initialized.");
        }

        Layer::ActivationType activationType;
         if (std::strcmp(activationTypeStr, "ReLU") == 0) {
            activationType = Layer::ActivationType::ReLU;
        } else if (std::strcmp(activationTypeStr, "Sigmoid") == 0) {
            activationType = Layer::ActivationType::Sigmoid;
        } else if (std::strcmp(activationTypeStr, "Tanh") == 0) {
           activationType = Layer::ActivationType::Tanh;
        } else if (std::strcmp(activationTypeStr, "Linear") == 0) {
            activationType = Layer::ActivationType::Linear;
        } else if (std::strcmp(activationTypeStr, "None") == 0) {
            activationType = Layer::ActivationType::None;
        } else {
            throw std::invalid_argument("Invalid activation type.");
        }

        g_neuralNetwork->addLayer(numOutputs, activationType);
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error adding layer: " << e.what() << std::endl;
        return false;
    }
}

extern "C" __declspec(dllexport) bool saveNetworkModel(const char* filename) {
    try {
        if (!g_neuralNetwork || !g_dataNormalization) {
            throw std::runtime_error("Network not initialized.");
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file to save model.");
        }

        // 1. Save Model Version
        file << g_modelVersion << "\n";

        // 2. Save Normalization Type
        file << static_cast<int>(g_dataNormalization->getNormalizationType()) << "\n";

        if (g_dataNormalization->getNormalizationType() == DataNormalization::NormalizationType::MinMax) {
            file << g_dataNormalization->getMinRange() << " " << g_dataNormalization->getMaxRange() << "\n";
        } else if (g_dataNormalization->getNormalizationType() == DataNormalization::NormalizationType::ZScore) {
             file << g_dataNormalization->getMean() << " " << g_dataNormalization->getStd() << "\n";
        }

        // 3. Save Neural Network
        g_neuralNetwork->saveModel(file); 

        file.close();

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false; 
    }
}

extern "C" __declspec(dllexport) bool loadNetworkModel(const char* filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file to load model.");
        }

        // 1. Load Model Version
        std::string modelVersion;
        std::getline(file, modelVersion);
        g_modelVersion = modelVersion;

        //2. Load Normalization Type
        int normalizationTypeInt;
        file >> normalizationTypeInt;
        DataNormalization::NormalizationType normalizationType = static_cast<DataNormalization::NormalizationType>(normalizationTypeInt);


        g_dataNormalization = std::make_unique<DataNormalization>(normalizationType);


       if(g_dataNormalization->getNormalizationType() == DataNormalization::NormalizationType::MinMax) {
           double min, max;
           file >> min >> max;
           g_dataNormalization->setMinMaxRange(min, max);
       } else if(g_dataNormalization->getNormalizationType() == DataNormalization::NormalizationType::ZScore) {
            double mean, std;
           file >> mean >> std;
           g_dataNormalization->setMeanStd(mean, std);
       }


        //3. Load Neural Network
        g_neuralNetwork = std::make_unique<NeuralNetwork>(); // Correctly create a new NeuralNetwork 
        g_neuralNetwork->loadModel(file);

        file.close();
        return true; 

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false; 
    }
}