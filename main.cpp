// main.cpp
#include <iostream>
#include <stdexcept>
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
extern "C" __declspec(dllexport)  std::vector<double> processData(const std::vector<BarData>& barData, 
                                                            const std::map<std::string, std::vector<double>>& indicatorData, 
                                                            bool useIndicators, bool isTraining);

// Global variables to store the neural network and normalization type
static NeuralNetwork* g_neuralNetwork = nullptr;
static DataNormalization::NormalizationType g_normalizationType = DataNormalization::NormalizationType::MinMax;

// Function to initialize the neural network
bool initializeNeuralNetwork(size_t numInputs, size_t numOutputs, DataNormalization::NormalizationType normalizationType) {
    try {
        if (g_neuralNetwork != nullptr) {
            delete g_neuralNetwork;
        }
        g_neuralNetwork = new NeuralNetwork(numInputs, numOutputs);
        g_normalizationType = normalizationType;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing neural network: " << e.what() << std::endl;
        return false;
    }
}



BOOL APIENTRY DllMain(HMODULE hModule,
                      DWORD  ul_reason_for_call,
                      LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}




extern "C" __declspec(dllexport) std::vector<double> processData(const std::vector<BarData>& barData, 
                                                            const std::map<std::string, std::vector<double>>& indicatorData, 
                                                            bool useIndicators, bool isTraining) {
    try {

        if (g_neuralNetwork == nullptr) {
            throw std::runtime_error("Neural network not initialized.");
        }
        InterfaceFunction interface(*g_neuralNetwork, g_normalizationType);
        interface.setTrainingMode(isTraining);
        return interface.processData(barData, indicatorData, useIndicators);

    } catch (const std::exception& e) {
        std::cerr << "Error processing data: " << e.what() << std::endl;
        return {}; // Return an empty vector to indicate an error
    }
}

extern "C" __declspec(dllexport) bool setNetworkParameters(size_t numInputs, size_t numOutputs, const char* normalizationTypeStr) {
    try {
      DataNormalization::NormalizationType normalizationType;
        if (std::strcmp(normalizationTypeStr, "MinMax") == 0) {
            normalizationType = DataNormalization::NormalizationType::MinMax;
        } else if (std::strcmp(normalizationTypeStr, "ZScore") == 0) {
            normalizationType = DataNormalization::NormalizationType::ZScore;
        } else {
            throw std::invalid_argument("Invalid normalization type specified.");
        }

        return initializeNeuralNetwork(numInputs, numOutputs, normalizationType);


    } catch (const std::exception& e) {
        std::cerr << "Error setting network parameters: " << e.what() << std::endl;
        return false;
    }

}


extern "C" __declspec(dllexport) bool addLayerToNetwork(size_t numOutputs, const char* activationTypeStr) {
    try {
         if (g_neuralNetwork == nullptr) {
            throw std::runtime_error("Neural network not initialized. Call setNetworkParameters first.");
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
        }

        else {
            throw std::invalid_argument("Invalid activation type provided.");
        }

        g_neuralNetwork->addLayer(numOutputs, activationType);
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error adding layer: " << e.what() << std::endl;
        return false;
    }
}




extern "C" __declspec(dllexport) bool saveNetworkModel(const char* filename)
{

    try {
         if (g_neuralNetwork == nullptr) {
            throw std::runtime_error("Neural network not initialized. Cannot save.");
        }
        g_neuralNetwork->saveModel(filename);
         return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
        return false; // Indicate failure
    }
}


extern "C" __declspec(dllexport) bool loadNetworkModel(const char* filename) {
    try {

         if (g_neuralNetwork == nullptr) {
             // Initialize with default values if not already initialized.  Adjust as needed.
              initializeNeuralNetwork(4, 1, DataNormalization::NormalizationType::MinMax);
        }
         g_neuralNetwork->loadModel(filename);
        return true; // Indicate success

    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false; // Indicate failure
    }
}