// interface_function.h
#ifndef INTERFACE_FUNCTION_H
#define INTERFACE_FUNCTION_H

#include <vector>
#include "data_storage.h"
#include "neural_network.h"
#include "data_normalization.h"
#include <stdexcept>

class InterfaceFunction {
public:
    InterfaceFunction(NeuralNetwork& neuralNetwork, DataNormalization::NormalizationType normalizationType = DataNormalization::NormalizationType::MinMax);

    std::vector<double> processData(const std::vector<BarData>& barData, const std::map<std::string, std::vector<double>>& indicatorData = {}, bool useIndicators = true);

    void setTrainingMode(bool isTraining);
    DataNormalization& getDataNormalization();

private:
    NeuralNetwork& neuralNetwork_;
    DataNormalization dataNormalization_;
    bool isTraining_ = false; 

    std::vector<double> createInputVector(const BarData& barData, const std::map<std::string, std::vector<double>>& indicatorData, bool useIndicators) const;

    void trainNetwork(const DataStorage& dataStorage);
};

#endif // INTERFACE_FUNCTION_H