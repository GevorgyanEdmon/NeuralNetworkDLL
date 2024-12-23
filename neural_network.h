#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "layer.h"
#include "data_storage.h"
#include <stdexcept>
#include <fstream> 
#include <iostream>


class NeuralNetwork {
public:
    NeuralNetwork(size_t numInputs = 0, size_t numOutputs = 0);

    void addLayer(size_t numOutputs, Layer::ActivationType activationType = Layer::ActivationType::ReLU);
    void addLayer(const Layer& layer);

    std::vector<double> predict(const std::vector<double>& input) const;

    void train(const DataStorage& trainingData, size_t epochs, double learningRate);

    void saveModel(std::ostream& file) const;
    void loadModel(std::istream& file);

    std::vector<Layer>& getLayers();
    size_t getNumInputs() const;
    size_t getNumOutputs() const;

    void setTrainingMode(bool isTraining);

private:
    std::vector<Layer> layers_;
    size_t numInputs_;
    size_t numOutputs_;

    double momentum_ = 0.9;
    std::vector<std::vector<std::vector<double>>> previousWeightUpdates_;
    std::vector<std::vector<double>> previousBiasUpdates_;

    std::vector<std::vector<double>> calculateDeltas(const std::vector<double>& target, const std::vector<double>& output, const Layer& layer) const;
    void backpropagate(const std::vector<double>& target, const std::vector<double>& output);
    void updateWeights(double learningRate, const std::vector<double>& input);

    static double activationFunction(double x, Layer::ActivationType type);
    static double activationDerivative(double x, Layer::ActivationType type);

    static double relu(double x);
    static double reluDerivative(double x);
    static double sigmoid(double x);
    static double sigmoidDerivative(double x);
    static double tanh(double x);

};

#endif // NEURAL_NETWORK_H