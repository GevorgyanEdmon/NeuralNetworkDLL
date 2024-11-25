// layer.h
#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include <random>
#include <stdexcept>

class Layer {
public:
    enum class ActivationType {
        ReLU,
        Sigmoid,
        Tanh,
        Linear,
        None 
    };

    Layer(size_t numInputs, size_t numOutputs, ActivationType activationType = ActivationType::ReLU);

    void setActivationFunction(ActivationType activationType);
    std::vector<double> forward(const std::vector<double>& input) const;

    void setWeights(const std::vector<std::vector<double>>& weights);
    std::vector<std::vector<double>> getWeights() const;
    void setBiases(const std::vector<double>& biases);
    std::vector<double> getBiases() const;
    
    size_t getInputSize() const;
    size_t getOutputSize() const;
    ActivationType getActivationFunction() const;


    void setDeltas(const std::vector<std::vector<double>>& deltas);
    std::vector<std::vector<double>> getDeltas() const;
    std::vector<double> getOutput() const; // To access output of a layer

private:
    size_t numInputs_;
    size_t numOutputs_;
    std::vector<std::vector<double>> weights_;
    std::vector<double> biases_;
    ActivationType activationType_; // Store the activation type

    mutable std::vector<double> output_;       // mutable для изменения в const методах
    mutable bool outputCalculated_ = false;  // mutable для изменения в const методах



    std::function<double(double)> activationFunction_;

    void initializeWeights();
    static double relu(double x);
    static double sigmoid(double x);
    static double tanh(double x);
    static double linear(double x);


};

#endif // LAYER_H