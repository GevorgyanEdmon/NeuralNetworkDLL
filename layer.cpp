// layer.cpp
#include "layer.h"
#include <cmath>
#include <limits> 
#include <stdexcept>
#include <random>

Layer::Layer(size_t numInputs, size_t numOutputs, ActivationType activationType) : 
    numInputs_(numInputs), numOutputs_(numOutputs), activationType_(activationType), activationFunction_(relu)
{
    if (numInputs == 0 || numOutputs == 0) {
        throw std::invalid_argument("Number of inputs and outputs must be greater than zero.");
    }

    weights_.resize(numOutputs_, std::vector<double>(numInputs_));
    biases_.resize(numOutputs_);

    initializeWeights();
    setActivationFunction(activationType);
}

void Layer::setActivationFunction(ActivationType activationType) {
    activationType_ = activationType; 
    switch (activationType) {
        case ActivationType::ReLU:
            activationFunction_ = relu;
            break;
        case ActivationType::Sigmoid:
            activationFunction_ = sigmoid;
            break;
        case ActivationType::Tanh:
            activationFunction_ = tanh;
            break;
        case ActivationType::Linear:
            activationFunction_ = linear;
            break;
        case ActivationType::None:
            activationFunction_ = nullptr; 
            break;
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) const {  // const  
    if (input.size() != numInputs_) {
        throw std::invalid_argument("Input size mismatch in Layer::forward()");
    }

    outputCalculated_ = true;

    std::vector<double> output(numOutputs_);
    for (size_t i = 0; i < numOutputs_; ++i) {
        output[i] = 0.0;
        for (size_t j = 0; j < numInputs_; ++j) {
            output[i] += input[j] * weights_[i][j];
        }
        output[i] += biases_[i];
        if (activationFunction_) {
            output[i] = activationFunction_(output[i]);
        }
    }

    output_ = output;
    outputCalculated_ = true; // Always set to true after successful calculation
    return output;
}

void Layer::setWeights(const std::vector<std::vector<double>>& weights) {
    if (weights.size() != numOutputs_ || weights[0].size() != numInputs_) {
        throw std::invalid_argument("Weight matrix dimensions mismatch in Layer::setWeights()");
    }
    weights_ = weights;
}

std::vector<std::vector<double>> Layer::getWeights() const {
    return weights_;
}

void Layer::setBiases(const std::vector<double>& biases) {
    if (biases.size() != numOutputs_) {
        throw std::invalid_argument("Bias vector size mismatch in Layer::setBiases()");
    }
    biases_ = biases;
}

std::vector<double> Layer::getBiases() const {
    return biases_;
}

size_t Layer::getInputSize() const {
    return numInputs_;
}

size_t Layer::getOutputSize() const {
    return numOutputs_;
}

Layer::ActivationType Layer::getActivationFunction() const {
    return activationType_;
}

void Layer::setDeltas(const std::vector<std::vector<double>>& deltas) {
    deltas_ = deltas;
}

std::vector<std::vector<double>> Layer::getDeltas() const {
    return deltas_;
}

std::vector<double> Layer::getOutput() const {
    if (!outputCalculated_) {
        throw std::runtime_error("Output not calculated yet. Call forward() first.");
    }
    return output_;
}

void Layer::initializeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0 / std::sqrt(numInputs_));

    for (size_t i = 0; i < numOutputs_; ++i) {
        for (size_t j = 0; j < numInputs_; ++j) {
            weights_[i][j] = distribution(gen);
        }
        biases_[i] = 0.0; 
    }
}

double Layer::relu(double x) {
    return std::max(0.0, x);
}

double Layer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Layer::tanh(double x) {
    return std::tanh(x);
}

double Layer::linear(double x) {
    return x;
}