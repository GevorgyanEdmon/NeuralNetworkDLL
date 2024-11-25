// neural_network.cpp
#include "neural_network.h"
#include <cmath>
#include <stdexcept>
#include <fstream> 
#include <sstream> 
#include <iostream>



// Конструктор с параметрами для установки количества входов и выходов
NeuralNetwork::NeuralNetwork(size_t numInputs, size_t numOutputs) : numInputs_(numInputs), numOutputs_(numOutputs) {}


void NeuralNetwork::addLayer(size_t numOutputs, Layer::ActivationType activationType) {
    size_t numInputs = layers_.empty() ? numInputs_ : layers_.back().getOutputSize();
    layers_.emplace_back(numInputs, numOutputs, activationType);
    numOutputs_ = numOutputs;  // Update the network's output size
}



void NeuralNetwork::addLayer(const Layer& layer) {
    if (!layers_.empty() && layers_.back().getOutputSize() != layer.getInputSize()) {
        throw std::invalid_argument("Number of inputs in new layer must match the number of outputs in the previous layer.");
    }
    layers_.push_back(layer);
    if (layers_.size() == 1) {
        numInputs_ = layer.getInputSize();
    }
    numOutputs_ = layer.getOutputSize();  // Update the network's output size
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) const {
    if (layers_.empty()) {
        throw std::runtime_error("Neural network is empty. Add layers before predicting.");
    }
    if (input.size() != numInputs_) {
        throw std::invalid_argument("Input size mismatch.");
    }

    std::vector<double> output = input;
    for (const auto& layer : layers_) {
        output = layer.forward(output);
    }
    return output;
}


void NeuralNetwork::train(const DataStorage& trainingData, size_t epochs, double learningRate) {
    if (layers_.empty()) {
        throw std::runtime_error("Neural network is empty. Add layers before training.");
    }

    if (trainingData.getBarDataSize() == 0) {
        throw std::runtime_error("Training data is empty. Provide data before training.");
    }

    if (numInputs_ != 4) {
        throw std::runtime_error("Input size must be 4 (OHLC) for this training.");
    }

    if (numOutputs_ != 1) {
        throw std::runtime_error("Output size must be 1 for this training data");
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < trainingData.getBarDataSize(); ++i) {
            BarData bar = trainingData.getBarData(i);
            std::vector<double> input = {bar.open, bar.close, bar.high, bar.low};
            std::vector<double> target = {bar.close};

            std::vector<double> output = predict(input);

            backpropagate(target, output);
            updateWeights(learningRate, input);
        }
    }
}}



void NeuralNetwork::saveModel(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to save model.");
    }

    file << numInputs_ << " " << numOutputs_ << "\n";

    for (const auto& layer : layers_) {
        file << layer.getInputSize() << " " << layer.getOutputSize() << " " << static_cast<int>(layer.getActivationFunction()) << "\n";

        const auto& weights = layer.getWeights();
        for (const auto& row : weights) {
            for (double weight : row) {
                file << weight << " ";
            }
            file << "\n";
        }

        const auto& biases = layer.getBiases();
        for (double bias : biases) {
            file << bias << " ";
        }
        file << "\n";
    }

    file.close();
}




void NeuralNetwork::loadModel(const std::string& filename) {
 std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to load model.");
    }

    layers_.clear(); 

    file >> numInputs_ >> numOutputs_;

    size_t numInputs, numOutputs;
    int activationTypeInt;
    while (file >> numInputs >> numOutputs >> activationTypeInt) {
        Layer::ActivationType activationType = static_cast<Layer::ActivationType>(activationTypeInt);
        Layer layer(numInputs, numOutputs, activationType);

        std::vector<std::vector<double>> weights(numOutputs, std::vector<double>(numInputs));
        for (size_t i = 0; i < numOutputs; ++i) {
            for (size_t j = 0; j < numInputs; ++j) {
                file >> weights[i][j];
            }
        }
        layer.setWeights(weights);

        std::vector<double> biases(numOutputs);
        for (size_t i = 0; i < numOutputs; ++i) {
            file >> biases[i];
        }
        layer.setBiases(biases);


        addLayer(layer); 

    }

    file.close();
}


std::vector<Layer>& NeuralNetwork::getLayers() {
    return layers_;
}

size_t NeuralNetwork::getNumInputs() const {
    return numInputs_;
}

size_t NeuralNetwork::getNumOutputs() const {
    return numOutputs_;
}


std::vector<std::vector<double>> NeuralNetwork::calculateDeltas(const std::vector<double>& target, const std::vector<double>& output, const Layer& layer) const {
    std::vector<std::vector<double>> deltas(1, std::vector<double>(layer.getOutputSize()));
    for (size_t i = 0; i < layer.getOutputSize(); ++i) {
        double error = target[i] - output[i];
        deltas[0][i] = error * activationDerivative(output[i], layer.getActivationFunction());
    }
    return deltas;
}


void NeuralNetwork::backpropagate(const std::vector<double>& target, const std::vector<double>& output) {
     if (layers_.empty()) {
        throw std::runtime_error("Cannot backpropagate on an empty network.");
    }

    if (target.size() != layers_.back().getOutputSize()) {
        throw std::invalid_argument("Target size mismatch with output layer size.");
    }



    std::vector<std::vector<double>> layerInputs;
     layerInputs.push_back(input); // Store initial input


    for (size_t i = 0; i < layers_.size() -1; ++i)
    {
         layerInputs.push_back(layers_[i].forward(layerInputs.back()));

    }



    std::vector<std::vector<std::vector<double>>> deltas;
    deltas.push_back(calculateDeltas(target, layers_.back().forward(layerInputs.back()), layers_.back()));  //Use correct input for last layer



    for (size_t i = layers_.size() - 2; i >= 0; --i) {
        std::vector<double> nextLayerWeightedSum(layers_[i].getOutputSize(), 0.0);
        for (size_t j = 0; j < layers_[i + 1].getOutputSize(); ++j) {
            for (size_t k = 0; k < layers_[i].getOutputSize(); ++k) {
                nextLayerWeightedSum[k] += deltas.back()[0][j] * layers_[i+1].getWeights()[j][k];
            }
        }

        //Correctly use already calculated layer output
         deltas.insert(deltas.begin(), calculateDeltas(nextLayerWeightedSum, layerInputs[i+1], layers_[i]));

    }


    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].setDeltas(deltas[i]);
    }
}



void NeuralNetwork::updateWeights(double learningRate, const std::vector<double>& input) {

    std::vector<double> layerInput = input;

    for (size_t i = 0; i < layers_.size(); ++i) {
        Layer& layer = layers_[i];
        std::vector<std::vector<double>>& weights = layer.getWeights();
        std::vector<double>& biases = layer.getBiases();
       
        std::vector<std::vector<double>> deltas = layer.getDeltas();

        for (size_t j = 0; j < layer.getOutputSize(); ++j) {
            for (size_t k = 0; k < layer.getInputSize(); ++k) {
                weights[j][k] += learningRate * deltas[0][j] * layerInput[k]; 
            }
            biases[j] += learningRate * deltas[0][j];
        }

        layerInput = layer.forward(layerInput); // Recalculate output for next layer
    }
}


double NeuralNetwork::activationFunction(double x, Layer::ActivationType type) {
    switch (type) {
    case Layer::ActivationType::ReLU:
        return relu(x);
    case Layer::ActivationType::Sigmoid:
        return sigmoid(x);
    case Layer::ActivationType::Tanh:
        return tanh(x);
    case Layer::ActivationType::Linear:
        return x;
    case Layer::ActivationType::None:
        return x;
    default:
        throw std::invalid_argument("Unknown activation type");

    }
}



double NeuralNetwork::activationDerivative(double x, Layer::ActivationType type) {
     switch (type) {
    case Layer::ActivationType::ReLU:
        return reluDerivative(x);
    case Layer::ActivationType::Sigmoid:
        return sigmoidDerivative(x);
    case Layer::ActivationType::Tanh:
        return 1.0 - std::pow(tanh(x), 2); 
    case Layer::ActivationType::Linear:
        return 1.0;
    case Layer::ActivationType::None:
        return 1.0;
    default:
        throw std::invalid_argument("Unknown activation type");
    }
}


double NeuralNetwork::relu(double x) {
    return std::max(0.0, x);
}

double NeuralNetwork::reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

double NeuralNetwork::tanh(double x) {
    return std::tanh(x);
}