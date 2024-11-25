#include "neural_network.h"
#include <cmath>
#include <stdexcept>
#include <fstream> 
#include <sstream> 
#include <iostream>
#include <random>



NeuralNetwork::NeuralNetwork(size_t numInputs, size_t numOutputs) : numInputs_(numInputs), numOutputs_(numOutputs) {}

void NeuralNetwork::addLayer(size_t numOutputs, Layer::ActivationType activationType) {
    size_t numInputs = layers_.empty() ? numInputs_ : layers_.back().getOutputSize();
    layers_.emplace_back(numInputs, numOutputs, activationType);
    numOutputs_ = numOutputs;

    if (!previousWeightUpdates_.empty()) {
        previousWeightUpdates_.resize(layers_.size());
        previousWeightUpdates_.back().resize(layers_.back().getOutputSize(), std::vector<double>(layers_.back().getInputSize(), 0.0));
    }

    if (!previousBiasUpdates_.empty()) {
        previousBiasUpdates_.resize(layers_.size());
        previousBiasUpdates_.back().resize(layers_.back().getOutputSize(), 0.0);
    }
}

void NeuralNetwork::addLayer(const Layer& layer) {
    if (!layers_.empty() && layers_.back().getOutputSize() != layer.getInputSize()) {
        throw std::invalid_argument("Number of inputs in new layer must match the number of outputs in the previous layer.");
    }
    layers_.push_back(layer);
    if (layers_.size() == 1) {
        numInputs_ = layer.getInputSize();
    }
    numOutputs_ = layer.getOutputSize();

     if (!previousWeightUpdates_.empty())
    {
        previousWeightUpdates_.resize(layers_.size());
         previousWeightUpdates_.back().resize(layers_.back().getOutputSize(), std::vector<double>(layers_.back().getInputSize(), 0.0));

    }

    if (!previousBiasUpdates_.empty())
    {

         previousBiasUpdates_.resize(layers_.size());
         previousBiasUpdates_.back().resize(layers_.back().getOutputSize(), 0.0);

    }

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


    if (previousWeightUpdates_.empty()) {
        previousWeightUpdates_.resize(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            previousWeightUpdates_[i].resize(layers_[i].getOutputSize(), std::vector<double>(layers_[i].getInputSize(), 0.0));
        }
    }

    if (previousBiasUpdates_.empty()) {
        previousBiasUpdates_.resize(layers_.size());
        for (size_t i = 0; i < layers_.size(); ++i) {
            previousBiasUpdates_[i].resize(layers_[i].getOutputSize(), 0.0);
        }

    }


    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < trainingData.getBarDataSize(); ++i) {
            BarData bar = trainingData.getBarData(i);
            std::vector<double> input = {bar.open, bar.close, bar.high, bar.low};
            std::vector<double> target = {bar.close};

            std::vector<double> output = predict(input);
            backpropagate(target, output, input); // Pass input to backpropagate
            updateWeights(learningRate, input);
        }
    }
}


void NeuralNetwork::saveModel(std::ostream& file) const {
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
}

void NeuralNetwork::loadModel(std::istream& file) {
    layers_.clear();
    previousWeightUpdates_.clear();
    previousBiasUpdates_.clear();


    size_t numInputs, numOutputs;
    file >> numInputs >> numOutputs;
    numInputs_ = numInputs;
    numOutputs_ = numOutputs;

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

void NeuralNetwork::backpropagate(const std::vector<double>& target, const std::vector<double>& output, const std::vector<double>& input) { // Added input parameter

    if (layers_.empty()) {
        throw std::runtime_error("Cannot backpropagate on an empty network.");
    }

    if (target.size() != layers_.back().getOutputSize()) {
        throw std::invalid_argument("Target size mismatch with output layer size.");
    }



 std::vector<std::vector<std::vector<double>>> deltas;
    deltas.push_back(calculateDeltas(target, output, layers_.back()));  



    for (size_t i = layers_.size() - 2; i >= 0; --i) {
        std::vector<double> nextLayerWeightedSum(layers_[i].getOutputSize(), 0.0);
        for (size_t j = 0; j < layers_[i + 1].getOutputSize(); ++j) {
            for (size_t k = 0; k < layers_[i].getOutputSize(); ++k) {
                nextLayerWeightedSum[k] += deltas.back()[0][j] * layers_[i + 1].getWeights()[j][k];
            }
        }


        std::vector<double> prevLayerOutput = (i == 0) ? input : layers_[i - 1].getOutput();
        std::vector<double> currentLayerOutput = layers_[i].forward(prevLayerOutput);



        deltas.insert(deltas.begin(), calculateDeltas(nextLayerWeightedSum, currentLayerOutput, layers_[i]));




    }


    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].setDeltas(deltas[i]);
    }
}


void NeuralNetwork::updateWeights(double learningRate, const std::vector<double>& input) {
    std::vector<double> layerInput = input;

    if (previousWeightUpdates_.empty()) {
       previousWeightUpdates_.resize(layers_.size());
       for(size_t i = 0; i < layers_.size(); ++i)
       {
           previousWeightUpdates_[i].resize(layers_[i].getOutputSize(), std::vector<double>(layers_[i].getInputSize(), 0.0));
       }

    }

    if(previousBiasUpdates_.empty())
    {
         previousBiasUpdates_.resize(layers_.size());

        for(size_t i = 0; i < layers_.size(); ++i)
        {
            previousBiasUpdates_[i].resize(layers_[i].getOutputSize(), 0.0);

        }
    }




    for (size_t i = 0; i < layers_.size(); ++i) {
        Layer& layer = layers_[i];
        std::vector<std::vector<double>>& weights = layer.getWeights();
        std::vector<double>& biases = layer.getBiases();
        std::vector<std::vector<double>> deltas = layer.getDeltas();

        for (size_t j = 0; j < layer.getOutputSize(); ++j) {
            for (size_t k = 0; k < layer.getInputSize(); ++k) {
                 double weightUpdate = learningRate * deltas[0][j] * layerInput[k] + momentum_ * previousWeightUpdates_[i][j][k];
                 weights[j][k] += weightUpdate;
                 previousWeightUpdates_[i][j][k] = weightUpdate;
            }


            double biasUpdate = learningRate * deltas[0][j] + momentum_ * previousBiasUpdates_[i][j];

            biases[j] += biasUpdate;
            previousBiasUpdates_[i][j] = biasUpdate;


        }

        layerInput = layer.forward(layerInput); 
    }
}


void NeuralNetwork::setTrainingMode(bool isTraining) {

    (void)isTraining;
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