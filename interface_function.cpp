#include "interface_function.h"
#include <iostream>



InterfaceFunction::InterfaceFunction(NeuralNetwork& neuralNetwork, DataNormalization::NormalizationType normalizationType) :
    neuralNetwork_(neuralNetwork), dataNormalization_(normalizationType) {}

std::vector<double> InterfaceFunction::processData(const std::vector<BarData>& barData, const std::map<std::string, std::vector<double>>& indicatorData, bool useIndicators) {

    DataStorage dataStorage;
    for (const auto& bar : barData) {
        dataStorage.addBarData(bar);
    }
    
    if(useIndicators) {
       for(const auto& pair : indicatorData) {
         dataStorage.addIndicatorData(pair.first, pair.second);
       }
    }



    if (isTraining_) {

        dataNormalization_.normalizeBarData(dataStorage);
        trainNetwork(dataStorage);

        return {};

    } else {

        std::vector<double> result;
        for(size_t i = 0; i < dataStorage.getBarDataSize(); ++i) {
            DataStorage singleBarStorage;
             singleBarStorage.addBarData(dataStorage.getBarData(i));

             if(useIndicators) {
               for(const auto& pair : indicatorData) {
                   std::vector<double> indicatorValuesForBar;

                    if (i < pair.second.size()) {
                        indicatorValuesForBar.push_back(pair.second[i]);
                    } else {

                         indicatorValuesForBar.push_back(0.0); 
                    }

                     singleBarStorage.addIndicatorData(pair.first, indicatorValuesForBar);

                 }

             }


        dataNormalization_.normalizeBarData(singleBarStorage);
        std::vector<double> inputVector = createInputVector(singleBarStorage.getBarData(0), singleBarStorage.getAllIndicatorData(), useIndicators);

        if (inputVector.size() != neuralNetwork_.getNumInputs()) {
                throw std::runtime_error("Input vector size mismatch.");
            }
             result.push_back(neuralNetwork_.predict(inputVector)[0]);

        }
        return result;



    }
}

void InterfaceFunction::setTrainingMode(bool isTraining) {
    isTraining_ = isTraining;
    neuralNetwork_.setTrainingMode(isTraining);
}

DataNormalization& InterfaceFunction::getDataNormalization() {
    return dataNormalization_;
}



std::vector<double> InterfaceFunction::createInputVector(const BarData& barData, const std::map<std::string, std::vector<double>>& indicatorData, bool useIndicators) const{
    std::vector<double> inputVector = {barData.open, barData.close, barData.high, barData.low};

    if(useIndicators){
        for(const auto& pair: indicatorData){

           if (pair.second.size() > 0) {
            inputVector.push_back(pair.second[0]);
            } else {

                throw std::runtime_error("Indicator data is missing");
                }
        }
    }

    return inputVector;
}

void InterfaceFunction::trainNetwork(const DataStorage& dataStorage) {
    try{
         if (neuralNetwork_.getNumInputs() != 4 && !dataStorage.getAllIndicatorData().empty()) {
            throw std::runtime_error("Input size of neural network and input vector must match.");
         }
    neuralNetwork_.train(dataStorage, 1, 0.1);


    }
      catch (const std::exception& e) {
        std::cerr << "Error training network: " << e.what() << std::endl;
       
    }
}