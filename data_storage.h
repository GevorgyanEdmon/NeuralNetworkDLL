// data_storage.h

#ifndef DATA_STORAGE_H
#define DATA_STORAGE_H

#include <vector>
#include <stdexcept>
#include <string>
#include <map>

class BarData {
public:
    double open;
    double close;
    double high;
    double low;

    BarData(double open = 0.0, double close = 0.0, double high = 0.0, double low = 0.0) : 
        open(open), close(close), high(high), low(low) {}
};

class DataStorage {
public:
    DataStorage() = default;

    void addBarData(const BarData& bar);
    void addBarData(double open, double close, double high, double low);
    
    std::vector<BarData> getBarData() const;
    BarData getBarData(size_t index) const;
    size_t getBarDataSize() const;
    
    void clear();

    // Методы для работы с данными индикаторов
    void addIndicatorData(const std::string& indicatorName, const std::vector<double>& indicatorData);
    std::vector<double> getIndicatorData(const std::string& indicatorName) const;
    std::map<std::string, std::vector<double>> getAllIndicatorData() const;
    bool hasIndicator(const std::string& indicatorName) const;
    void removeIndicator(const std::string& indicatorName);
    size_t getIndicatorCount() const;




private:
    std::vector<BarData> barData_;
    std::map<std::string, std::vector<double>> indicatorData_; //  Хранение данных индикаторов,  ключ - имя индикатора
};

#endif // DATA_STORAGE_H


// data_storage.cpp

#include "data_storage.h"


void DataStorage::addBarData(const BarData& bar) {
    barData_.push_back(bar);
}

void DataStorage::addBarData(double open, double close, double high, double low) {
    barData_.emplace_back(open, close, high, low);
}

std::vector<BarData> DataStorage::getBarData() const {
    return barData_;
}

BarData DataStorage::getBarData(size_t index) const {
    if (index >= barData_.size()) {
        throw std::out_of_range("Index out of range in getBarData");
    }
    return barData_[index];
}

size_t DataStorage::getBarDataSize() const {
    return barData_.size();
}


void DataStorage::clear() {
    barData_.clear();
    indicatorData_.clear();
}

void DataStorage::addIndicatorData(const std::string& indicatorName, const std::vector<double>& indicatorData) {
    indicatorData_[indicatorName] = indicatorData;
}

std::vector<double> DataStorage::getIndicatorData(const std::string& indicatorName) const {
    auto it = indicatorData_.find(indicatorName);
    if (it == indicatorData_.end()) {
        throw std::invalid_argument("Indicator not found: " + indicatorName);
    }
    return it->second;
}


std::map<std::string, std::vector<double>> DataStorage::getAllIndicatorData() const {
    return indicatorData_;
}

bool DataStorage::hasIndicator(const std::string& indicatorName) const {
    return indicatorData_.count(indicatorName) > 0;
}

void DataStorage::removeIndicator(const std::string& indicatorName) {
    indicatorData_.erase(indicatorName);
}

size_t DataStorage::getIndicatorCount() const {
    return indicatorData_.size();
}