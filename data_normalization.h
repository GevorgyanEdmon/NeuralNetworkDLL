// data_normalization.h

#ifndef DATA_NORMALIZATION_H
#define DATA_NORMALIZATION_H

#include <vector>
#include <stdexcept>
#include <algorithm> // Make sure to include this for min/max operations
#include "data_storage.h" // Include your data storage header


class DataNormalization {
public:
    enum class NormalizationType {
        MinMax,
        ZScore
    };

    DataNormalization(NormalizationType type = NormalizationType::MinMax);
    
    void normalizeBarData(DataStorage& dataStorage); // Modifies the DataStorage object directly
    std::vector<BarData> normalizeBarData(const std::vector<BarData>& barData);


    void setNormalizationType(NormalizationType type);

    // MinMax specific
    void setMinMaxRange(double min, double max);

    // ZScore specific
    void calculateMeanStd(const std::vector<BarData>& barData);
    void setMeanStd(double mean, double std);


private:
    NormalizationType type_;

    // MinMax
    double minRange_ = 0.0;
    double maxRange_ = 1.0;

    //ZScore
    double mean_ = 0.0;
    double std_ = 1.0;


    std::vector<BarData> normalizeMinMax(const std::vector<BarData>& barData) const;
    std::vector<BarData> normalizeZScore(const std::vector<BarData>& barData) const;
    BarData normalizeMinMax(const BarData& bar) const;
    BarData normalizeZScore(const BarData& bar) const;



};

#endif // DATA_NORMALIZATION_H