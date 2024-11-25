// data_normalization.cpp

#include "data_normalization.h"
#include <cmath>
#include <numeric>



DataNormalization::DataNormalization(NormalizationType type) : type_(type) {}



void DataNormalization::normalizeBarData(DataStorage& dataStorage)
{
    if(type_ == NormalizationType::ZScore && std_ == 0.0)
    {
        calculateMeanStd(dataStorage.getBarData());
    }

    std::vector<BarData> normalizedBarData = normalizeBarData(dataStorage.getBarData());
    dataStorage.clear(); //Clear existing data
    for(const auto& bar : normalizedBarData)
    {
        dataStorage.addBarData(bar);
    }


}




std::vector<BarData> DataNormalization::normalizeBarData(const std::vector<BarData>& barData) {
    switch (type_) {
        case NormalizationType::MinMax:
            return normalizeMinMax(barData);
        case NormalizationType::ZScore:
            if (std_ == 0.0) {
                 calculateMeanStd(barData);
            }

            return normalizeZScore(barData);
        default:
            throw std::runtime_error("Unknown normalization type."); // Or return a copy of the input data
    }
}


void DataNormalization::setNormalizationType(NormalizationType type) {
    type_ = type;
}

void DataNormalization::setMinMaxRange(double min, double max) {
    minRange_ = min;
    maxRange_ = max;
}

void DataNormalization::calculateMeanStd(const std::vector<BarData>& barData) {

    if(barData.empty())
    {
        return; // or throw an exception
    }
    double sum = 0.0;
    for (const auto& bar : barData) {
        sum += bar.close; // Or calculate based on another field or combination of fields
    }

    mean_ = sum / barData.size();
    double sq_sum = 0.0;
    for (const auto& bar : barData) {
        sq_sum += std::pow(bar.close - mean_, 2);
    }

    std_ = std::sqrt(sq_sum / barData.size());

}


void DataNormalization::setMeanStd(double mean, double std)
{
    mean_ = mean;
    std_ = std;
}

std::vector<BarData> DataNormalization::normalizeMinMax(const std::vector<BarData>& barData) const {

    std::vector<BarData> normalizedData;
    normalizedData.reserve(barData.size());


    for(const auto& bar : barData)
    {
        normalizedData.push_back(normalizeMinMax(bar));
    }
    return normalizedData;

}

BarData DataNormalization::normalizeMinMax(const BarData& bar) const
{
       double minVal = std::min({bar.open, bar.close, bar.high, bar.low});
       double maxVal = std::max({bar.open, bar.close, bar.high, bar.low});

        if(minVal == maxVal)
        {
            //Handle the edge case to avoid division by zero.  Return the minRange
             return  BarData(minRange_, minRange_, minRange_, minRange_);

        }

        BarData normalizedBar;
        normalizedBar.open = minRange_ + (bar.open - minVal) * (maxRange_ - minRange_) / (maxVal - minVal);
        normalizedBar.close = minRange_ + (bar.close - minVal) * (maxRange_ - minRange_) / (maxVal - minVal);
        normalizedBar.high = minRange_ + (bar.high - minVal) * (maxRange_ - minRange_) / (maxVal - minVal);
        normalizedBar.low = minRange_ + (bar.low - minVal) * (maxRange_ - minRange_) / (maxVal - minVal);

        return normalizedBar;



}




std::vector<BarData> DataNormalization::normalizeZScore(const std::vector<BarData>& barData) const {
    std::vector<BarData> normalizedData;
    normalizedData.reserve(barData.size());

    for(const auto& bar : barData)
    {
         normalizedData.push_back(normalizeZScore(bar));
    }
    return normalizedData;

}

BarData DataNormalization::normalizeZScore(const BarData& bar) const
{
        if (std_ == 0.0) {
            // Handle the case where standard deviation is zero to avoid division by zero
            return bar; // Or throw an exception, or return a specific value
        }


        BarData normalizedBar;
        normalizedBar.open = (bar.open - mean_) / std_;
        normalizedBar.close = (bar.close - mean_) / std_;
        normalizedBar.high = (bar.high - mean_) / std_;
        normalizedBar.low = (bar.low - mean_) / std_;

        return normalizedBar;
}