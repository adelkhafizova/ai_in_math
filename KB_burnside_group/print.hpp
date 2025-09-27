#pragma once
#include <iostream>
#include <vector>
#include <utility>

inline std::ostream& operator<<(std::ostream& os, const std::vector<std::pair<std::vector<int>, std::vector<int>>>& data) {
    os << "[";
    for (size_t i = 0; i < data.size(); ++i) {
        os << "{";
        os << "[";
        for (size_t j = 0; j < data[i].first.size(); ++j) {
            os << data[i].first[j];
            if (j + 1 < data[i].first.size()) os << ",";
        }
        os << "]";
        os << ",";
        os << "[";
        for (size_t j = 0; j < data[i].second.size(); ++j) {
            os << data[i].second[j];
            if (j + 1 < data[i].second.size()) os << ",";
        }
        os << "]";
        os << "}";
        if (i + 1 < data.size()) os << ", ";
    }
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<int>& data) {
    os << "[";
    for (size_t i = 0; i < data.size(); ++i) {
        os << data[i];
        if (i + 1 < data.size()) os << ",";
    }
    os << "]";
    return os;
}