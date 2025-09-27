#include "expWords.hpp"
#include <vector>

std::vector<int> triple_word(const std::vector<int>& w)
{
    std::vector<int> res;
    res.reserve(w.size() * 3);
    res.insert(res.end(), w.begin(), w.end());
    res.insert(res.end(), w.begin(), w.end());
    res.insert(res.end(), w.begin(), w.end());
    return res;
}

std::vector<int> quadruple_word(const std::vector<int>& w)
{
    std::vector<int> res;
    res.reserve(w.size() * 4);
    res.insert(res.end(), w.begin(), w.end());
    res.insert(res.end(), w.begin(), w.end());
    res.insert(res.end(), w.begin(), w.end());
    res.insert(res.end(), w.begin(), w.end());
    return res;
}