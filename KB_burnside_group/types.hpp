#pragma once
#include <vector>
#include <utility>

using Word = std::vector<int>;
using Rule = std::pair<Word, Word>;
using Rules = std::vector<Rule>;
using CritPair  = std::pair<Word, Word>;
using CritPairs = std::vector<CritPair>;