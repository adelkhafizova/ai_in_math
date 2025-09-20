#pragma once
#include <vector>
#include <utility>
#include "types.hpp"

const Word& rule_left(const Rule& r);
const Word& rule_right(const Rule& r);
CritPairs& make_crit_pairs(const Rules& rules,
                            CritPairs& crit_pairs,
                            bool reduce_immediately = true,
                            std::size_t start_at_rule = 0,
                            std::size_t max_crit_length = 0,
                            bool reset_pairs = true,
                            bool print_progress = false,
                            bool print_progress_pct = false);

bool check_confluence(const Rules&          rules,
                      CritPairs*           crit_pairs = nullptr,   // ← optional
                      std::size_t          max_crit_length = 0,
                      bool                 erase_pair_list = true,
                      bool                 print_progress = false);