#pragma once
#include <utility>
#include <vector>
#include <functional>
#include "ordering.hpp"
#include "types.hpp"

bool resolve_all_crit_pairs(std::vector<Rule>&               rules,
                            CritPairs&                           crit_pairs,
                            const std::function<int(const Word&, const Word&)>& order = shortlex_default,
                            bool                             reduce_crit_immediately = true,
                            bool                             go_from_the_end = true,
                            bool                             print_progress = false,
                            bool                             print_crit_pairs_progress = false);