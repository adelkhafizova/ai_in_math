#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <cassert>
#include "types.hpp"

int knuth_bendix(
    Rules& rules,
    const std::function<int(const Word&, const Word&)>& order = shortlex_default,
    bool reduce_crit_immediately = true,
    int max_rounds = -1,
    bool recheck_old_rules = false,
    bool print_progress = false,
    bool print_crit_pairs_progress = false);