#pragma once
#include <vector>
#include <utility>
#include "types.hpp"

std::pair<bool, std::size_t> eliminate_redundancy(Rules& rules, std::size_t start_at_rule = 0, bool go_from_the_end = true);
