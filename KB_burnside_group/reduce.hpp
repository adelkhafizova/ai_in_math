#pragma once
#include <vector>
#include <utility>
#include "types.hpp"

Word reduce(Word word, const Rules& rules, int skip_rule = -1, bool print_progress = false);