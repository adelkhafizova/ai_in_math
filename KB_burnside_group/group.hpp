#pragma once
#include <vector>
#include <utility>

void symmetrize_group_rule(std::vector<std::pair<std::vector<int>, std::vector<int>>>& rules,
                             std::size_t i,
                             const std::function<int(const std::vector<int>&, const std::vector<int>&)>& order = nullptr);

void add_free_group_rules(std::vector<std::pair<std::vector<int>, std::vector<int>>>& rules);