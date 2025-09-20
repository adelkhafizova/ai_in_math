#include <vector>
#include <utility>
#include <cstddef>
#include <algorithm>
#include "reduce.hpp"
#include "crit_pairs.hpp"
#include "types.hpp"

/**
 * @brief Reduces the right‑hand side of rule *rule_number* using the whole
 *        system *rules*.
 *
 * @param rules        the rewriting system (modified in‑place)
 * @param rule_number  index of the rule to try to shortcut
 *
 * @return true  if the RHS was changed (a shortcut was made)
 * @return false otherwise
 */
bool shortcut(Rules& rules, std::size_t rule_number)
{
    const Word& lhs = rule_left (rules[rule_number]);
    const Word& rhs = rule_right(rules[rule_number]);

    Word reduced_rhs = reduce(rhs, rules);

    if (reduced_rhs != rhs) {
        rules[rule_number].second = std::move(reduced_rhs);
        return true;
    }
    return false;
}

/**
 * @brief Reduces the RHS of **all** rules once.
 *
 * @param rules  the rewriting system (modified in‑place)
 *
 * @return true  if at least one rule was shortened
 * @return false otherwise
 */
bool make_all_shortcuts(Rules& rules)
{
    bool changes_made = false;
    for (std::size_t i = 0; i < rules.size(); ++i) {
        changes_made = shortcut(rules, i) || changes_made;
    }
    return changes_made;
}
