#include <vector>
#include <utility>
#include <cstddef>
#include <algorithm>
#include <optional>
#include "crit_pairs.hpp"
#include "reduce.hpp"
#include "types.hpp"

// -----------------------------------------------------------------------------
/// @brief  Checks whether rule *rule_number* is redundant.
///
/// A rule is redundant when its left‑hand side and right‑hand side reduce to
/// the same word when **all other** rules are available for rewriting.
///
/// @param rules        the whole system (read‑only)
/// @param rule_number  index of the rule to test
/// @return true  iff the rule is redundant
/// @return false otherwise
bool is_rule_redundant ( const Rules& rules, std::size_t rule_number )
{
    const Word& lhs = rule_left ( rules[rule_number] );
    const Word& rhs = rule_right( rules[rule_number] );

    // Reduce while *skipping* the rule we are testing
    Word lhs_red = reduce( lhs, rules, rule_number );
    Word rhs_red = reduce( rhs, rules, rule_number );

    return lhs_red == rhs_red;
}

/// @brief  Removes all redundant rules from *rules*.
///
/// The function can walk the vector from the beginning or from the end
/// (controlled by *go_from_the_end*).
///   – when a rule is erased we do **not** advance the loop counter,
///   – the variable *new_first_rule* is adjusted whenever a rule that lies
///     before the current “first new rule” is deleted.
///
/// @param rules            the rewriting system (modified in‑place)
/// @param start_at_rule    the index of the first rule that will be used later
///                         when critical pairs are created (default = 0)
/// @param go_from_the_end  if true the vector is scanned from the back,
///                         otherwise from the front (default = true)
/// @return a pair *(redundancies_were_present, new_first_rule)*
///         – *redundancies_were_present* is true iff at least one rule was
///           eliminated,
///         – *new_first_rule* is the possibly‑shifted index to be returned to
///           the caller.
std::pair<bool, std::size_t>
eliminate_redundancy (Rules& rules, std::size_t  start_at_rule, bool go_from_the_end)
{
    bool        redundancies_were_present = false;
    std::size_t i                         = 0;
    std::size_t new_first_rule            = start_at_rule;

    if ( go_from_the_end )
    {
        while ( i < rules.size() )
        {
            std::size_t idx = rules.size() - 1 - i;

            if ( is_rule_redundant( rules, idx ) )
            {
                redundancies_were_present = true;

                if ( idx < new_first_rule )
                    --new_first_rule;

                rules.erase( rules.begin() + static_cast<std::ptrdiff_t>( idx ) );
            }
            else
                ++i;
        }
    }
    else
    {
        while ( i < rules.size() )
        {
            if ( is_rule_redundant( rules, i ) )
            {
                redundancies_were_present = true;

                rules.erase( rules.begin() + static_cast<std::ptrdiff_t>( i ) );

                if ( i < new_first_rule )
                    --new_first_rule;
            }
            else
                ++i;
        }
    }

    return { redundancies_were_present, new_first_rule };
}
