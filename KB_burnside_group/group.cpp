#include <vector>
#include <utility>
#include <algorithm>
#include <functional>
#include <cassert>
#include "crit_pairs.hpp"
#include "types.hpp"

//  Returns the word obtained by negating every generator and then reversing
inline Word group_inverse(const Word& w)
{
    Word inv;
    inv.reserve(w.size());
    // negate while we copy
    for (auto it = w.rbegin(); it != w.rend(); ++it)
        inv.push_back(-(*it));
    return inv;                // e.g. [a,b] → [-b,-a]
}

//  For a rule *i* (u → v) we create the word u · v⁻¹ and then add
//  all cyclic shifts of that word and of its inverse as new rules.
void symmetrize_group_rule(Rules&                       rules,
                             std::size_t                  i,
                             const std::function<int(const Word&, const Word&)>& order)
{
    assert(i < rules.size());

    Word w = rule_left(rules[i]);
    Word rhs_inv = group_inverse(rule_right(rules[i]));
    w.insert(w.end(), rhs_inv.begin(), rhs_inv.end());

    for (std::size_t shift = 0; shift < w.size(); ++shift) {

        Word shifted;
        shifted.reserve(w.size());
        shifted.insert(shifted.end(), w.begin() + shift, w.end());
        shifted.insert(shifted.end(), w.begin(), w.begin() + shift);

        for (std::size_t split = 0; split <= shifted.size(); ++split) {

            Word left (shifted.begin(), shifted.begin() + split);
            Word right_suffix(shifted.begin() + split, shifted.end());

            Word right = group_inverse(right_suffix);

            int cmp = order ? order(left, right) : 0;
            if (cmp == 1) {
                rules.emplace_back(left,  right);
            } else if (cmp == -1) {
                rules.emplace_back(right, left);
            } else {
                // cmp == 0 : the two sides are equal – nothing to add
            }
        }
    }

    w = group_inverse(w);

    for (std::size_t shift = 0; shift < w.size(); ++shift) {
        Word shifted;
        shifted.reserve(w.size());
        shifted.insert(shifted.end(), w.begin() + shift, w.end());
        shifted.insert(shifted.end(), w.begin(), w.begin() + shift);

        for (std::size_t split = 0; split <= shifted.size(); ++split) {

            Word left (shifted.begin(), shifted.begin() + split);
            Word right = group_inverse(Word(shifted.begin() + split, shifted.end()));

            int cmp = order ? order(left, right) : 0;
            if (cmp == 1) {
                rules.emplace_back(left,  right);
            } else if (cmp == -1) {
                rules.emplace_back(right, left);
            }
        }
    }
}

//  Adds the cancelling pairs for every generator
//  that occurs (in either sign) in the current system.
//
//  With the pair representation we store the rule as
//        lhs = {a,b} , rhs = {}.
// ---------------------------------------------------------------------------
void add_free_group_rules(Rules& rules)
{
    int max_abs = 1;

    for (const Rule& r : rules) {
        for (int x : r.first)  max_abs = std::max(max_abs, std::abs(x));
        for (int x : r.second) max_abs = std::max(max_abs, std::abs(x));
    }

    for (int i = max_abs; i >= 1; --i) {
        rules.insert(rules.begin(), Rule(Word{ i, -i }, Word{}));
        rules.insert(rules.begin(), Rule(Word{ -i, i }, Word{}));
    }
}
