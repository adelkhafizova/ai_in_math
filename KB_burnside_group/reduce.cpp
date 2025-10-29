#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include "types.hpp"

Word apply_rule(Word word, const std::pair<Word, Word>& rule, bool& applied, std::size_t place = 0)
{
    const Word& lhs = rule.first;
    const Word& rhs = rule.second;
    if (place > word.size()) return word;
    for (std::size_t i = place; i + lhs.size() <= word.size(); ++i){
        if (std::equal(lhs.begin(), lhs.end(), word.begin() + i)){
            applied = true;
            word.erase(word.begin() + i, word.begin() + i + lhs.size());
            word.insert(word.begin() + i, rhs.begin(), rhs.end());
            break;
        }
    }
    return word;
}

// Must be updated ------------------------------------------------------------------
//  * Repeatedly scans the list of rewrite *rules* and applies the first
//    applicable rule at the first possible position
//  * *skip_rule* can be used to temporarily ignore a rule – the default
//    value -1 means “do not skip any rule”.  It is useful for the
//    redundancy‑check of the Knuth‑Bendix algorithm.
//  * The function returns the (fully) reduced word.
// ------------------------------------------------------------------
Word reduce(Word word, const Rules& rules, int skip_rule = -1) {
    if (word.empty()) return word;
    bool global_applied = true;
    while (global_applied) {
        global_applied = false;
        for (std::size_t i = 0; i < rules.size(); ++i) {
            bool rule_applied = false;
            if (static_cast<int>(i) == skip_rule) continue;
            word = apply_rule(word, rules[i], rule_applied);
            if (rule_applied) {
                global_applied = true;
                if (word.empty()) return word;
                break;
            }
        }
    }
    return word;
}

// reorder_rules not quite working
void reorder_rules(Rules& rules, std::function<int(Word, Word)> &order) {
  //'' for each rule u->v, rewrite it as u->v or v->u according to order '''
  for (std::size_t i = 0; i < rules.size(); ++i)   // i is an unsigned index
  {
    if(order(rules[i].first, rules[i].second) == -1)
      swap(rules[i].first,rules[i].second);
  }
}
