#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include "reduce.hpp"
#include "crit_pairs.hpp"
#include "ordering.hpp"
#include "types.hpp"

/**
 * @brief Resolve one critical pair.
 *
 * @param rules          The (mutable) list of current rules.
 * @param crit_pairs     The mutable list of critical pairs.
 * @param idx            Index of the critical pair that shall be resolved.
 * @param order          Ordering function (e.g. shortlex). Must return 1,
 *                       0 or -1.
 * @param print_progress If true, progress messages are printed.
 * @return int           1 if a new rule was added, 0 otherwise.
 */
int resolve_crit_pair(std::vector<Rule>&               rules,
                      CritPairs&                          crit_pairs,
                      std::size_t                      idx,
                      const std::function<int(const Word&, const Word&)>& order,
                      bool                             print_progress = false)
{
    if (print_progress) {
        std::cout << "Resolving " << "["
                  << crit_pairs[idx].first.size() << ", "
                  << crit_pairs[idx].second.size() << "]\n";
    }

    Word w1 = reduce(crit_pairs[idx].first,  rules, print_progress);
    Word w2 = reduce(crit_pairs[idx].second, rules, print_progress);

    if (w1 != w2) {
        if (order(w1, w2) == 1) {
            rules.emplace_back(std::move(w1), std::move(w2));
        } else {
            rules.emplace_back(std::move(w2), std::move(w1));
        }

        crit_pairs.erase(crit_pairs.begin() + static_cast<std::ptrdiff_t>(idx));

        if (print_progress) {
            const Rule& added = rules.back();
            std::cout << "Added rule: [";
            for (int x : added.first) std::cout << x << ' ';
            std::cout << "-> ";
            for (int x : added.second) std::cout << x << ' ';
            std::cout << "]\n";
        }
        return 1;
    }

    crit_pairs.erase(crit_pairs.begin() + static_cast<std::ptrdiff_t>(idx));

    if (print_progress) {
        std::cout << "No rule added\n";
    }
    return 0;
}

/**
 * @brief Resolve *all* critical pairs (optionally generated on the fly).
 *
 * @param rules                The mutable rule set.
 * @param crit_pairs           Optional pre‑computed critical‑pair list.
 *                             If empty, it will be generated with
 *                             make_crit_pairs().
 * @param order                Ordering function (default = shortlex).
 * @param reduce_crit_immediately  Passed to make_crit_pairs().
 * @param go_from_the_end     If true resolve from the back of the list,
 *                             otherwise from the front.
 * @param print_progress      If true, progress information is printed.
 * @param print_crit_pairs_progress  Passed to make_crit_pairs().
 * @return bool                true if at least one new rule was added.
 */
bool resolve_all_crit_pairs(std::vector<Rule>&               rules,
                            CritPairs                           crit_pairs,
                            const std::function<int(const Word&, const Word&)>& order,
                            bool                             reduce_crit_immediately,
                            bool                             go_from_the_end,
                            bool                             print_progress,
                            bool                             print_crit_pairs_progress)
{
    if (crit_pairs.empty()) {
        crit_pairs = make_crit_pairs(rules,
                                    crit_pairs,
                                    reduce_crit_immediately,
                                    0,
                                    0,
                                    true,
                                    print_progress,
                                    print_crit_pairs_progress);
    }

    int rule_counter = 0;

    if (print_progress) {
        std::cout << "Resolving critical pairs (" << crit_pairs.size() << " total)...\n";
    }

    while (!crit_pairs.empty()) {
        std::size_t idx = go_from_the_end ? crit_pairs.size() - 1 : 0;
        rule_counter += resolve_crit_pair(rules,
                                          crit_pairs,
                                          idx,
                                          order,
                                          print_progress);
    }

    if (print_progress) {
        std::cout << "Finished resolving critical pairs.\n";
    }

    if (rule_counter > 0) {
        if (print_progress) {
            std::cout << "Added " << rule_counter << " new rule"
                      << (rule_counter == 1 ? "" : "s") << "\n";
        }
        return true;
    }
    return false;
}