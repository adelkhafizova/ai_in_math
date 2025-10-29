#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <cassert>
#include "shortcut.hpp"
#include "ordering.hpp"
#include "redundancy.hpp"
#include "crit_pairs.hpp"
#include "resolve.hpp"
#include "types.hpp"

/**
 * @brief Runs the Knuth‑Bendix completion algorithm.
 *
 * @param rules                     the rewriting system (modified in‑place)
 * @param order                     total order on words (defaults to shortlex)
 * @param reduce_crit_immediately   whether a critical pair is reduced as soon as it is created
 * @param max_rounds                limit on the number of rounds (‑1 = no limit)
 * @param recheck_old_rules        if true, critical pairs are recomputed for *all* rules each round
 * @param print_progress            enable textual progress output
 * @param print_crit_pairs_progress enable progress output while generating critical pairs
 *
 * @return number of rounds performed, or –1 if the algorithm stopped because
 *         `max_rounds` was hit.
 */
int knuth_bendix(
    Rules& rules,
    const std::function<int(const Word&, const Word&)>& order,
    bool reduce_crit_immediately,
    int max_rounds,
    bool recheck_old_rules,
    bool print_progress,
    bool print_crit_pairs_progress)
{
    bool changes_made = true;
    int  rounds       = 0;
    std::size_t first_new_rule = 0;
    CritPairs crit_pairs;

    while (changes_made) {
        ++rounds;
        if (max_rounds >= 0 && rounds > max_rounds) {
            std::cout << "Max rounds reached: " << max_rounds << '\n';
            std::cout << "Making shortcuts and removing redundancies, then stopping.\n";
            return -1;
        }
        if (print_progress) {
            std::cout << "Round " << rounds << '\n';
        }
        bool shortcuts_made          = false;
        bool redundancies_eliminated = false;
        bool new_rules_added        = false;
        shortcuts_made = make_all_shortcuts(rules);
        if (print_progress && shortcuts_made) {
            std::cout << "Shortcuts made. " << rules.size() << " rules total.\n";
        }
        std::tie(redundancies_eliminated, first_new_rule) = eliminate_redundancy(rules, first_new_rule);
        if (print_progress && redundancies_eliminated) {
            std::cout << "Redundancies eliminated. " << rules.size() << " rules total.\n";
        }
        if (max_rounds >= 0 && rounds > max_rounds) {
            if (print_progress) {
                std::cout << "Stopping because max rounds reached: " << max_rounds << '\n';
            }
            return -1;
        }
        assert(crit_pairs.empty());
        if (recheck_old_rules) {
            make_crit_pairs(rules,
                            crit_pairs,
                            reduce_crit_immediately,
                            0,
                            print_crit_pairs_progress,
                            print_progress);
        } else {
            make_crit_pairs(rules,
                            crit_pairs,
                            reduce_crit_immediately,
                            first_new_rule,
                            print_crit_pairs_progress,
                            print_progress);
        }

        if (print_progress) {
            std::cout << "Crit pairs: " << crit_pairs.size() << '\n';
        }

        first_new_rule = rules.size();

        // Add memory preallocation here
        new_rules_added = resolve_all_crit_pairs(rules,
                                                 crit_pairs,
                                                 order,
                                                 reduce_crit_immediately,
                                                 print_progress);
        if (print_progress && new_rules_added) {
            std::cout << "New rules added. " << rules.size() << " rules total.\n";
        }

        changes_made = shortcuts_made || redundancies_eliminated || new_rules_added;

        if (print_progress) {
            std::cout << rules.size()
                      << " rules total at the end of round " << rounds << '\n';
        }

        crit_pairs.clear();
    }

    return rounds;
}
