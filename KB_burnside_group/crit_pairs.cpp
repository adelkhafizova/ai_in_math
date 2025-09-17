#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>
#include <cstddef>
#include "reduce.hpp"
#include "types.hpp"

//  Helper functions
static inline Word concat(const Word& a, const Word& b)
{
    Word r; r.reserve(a.size() + b.size());
    r.insert(r.end(), a.begin(), a.end());
    r.insert(r.end(), b.begin(), b.end());
    return r;
}

static inline void print_word(const Word& w)
{
    std::cout << '[';
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::cout << w[i];
        if (i + 1 < w.size()) std::cout << ' ';
    }
    std::cout << ']';
}

inline const Word& rule_left (const Rule& r) { return r.first;  }
inline const Word& rule_right(const Rule& r) { return r.second; }

/**
 * @brief Build the list of critical pairs for a given set of rules.
 *
 * @param rules                the rewriting system (vector of pairs)
 * @param crit_pairs           container that will receive the pairs.
 * @param reduce_immediately   if true, each candidate pair is reduced immediately;
 *                             only pairs whose two reductions differ are stored.
 * @param start_at_rule        only pairs that involve a rule with index greater than or equal to this value
 *                             are examined (used for incremental Knuth‑Bendix).
 * @param max_crit_length      0 -> no length limit; otherwise discard pairs whose
 *                             concatenated word would be longer than this.
 * @param reset_pairs          if true, clear `crit_pairs` before filling it.
 * @param print_progress       verbose printing of each candidate pair.
 * @param print_progress_pct   fake “percentage” mode; we simply iterate with a
 *                             different loop variable – the original used tqdm.
 *
 * @return reference to the filled `crit_pairs` (for convenience)
 */
CritPairs& make_crit_pairs(const Rules& rules,
                          CritPairs& crit_pairs,
                          bool reduce_immediately,
                          std::size_t start_at_rule,
                          std::size_t max_crit_length,
                          bool reset_pairs,
                          bool print_progress,
                          bool print_progress_pct)
{
    if (reset_pairs) crit_pairs.clear();
    if (print_progress_pct) std::cout << "Building crit pairs – overlap:\n";

    auto i_range = (print_progress_pct ? std::vector<std::size_t>{} : std::vector<std::size_t>{});
    for (std::size_t i = 0; i < rules.size(); ++i) {
        for (std::size_t j = 0; j < rules.size(); ++j) {
            if (i < start_at_rule && j < start_at_rule) continue;

            const Word& lhs1 = rule_left (rules[i]);
            const Word& lhs2 = rule_left (rules[j]);

            std::size_t max_k = std::min(lhs1.size(), lhs2.size());
            for (std::size_t k = 1; k < max_k; ++k) {
                if (!std::equal(lhs1.end() - k, lhs1.end(), lhs2.begin())) continue;

                // length check (optional)
                if (max_crit_length != 0 &&
                    lhs1.size() + lhs2.size() - k > max_crit_length) continue;

                Word w1 = concat(rule_right(rules[i]), Word(lhs2.begin() + k, lhs2.end()));
                Word w2 = concat(Word(lhs1.begin(), lhs1.end() - k), rule_right(rules[j]));

                if (reduce_immediately) {
                    Word r1 = reduce(w1, rules);
                    Word r2 = reduce(w2, rules);
                    if (print_progress) {
                        std::cout << "Overlap: ";
                        print_word(concat(lhs1, Word(lhs2.begin() + k, lhs2.end())));
                        std::cout << "  →  ";
                        print_word(r1); std::cout << " , ";
                        print_word(r2); std::cout << '\n';
                    }
                    if (r1 != r2) crit_pairs.emplace_back(std::move(r1), std::move(r2));
                } else {
                    if (print_progress) {
                        std::cout << "Overlap (raw): ";
                        print_word(concat(lhs1, Word(lhs2.begin() + k, lhs2.end())));
                        std::cout << "  vs  ";
                        print_word(w1); std::cout << " , ";
                        print_word(w2); std::cout << '\n';
                    }
                    crit_pairs.emplace_back(std::move(w1), std::move(w2));
                }
            }
        }
    }

    if (print_progress_pct) std::cout << "Building crit pairs – inclusion:\n";

    for (std::size_t i = 0; i < rules.size(); ++i) {
        const Word& lhs1 = rule_left(rules[i]);

        for (std::size_t j = start_at_rule; j < rules.size(); ++j) {
            if (i == j) continue;
            const Word& lhs2 = rule_left(rules[j]);

            if ((max_crit_length == 0 || lhs2.size() <= max_crit_length) &&
                lhs2.size() >= lhs1.size())
            {
                for (std::size_t k = 0; k + lhs1.size() <= lhs2.size(); ++k) {
                    if (!std::equal(lhs1.begin(), lhs1.end(), lhs2.begin() + k)) continue;

                    Word w1 = concat(Word(lhs2.begin(), lhs2.begin() + k),
                                    concat(rule_right(rules[i]),
                                           Word(lhs2.begin() + k + lhs1.size(), lhs2.end())));
                    Word w2 = rule_right(rules[j]);

                    if (reduce_immediately) {
                        Word r1 = reduce(w1, rules);
                        Word r2 = reduce(w2, rules);
                        if (print_progress) {
                            std::cout << "Inclusion (raw): ";
                            print_word(lhs2); std::cout << " >= ";
                            print_word(lhs1); std::cout << "  -->  ";
                            print_word(r1); std::cout << " , ";
                            print_word(r2); std::cout << '\n';
                        }
                        if (r1 != r2) crit_pairs.emplace_back(std::move(r1), std::move(r2));
                    } else {
                        if (print_progress) {
                            std::cout << "Inclusion (raw): ";
                            print_word(lhs2); std::cout << " >= ";
                            print_word(lhs1); std::cout << "  -->  ";
                            print_word(w1); std::cout << " , ";
                            print_word(w2); std::cout << '\n';
                        }
                        crit_pairs.emplace_back(std::move(w1), std::move(w2));
                    }
                }
            }
            else if (i < start_at_rule && j >= start_at_rule &&
                     (max_crit_length == 0 || lhs1.size() <= max_crit_length) &&
                     lhs1.size() > lhs2.size())
            {
                for (std::size_t k = 0; k + lhs2.size() <= lhs1.size(); ++k) {
                    if (!std::equal(lhs2.begin(), lhs2.end(), lhs1.begin() + k)) continue;

                    Word w1 = concat(Word(lhs1.begin(), lhs1.begin() + k),
                                    concat(rule_right(rules[j]),
                                           Word(lhs1.begin() + k + lhs2.size(), lhs1.end())));
                    Word w2 = rule_right(rules[i]);

                    if (reduce_immediately) {
                        Word r1 = reduce(w1, rules);
                        Word r2 = reduce(w2, rules);
                        if (print_progress) {
                            std::cout << "Inclusion (swap): ";
                            print_word(lhs1); std::cout << " <= ";
                            print_word(lhs2); std::cout << "  -->  ";
                            print_word(r1); std::cout << " , ";
                            print_word(r2); std::cout << '\n';
                        }
                        if (r1 != r2) crit_pairs.emplace_back(std::move(r1), std::move(r2));
                    } else {
                        if (print_progress) {
                            std::cout << "Inclusion (swap): ";
                            print_word(lhs1); std::cout << " <= ";
                            print_word(lhs2); std::cout << "  -->  ";
                            print_word(w1); std::cout << " , ";
                            print_word(w2); std::cout << '\n';
                        }
                        crit_pairs.emplace_back(std::move(w1), std::move(w2));
                    }
                }
            }
        }
    }

    return crit_pairs;
}

bool check_confluence(const Rules&          rules,
                      CritPairs*           crit_pairs,
                      std::size_t          max_crit_length,
                      bool                 erase_pair_list,
                      bool                 print_progress)
{
    CritPairs local_storage;
    CritPairs* cp = crit_pairs ? crit_pairs : &local_storage;
    if (cp->empty()) {
        make_crit_pairs(rules, *cp,
                        true,
                        0,
                        max_crit_length,
                        true,
                        print_progress,
                        false);
    }
    for (std::size_t i = 0; i < cp->size(); ++i) {
        const auto& cpair = (*cp)[i];

        if (print_progress) {
            std::cout << "testing pair " << i << ": ";
            print_word(cpair.first);  std::cout << "  ,  ";
            print_word(cpair.second); std::cout << '\n';
        }

        Word w1 = reduce(cpair.first,  rules, -1, false);
        Word w2 = reduce(cpair.second, rules, -1, false);

        if (w1 != w2) {
            if (print_progress) {
                std::cout << "Confluence failure for pair ";
                print_word(cpair.first);  std::cout << " , ";
                print_word(cpair.second); std::cout << "\n   → ";
                print_word(w1);           std::cout << " , ";
                print_word(w2);           std::cout << '\n';
            }
            if (erase_pair_list) cp->clear();
            return false;
        }
    }

    if (erase_pair_list) cp->clear();
    return true;
}