#pragma once
#include <algorithm>
#include <iterator>
#include <vector>
#include <utility>
#include <iostream>    // optional, for debugging

namespace grp {

/* ------------------------------------------------------------------
 *  Helper: inverse of a word.
 *  The inverse is obtained by reversing the word and flipping the sign
 *  of every generator.
 * ------------------------------------------------------------------ */
inline std::vector<int> group_inverse(const std::vector<int>& w)
{
    std::vector<int> inv;
    inv.reserve(w.size());
    for (auto it = w.rbegin(); it != w.rend(); ++it)
        inv.push_back(-(*it));
    return inv;
}

/* ------------------------------------------------------------------
 *  Helper: test whether a word is already stored in the container.
 *  (Linear search – identical to Python's `in` operator.)
 * ------------------------------------------------------------------ */
inline bool contains(const std::vector<std::vector<int>>& container,
                     const std::vector<int>& word)
{
    return std::find(container.begin(), container.end(), word) != container.end();
}

/* ------------------------------------------------------------------
 *  generate_group_words:
 *  Generates cyclically reduced group words on number_gens group generators up to length max_length
 * ------------------------------------------------------------------ */
inline void generate_group_words(
        std::vector<std::vector<int>>& words,
        int number_gens = 2,
        int max_length  = 4,
        bool exact_length = false,
        bool remove_cyc_reducible = true,
        bool remove_shifts = true,
        bool remove_inverses = true)
{
    /* ----- 1. initialise length‑1 words --------------------------- */
    words.clear();
    for (int i = 1; i <= number_gens; ++i) {
        words.push_back({i});
        words.push_back({-i});
    }

    /* ----- 2. grow words up to max_length -------------------------- */
    for (int length = 2; length <= max_length; ++length) {
        std::vector<std::vector<int>> new_words;
        new_words.reserve(words.size() * number_gens * 2);

        for (const auto& w : words) {
            if (static_cast<int>(w.size()) != length - 1) continue;

            for (int gen = 1; gen <= number_gens; ++gen) {
                if (w.back() !=  gen) {
                    std::vector<int> tmp = w;
                    tmp.push_back(-gen);
                    new_words.push_back(std::move(tmp));
                }
                if (w.back() != -gen) {
                    std::vector<int> tmp = w;
                    tmp.push_back(gen);
                    new_words.push_back(std::move(tmp));
                }
            }
        }
        words.insert(words.end(),
                     std::make_move_iterator(new_words.begin()),
                     std::make_move_iterator(new_words.end()));
    }

    /* ----- 3. post‑processing -------------------------------------- */
    if (exact_length) {
        /* ---- keep only words of length max_length ----------------- */
        words.erase(
            std::remove_if(words.begin(), words.end(),
                [max_length](const std::vector<int>& w){ return static_cast<int>(w.size()) != max_length; }),
            words.end());

        if (remove_inverses) {
            for (size_t i = 0; i < words.size(); ++i) {
                const auto inv = group_inverse(words[i]);
                if (contains(words, inv))
                    words.erase(std::remove(words.begin(), words.end(), inv), words.end());
            }
        }

        if (remove_shifts) {
            for (size_t i = 0; i < words.size(); ++i) {
                const auto& w = words[i];
                const int n = static_cast<int>(w.size());
                for (int sh = 1; sh < n; ++sh) {
                    std::vector<int> shifted(w.begin() + sh, w.end());
                    shifted.insert(shifted.end(), w.begin(), w.begin() + sh);
                    if (shifted != w && contains(words, shifted))
                        words.erase(std::remove(words.begin(), words.end(), shifted), words.end());
                }
            }
        }
        return;        // exact_length mode ends here
    }

    /* ---- a) remove cyclically reducible words ------------------- */
    if (remove_cyc_reducible) {
        for (auto it = words.begin(); it != words.end(); ) {
            if (it->size() > 2 && (*it)[0] == -static_cast<int>((*it).back())) {
                it = words.erase(it);               // erase returns next iterator
            } else {
                ++it;
            }
        }
    }

    /* ---- b) remove inverse duplicates --------------------------- */
    if (remove_inverses) {
        for (size_t i = 0; i < words.size(); ++i) {
            const auto inv = group_inverse(words[i]);
            if (contains(words, inv)) {
                words.erase(std::remove(words.begin(), words.end(), inv), words.end());
            }
        }
    }

    /* ---- c) remove shifted duplicates -------------------------- */
    if (remove_shifts) {
        for (size_t i = 0; i < words.size(); ++i) {
            const auto& w = words[i];
            const int n = static_cast<int>(w.size());
            for (int sh = 1; sh < n; ++sh) {
                std::vector<int> shifted(w.begin() + sh, w.end());
                shifted.insert(shifted.end(), w.begin(), w.begin() + sh);
                if (shifted != w && contains(words, shifted))
                    words.erase(std::remove(words.begin(), words.end(), shifted), words.end());
            }
        }
    }

    /* ---- d) remove shifts of inverses (only if both options set) */
    if (remove_shifts && remove_inverses) {
        for (size_t i = 0; i < words.size(); ++i) {
            const auto inv = group_inverse(words[i]);
            const int n = static_cast<int>(inv.size());
            for (int sh = 0; sh < n; ++sh) {
                std::vector<int> shifted(inv.begin() + sh, inv.end());
                shifted.insert(shifted.end(), inv.begin(), inv.begin() + sh);
                if (shifted != words[i] && contains(words, shifted))
                    words.erase(std::remove(words.begin(), words.end(), shifted), words.end());
            }
        }
    }
}

/* ------------------------------------------------------------------
 *  generate_all_words
 * ------------------------------------------------------------------ */
inline void generate_all_words(
        std::vector<std::vector<int>>& words,
        int number_gens = 2,
        int max_length  = 4,
        bool exact_length = false)
{
    words.clear();
    for (int i = 1; i <= number_gens; ++i) {
        words.push_back({ i });
        words.push_back({ -i });
    }

    for (int length = 2; length <= max_length; ++length) {
        std::vector<std::vector<int>> new_words;
        new_words.reserve(words.size() * number_gens * 2);

        for (const auto& w : words) {
            if (static_cast<int>(w.size()) != length - 1) continue;

            for (int gen = 1; gen <= number_gens; ++gen) {
                std::vector<int> t1 = w; t1.push_back(-gen);
                std::vector<int> t2 = w; t2.push_back(gen);
                new_words.push_back(std::move(t1));
                new_words.push_back(std::move(t2));
            }
        }
        words.insert(words.end(),
                     std::make_move_iterator(new_words.begin()),
                     std::make_move_iterator(new_words.end()));
    }

    if (exact_length) {
        words.erase(
            std::remove_if(words.begin(), words.end(),
                [max_length](const std::vector<int>& w){ return static_cast<int>(w.size()) != max_length; }),
            words.end());
    }
}

}

using Rule  = std::pair<std::vector<int>, std::vector<int>>;
using Rules = std::vector<Rule>;

/* ------------------------------------------------------------------
 *  presentation_to_rules
 *
 *  Input  : `relators` – a list of relators, each relator is a word
 *           represented by `std::vector<int>`.
 *
 *  Output : `Rules` – for every relator we create a rule whose left
 *           hand side is the relator itself and whose right hand side
 *           is the empty word (`std::vector<int>{}`).
 * ------------------------------------------------------------------ */
inline Rules presentation_to_rules(const std::vector<std::vector<int>>& relators)
{
    Rules rules;
    rules.reserve(relators.size());

    for (const auto& rel : relators) {
        rules.emplace_back(rel, std::vector<int>{});
    }
    return rules;
}

/* ------------------------------------------------------------------
 *  Overload that accepts an r‑value list of relators.  This allows
 *  callers to pass a temporary without an extra copy.
 * ------------------------------------------------------------------ */
inline Rules presentation_to_rules(std::vector<std::vector<int>>&& relators)
{
    Rules rules;
    rules.reserve(relators.size());

    for (auto& rel : relators) {
        rules.emplace_back(std::move(rel), std::vector<int>{});
    }
    return rules;
}