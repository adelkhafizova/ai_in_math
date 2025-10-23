/********************************************************************
 *  Assumptions
 *  ------------------------------------------------------
 *  * a *word*          → std::vector<int>
 *  * a collection of words → std::vector<std::vector<int>>
 *  * the inverse of a word is obtained by reversing the word and
 *    changing the sign of every generator.
  ********************************************************************/

#include <algorithm>
#include <iterator>
#include <vector>
#include "word_generator.hpp"

namespace grp {
/* ------------------------------------------------------------------
 *  1. detect_inverses
 *
 *  Returns true iff there exists a word w in *words* such that
 *  the inverse of w also belongs to *words*.
 * ------------------------------------------------------------------ */
inline bool detect_inverses(const std::vector<std::vector<int>>& words)
{
    for (const auto& w : words) {
        if (contains(words, group_inverse(w)))
            return true;
    }
    return false;
}

/* ------------------------------------------------------------------
 *  2. detect_duplicates
 *
 *  Returns true iff at least one word occurs more than once in *words*.
 * ------------------------------------------------------------------ */
inline bool detect_duplicates(const std::vector<std::vector<int>>& words)
{
    for (size_t i = 0; i < words.size(); ++i) {
        // count how many times words[i] appears in the whole container
        if (std::count(words.begin(), words.end(), words[i]) > 1)
            return true;
    }
    return false;
}

/* ------------------------------------------------------------------
 *  3. detect_free_inclusion
 *
 *  a) free reduction by cancelling adjacent inverse pairs,
 *  b) cyclic reduction (first = -last),
 *  c) checking all cyclic shifts of the reduced word and of its inverse.
 *
 *  Parameters
 *      word  – the word to test (will be copied, the original is unchanged)
 *      words – the set of words against which we test inclusion
 *
 *  Returns true if the (possibly reduced) word, a cyclic shift of  or a cyclic shift of its inverse is present in *words*.
 * ------------------------------------------------------------------ */
inline bool detect_free_inclusion(std::vector<int> word,
                                  const std::vector<std::vector<int>>& words)
{
    /* --------------------------------------------------------------
     * Stage A : free reduction – cancel i,i+1 when they are inverses.
     * -------------------------------------------------------------- */
    bool cancellation_made = true;
    while (word.size() > 1 && cancellation_made) {
        cancellation_made = false;
        for (size_t i = 0; i + 1 < word.size(); ++i) {
            if (word[i] == -word[i + 1]) {
                word.erase(word.begin() + i, word.begin() + i + 2);
                cancellation_made = true;
                if (word.empty())
                    return true;
                break;
            }
        }
    }

    /* --------------------------------------------------------------
     * Stage B : cyclic reduction – repeatedly delete the outermost
     *           inverse pair (first, last) while they are opposite.
     * -------------------------------------------------------------- */
    while (word.size() > 1 && word.front() == -word.back()) {
        word.erase(word.begin());
        word.pop_back();
        if (word.empty())
            return true;
    }

    /* --------------------------------------------------------------
     * Helper lambda: does any cyclic shift of `w` occur in `words` ?
     * -------------------------------------------------------------- */
    auto shift_in_container = [&](const std::vector<int>& w) -> bool {
        const size_t n = w.size();
        for (size_t shift = 0; shift < n; ++shift) {
            std::vector<int> shifted;
            shifted.reserve(n);
            shifted.insert(shifted.end(), w.begin() + shift, w.end());
            shifted.insert(shifted.end(), w.begin(), w.begin() + shift);
            if (contains(words, shifted))
                return true;
        }
        return false;
    };

    /* --------------------------------------------------------------
     * Stage C : check the (reduced) word and the inverse of the word.
     * -------------------------------------------------------------- */
    if (shift_in_container(word))
        return true;

    const std::vector<int> inv = group_inverse(word);
    if (shift_in_container(inv))
        return true;

    return false;
}
}