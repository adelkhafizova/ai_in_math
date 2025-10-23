#pragma once
#include <vector>
#include <functional>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include "types.hpp"

using AlphabetOrder  = std::function<int(int, int)>;


// Basic alphabet orders

/*  x_1 < x_1⁻¹ < x_2 < x_2⁻¹ < ...  */
inline int alphabet_order(int a, int b)
{
    if (std::abs(a) > std::abs(b))      return  1;
    if (std::abs(a) < std::abs(b))      return -1;
    if (a < b)                          return -1;
    if (a > b)                          return  1;
    return 0;
}

/*  x_1 < x_2 < ... < x_n < x_1⁻¹ < x_2⁻¹ < ...  */
inline int alphabet_order_positive_first(int a, int b)
{
    if (a > 0 && b < 0)                 return -1;
    if (a < 0 && b > 0)                 return  1;

    if (a > 0 && b > 0) {
        if (a > b) return  1;
        if (a < b) return -1;
        return 0;
    }

    if (a < 0 && b < 0) {
        if (a < b) return  1;
        if (a > b) return -1;
        return 0;
    }

    return 0;   // should never be reached
}

/*  Chenadec order
    x_2p > x_2p⁻¹ > x_{2p-2} > ... > x_2⁻¹ > x_1 > x_1⁻¹ > ... > x_{2p-1}⁻¹   */
inline int chenadec_order(int a, int b)
{
    const bool a_even = (a % 2 == 0);
    const bool b_even = (b % 2 == 0);

    if (a_even && !b_even)  return  1;
    if (!a_even && b_even)  return -1;

    if (a_even && b_even) {               // both even
        if (std::abs(a) > std::abs(b))   return  1;
        if (std::abs(a) < std::abs(b))   return -1;
        if (a > b)                       return  1;
        if (a < b)                       return -1;
        return 0;
    }

    // both odd
    if (std::abs(a) < std::abs(b))       return  1;
    if (std::abs(a) > std::abs(b))       return -1;
    if (a > b)                           return  1;
    if (a < b)                           return -1;
    return 0;
}

// Word orderings that use an alphabet order


/* Short-lex : first compare length, then compare letter-by-letter
   using the supplied alphabet order                                 */
inline int shortlex(const Word& w1,
                    const Word& w2,
                    const AlphabetOrder& alph = alphabet_order)
{
    if (w1.size() > w2.size()) return  1;
    if (w1.size() < w2.size()) return -1;

    for (std::size_t i = 0; i < w1.size(); ++i) {
        int c = alph(w1[i], w2[i]);
        if (c != 0) return c;   // either 1 or -1
    }
    return 0;
}

// Wrappers
inline int shortlex_default(const Word& a, const Word& b)
{
    return shortlex(a, b);
}
inline int shortlex_with_chenadec(const Word& a, const Word& b)
{
    return shortlex(a, b, chenadec_order);
}
inline int shortlex_with_positive_first(const Word& a, const Word& b)
{
    return shortlex(a, b, alphabet_order_positive_first);
}

/* Pure lexicographic (no length check first)*/
inline int lex(const Word& w1,
               const Word& w2,
               const AlphabetOrder& alph = alphabet_order)
{
    const std::size_t m = std::min(w1.size(), w2.size());
    for (std::size_t i = 0; i < m; ++i) {
        int c = alph(w1[i], w2[i]);
        if (c != 0) return c;
    }
    if (w1.size() > w2.size()) return  1;
    if (w1.size() < w2.size()) return -1;
    return 0;
}

// Recursive Path Order (RPO)

/* Forward declaration (used by the DP version) */
inline int recursive_path_order_dyn_prog(const Word& w1,
                                          const Word& w2,
                                          const AlphabetOrder& alph);

/* The classic recursive definition (with optional DP shortcut).    */
inline int recursive_path_order(const Word& w1,
                                const Word& w2,
                                const AlphabetOrder& alph = alphabet_order,
                                bool use_dynprog = true)
{
    // DP optimisation - only used when at least one word is “big”.
    if (use_dynprog && (w1.size() > 20 || w2.size() > 20))
        return recursive_path_order_dyn_prog(w1, w2, alph);

    if (w1.empty() && w2.empty()) return 0;
    if (w1.empty())               return -1;
    if (w2.empty())               return  1;

    if (w1.size() == 1 && w2.size() == 1)
        return alph(w1[0], w2[0]);

    if (w1[0] == w2[0]) {
        Word w1_tail(w1.begin() + 1, w1.end());
        Word w2_tail(w2.begin() + 1, w2.end());
        return recursive_path_order(w1_tail, w2_tail, alph, use_dynprog);
    }

    if (alph(w1[0], w2[0]) == 1) {
        Word w2_tail(w2.begin() + 1, w2.end());
        if (recursive_path_order(w1, w2_tail, alph, use_dynprog) == 1)
            return 1;
    }

    Word w1_tail(w1.begin() + 1, w1.end());
    int ord = recursive_path_order(w1_tail, w2, alph, use_dynprog);
    if (ord == 1 || ord == 0) return 1;
    return -1;
}

// Iterative (dynamic-programming) version of RPO
inline int recursive_path_order_dyn_prog(const Word& w1,
                                          const Word& w2,
                                          const AlphabetOrder& alph)
{
    if (w1.size() == 1 && w2.size() == 1)
        alph(w1[0], w2[0]);
    if (w1.empty() && w2.empty()) return 0;
    if (w1.empty())               return -1;
    if (w2.empty())               return  1;

    const std::size_t n = w1.size();
    const std::size_t m = w2.size();

    /* DP table: dp[i][j] stores the result of RPO( w1[i..], w2[j..] )
       i ranges 0..n, j ranges 0..m.  The extra row/column correspond to
       the empty suffix.                                             */
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (std::size_t j = 0; j < m; ++j) dp[n][j] = -1;
    for (std::size_t i = 0; i < n; ++i) dp[i][m] =  1;
    dp[n][m] = 0;

    for (int diag = static_cast<int>(n + m - 2); diag >= 0; --diag) {
        const int i_start = std::max(0, diag - static_cast<int>(m) + 1);
        const int i_end   = std::min(diag, static_cast<int>(n) - 1);

        for (int i = i_start; i <= i_end; ++i) {
            const int j = diag - i;
            assert(j >= 0 && static_cast<std::size_t>(j) < m);

            if (w1[i] == w2[j]) {
                dp[i][j] = dp[i + 1][j + 1];
            }
            else if (alph(w1[i], w2[j]) == 1 && dp[i][j + 1] == 1) {
                dp[i][j] = 1;
            }
            else if (dp[i + 1][j] >= 0) {
                dp[i][j] = 1;
            }
            else {
                dp[i][j] = -1;
            }
        }
    }

    return dp[0][0];
}

// Convenience wrappers for the specialised orders
inline int recursive_path_order_with_chenadec(const Word& a,
                                               const Word& b)
{
    return recursive_path_order(a, b, chenadec_order, true);
}
inline int recursive_path_order_with_positive_first(const Word& a,
                                                     const Word& b)
{
    return recursive_path_order(a, b,
                                alphabet_order_positive_first,
                                true);
}

/* -------------------------------------------------------------
   6  Placeholder for “wreath order” (not implemented in the
        original Python file)
   ------------------------------------------------------------- */
inline int wreath_order(const Word&, const Word&,
                        const AlphabetOrder& = alphabet_order)
{
    return 0;   // stub
}