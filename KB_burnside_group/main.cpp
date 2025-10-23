#include "ordering.hpp"
#include "vector.hpp"
#include "reduce.hpp"
#include "crit_pairs.hpp"
#include "resolve.hpp"
#include "shortcut.hpp"
#include "redundancy.hpp"
#include "knuth-bendix.hpp"
#include "group.hpp"
#include "print.hpp"
#include "detect.hpp"
#include "word_generator.hpp"
#include "expWords.hpp"

int main()
{
  // Short Test Cases:
  /*
  // Abelian group on a, b
  Rules rules = {{{1,2,-1,-2},{}}}; //Z^2

  add_free_group_rules(rules);
  std::cout << "Starting rules: " << rules << "\n";
  int rounds = knuth_bendix(rules, shortlex_default, true, 10, true, false, false);
  if(rounds != -1)
    std::cout << "Finished in " << rounds << " rounds.\nTotal number of rules:" << rules.size() << "\n";
  else
    std::cout << "Terminated by reaching max_rounds.\nTotal number of rules:" << rules.size() << "\n";
  std::cout << rules << "\n";
  std::cout << "Confluence:" << check_confluence(rules) << "\n";
  std::cout << "---------\n";
  */

  // Burnside Test Case:
  // 1. generate all reduced words on 2 generators of length ≤ 4
    std::vector<std::vector<int>> group_words;
    grp::generate_group_words(group_words, 2, 3, false, false, false, false);

    // If you want the “no‑reduction” variant, just uncomment the call
    // below and comment the one above (exact same arguments, but the three
    // boolean flags set to false).
    /*
    grp::generate_group_words(
        group_words,
        2, 3, false,
        false,   // remove_cyc_reducible
        false,   // remove_shifts
        false);  // remove_inverses
    */

    // 2. build the list `all_starting_relators = [word*4 for word in group_words]`
    std::vector<std::vector<int>> all_starting_relators;
    all_starting_relators.reserve(group_words.size());
    for (const auto& w : group_words) {
        all_starting_relators.push_back(quadruple_word(w));
    }

    // 3. turn each relator into a rule (rhs = empty word)
    Rules all_starting_rules = presentation_to_rules(all_starting_relators);

    // 4. copy the rules into the mutable container `rules`
    Rules rules = all_starting_rules;

    // 5. finally add the free‑group rules
    add_free_group_rules(rules);

    /*
    // ------------------------------------------------------------------
    // optional: show how many rules we ended up with
    // ------------------------------------------------------------------
    std::cout << "Number of generated rules: " << rules.size() << '\n';

    // (you can also print a few rules if you like)
    for (size_t i = 0; i < std::min(rules.size(), size_t{16}); ++i) {
        const auto& lhs = rules[i].first;
        const auto& rhs = rules[i].second;   // always empty in this stage
        std::cout << "Rule " << i << ":  LHS = [";
        for (size_t j = 0; j < lhs.size(); ++j) {
            std::cout << lhs[j];
            if (j + 1 < lhs.size()) std::cout << ',';
        }
        std::cout << "]  RHS = [";
        for (size_t j = 0; j < rhs.size(); ++j) {
            std::cout << rhs[j];
            if (j + 1 < rhs.size()) std::cout << ',';
        }
        std::cout << "]\n";
    }
    */

    /*
    print("B(2,3). Starting rules:", rules)
    print("Trying shortlex:")
    rounds = knuth_bendix(rules, order=shortlex, reduce_crit_immediately=True, max_rounds=4, print_progress=False)
    if rounds != -1:
      print("Finished in", rounds, "rounds.\nTotal number of rules:", len(rules))
    else:
      print("Terminated by reaching max_rounds.\nTotal number of rules:", len(rules))
    print(rules)
    print("Confluence:", check_confluence(rules,print_progress=False))
    print("---------")
    */
  

    std::cout << "B(2,4). Number of starting rules:\n" << rules.size() << '\n';
    std::cout << "Trying RPO pos first:\n";

    int rounds = knuth_bendix(rules, recursive_path_order_with_positive_first, true, 4, false, true, false);
    if(rounds != -1)
      std::cout << "Finished in " << rounds << " rounds.\nTotal number of rules:" << rules.size() << "\n";
    else
      std::cout << "Terminated by reaching max_rounds.\nTotal number of rules:" << rules.size() << "\n";

    std::cout << "Confluence:" << check_confluence(rules) << "\n"; // Pass the non-const object
    std::cout << rules << "\n";
    std::cout << "---------\n";

    // Other Tests:
    /*
    std::vector<std::vector<int>> testerWords;
    grp::generate_group_words(testerWords, 2, 3, false, false, false, false);
    std::cout << "Generated " << testerWords.size() << " words" << std::endl;
    std::cout << "[";
    for (const auto& word : testerWords) {
        std::cout << word << ",";
    }
    std::cout << "]";
    */

  return 0;
}