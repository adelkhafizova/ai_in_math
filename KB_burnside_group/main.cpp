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
#include <chrono>
#include <iostream>

int main()
{
  // B(2, 3) Test Case (Starting Presentation: Rules of length up to 4 to the power of 4 with cyclic shifts and inversions removed):
    std::vector<std::vector<int>> group_words;
    grp::generate_group_words(group_words, 2, 4, false, true, true, true);
    std::vector<std::vector<int>> all_starting_relators;
    all_starting_relators.reserve(group_words.size());
    for (const auto& w : group_words) { 
      all_starting_relators.push_back(triple_word(w));
    }
    Rules all_starting_rules = presentation_to_rules(all_starting_relators);
    Rules rules = all_starting_rules;
    add_free_group_rules(rules);
    std::cout << "B(2,3). Number of starting rules:\n" << rules.size() << '\n';
    std::cout << "Trying RPO pos first:\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    int rounds = knuth_bendix(rules, recursive_path_order_with_positive_first, true, -1, false, true, false);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    auto hours = std::chrono::duration_cast<std::chrono::hours>(total_ms);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(total_ms - hours);
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(total_ms - hours - minutes);
    auto ms = total_ms - hours - minutes - seconds;
    std::cout << "Execution time: " << hours.count() << "h "
              << minutes.count() << "m "
              << seconds.count() << "s "
              << ms.count() << "ms\n";
    if(rounds != -1)
      std::cout << "Finished in " << rounds << " rounds.\nTotal number of rules:" << rules.size() << "\n";
    else
      std::cout << "Terminated by reaching max_rounds.\nTotal number of rules:" << rules.size() << "\n";
    std::cout << "Confluence:" << check_confluence(rules) << "\n";
    std::cout << rules << "\n";
    std::cout << "---------\n";
    return 0;
}