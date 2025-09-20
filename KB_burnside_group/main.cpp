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

int main()
{
  // Abelian group on a, b
  Rules rules = {{{1,2,-1,-2},{}}}; //Z^2
  // CritPairs crit_pairs={};

  //rules.insert(rules.begin(), Rule(Word{1,2,-1,-2}, Word{}));
  add_free_group_rules(rules);
  std::cout << "Starting rules: " << rules << "\n";
  int rounds = knuth_bendix(rules, shortlex_default, true, 10, true, false, false);
  if(rounds != -1)
    std::cout << "Finished in " << rounds << " rounds.\nTotal number of rules:" << rules.size() << "\n";
  else
    std::cout << "Terminated by reaching max_rounds.\nTotal number of rules:" << rules.size() << "\n";
  std::cout << rules << "\n";
  std::cout << "Confluence:" << check_confluence(rules) << "\n"; // Pass the non-const object
  std::cout << "---------\n";

  return 0;
}