#include <fstream>
#include <vector>
#include <iostream>
#include "Word.h"
#include "FreeGroup.h"
#include "SubgroupFG.h"
#include <GraphConcept.h>

int main( )
{

  std::vector<std::vector<std::vector<int>>> all_subgroups = {{{1}, {2}}, {{2, -1}, {1, 1, -1}, {1, 2, -1},  {1, 1}, {1, 2}}, {{1}, {1, -2}, {2, 1}, {2, 2}, {2, 2, -2}}, {{1, 1}, {1, 2, -1}, {1, 1, -1}}, {{1, 1, 1}, {1, 1, 2}, {1, 1, 2, -1, -1}}, {{1}, {2, 2, -2}, {2, 2}}, {{1, 1}, {1, 2, -1}, {1}}, {{1, 1, -1}, {1, 1}, {2, 2, -2}, {2, 2}}, {{2, -1}, {1}, {1, 1}, {1, 2}, {1, 2, -1}}, {{1, 1, -1}, {1, 2, -1}, {1, 1}}, {{2, -1}, {1, 2, -1}, {1, 2}, {1, 1, 1}}, {{1}, {2, 2, -2}, {2, 2}, {1, 1}}, {{1, 1}, {1, 1, -1}, {2, 1, 2, -1, -2}, {2, 1, 2, -1}}, {{1, 1, 1}, {1, 1, 2, -1, -1}}, {{2, 2, -2}, {2, 2}, {1, 1, 1}}, {{1}, {2}, {2, 2}}, {{1, 1}, {1, 2}, {1, 2, -1}, {1}, {2, -1}}, {{2, -1}, {1, 1, -1}, {1, 1}, {2, 1, 2}}, {{2, -1}, {2}, {1}, {1, 1}, {1, 2}}, {{1, 1, -1}, {1, 2, -1}, {1, 2}, {1, 1}, {2, -1}}, {{1, 2}, {1, 1, 1}, {1, 1, 2, -1}}, {{1}, {1, -2}, {2, 1}, {2, 2, 2}}, {{1, 1}, {1, 2, -1}, {1, 1, -1}, {2, 1, 2, -1}}, {{1, 1, 1}, {1, 1, 2, -1}, {1, 1, 2, -1, -1}, {1, 2, -1, -1}}, {{1, 1, 1}, {2, 1, 1, 2}}, {{1}, {2}, {2, -1}, {1, 1}, {1, 2}}, {{1, 1}, {1, 2}, {1, 1, -1}, {1, 2, -1}, {2, -1}}, {{2, -1}, {1, 2, -1, -1}, {1, 1, 1}, {1, 1, 2}}, {{2, -1}, {2}, {1, 1, -1}, {1, 1}, {1, 2}}, {{2, 1, -2}, {2, 2, -2}, {2, 2}}, {{1, 1, -1}, {1, 1}, {1, 2}, {2, 2, -1}}, {{1}, {2, 2, 1}, {2, 2, 2}}, {{1, 1, 1}, {1, 1, 2, -1, -1}, {1, 1, 2, -1}, {1, 2, -1, -1}}, {{1, 1}, {1, 1, -1}, {1, 2, -1}, {2, 1, 2, -1}}, {{2, 2, 1}, {2, 2, 2}}, {{1}, {2, 2, 2}}, {{1, 1}, {1, 2, -1}, {1}, {2, 1, 2, -1}}, {{1, 1}, {1, 2, -1}, {1}, {2, 1, 2, -1}}, {{1, 1, -1}, {1, 1}, {2, 2, 2}}, {{2, -1}, {1}, {1, 1}, {2, 1, 2}}, {{1, 1, -1}, {1, 2, -1}, {1, 1}, {2, 1, 2, -1}}, {{1, 2, -1, -1}, {1, 2, -1}, {1, 1, 1}, {1, 1, 2, -1}}, {{2, -1}, {1, 1, 1}, {2, 1, 2}}, {{1}, {1, 1}, {1, 2}, {2, 2, -1}}, {{1, 2, -1, -1}, {1, 2, -1}, {1, 1, 1}, {1, 1, 2, -1}}, {{1, 1, -1}, {1, 2, -1}, {1, 1}, {2, 1, 2, -1}}, {{1, 2}, {2, 2, -1}, {1, 1, 1}}, {{1}, {1, 1}, {2, 2, 2}}, {{1, 1}, {1, 1, -1}, {2, 2, 1, 2, -1}}, {{1, 1, 1}, {1, 1, 2, -1, -1}, {2, 1, 1, 2, -1, -1}}, {{1, 1, 1}, {1, 1, 2, -1, -1}, {2, 1, 1, 2, -1, -1}}, {{1, 1, 1}, {2, 2, 2}}};
  
  std::vector<int> a1(3, 0);
  a1[0] = 1;a1[1] = 1;a1[2] = 1;
  Word w1(a1);
  std::vector<int> a2(3, 0);
  a2[0] = 2;a2[1] = 2;a2[2] = 2;
  Word w2(a2);


  for (size_t i = 0; i < all_subgroups.size(); ++i) {
      int N = all_subgroups[i].size();
      SubgroupFG Sbgp(N);
      for (size_t j = 0; j < all_subgroups[i].size(); ++j){
        Word w(all_subgroups[i][j]);
        Sbgp += w;
      }
      SubgroupFG Sbgp1(2);
      if (Sbgp.doesBelong( w1 )){
          Word d1 = Sbgp.express( w1 );
          Sbgp1 += d1;
      }

      if (Sbgp.doesBelong( w2 )){
          Word d2 = Sbgp.express( w2 );
          Sbgp1 += d2;
      }

      cout << Sbgp1.getIndex() << endl;

  }






  
  return 0;
}