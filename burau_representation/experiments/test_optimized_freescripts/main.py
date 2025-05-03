import sys
sys.path.append('../../scripts')
import burau_enchanced as lmp
import free_scripts as bf
import free_optimized as bf_optimized
import matplotlib.pyplot as plt
import sys
from multiprocessing import freeze_support
import time

if __name__ == '__main__':
    freeze_support()

    n = 10
    mod = 2
    matrices_mod = {
        "A": lmp.A.convert_to_modulo(mod),
        "B": lmp.B.convert_to_modulo(mod),
        "a": lmp.a.convert_to_modulo(mod),
        "b": lmp.b.convert_to_modulo(mod)
    }
    s = time.time()
    b = bf_optimized.extend_in_all_ways_p(matrices_mod,matrices_mod,n)
    print(f"created b in{time.time()-s}")
    for i in range(50):

        tier_percentages = [1,1,1]
        remaining_percentage = 0
        min_num = 2000000
        max_num = 4000000
        c = bf.tiered_sampling(c,tier_percentages,remaining_percentage,min_num,max_num, invariant=bf.largest_power_range)
        #plotting
        degrees =bf.get_invariant_picture(c,bf.largest_power_range)
        keys = list(degrees.keys())
        values = list(degrees.values())
        plt.clf()
        plt.bar(keys, values)
        for x, y in zip(keys, values):
            plt.text(x, y + 0.2, str(y), ha='center', va='bottom')
        plt.xlabel('Degree')
        plt.ylabel('Number of words')
        plt.title(f"Histogram for length n = {len(next(iter(c)))} with parameters {tier_percentages,min_num,max_num}")
        plt.savefig(f"testplotsmod{mod}/histogram_{len(next(iter(c)))}.png")
        #plotting

        c = bf_optimized.extend_in_all_ways_p(matrices_mod,c,1)
        print(len(next(iter(c))),bf.min_invariant_in_array(c,bf.largest_power_range))
    