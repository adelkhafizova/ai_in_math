# AC-Distance in Finite Linear Groups

This SageMath program computes the shortest number of AC-moves needed to transform a given generator pair (x, y) into the Akbulut-Kirby pair (r_n(x, y), s_n(x, y)) in the AC-graph of certain finite groups.

## Running the Program

Open SageMath and run one of the following functions:

```python
attach('AC.sage')

ac_distance_PSL(n=2, q=7)
ac_distance_PSL(n=3, q=5)

ac_distance_F4(q=3)

ac_distance_G2(q=5)
