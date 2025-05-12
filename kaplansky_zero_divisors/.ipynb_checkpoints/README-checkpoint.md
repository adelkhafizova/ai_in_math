Author: Winston Lee (wlee10@stevens.edu)

This directory contains the working code for searching for a counterexample to the Kaplansky Zero Divisor Conjecture. A summary of the current approach can be found in the [following GitHub Issue.](https://github.com/adelkhafizova/ai_in_math/issues/1)

This directory also contains obsolete code for a previous approach involving the Taiko construction of Mineyev. The relevant paper can be found [here](https://arxiv.org/abs/2501.07646) for those who are interested, and the code will remain here in case this approach needs to be revived. A more up-to-date version of this code can be found in [this Google Colab Notebook.](https://colab.research.google.com/drive/1V8ZjJL_ft4OsDCH3UV8Gk_xhPBlXo6Pm?authuser=1)

For those who are interested in contributing to this project, a good first place to start would be the [aforementioned GitHub Issue](https://github.com/adelkhafizova/ai_in_math/issues/1), which contains a summary of the approach and some relevant papers/resources for the theory. You may also contact me (Winston Lee) via email at wlee10@stevens.edu

A summary of the files in this directory is given below:
- `find_counterexamples.ipynb` is the main Jupyter notebook for the current approach. It is currently configured to find counterexamples to the Kaplansky units conjecture for the Promislow group.
- `kaplansky-sat-solver-script.py` is a Python script that condenses the code of `find_counterexamples.ipynb` into a single script. It is meant for ease of use on a GUI-less virtual machine.
- `output.txt` is the output text file for the `kaplansky-sat-solver-script.py` script. It describes the support of the two factors in some obtained counterexample.
- `ppo_zerodivisor.zip` is the saved PPO model for the RL component of `find_counterexamples.ipynb`

Deprecated Files for the Taiko apporach (see [this Google Colab Notebook](https://colab.research.google.com/drive/1V8ZjJL_ft4OsDCH3UV8Gk_xhPBlXo6Pm?authuser=1) instead):
- `taiko.py`
- `taiko_dfs.py`
- `draw_taiko.py`
- `test.py`
