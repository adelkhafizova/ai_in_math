# DQN for Burau Representation



---

## 🧩 Problem Description

We model the exploration of words in the Burau representation as a reinforcement learning problem.

### 🔢 Modulo Arithmetic
- All matrices are computed **modulo `MODULO = 2`**

### 🔤 State Space
- All words over the alphabet `{A, B, a, b}` up to a maximum length of **`MAX_LENGTH = 32`**.

### 🎯 Action Space
- Append one of the letters `A`, `B`, `a`, or `b` to the current word.

---

## 🏆 Reward Function

- **Illegal moves penalty:**  
  If the resulting word includes subwords like `Aa`, `aA`, `Bb`, or `bB`, the agent receives a heavy penalty of **`-10000`**.

- **Valid moves reward:**  
  For valid moves, the reward is calculated as: `reward = 10 / (1 + degree)` where `degree` is maximal abosulte value of degrees of all monomials in all matrix entries. Where matrix is matrix corresponding to a word.


## 🧠 DQN Hyperparameters

| Parameter              | Value     |
|------------------------|-----------|
| `MODULO`               | 2         |
| `MAX_LENGTH`           | 32        |
| `BATCH_SIZE`           | 500       |
| `GAMMA`                | 0.99      |
| `EPSILON_START`        | 1.0       |
| `EPSILON_END`          | 0.1       |
| `EPSILON_DECAY`        | 5000      |
| `TARGET_UPDATE_FREQ`   | 50        |
| `REPLAY_CAPACITY`      | 100000    |
| `LEARNING_RATE (LR)`   | 1e-3      |

---

## 🚀 Goal

Teach agent to obtain reduced word of length `32` (hopefully lower) which gives identity (equivalently has reward function `10`) modulo `2`. It has to exist since it was found using deterministic search.



