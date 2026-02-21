# Evaluation Results Table

## Mean Reward per Step

This table shows the mean reward per step for each model (trained without a specific robot) evaluated on different robots.

| Model (Without) | Go1      | Go2      | A1       | Minicheetah |
|-----------------|----------|----------|----------|-------------|
| Go1             | 0.019940 | 0.020887 | 0.020282 | 0.018648    |
| Go2             | 0.021539 | 0.010508 | 0.019970 | 0.018276    |
| A1              | 0.021612 | 0.020936 | 0.015892 | 0.015490    |
| Minicheetah     | 0.021740 | 0.018441 | 0.020495 | -0.013319   |

## Detailed Values

| Model (Without) | Go1      | Go2      | A1       | Minicheetah |
|-----------------|----------|----------|----------|-------------|
| Go1             | 0.019940 | 0.020887 | 0.020282 | 0.018648    |
| Go2             | 0.021539 | 0.010508 | 0.019970 | 0.018276    |
| A1              | 0.021612 | 0.020936 | 0.015892 | 0.015490    |
| Minicheetah     | 0.021740 | 0.018441 | 0.020495 | -0.013319   |

## Observations

- **Diagonal values** (evaluating the model on the robot that was excluded from training) show lower performance:
  - Go1without evaluated on Go1: 0.019940 (lower than others)
  - Go2without evaluated on Go2: 0.010508 (lowest)
  - A1without evaluated on A1: 0.015892 (lower than others)
  - Minicheetahwithout evaluated on Minicheetah: -0.013319 (negative, very poor)

- **Off-diagonal values** (cross-robot evaluation) show that models trained on multiple robots can transfer to unseen robots, but performance varies.

- The **best cross-robot performance** is achieved by Minicheetahwithout on Go1 (0.021740).

- The **worst performance** overall is Minicheetahwithout on Minicheetah (-0.013319), indicating that training without Minicheetah data severely impacts performance on Minicheetah.
