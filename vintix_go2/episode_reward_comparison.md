# Episode Reward Comparison: Expert vs All_onegroup

## Mean Reward per Episode

| Model | Go1 | Go2 | A1 | Minicheetah |
|-------|-----|-----|----|----|----|
| Expert | 22.502 | 21.325 | 20.860 | 21.052 |
| All_onegroup | 21.460 | 11.908 | 20.700 | 20.669 |

## Standard Deviation of Reward per Episode

| Model | Go1 | Go2 | A1 | Minicheetah |
|-------|-----|-----|----|----|----|
| Expert | 0.028 | 0.011 | 0.019 | 0.035 |
| All_onegroup | 4.522 | 9.873 | 1.443 | 0.257 |

## Detailed Values

### Mean Reward per Episode

**Expert Policy:**
- Go1: 22.501968
- Go2: 21.325001
- A1: 20.860417
- Minicheetah: 21.051752

**All_onegroup Model (15th epoch):**
- Go1: 21.460145
- Go2: 11.908149
- A1: 20.700345
- Minicheetah: 20.668514

### Standard Deviation of Reward per Episode

**Expert Policy:**
- Go1: 0.028277
- Go2: 0.011296
- A1: 0.019473
- Minicheetah: 0.034513

**All_onegroup Model (15th epoch):**
- Go1: 4.522323
- Go2: 9.872996
- A1: 1.442779
- Minicheetah: 0.256970

## Observations

1. **Expert Policy** shows very consistent performance across all robots with low standard deviations (0.01-0.03 range).

2. **All_onegroup Model**:
   - **Go1**: Slightly lower mean than Expert, but with much higher variance (Std = 4.52)
   - **Go2**: Significantly lower mean (11.91 vs 21.33) with very high variance (Std = 9.87)
   - **A1**: Comparable mean to Expert (20.70 vs 20.86), with moderate variance (Std = 1.44)
   - **Minicheetah**: Comparable mean to Expert (20.67 vs 21.05), with low variance (Std = 0.26)

3. The Go2 evaluation for All_onegroup shows concerning results with low mean reward and high variance, suggesting potential instability in that configuration.
