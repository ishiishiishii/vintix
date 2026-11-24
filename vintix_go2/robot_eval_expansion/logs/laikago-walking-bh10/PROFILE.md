# Laikago PPO bh10 retune (1-factor height penalty)

Prior term20 (term=20 / h=0.45 / vx=0.40 / kp=20 / base_height=-50): completion 0%, mean len≈7 → still collapse.
1-factor: height penalty -50→**-10** (termination 20° / height target / cmd / PD unchanged).
Aliengo early-collapse lesson. Next if fail: kp40 (PD).
save_interval=20 (pre-seeded cfgs.pkl).
