# Laikago PPO kp40 retune (1-factor PD)

Prior bh10 (term=20 / h=0.45 / vx=0.40 / kp=20 / base_height=-10): completion 0%, mean len≈8 → still collapse.
1-factor PD: kp 20→**40**, kd 0.5→**1.0** (termination 20° / height penalty -10 / cmd unchanged).
ANYmal-B lesson: PD unlocked walking when height/cmd alone failed.
save_interval=20 (pre-seeded cfgs.pkl).
