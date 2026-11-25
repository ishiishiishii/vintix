# Spot PPO Isaac Lab height profile (#16 retrain)

- init z = 0.50, base_height_target = 0.50, feet_height_target = 0.10 (Isaac Lab SpotFlat / OSU)
- lin_vel_x = 0.70 (body 0.9 previously failed completion; IFU Med 0.9 kept as catalog reference only)
- kp/kd = 40/1, action_scale = 0.25
- save_interval = 20 via train.py robot list
- Old logs (`spot-walking`, `spot-walking-body`, `spot-walking-mid07`) not overwritten
