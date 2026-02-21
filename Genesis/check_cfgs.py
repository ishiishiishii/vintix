import pickle

with open("logs/go2-walking/cfgs.pkl", "rb") as f:
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

print("env_cfg:", env_cfg)
print("obs_cfg:", obs_cfg)
print("reward_cfg:", reward_cfg)
print("command_cfg:", command_cfg)
print("train_cfg:", train_cfg)
