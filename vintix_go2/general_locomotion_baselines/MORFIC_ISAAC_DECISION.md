# MorFiC / Isaac Gym 判断（更新 2026-11-24）

旧案 C（主表から外す）は **撤回**。ユーザー指示と文献調査（`SIMULATOR_POLICY.md`）により:

- 手法検証は **公式シミュレータ**で行う
- MorFiC には **Isaac Gym が必要** → 導入する（本線 conda とは分離）

## 次の一手

1. ~~NVIDIA から Isaac Gym Preview 4 tarball を取得し導入~~（2026-11-24:  
   `bash install_isaac_gym.sh /workspace/IsaacGym_Preview_4_Package.tar` → env `isaacgym_py38`、import OK）
2. 別 env で MorFiC `play.py --robot go2` を実行し、生存・速度を JSON 化  
   `bash wrappers/morfic/run_morfic_official_eval.sh`（GPU が空いてから）
3. `results/MAIN_TABLE.md` に MorFiC 行を復帰

tarball は再配布不可。マシン上のローカル配置のみ。
