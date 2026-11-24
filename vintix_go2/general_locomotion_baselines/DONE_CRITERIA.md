# #22 完了条件（更新 2026-11-24）

ユーザー合意: シミュレータは手法ごとに公式のものを使ってよい（文献根拠は `SIMULATOR_POLICY.md`）。
**各ベースラインが、論文で動かしていたロボットで安定歩行できること**が完了条件。

## 必須チェックリスト

| 手法 | 公式シミュレータ | 検証ロボット（論文） | 安定歩行 | 状態 |
|------|------------------|----------------------|----------|------|
| GenLoco | PyBullet | Go1（pace）等 | 生存 300/300 ×3 | **済** |
| URMA | MuJoCo / RL-X | Go2 | 生存 1000/1000 ×3 | **済** |
| MorFiC | **Isaac Gym** | Go2（学習ソース） | 公式 `play.py` | **済**（`results/morfic_isaac_go2_play.json`・2026-11-24） |

## やること（残り）

- [x] Isaac Gym Preview を別 conda env に導入（`install_isaac_gym.sh` + `/workspace/IsaacGym_Preview_4_Package.tar`）
- [x] MorFiC `scripts/play.py --robot go2` で安定歩行を定量記録し主表へ戻す  
      （`wrappers/morfic/run_morfic_official_eval.sh` → `results/morfic_isaac_go2_play.json`）
- [ ] （任意）Go1 など論文記載の他機体でも再生

**必須チェックはすべて済** → #22 を close 可。Genesis 近似は参考列のみ。完了判定に使わない。

### Isaac Gym 導入メモ（2026-11-24）

- conda env: `isaacgym_py38`（本線 Genesis と分離）
- `libpython3.8.so.1.0` は `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` が必要（`activate.d` と公式評価ラッパで設定）
- MorFiC 側: `gym==0.23.1`（0.21.0 は新 setuptools で metadata 失敗）、`numpy==1.23.5`
