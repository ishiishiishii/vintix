# 観測・行動対応表（clone 成果物ベース・2026-11-18）

本リポ標準 Env（`train.py`）: **obs=45** = ang_vel(3)+gravity(3)+commands(3)+dof_pos(12)+dof_vel(12)+actions(12)、**actions=12**（default 姿勢への PD 目標オフセット）。

完全なバイナリ互換は狙わない。比較実験では (a) 各手法公式 sim での評価、(b) Genesis へ観測を再構成しての評価、のどちらかを NOTES で選ぶ。

## MorFiC（P0・配線優先候補）

| 項目 | 値 | 出典 |
|------|-----|------|
| 公式 sim | Isaac Gym（`go2_gym`） | README |
| 単ステップ obs（pretrained） | **70** | `scripts/train.py`（`num_observations=70`）+ `ac_weights_last.pt` |
| 履歴（pretrained） | **30** → flat **2100** | `num_observation_history=30`；adaptation 入力 (256, 2100) |
| 形態ベクトル m | **11** → latent **64** | `morphology.py`；morph_encoder (128,11)→(64,128) |
| 行動 | **12** | actor_body 最終層 (12, 128)；actor 入力 **2166**=2100+64+2 |
| Pretrained | Go2 学習 `runs/.../ac_weights_last.pt` | clone 済 |
| Genesis 接続 | `wrappers/morfic/`（obs アダプタ＋CPU スモーク済）。Env 接続・再生は GPU 空き後 | |

70 内訳: gravity3 + commands15 + dof_pos12 + dof_vel12 + actions12 + last_actions12 + clock4。  
（`config_shared` の 42 はデフォルト。**pretrained は 70×30**。旧メモの 42×15 は誤り。）

形態キー: `thigh_mass, calf_mass, hip_mass, foot_mass, base_mass, hip/thigh/calf limit min/max`。

## GenLoco（P0）

| 項目 | 値 | 出典 |
|------|-----|------|
| 公式 sim | PyBullet（`motion_imitation`） | README |
| 観測 | MotorAngle(12) + IMU + LastAction、各 **history=15**（phase_only 時 IMU は RPY+角速度） | `env_builder.py` |
| 行動 | 12 DoF 位置オフセット（模倣） | robots |
| Pretrained | `morphology_generator_{pace,spin,inverse}.zip` | clone 済 |
| Genesis 接続方針 | 模倣＋位相が本リポ速度追従とタスクが違う。比較は **速度／完走を別定義で揃える**か、公式 sim 評価を主にする | |

## URMA（P0）

| 項目 | 値 | 出典 |
|------|-----|------|
| 公式 sim | 独自 Env + RL-X | README |
| 観測 | 関節ごと description ベクトル + pos/vel/prev_action、trunk 速度・重力・goal 速度・接触 等（可変 DoF） | `environments/unitree_go2/environment.py` |
| 行動 | 可変長（multi_robot_max_action_size でパディング） | 同上 |
| Pretrained | `experiments/pre_trained_model` | clone 済 |
| Genesis 接続方針 | 四足サブセットのみ。関節 description の定義を URMA コードから抽出し、本リポ URDF から同ベクトルを生成 | |

依存: RL-X を別途 pin commit で入れる必要あり（重い。評価スモークは公式 `test.sh` を GPU 空き時に）。

## McARL（P1・コードなし）

| 項目 | 値 |
|------|-----|
| リポ内容 | LICENSE / README stub / docs 画像のみ |
| 接続 | 不可。論文の形態ベクトル定義を読んで再実装するか、公開を待つ |

## 配線の優先順（GPU 空き後）

1. **MorFiC**: pretrained があり、問い（形態ゼロショット）が本線に最も近い
2. **GenLoco**: 古典ベースライン。公式 PyBullet 評価でも比較表に載せられる
3. **URMA**: 依存（RL-X）が重いので後回し可
