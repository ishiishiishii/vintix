# 拡充ロボットの指令速度・胴体高さ（インターネット検証版）

更新: 2026-11-24（**再監査**）。公式サイト・公開 RL・論文を **ワークスペース非依存で再取得**し、採用値を客観判定した。  
前回案（Go2 と同 Froude で \(v\) をスケール、カタログ立位をそのまま高さ目標に近い値として扱う）は一部 **不適切**。

## 再監査の結論（2026-11-24 2回目）

**現行採用表は維持して実験を進めてよい。** 強根拠の項目は変更不要。弱根拠は表末の「信頼度」を参照（失敗時にそこから再チューン）。

| 項目 | 判定 | 根拠の質 |
|------|------|----------|
| 外寸 ≠ `base_height_target` | **維持** | B1 マニュアル立位≈63 cm 外寸 vs RL stand **0.54**（Improbable-AI）で再確認 |
| Froude で巡航を決める | **不採用のまま** | 再検索でも遷移閾値用途が主 |
| Spot \(v_x=0.90\) | **OK** | BD IFU v2.1.2: Med **default = 0.9** / Slow 0.5 / Fast 1.6 |
| Spot \(h=0.50\) | **OK** | OSU `height=(0.5,0.5)` / Isaac Lab Spot spawn≈0.5。カタログ歩行 610 mm は外寸 |
| Aliengo \(h=0.30\) | **OK** | HIMLoco `aliengo_config.py` raw: `base_height_target = 0.30`, init z=0.55 |
| Aliengo \(v_x=0.75\) | **OK（中）** | 公式 vmax>1.5。メーカー巡航なし → \(0.5\,v_{\max}\) は慣用ヒューリスティック |
| B1 \(h=0.55\) / \(v_x=0.60\) | **OK** | RL 0.54；安定 vmax **1.2**（Weston / IRIS）。0.55 は 0.54 近傍 |
| ANYmal-B \(h=0.50\) / \(v_x=0.60\) | **OK（中）** | Hutter: 0.5 m tall；walk≈0.3 / trot≈0.8。0.60 は中間巡航 |
| Laikago \(h=0.45\) / \(v_x=0.40\) | **OK（弱）** | New Atlas / ロボスタ: 外寸≈0.6 m, vmax≈0.8。**公開 RL cfg なし** → 最初の失敗点候補 |
| matched \(v_x=0.5\) | **OK** | 本線・Spot Slow・比較用固定点 |

## 結論（設計原則）

| 判断 | 内容 |
|------|------|
| カタログの「立位高さ」 | **外寸**。RL の `base_height_target`（base/CoM）とは別物。混同すると過大になる（B1 で顕著） |
| Froude で Go2 の \(v=0.5\) を相似拡大 | 論文では主に **歩容遷移の閾値**に使う。単一点の巡航指令を決める標準ではない → **主根拠から降格** |
| 公開 RL の速度 | 多くは **範囲**（例 ±1 m/s）。単一点専門家を取るなら「メーカー常用速度」か「\(0.4\text{–}0.5\,v_{\max}\)」か「範囲の中央付近」が妥当 |
| Go2 同条件 \(v_x=0.5\) | **妥当**（本線・Spot Slow・SubT ANYmal≈0.45 と同帯） |

## インターネット根拠（一次寄り）

### 公式スペック

| 機体 | 出典 | 立位・速度 |
|------|------|------------|
| Go2 | [unitree.com/go2](https://www.unitree.com/go2/) | 立位外寸 70×31×**40 cm**。速度帯 0–2.5〜3.7 m/s |
| Aliengo | [unitree.com/aliengo](https://www.unitree.com/aliengo/) | 立位 **0.65×0.31×0.60 m**。最大歩行 **>1.5 m/s** |
| B1 | [unitree.com/b1](https://www.unitree.com/b1/) + マニュアル類 | 立位 **1126×467×636 mm**。安定最大歩行 **1.2 m/s**（FAQ は 1.8） |
| Spot | [bostondynamics.com/products/spot](https://bostondynamics.com/products/spot/) + IFU | 歩行デフォルト高さ **610 mm**（min 520 / max 700）。最大 **1.6 m/s**。操作設定 Slow **0.5** / Med **0.9** / Fast **1.6** |
| ANYmal-B | Hutter et al. 論文 | 高さ ≈**0.5 m**。歩行 ≈0.3、トロット ≈**0.8** m/s |
| Laikago | Unitree 動画・報道 | 立位 ≈**0.60 m**。最大 ≈**0.8 m/s** |

### 公開 RL 設定（高さ・指令の“実使用値”）

| 機体 | 出典 | `base_height` / init | 指令速度 |
|------|------|----------------------|----------|
| Go2 | [unitree_rl_gym go2_config.py](https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/go2/go2_config.py) | target **0.25**、init z **0.42** | 親 cfg 系で概ね ±1 m/s |
| Aliengo | [HIMLoco aliengo_config.py](https://raw.githubusercontent.com/OpenRobotLab/HIMLoco/main/legged_gym/legged_gym/envs/aliengo/aliengo_config.py) | target **0.30**、init z **0.55**、kp=40 | 標準 legged_gym 範囲（±1 系） |
| B1 | [Improbable-AI learning-compliance](https://github.com/Improbable-AI/learning-compliance) | stand height / init **0.54**；指令 x∈**[-1,1]** | 同上 |
| B1 | [harelab-ucsc b1_rl_locomotion](https://github.com/harelab-ucsc/b1_rl_locomotion) | 理想高さ **0.54–0.565** | 起立タスクで明示 |
| Spot | Isaac Lab SpotFlat（NVIDIA ブログ / 公開 cfg） | 高さ指令・目標まわり **0.5 m**、足上げ 0.1 | 学習範囲例 **x∈[-2, 3〜4]**（広い） |
| Spot | OSU quadruped_locomotion Spot | height 指令 **0.5** 固定 | カテゴリで ±2 等 |
| ANYmal | [IsaacGymEnvs Anymal.yaml](https://raw.githubusercontent.com/isaac-sim/IsaacGymEnvs/main/isaacgymenvs/cfg/task/Anymal.yaml) | init z **0.62** | x∈**[-2,2]** |
| ANYmal | Raisim gym 例 | 観測平均高さ **0.44** | — |
| ANYmal SubT | CERBERUS 仕様 | 実運用の固定速 ≈**±0.45** | 競技用の遅い巡航 |

### Froude（論文での使い方）

- Fu et al. 2111.01674: A1 で walk Fr≈0.059（≈0.375 m/s）、trot≈0.316（≈0.9 m/s）
- 2026 gait-aware 論文: Fr≈0.15 / 0.74 を **歩容切替閾値**に使用（Barkour 脚長 0.41 m → 0.7 / 1.7 m/s）
- → 「Go2 の 0.5 m/s と同 Fr に揃える」は **一般的な RL 設計慣行ではない**。補助指標には使える。

## 前回案のどこが弱かったか

| 項目 | 前回 | ネット検証 | 判定 |
|------|------|------------|------|
| B1 `h=0.64` | カタログ 636 mm をほぼ採用 | 公開 RL は **0.54** 前後 | **過大 → 修正** |
| Spot `h=0.55` | カタログ 610 mm 寄り | Isaac Lab / OSU は **0.50** | **やや高 → 0.50 に合わせる** |
| Aliengo `h=0.36` | MorFiC/sim 寄り | HIMLoco は **0.30** | **やや高 → 0.30 推奨** |
| Spot `vx=0.90` | BD Med | IFU の常用設定として **妥当** | **維持** |
| B1 `vx=0.70` | Fr スケール＋clip | ±1 学習帯・1.2 vmax 内。**許容**（0.6 でも可） | 維持可 |
| Aliengo `vx=0.55` | Fr | 0.5·vmax=0.75 の方が「体格巡航」らしい | **0.75 に修正** |
| Froude 主根拠 | 主ルール | 遷移用が主。巡航の主根拠にしない | **降格** |

## 改訂採用値（専門家の単一点指令）

本リポジトリは固定指令の専門家を取るため、文献の「範囲」ではなく **評価・学習の代表点**を選ぶ。

| 機体 | \(h\) target | init z | feet_z | body \(v_x\) | 根拠（短い） | matched \(v_x\) |
|------|--------------|--------|--------|--------------|--------------|-----------------|
| Go2 | 0.30 | 0.42 | 0.075 | **0.50** | 本線。unitree_rl は target 0.25 だが本線と整合を優先 | 0.50 |
| Aliengo | **0.30** | 0.55（学習時）/運用は安定優先 | 0.08 | **0.75** | HIMLoco h=0.30；0.5·vmax(1.5) | **0.50** |
| B1 | **0.55** | **0.58** | 0.10 | **0.60** | RL 0.54；0.5·1.2vmax | **0.50** |
| ANYmal-B | **0.50** | **0.55** | 0.09 | **0.60** | 論文トロット帯のやや下；SubT≈0.45 より速い巡航 | **0.50** |
| Spot | **0.50** | **0.55** | 0.10 | **0.90** | Isaac h=0.5；IFU Med | **0.50** |
| Laikago | **0.45** | 0.50 | 0.09 | **0.40** | 0.5·vmax(0.8)；外寸0.6≠CoM | **0.50** |

注意: Aliengo の **現行成功 ckpt** は h≈0.36 / vx=0.5。ネット推奨 h=0.30 への変更は **再学習時**に適用。動いている Spot matched（vx=0.5, 旧 h=0.48）は Isaac の 0.5 に近く、matched としては許容。

## 実装

- `train.py` の `get_*_cfgs()` デフォルト = 上表の **body**
- Go2 同条件: `--cmd_profile matched_go2` または `--cmd_vx 0.5`
- 高さは cfg 本体を変える（指令だけ上書きしても高さは変わらない）

## 文献表（採否）

| 文献・ソース | 変える量 | 本実験への採否 |
|--------------|----------|----------------|
| Unitree / BD 公式 | 外寸・vmax・Spot Med | **採用**（vmax・常用速度） |
| HIMLoco Aliengo cfg | h=0.30, init=0.55 | **採用**（Aliengo 高さ） |
| unitree_rl_gym Go2 | h=0.25, init=0.42 | **参考**（本線 0.30 を維持） |
| Improbable-AI / harelab B1 | h≈0.54, vx±1 | **採用**（B1 高さ） |
| Isaac Lab Spot | h≈0.5, 広い vx 範囲 | **採用**（高さ）；範囲は単一点評価には不使用 |
| Hutter ANYmal 論文 | 0.3 walk / 0.8 trot | **採用**（ANYmal 巡航帯） |
| Fu et al. Froude gaits | Fr→歩容 | **参考**（遷移）。巡航の主式にはしない |
| 前回の \(v=0.5\sqrt{h/0.3}\) | — | **不採用（主根拠）** |
