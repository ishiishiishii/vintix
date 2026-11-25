# 論文公開設定で拡充ロボットの専門家を用意する案

更新: 2026-11-25（日本時間）。

MorFiC に機体別 config があるものはそれを優先。**Spot / Laikago は MorFiC に学習用 config が無い**ので、**Isaac Lab（Spot）と RaisimGym（Laikago）**を参照する。GenLoco の動作模倣用ロボット定義は参考にしない。

## 機体ごとの参照

| 機体 | 主参照 | いまの状態 | 方針 |
|------|--------|------------|------|
| Aliengo | MorFiC `aliengo_config`（URDF も同系統） | MorFiC 揃えで再学習待ち | そのまま進める |
| B1 | MorFiC `b1_config`（kp90 / kd2.5 / h=0.56 / as=0.35 / 指令±0.6） | 採用中は Go2 系（指令 0.5） | **unitree URDF + MorFiC 制御値**で新規学習（MorFiC URDF は立位失敗あり） |
| ANYmal-B | MorFiC は **C** のみ（kp80 / h=0.55） | 採用中は B + kp40 | C 設定を B に寄せる近似 |
| Spot | **Isaac Lab SpotFlat**（init z≈**0.5**、足上げ 0.1。公開 runner は最大数万 iter） | 採用中は指令 0.7 | 高さ 0.5・指令は IFU Med 0.9 または学習しやすい 0.7。報酬一式の丸コピーはしない |
| Laikago | **RaisimGym**（init z≈**0.46**、観測平均高さ≈**0.44**、kp=**40** / kd=**1**、action_scale≈0.6） | まだ歩けていない | RaisimGym の高さ・PD を優先。関連論文は kp=40・最大≈1000 iter。GenLoco は使わない |

MorFiC 共有指令帯（Aliengo/B1 向け）: \(v_x,v_y\in[-0.6,0.6]\)、yaw \(\in[-1,1]\)。

## 時間の目安（GPU 1 本・4096 env）

301 iter の実測はだいたい **20〜50 分**。足りない機体は `ITERATION_POLICY.md` に従い **601 / 901** まで延長（追加で同程度〜数倍）。

| 作業 | 台数 | 301 の目安 | 延長時 |
|------|------|------------|--------|
| Aliengo MorFiC | 1 | ≈0.5 h | 品質未達かつ上昇中なら 601 |
| B1 MorFiC 制御値 | 1 | ≈0.5–1 h | 同上（途中崩壊なら設定直し） |
| ANYmal-B ← C 寄せ | 1 | ≈0.5–1 h | 同上 |
| Spot ← Isaac Lab 高さ | 1 | ≈0.5–1 h | 上昇中なら 601→901 |
| Laikago ← RaisimGym | 1+ | ≈0.5–1 h × 試行 | 崩壊なら iter より URDF/姿勢/PD を先に |

直列・再試行なしなら **おおよそ 3〜6 時間**（301 のみ）。901 まで伸ばす台があると **半日前後**になりうる。

## 注意

- B1 は MorFiC の **URDF** が Genesis で沈む実績あり。制御パラメータだけ移植する。
- Spot / Laikago に MorFiC 公式の報酬設計は無い。Isaac Lab / RaisimGym の **高さ・PD・初期姿勢**を移植し、報酬は既存 Genesis env を使う。
- 採用中の専門家とは別ログディレクトリ。上書きしない。
- イテレーションの増やし方は `ITERATION_POLICY.md`。
