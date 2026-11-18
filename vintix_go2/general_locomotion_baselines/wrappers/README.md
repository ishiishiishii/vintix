# Genesis 評価ラッパ（骨格）

各手法の公式コードは Isaac Gym / PyBullet 前提が多い。ここでは **共通コードを改変せず**、
本リポの評価入口（例: `Genesis/examples/locomotion/eval.py` 相当の観測・行動）へ接続する薄い層を置く。

## レイアウト

```
wrappers/
  obs_action_map.md   # 手法ごと観測・行動次元の対応表
  morfic/             # P0: 70×30 / actor 2166 — CPU スモーク済
  genloco/            # P0: 451→12 — CPU スモーク済
  urma/               # P0: Go2 obs 380 — 寸法スモーク済（actor は RL-X 待ち）
  mcarl/              # P1: コード無し
```

## CPU スモーク（GPU 不要）

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/morfic/smoke_cpu_check.py
python wrappers/genloco/smoke_cpu_check.py
python wrappers/urma/smoke_cpu_check.py
```

Genesis Env 接続・pretrained 長時間再生は #18 GPU 空き後。

## 制約

- 重い学習・大規模評価は #18/#21 と GPU を分けてから
- 完全なシミュレータ互換は目指さない。差分は NOTES に書く
