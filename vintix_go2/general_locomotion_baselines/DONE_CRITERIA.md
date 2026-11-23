# #22 完了条件（案）

更新: 2026-11-23（autopilot）。分かりやすい言葉で書く。

## いま満たしていること

- [x] 代表手法の文献調査表（`LITERATURE_SURVEY.md`）
- [x] GenLoco / URMA / MorFiC の OSS クローン手順とライセンス記録
- [x] Genesis への薄い配線（近似評価）と軽量スモーク
- [x] **公式に近いシミュレータ**での数値:
  - GenLoco（PyBullet）: Go1 pace・300 ステップ生存 3/3
  - URMA（MuJoCo / RL-X）: Go2・1000 ステップ生存 3/3、平均 return ≈53.4
- [x] 本線指標との対応表（`METRICS_CROSSWALK.md`）
- [x] MorFiC の Isaac Gym は仮置きで主表外（`MORFIC_ISAAC_DECISION.md`・案 C）

## まだ足りない／任意

- [x] 論文用の「同じ見方」の主表を 1 枚に固定 → [`results/MAIN_TABLE.md`](results/MAIN_TABLE.md)（主表 = GenLoco + URMA、MorFiC は参考外）
- [ ] MorFiC を主表に残すなら別環境で Isaac Gym 再生（ユーザー判断。このマシンへの導入は明示後）
- [ ] URMA の GPU 再生（cuDNN 揃え）は任意。CPU 数値で主張は足りる

## 閉じ方の提案

**最小完了の文書条件は満たした**（2026-11-23）。ユーザー合意があれば #22 を close 可:

1. **最小完了（推奨・文書済）:** 主表 GenLoco + URMA の公式再生、Genesis 近似は参考、MorFiC は主表外
2. **厚め完了:** 上記 + MorFiC 公式数値（別マシン）まで

GPU は #16 Aliengo 履歴収集を優先。#22 は合意待ちで閉じられる。
