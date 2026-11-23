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

- [ ] 論文用の「同じロボット・同じ指標」での横並び表を 1 枚に固定（主表 = GenLoco + URMA）
- [ ] MorFiC を主表に残すなら別環境で Isaac Gym 再生（ユーザー判断。このマシンへの導入は明示後）
- [ ] URMA の GPU 再生（cuDNN 揃え）は任意。CPU 数値で主張は足りる

## 閉じ方の提案

次のどちらかで **close 可** とする（ユーザー合意待ちでも、案として固定）:

1. **最小完了（推奨）:** 主表が GenLoco + URMA の公式再生数値で埋まり、Genesis 近似は参考列、MorFiC は主表外と明記 → Issue を close
2. **厚め完了:** 上記 + MorFiC 公式数値（別マシン）まで

最小完了なら、残りはドキュメント整備のみで GPU 不要。#16（機体拡充）と切り分けて先に閉じられる。
