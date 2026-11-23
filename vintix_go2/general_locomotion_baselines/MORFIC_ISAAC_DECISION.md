# MorFiC 公式再生（Isaac Gym）の判断メモ

Issue: [#22](https://github.com/ishiishiishii/vintix/issues/22)  
更新: 2026-11-23（autopilot）

## いまの事実

- MorFiC の学習済み重みの **公式再生**は Isaac Gym + 同梱 `scripts/play.py`
- **このコンテナには Isaac Gym が入っていない**
- Genesis への近似配線はあるが、Go2・4 env・500 ステップで完走率 **0%**（即転倒）→ 配線差が主因の可能性が高く、手法そのものの評価には使えない

## 選択肢（ユーザー判断用）

| 案 | 内容 | 利点 | 欠点 |
|----|------|------|------|
| A | このマシンに Isaac Gym を入れる | 公式数値をここに揃えられる | インストールが重い・CUDA/ドライバ依存・既存環境を壊すリスク |
| B | 別マシン／別コンテナで公式再生だけ回し、JSON を持ち帰る | 本線コンテナを汚さない | 運用が分かれる・再現手順の維持が必要 |
| C | 第1波の比較から MorFiC を外し、GenLoco + URMA を主にする | すぐ論文表が書ける | MorFiC は文献調査に残るが数値なし |

## 推奨（自律探索の仮置き）

**当面は C（主表は GenLoco + URMA）＋ B を将来オプション**とする。

- 理由: GenLoco（PyBullet）と URMA（MuJoCo）はすでに公式側で「歩ける」数値がある。MorFiC だけ公式が欠ける状態で Isaac Gym 導入に時間を使うより、#16 の評価機体拡充の方が本線に効く
- A はユーザーが「このマシンで揃える」と明示したときだけ

## 判断待ちでも進めること

- GenLoco / URMA の公式数値の厚み（シード・ロボット数）
- 指標対応表（`METRICS_CROSSWALK.md`）
- MorFiC は Genesis 列を「参考・未公式」と明記したまま維持
