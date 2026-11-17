# 同体のみ学習 Vintix での体スケール評価（#18）

Issue: [#18](https://github.com/ishiishiishii/vintix/issues/18)  
ブランチ: `exp/robot-body-scale-zeroshot`（#10/#15 と同じ線。評価は `robot_body_scale_zeroshot` の格子を再利用）  
Notion: [4台体スケールのゼロショット限界](https://app.notion.com/p/41ad65e6123f816093d0ebc4bb281f25)

## 目的・問い

leave-one-out（`<robot>_without`、#10）と、**評価ロボット自身のデータだけで学習した Vintix** で、質量・大きさスケールの崖がどう変わるか。

## 仮説

同体学習の方が通常体付近で歩ける（特に Go2 / Mini Cheetah のように leave-one-out が 1.0× で失敗する台）。崖の位置も leave-one-out とずれる可能性がある。

## 文献調査（実装前・5本以上）

| 論文 | 変える量 | 指標 | 正規化 | 本実験への採否 |
|------|----------|------|--------|----------------|
| Laskin et al., Algorithm Distillation (ICLR 2023) | 多タスクの学習履歴を蒸留。単一タスク履歴でも in-context 改善が目的 | return / 新タスク汎化 | タスク横断の系列モデル | **採用**: 同体のみ = 単一タスク履歴の AD。leave-one-out との差は「履歴のロボット多様性」の効果 |
| Cross-Embodiment Offline RL (arXiv:2602.18025) | leave-one-out 事前学習 vs 単一 embodiment | 学習曲線・最終性能 | 異種データ混合 | **採用**: LOO 事前学習は fine-tune を速めるが、劣データ＋多様性では負の転移も。同体のみ比較の根拠 |
| ManyQuadrupeds (arXiv:2310.10486) | 16 体の多ロボット同時学習 vs 単体 | 速度・sim2real | CPG+IK で形態差を吸収 | **参考**: 多体共有の利点を示すが、本実験は AD/Transformer で形態条件なし |
| QWM morphology-conditioned WM (arXiv:2604.08780) | LOO で 1 台除外しゼロショット | reward / episode length（5 seed） | 形態ベクトル μ | **参考**: LOO プロトコル。本実験は形態条件なしの同体 vs LOO |
| MorFiC (arXiv:2603.14554) | 単一ソース＋形態 DR → 他形態ゼロショット | 速度・最長走行 | critic 側形態条件 | **参考**: 単一ソースでも転移可能という対照。本実験は「学習データが評価体そのものか」を切る |
| One Policy to Run Them All / URMA (CoRL 2024) | 多ロボット MTRL vs 単タスク | ゼロショット実機含む | 統一観測 | **参考**: 多体学習の利点。本実験の LOO 側に対応 |

設計決定:

- **比較軸**: #10 の leave-one-out 結果を正本にし、同体モデルで同じ格子・同じシードを評価
- **モデル条件**: 参考 `models/go2_data1M_5epochs`（epochs=5, context=2048, sparsity=128, seed=5, group=`go2_locomotion`）。不足 3 台は同条件で from-scratch
- **注意**: 参考フォルダ名は 5epochs だが実チェックポイントは `0000`/`0001` のみ → Go2 評価は最新 `0001_epoch` を使い、README に明記

## 方法・設定

| 項目 | 値 |
|------|-----|
| 対象 | Go1・Go2・A1・Mini Cheetah |
| Go2 モデル | 既存 `models/go2_data1M_5epochs/.../0001_epoch` |
| 不足モデル | 本ディレクトリ `train_missing_models.sh`（Go1→A1→MC、合計見積 ~6 GPU 時） |
| 評価格子 | #10 と同じ（`robot_body_scale_zeroshot/robots.py`） |
| 指標 | #15 の `*_metrics.json` / episodes loco 列を利用 |

## 実行

```bash
cd /workspace/vintix_go2
setsid nohup bash same_body_scale_zeroshot/train_missing_models.sh \
  > same_body_scale_zeroshot/logs/train_all.out 2>&1 &
```

学習完了後に Go2 から順に体スケール評価（オーケストレーションは次 tick）。

## 実行ログ

### 2026-11-17（autopilot tick 20261117_232448）

- 文献調査 6 本を上記表に記録
- 不足 3 台の dataset YAML と `train_missing_models.sh` を追加し学習を起動
- 次: 学習完了監視 → Go2 同体モデルで #10 格子の評価 → leave-one-out 差分図
