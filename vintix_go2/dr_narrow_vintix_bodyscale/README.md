# DR narrow 履歴で学習した Vintix の Go1 体スケール評価

Issue: [#21](https://github.com/ishiishiishii/vintix/issues/21)  
ブランチ: `exp/dr-narrow-vintix-bodyscale`  
Notion: [DR narrow履歴Vintixの体スケール限界](https://app.notion.com/p/DR-narrow-Vintix-41cd65e6123f81d3b85ddb5c838740d0)（`41cd65e6-123f-81d3-b85d-db5c838740d0`）

## 目的・問い

すでに得られた **narrow 100 体**の PPO 学習履歴のうち成功体だけを使い、Vintix を新規学習したとき、  
Go1 の質量／大きさ／密度一定（3 軸）でどこまで歩けるか。学習に使った質量・脚長の範囲を数直線に示し、崖と対応付ける。

## 仮説

形態 DR 由来の多様な履歴は、同体のみ（#18）より体スケール崖を外側へ押し、leave-one-out（#10）とも異なる境界を持つ。崖が学習分布の外に出るなら範囲拡張が次、分布の内で落ちるなら表現・正規化の問題が疑わしい。

## 文献調査（実装前・5本以上）

| 論文 | 変える量 | 指標 | 正規化 | 本実験への採否 |
|------|----------|------|--------|----------------|
| GenLoco (CoRL 2022, arXiv:2209.05309) | 形態テンプレの α・リンク寸法・質量・摩擦・PD 等をエピソードごとに乱択 | 速度・実機ゼロショット | 形態条件なしの単一方策 | **採用**: narrow プロファイルの出典。成功履歴の多様性が汎化の前提 |
| ManyQuadrupeds (arXiv:2310.10486) | 2–200 kg・身長・DoF の広い多体 | 速度・sim2real | CPG+Pattern Formation | **参考**: 広い質量範囲の単一方策。本実験は AD/Transformer |
| MorFiC (arXiv:2603.14554) | 単一ソース＋形態 DR → 他形態ゼロショット | 速度・最長走行 | critic 側形態 FiLM | **参考**: DR→ゼロショットの問い。本実験は形態条件なし AD |
| Algorithm Distillation (ICLR 2023) | 多タスク学習履歴の系列蒸留 | return / in-context 改善 | タスク横断 | **採用**: 履歴を Transformer に蒸留する枠組み |
| Vintix (ICML 2025, arXiv:2501.19400) | クロスドメイン AD で行動モデル | in-context 適応 | タスク／グループ | **採用**: 実装・学習ハイパーの基準（epochs/context/sparsity） |
| McARL (arXiv:2505.18418) | 形態・制御パラメータを意識した汎化 | 多形態歩行 | 形態条件付き | **参考**: 形態条件の別解。本実験は条件なしで崖を測る |

設計決定:

- **データ**: narrow 成功体のみ（`reward_last ≥ 15` → 現状 65/100）。失敗体の崩れた履歴は入れない
- **グループ名**: `quadruped_locomotion`（Go1 未知タスク追加時に共有デコーダへ載せるため）
- **評価**: #18 と同じ Go1 3 軸格子・10 seeds。比較対象は同体 Vintix と leave-one-out
- **数直線**: 学習体の質量 [kg]・脚長 [m] の範囲＋評価 Success rate の対応（物理量軸。倍率図は別）

## 方法・設定

| 項目 | 値 |
|------|-----|
| 履歴元 | `dr_robots_ppo_history/dr_narrow_genloco/generated/*/ppo_history` |
| フィルタ | `outputs/dr_narrow_expert_rewards.csv` の reward_last ≥ 15 |
| 学習 | `train.sh` → epochs=5, context=2048, sparsity=128, seed=5 |
| モデル | `models/dr_narrow_success_5epochs/.../0004_epoch` |
| 評価 | `eval_go1_pipeline.py`（#18 と同格子） |
| 起動条件 | **#18 Go1 評価が終わってから**（`wait_and_run.sh`） |
| レポート | `REPORT.md` ＋ Notion 同ページ。数直線は `figures/` |

## 実行

```bash
cd /workspace/vintix_go2
# #18 完了を待ってから学習→評価→レポート（推奨）
setsid nohup bash dr_narrow_vintix_bodyscale/wait_and_run.sh \
  > dr_narrow_vintix_bodyscale/logs/wait_and_run.out 2>&1 < /dev/null &

# 個別
python dr_narrow_vintix_bodyscale/build_dataset.py
bash dr_narrow_vintix_bodyscale/train.sh
python -u dr_narrow_vintix_bodyscale/eval_go1_pipeline.py
python dr_narrow_vintix_bodyscale/make_numberline_report.py
```

## 実行ログ

### 2026-11-18: セットアップ（#18 評価中のため学習は待機）

- Issue #21・実験ディレクトリ・Notion ページを作成
- `wait_and_run.sh` で Go1 同体 3 軸評価完了後に学習を開始するよう予約
- データは既存 narrow 成功履歴を流用（新規 PPO 収集なし）

### 2026-11-19 11:12（autopilot）: #18 完了検知修正 → 学習起動

- **原因:** waiter が `done>=total` のみ見ており、early-stop の `skipped=54` を無視 → #18 完了後も永久待機
- **修正:** `wait_and_run.sh` で `done+skipped >= total` に変更。stuck waiter を止めて再起動
- **結果:** dataset 65 tasks 構築後、`train_vintix`（5 epochs）開始。親 PID 1056988 / train PID 1057017
- **次:** 学習完了 → Go1 3 軸評価 → `make_numberline_report.py`
