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

### 2026-11-20 03:23（autopilot）: 学習完了 → early-stop 評価を再起動

- **学習:** 5/5 epoch 完了 → `models/.../0004_epoch`（85280 steps/ep）
- **問題:** 初回 `eval_go1_pipeline` に early-stop が無く、ジョブ順も seed→軸 だった（#18 と不一致）
- **対応:** #18 と同仕様の early-stop＋軸→|log scale|→seed を移植。wait_and_run ツリーを止めて
  `run_eval_and_report.sh`（PID **1100526**）で評価→数直線まで再開
- **学習分布（成功 65）:** 質量 10.63–19.52 kg、脚長 0.206–0.587 m
- **次:** 評価完走 → 数直線・同体/LOO 比較 → Notion 上部更新

### 2026-11-20 19:25（autopilot）: 質量途中結果 — 同体崖の外側まで高完走

- **進捗:** 68/470（mass 軸。1.25× seed8–9 / 1.3× seed0–1 走行中）。cliff 未確定。failed なし
- **質量（暫定）:** 0.8–1.0× **100%** → 1.15× **98.8±1.0%** → 1.2× **96.3±1.9%** → 1.25× **87.4±4.0%**（n=8）
- **比較:** 同体(#18) は **1.15× で全 0%**。narrow-DR は同倍率でほぼ完走 → 学習履歴の多様性が質量帯を広げる仮説と整合（暫定）
- **学習分布との対応:** Go1 1.0× ≈13.1 kg。学習質量 10.6–19.5 kg ≈ **0.81–1.49×**。いまの 1.25× は分布内。崖が分布外かは未
- **図:** `figures/go1_mass_narrow_vs_samebody_loo_completion_mean_std_20261120.png`
- **次:** 質量崖確定 → size / size_density → 数直線。成功ジョブが長く ETA は伸びやすい（≈4 jobs/h）

### 2026-11-21 11:33（autopilot）: 質量ほぼ確定 — 高側崖 1.5×、1.3× で LOO 超え

- **進捗:** 158+50skip/470。mass hi cliff=**1.5**（1.75×以降スキップ）。size 1.1× 開始。failed なし
- **質量（ほぼ確定・0.25× は n=8）:** 0.25–1.0× **100%** → 1.15× **98.8%** → 1.25× **87.8%** → **1.3× 57.7%** → 1.35× 10.9% → **1.5× 0%**
- **比較:** 同体は 1.15×で全滅。LOO は **1.3×で 0%** なのに narrow-DR は **57.7%** → 学習多様性が重い側を LOO より伸ばす（質量軸・暫定確定）
- **軽い側:** 同体は ≤0.5×で 0%、narrow-DR/LOO は 0.25×でも高完走
- **図:** `figures/go1_mass_narrow_vs_samebody_loo_completion_mean_std_20261121.png`
- **次:** size / size_density 完走 → 数直線。並行で #16 立位角配線済（`--run` は GPU 空き後）
