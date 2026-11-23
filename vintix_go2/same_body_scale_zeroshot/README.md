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
| 対象（評価の優先） | Go1 **完了**（2026-11-19）。次は **A1**（2026-11-23 起動）→ MC → Go2 |
| 評価 3 軸 | ①質量だけ ②大きさだけ（質量固定）③大きさ＋質量連動（密度一定） |
| Go1 モデル | `models/go1_data1M_5epochs/.../0004_epoch` |
| A1 モデル | `models/a1_data1M_5epochs/.../0004_epoch` |
| Go2 モデル | 既存 `models/go2_data1M_5epochs/.../0001_epoch` |
| 不足モデル | `train_missing_models.sh`（Go1→A1→MC）**学習完了 2026-11-18** |
| 評価格子 | Go1 mass/size = `go1_body_scale_zeroshot` FULL_GRID。A1/MC/Go2 = #10 `GRIDS`。size_density = #10 格子 |
| 比較 | Vintix 同体のみ再評価。PPO/Expert は過去実験を再利用 |
| 結果の置き場 | `same_body_scale_zeroshot/results/`（LOO 結果と混在させない） |
| 指標 | #15 の `*_metrics.json` / episodes loco 列を利用 |

## 実行

```bash
cd /workspace/vintix_go2
# 学習（完了済み）
setsid nohup bash same_body_scale_zeroshot/train_missing_models.sh \
  > same_body_scale_zeroshot/logs/train_all.out 2>&1 &

# Go1 同体・3軸 Vintix 評価（完了）
# 早期打ち切り: 完走≈0% の崖より遠い倍率は起動しない（2026-11-19）
setsid nohup python -u same_body_scale_zeroshot/eval_go1_pipeline.py \
  > same_body_scale_zeroshot/logs/eval_go1.log 2>&1 < /dev/null &
python same_body_scale_zeroshot/eval_go1_pipeline.py --dry_run

# A1 同体・3軸（#10 格子・early-stop。停止は STOP_A1）
setsid nohup python -u same_body_scale_zeroshot/eval_a1_pipeline.py \
  > same_body_scale_zeroshot/logs/eval_a1.log 2>&1 < /dev/null &
python same_body_scale_zeroshot/eval_a1_pipeline.py --dry_run
```

## 実行ログ

### 2026-11-17（autopilot tick 20261117_232448）

- 文献調査 6 本を上記表に記録
- 不足 3 台の dataset YAML と `train_missing_models.sh` を追加し学習を起動
- 次: 学習完了監視 → 評価

### 2026-11-18: 学習完了 → Go1 3軸評価を起動

- Go1/A1/MC 同体モデル 5 epoch 完了
- 方針: 評価は Go1・質量／大きさ／密度一定の 3 条件、Vintix のみ（#18 コメント）
- `eval_go1_pipeline.py` 起動（470 jobs、4 並列）。結果は `results/` へ
- 関連: シード差の「学習データ欠落」仮説は #20

### 2026-11-18 06:15（autopilot）: 評価監視と seed0 質量速報

- 進捗 8/470、failed なし。PID 709234。#21 waiter PID 715125
- seed0 質量: 0.9×94% / 1.0×92% / 1.1×63% / 1.15×以降 0%（LOO seed0 は 1.25×まで高完走）
- 暫定考察: 同体の重い側崖が LOO より内側に見えるが 1 シードのみ → 完走まで保留
- #20: LOO のシード差は「学習にそのシード軌跡が無い」では説明不可（Go1 軌跡自体がゼロ）

### 2026-11-19 01:00（autopilot idle_research）: 質量軸 7 シード完了・LOO 比較

- 進捗 **309/470**、failed なし。PID 709234（size seed6 進行中）。#21 waiter 継続
- **質量（同体・seed0–6）:** 0.6–1.0× で高完走（1.0× **91.1±0.6%**）→ 1.1× **62.9±3.1%** → **1.15× 以降全シード 0%**。軽い側 ≤0.5× も全 0%（0.55× のみ 16.9%）
- **LOO（#10・10 seeds）との差:** LOO は軽い側も高完走・重い側は **1.25× まで平均 ≥80%** → 同体の方が崖が**内側**（仮説「同体で通常体付近が広がる」は Go1 質量では不支持）
- **#20:** 同体は崖付近のシード分散がほぼゼロ（1.15× 全 0%）。LOO の 1.15–1.25× 大ばらつきは評価ノイズ一般では説明しにくい
- 図: `figures/go1_mass_samebody_vs_loo_completion_mean_std_20261119.png` / 表 `figures/go1_mass_samebody_vs_loo_table_20261119.md`
- 次: size / size_density 完走 → 3 軸図・Notion 上部の確定更新 → #21 自動学習

### 2026-11-19 — 早期打ち切り方針をパイプラインに反映

- **実行内容:** ユーザー指示どおり、完走≈0% の崖より遠い倍率をスキップ。`eval_go1_pipeline.py` 更新、`STOP` で旧ランチャ停止
- **結果:** dry-run で mass 0.5/1.15・size 0.55/1.15・size_density 0.8/1.1 が cliff。skip≈54・残り≈41
- **次:** 稼働ワーカー完了後に early-stop 版を再起動 → 残り近傍シード消化 → #21

### 2026-11-19 09:04（autopilot change）: 質量 10 シード確定・size 中間・残り 16

- **実行内容:** early-stop 再起動後の進捗を集計（PID 996590）。質量図を 10 シードで更新、size 比較図を新規作成。稼働ジョブには非介入
- **進捗:** **400/470**（skipped 54・remaining **16**）。cliffs mass 0.5/1.15・size 0.55/1.15・size_density 0.8/1.1。ETA ≈+1.5–2h
- **質量（同体・10 seeds）:** 0.6–1.0× 高完走（1.0× **91.2±0.7%**）→ 1.1× **61.0±4.6%** → **1.15× 全 0%**。≤0.5× も全 0%（0.55× 17.3±5.0%）
- **大きさ（暫定・8–10 seeds）:** 0.65–0.9× 高完走 → 1.1× ~59% → **1.15× 全 0%**。0.55× 以下は 0%。LOO より崖が内側（質量と同型）
- **密度一定（暫定・8 seeds）:** 0.9× ~98% / 0.8×・1.1× は 0%（崖点）。seed8–9 が残り
- **図:** `figures/go1_mass_samebody_vs_loo_completion_mean_std_20261119.png` / `figures/go1_size_samebody_vs_loo_completion_mean_std_20261119.png`
- **次:** 残り 16 完走 → size_density 確定図 → Notion 上部確定 → #21 自動学習

### 2026-11-19 11:12（autopilot change）: Go1 3軸評価完了・#21 学習起動

- **実行内容:** #18 は `done=416 + skipped=54 = 470` で完了（early-stop）。#21 waiter が skipped 未計上で永久待機していたため `wait_and_run.sh` を修正して再起動 → dataset 65 tasks → `train_vintix` 開始（PID 親 1056988 / train 1057017）
- **質量（同体・10 seeds・確定）:** 0.6–1.0× 高完走（1.0× **91.2±0.7%**）→ 1.1× **61.0±4.6%** → **1.15× 全 0%**。≤0.5× も全 0%
- **大きさ（10 seeds・確定）:** 0.65–0.9× 高完走 → 1.1× **58.9±2.9%** → **1.15× 全 0%**。0.55× 以下 0%。LOO より崖が内側
- **密度一定（10 seeds・確定）:** **0.9× 98.2±1.2%** のみ高完走。**0.8×・1.1× は全 0%**（LOO は 0.8× で 91.9%）。帯が最も狭い
- **考察:** Go1 では「同体学習でスケール帯が広がる」仮説は 3 軸とも不支持。LOO の多様性優位。特に密度一定 0.8× の差が大きい
- **図:** `figures/go1_{mass,size,size_density}_samebody_vs_loo_completion_mean_std_20261119.png`
- **次:** #21 学習完了 → Go1 3 軸評価 → 数直線レポート。他台の同体評価は後続

### 2026-11-23 07:29（autopilot idle_research）: A1 同体 3軸評価を起動

- **実行内容:** `eval_a1_pipeline.py` / `eval_a1_worker.py` を追加（#10 `GRIDS`・early-stop・`STOP_A1`）。親 PID **1861311**、4 並列で mass 1.0× seed0–3 から開始。モデル `a1_data1M_5epochs/.../0004_epoch`
- **進捗:** total **320**（done 0）。dry-run OK。GPU ≈100%
- **LOO 参照（#10 A1・10 seeds）:** 質量 1.0× **59.1±8.0%**・**1.1×〜0%**。大きさも 1.1×〜0%。密度一定は 0.9× 100% / 0.8× 58% / 1.1× 0%
- **見積もり:** early-stop 込み壁時計 ≈10–14h（崖は LOO どおり重い側 1.1× 付近の見込み）
- **次:** 質量近傍シード消化 → 崖確定 → size / size_density → 同体 vs LOO 図。MC/Go2 は後続

