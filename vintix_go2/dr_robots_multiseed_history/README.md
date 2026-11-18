# dr_robots_multiseed_history: DR ロボット 200 体 × PPO 学習シード 10 の学習履歴（Vintix 用データ）

`dr_robots_ppo_history` で作った DR ロボット（狭い範囲 100 体 + 広い範囲 100 体）それぞれを、
PPO で学習シードを変えて最大 10 回ずつ学習し、その学習履歴（下手 → 上手の上達過程）を
Vintix の学習データとして保存する。

- Issue: #11
- 依頼: 「それぞれのロボットの体に対して、Train.py でシードを変えて 10 回位 PPO で普通のモデルを訓練して、
  それの履歴をデータとして保存して（Vintix の学習のデータセットとして用意しておきたい）」
- 開始: **#13 完了後**（キュー: #10 → #15 → #18 → #13 → **#11**）。`start_after_body_scale.sh` は #10 直後には起動しない（2026-11-16）

## 設定（`dr_robots_ppo_history/EXPERIMENT.md` と同じ）

| 項目 | 値 |
|------|-----|
| 学習 | `vintix_go2/scripts/train_with_history.py`（`train.py` と同じ PPO 学習に履歴の記録を足したもの）。4096 環境、301 反復 |
| 報酬 | 各体の `auto_reward_design.py` の自動設計（未適用の体だけ最初に 1 回実行） |
| 履歴の保存 | 先頭 256 環境の全遷移を HDF5（1 回あたり約 330 MB） |
| 学習シード | 1〜10。シード 1 は既存の配置（`model/`, `ppo_history/`）、シード k≥2 は `model_seed<k>/`, `ppo_history_seed<k>/`（各ロボットのパッケージ内） |
| 順番（2026-11-18） | **ロボット多様性優先・narrow 先**: 同一シードで narrow 100 体を一通り → シードを上げてまた 100 体。広い範囲は後回し（`--sets narrow` が既定）。同じロボットでシードを連続消化する順にはしない |
| 止まる条件 | ディスクの空きが 150 GB 未満、または `STOP` ファイル |

## 規模と所要時間

1 回 約 43 分（狭い範囲 1 体の実績）。**既定は narrow 100 体のみ**（wide は後回し・`--sets` で opt-in）。

- シード 1 の narrow 100 体は **既に済み**
- 次はシード 2 で同じ 100 体を一通り（≈ 100×43 分 ≈ **3 日**）、以降シードを増やして繰り返す
- 旧見積の 200 体×10 シード（≈60 日）は採らない。途中経過は `status.json` と Notion に記録

## 実行

```bash
cd /workspace/vintix_go2
python dr_robots_multiseed_history/collect.py --dry_run    # シードごとの完了数と空きディスク
setsid nohup python -u dr_robots_multiseed_history/collect.py > dr_robots_multiseed_history/logs/collect.log 2>&1 < /dev/null &
```

1 回ごとのログは `logs/<narrow|wide>/<robot_id>_seed<k>.log`、進捗は `status.json`。

## 実行ログ

### 2026-11-12 準備

- 実行内容: 既存の状況を確認（狭い範囲 100 体はシード 1 の学習と履歴が済み、広い範囲 100 体は以前の一括実行がスクリプトの文法エラーで始まっていなかった）。
  `collect.py` を実装し、`robot_body_scale_zeroshot` の完了を待って自動で始まるように起動
- 次: 体スケール実験の終了後に開始（#11）
