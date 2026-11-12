# dr_robots_multiseed_history: DR ロボット 200 体 × PPO 学習シード 10 の学習履歴（Vintix 用データ）

`dr_robots_ppo_history` で作った DR ロボット（狭い範囲 100 体 + 広い範囲 100 体）それぞれを、
PPO で学習シードを変えて最大 10 回ずつ学習し、その学習履歴（下手 → 上手の上達過程）を
Vintix の学習データとして保存する。

- Issue: #11
- 依頼: 「それぞれのロボットの体に対して、Train.py でシードを変えて 10 回位 PPO で普通のモデルを訓練して、
  それの履歴をデータとして保存して（Vintix の学習のデータセットとして用意しておきたい）」
- 開始: `robot_body_scale_zeroshot` のパイプラインが終わったら自動で始まる（`start_after_body_scale.sh`）

## 設定（`dr_robots_ppo_history/EXPERIMENT.md` と同じ）

| 項目 | 値 |
|------|-----|
| 学習 | `vintix_go2/scripts/train_with_history.py`（`train.py` と同じ PPO 学習に履歴の記録を足したもの）。4096 環境、301 反復 |
| 報酬 | 各体の `auto_reward_design.py` の自動設計（未適用の体だけ最初に 1 回実行） |
| 履歴の保存 | 先頭 256 環境の全遷移を HDF5（1 回あたり約 330 MB） |
| 学習シード | 1〜10。シード 1 は既存の配置（`model/`, `ppo_history/`）、シード k≥2 は `model_seed<k>/`, `ppo_history_seed<k>/`（各ロボットのパッケージ内） |
| 順番 | シード順（全 200 体のシード k が終わってから k+1）。いつ止めても体の偏りが出ない |
| 止まる条件 | ディスクの空きが 150 GB 未満、または `STOP` ファイル |

## 規模と所要時間

1 回 約 43 分（狭い範囲 100 体の実績）。シード 1 は狭い範囲 100 体が既に済んでいて、残りは広い範囲 100 体（約 3 日）。
シード 2 以降は 200 体で 1 シードあたり約 6 日。10 シードすべてだと約 60 日・約 650 GB になるので、
シード順に進め、途中経過は status.json と Notion に記録する。

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
