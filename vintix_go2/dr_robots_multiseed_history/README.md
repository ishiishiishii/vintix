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
- シード 2: **75/100**（2026-11-26 08:52 UTC 時点。いま `dr_narrow_seed076`）。再開後も中央 ≈**22 分/体** → 残り ≈**9 時間**（完了見込み ≈2026-11-26 18:15 UTC）
- 以降シードを増やして同じ 100 体を繰り返す。旧見積の 200 体×10 シード（≈60 日）は採らない。途中経過は `status.json` と Notion に記録

## 実行

```bash
cd /workspace/vintix_go2
# Genesis / rsl-rl-lib が入っている base conda を使う（isaacgym_py38 だと import 失敗する）
/opt/conda/bin/python dr_robots_multiseed_history/collect.py --dry_run
PATH="/opt/conda/bin:$PATH" setsid nohup /opt/conda/bin/python -u dr_robots_multiseed_history/collect.py \
  > dr_robots_multiseed_history/logs/collect.log 2>&1 < /dev/null &
```

子プロセスも起動時のインタプリタ（`sys.executable`）を使う。

1 回ごとのログは `logs/<narrow|wide>/<robot_id>_seed<k>.log`、進捗は `status.json`。

## 実行ログ

### 2026-11-26 09:01 UTC: seed2 が 75/100 まで進んだ

- **分かったこと:** 前回記録の 54/100 から約 8 時間で **75/100**。失敗なし。ペースは再開後と同程度（055–075 の中央 ≈22.1 分/体）
- **実行内容:** 監視のみ（collect PID **2118916** / 現ジョブ `dr_narrow_seed076` seed2、iteration ≈115/301）
- **結果:** `status.json` seed2 done=**75**/100、done_total=**175**/1000、空き ≈809 GB。残り ≈25 体 × ≈22 分 ≈ **9 時間**（完了見込み ≈2026-11-26 18:15 UTC）。成果物確認: `dr_narrow_seed075` に `model_seed2/model_300.pt` と `ppo_history_seed2/`（≈334 MB）
- **次:** seed2 完了で再記録 → seed3

### 2026-11-26 00:59 UTC: seed2 が 50/100 を超え 54/100

- **分かったこと:** #16 後の再開（33/100）から約 8 時間で **54/100**。失敗なし。ペースは再開前と同程度（中央 ≈22 分/体）
- **実行内容:** 監視のみ（collect PID **2118916** / 現ジョブ `dr_narrow_seed055` seed2）
- **結果:** `status.json` seed2 done=**54**/100、done_total=**154**/1000、空き ≈817 GB。残り ≈46 体 × ≈22 分 ≈ **17 時間**（完了見込み ≈2026-11-26 17:40 UTC）
- **次:** seed2 完了で再記録 → seed3

### 2026-11-25 16:54 UTC: #16 完了後に seed2 収集を再開（33/100）

- **分かったこと:** #16 で Laikago を含む 5 機専門家が揃った。#11 は `STOP` で seed2 **33/100** で止まっていた
- **実行内容:** `STOP` 削除後、PATH 先頭の `isaacgym_py38` だと `h5py`/`rsl-rl-lib` が無く即失敗したため、`/opt/conda/bin/python` で再起動。`collect.py` は子も `sys.executable` を使うよう修正
- **結果:** `dr_narrow_seed034` seed2 が学習中（PID **2118916** / 子 **2118917**）。空き ≈825 GB。残り 67 体 × ≈23 分 ≈ **26 時間**（完了見込み ≈2026-11-26 19:00 UTC）
- **次:** seed2 完了で再記録 → seed3。区切りは 50/100 と完了時

### 2026-11-25 00:44 UTC: seed2 が 23/100 まで進んだ

- **分かったこと:** 起動から約 9 時間で 23 体完了。1 体 ≈23 分で安定。履歴・ckpt 確認済み（失敗なし）
- **実行内容:** 監視のみ（collect PID **2031757** / 現ジョブ `dr_narrow_seed024` seed2）
- **結果:** `status.json` done_total=123/1000、空き ≈831 GB。seed2 完了見込み ≈2026-11-26 06:00 UTC
- **次:** 50/100（≈11:00 UTC）と seed2 完了で再記録 → seed3

### 2026-11-24 15:55 UTC: シード2 収集開始（#16 第1波クローズ後）

- **分かったこと:** #16 第1波を合格 4 機で区切ったあと、合意キューどおり #11 に移行。seed1 narrow 100 体は済み、seed2 は 0/100
- **実行内容:** `setsid nohup python -u collect.py`（PID **2031754**）。先頭は `dr_narrow_seed001` seed2（`model_seed2/` / `ppo_history_seed2/`）
- **結果:** ログ `logs/collect.log`。空きディスク ≈839 GB（停止閾値 150 GB）
- **次:** seed2×100 体を監視（1 体 ≈43 分・全体 ≈3 日）。区切りごとに Notion / Issue #11 へ進捗

### 2026-11-12 準備

- 実行内容: 既存の状況を確認（狭い範囲 100 体はシード 1 の学習と履歴が済み、広い範囲 100 体は以前の一括実行がスクリプトの文法エラーで始まっていなかった）。
  `collect.py` を実装し、`robot_body_scale_zeroshot` の完了を待って自動で始まるように起動
- 次: 体スケール実験の終了後に開始（#11）
