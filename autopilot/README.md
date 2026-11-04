# autopilot: 定期的にエージェントを起動して研究を進める仕組み

コンテナ内で Cursor の CLI（`~/.local/bin/agent`）を一定間隔で非対話実行し、
実験の監視・結果の分析と考察・記録（Issue / REPORT.md / Notion）・次の実験の起動までを自律的に行う。
IDE やチャットを閉じていても動く（コンテナが動いている限り）。

## ON / OFF

```bash
cd /workspace
autopilot/autopilot.sh on        # 有効化。すぐ 1 回実行し、以後 config.env の間隔（既定 2 時間）ごと
autopilot/autopilot.sh on 4      # 間隔を 4 時間にして有効化（次回以降も 4 時間）
autopilot/autopilot.sh off       # 無効化（実行中の 1 回は最後まで走る）
autopilot/autopilot.sh off --now # 実行中の 1 回も止める
autopilot/autopilot.sh status    # 状態と最後のログ
autopilot/autopilot.sh run-once  # 1 回だけ手元で実行（動作確認用）
```

コンテナを再起動したらスケジューラは止まるので、必要なら `on` し直す。

## 初回の準備（ユーザー）

`/workspace/.env`（Git 管理外）に次を書く:

```bash
CURSOR_API_KEY=...   # Cursor のダッシュボードで発行（Integrations / API Keys）
GH_TOKEN=...         # GitHub の Fine-grained token（ishiishiishii/vintix に Issues: Read and write、Contents: Read）
```

## 1 回の実行でやること

`PROMPT.md` が指示書。要点:

1. `AGENTS.md` と `research-experiment-workflow` スキル、前回までの記録 `state/journal.md`、open Issue、稼働中のジョブを読む
2. 自分が起動したジョブの失敗に対処する
3. 新しい結果を集計し、分析・考察する
4. Issue に進捗コメント、区切りなら REPORT.md と Notion を更新する
5. 次に必要な実験を Issue にする。条件を満たせばそのまま起動する
6. `state/journal.md` に記録して終了する

## ガードレール（`PROMPT.md` に記載）

- 新しい実験を相談なしで起動してよいのは、`autopilot` ラベルの Issue の範囲内か既存実験の次の手順で、
  見積もり GPU 時間が `config.env` の `MAX_NEW_GPU_HOURS`（既定 12 時間）以下、GPU に空きがあり、共通コードを変えない場合だけ
- それ以外は Issue に `needs-decision` を付けて止まる → ユーザーが判断してラベルを外す／コメントで指示
- 自分が起動していないプロセスには触れない、データを消さない、`main` を書き換えない、秘密情報を書き出さない

## ファイル

| ファイル | 内容 |
|----------|------|
| `autopilot.sh` | ON/OFF・スケジューラ・1 回の実行 |
| `PROMPT.md` | 毎回エージェントに渡す指示書（ガードレール含む） |
| `config.env` | 間隔・タイムアウト・GPU 予算・モデル |
| `state/` | 有効フラグ、PID、`journal.md`（Git 管理外） |
| `logs/` | 各回の出力 `tick_<日時>.log`（Git 管理外） |
