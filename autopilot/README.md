# autopilot: 定期的にエージェントを起動して研究を進める仕組み

コンテナ内で Cursor の CLI（`~/.local/bin/agent`）を一定間隔で非対話実行し、
実験の監視・結果の分析と考察・記録（Issue / REPORT.md / Notion）・次の実験の起動までを自律的に行う。
IDE やチャットを閉じていても動く（コンテナが動いている限り）。

## ON / OFF

```bash
cd /workspace
autopilot/autopilot.sh on        # 有効化。すぐ 1 回確認し、以後 8 時間ごと（config.env の INTERVAL_HOURS）
autopilot/autopilot.sh on 12     # 間隔を 12 時間にして有効化（次回以降も 12 時間）
autopilot/autopilot.sh off       # 無効化（実行中の AI は最後まで走る）
autopilot/autopilot.sh off --now # 実行中の AI も止める
autopilot/autopilot.sh status    # 状態・今日の AI 実行回数・最近の確認結果
autopilot/autopilot.sh check     # 次の確認で AI が呼ばれるか（AI は呼ばない）
autopilot/autopilot.sh run-once  # 変化がなくても AI を 1 回実行（動作確認用）
```

コンテナを再起動したらスケジューラは止まるので、必要なら `on` し直す。

## 利用枠を使いすぎない仕組み

| 仕組み | 内容（`config.env`） |
|--------|----------------------|
| 変化がなければ AI を呼ばない | 8 時間ごとの確認はスクリプト（`fingerprint.py`）だけで行い、前回の AI 実行のあとに Issue・コメント・ラベル、実験の進み具合・失敗、コミットのどれかが変わったときだけ AI を呼ぶ。確認だけなら Cursor の利用枠は使わない |
| 1 日の上限 | AI は 1 日 3 回まで（`MAX_AI_TICKS_PER_DAY`） |
| 自動停止（変化なし） | 変化のない確認が 3 回（24 時間）続いたら OFF（`AUTO_OFF_IDLE_CHECKS`） |
| 自動停止（判断待ち） | AI が 2 回続けて「判断待ちで進められない」と報告したら OFF（`AUTO_OFF_WAITING_TICKS`） |
| 停止の通知 | 自動で OFF になったら Issue #6 にコメント（理由と判断待ちの Issue 番号）。再開は判断を書いてから `on` |
| 記録を読みすぎない | AI は作業記録 `state/journal.md` の直近 5 回分だけ読む。結果が変わらない回は Notion に書かない |

確認の履歴は `logs/checks.log`、AI の各回の出力は `logs/tick_<日時>.log`。

## 初回の準備

- GitHub: `gh` に `vintix` 専用の Fine-grained token でログイン済み（同じ値が `/workspace/.env` の `GH_TOKEN`）
- Cursor: `agent login`（表示される URL をブラウザで開いて許可）。または `/workspace/.env` に `CURSOR_API_KEY=...`

## 1 回の実行でやること

`PROMPT.md` が指示書。要点:

1. `AGENTS.md` と `research-experiment-workflow` スキル、前回までの記録 `state/journal.md`、open Issue、稼働中のジョブを読む
2. 自分が起動したジョブの失敗に対処する
3. 新しい結果を集計し、分析・考察する
4. 結果・考察・次の実行内容を、研究の目的ごとの Notion ページ（`RESEARCH_OBJECTIVES.md` の台帳）の
   「実行ログ」と実験ディレクトリの `README.md` の「実行ログ」に追記し、Issue にもコメントする
   （実行ごとにページは作らず、同じ目的なら 1 ページに追記していく）
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
