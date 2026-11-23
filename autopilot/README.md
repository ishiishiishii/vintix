# autopilot: 定期的にエージェントを起動して研究を進める仕組み

コンテナ内で Cursor の CLI（`~/.local/bin/agent`）を一定間隔で非対話実行し、
実験の監視・結果の分析と考察・記録（Issue / REPORT.md / Notion）・次の実験の起動までを自律的に行う。
IDE やチャットを閉じていても動く（コンテナが動いている限り）。

ユーザー起票の Issue を消化するのが基本。**手が空いたら** `RESEARCH_GOAL.md` を読んで、
文献調査と考察から次の実験を自分で起こす（idle_research）。

**実験中でも非実験は進める（2026-11-18）:** Issue は「実験」（学習・大規模評価・データ収集）と
「非実験」（文献調査・設計・小さな実装・記録・壁打ち整理など）に分かれる。
長時間の実験ジョブが走っていても、GPU を大量に使わない／実験不要なタスクは後回しにせず同じ tick で進める。
実験待機だけを理由に `waiting` にしない（詳細は `PROMPT.md`）。

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
設定（`config.env` / `PROMPT.md`）を変えたあとも、反映には `off` → `on` が必要
（実験ジョブは止まらない。デーモンだけ差し替える）。

## 利用枠を使いすぎない仕組み

| 仕組み | 内容（`config.env`） |
|--------|----------------------|
| 動的間隔（2026-11-18） | 非実験タスクあり → **30 分**（`INTERVAL_MINUTES_ACTIVE`）。実験待ちだけ → **8 時間**（`INTERVAL_HOURS_DEFAULT`）。終了見込みが分かるときは `state/experiment_eta_epoch` まで寝て、完了後は 8 時間に戻す（`choose_schedule.py`） |
| 変化がなければ AI を呼ばない | 確認は安価な `fingerprint.py`。変化時または idle_research 条件で AI を呼ぶ |
| Issue → Notion DB | `sync_notion_issues_db.py` で GitHub の OPEN/CLOSED を Notion「状態」に同期。ページ内は**オープン表**と**クローズ表**に分けて表示（全件表は参照用） |
| 待機時の自律探索 | 変化のない確認が `IDLE_RESEARCH_AFTER_CHECKS`（既定 2）回続いたら `idle_research` |
| 1 日の上限 | 待ちモード 3 回（`MAX_AI_TICKS_PER_DAY_WAITING`）／アクティブ 24 回（`MAX_AI_TICKS_PER_DAY_ACTIVE`） |
| 自動停止（変化なし） | 変化のない確認が 6 回続き、かつ idle research も発火できなかったら OFF（`AUTO_OFF_IDLE_CHECKS`） |
| 自動停止（判断待ち） | AI が 2 回続けて「判断待ちで進められない」と報告したら OFF（`AUTO_OFF_WAITING_TICKS`）。`needs-decision` があっても別の自律一歩が取れるなら `waiting` にしない |
| 停止の通知 | 自動で OFF になったら Issue #6 にコメント（理由と判断待ちの Issue 番号）。再開は判断を書いてから `on` |
| 記録を読みすぎない | AI は作業記録 `state/journal.md` の直近 5 回分だけ読む。結果が変わらない回は Notion に書かない |

確認の履歴は `logs/checks.log`、AI の各回の出力は `logs/tick_<日時>.log`。

## 初回の準備

- GitHub: `gh` に `vintix` 専用の Fine-grained token でログイン済み（同じ値が `/workspace/.env` の `GH_TOKEN`）
- Cursor: `agent login`（表示される URL をブラウザで開いて許可）。または `/workspace/.env` に `CURSOR_API_KEY=...`

## 1 回の実行でやること

`PROMPT.md` が指示書。要点:

1. `AGENTS.md` と `research-experiment-workflow` スキル、`RESEARCH_GOAL.md`、前回までの記録 `state/journal.md`、open Issue、稼働中のジョブを読む
2. 自分が起動したジョブの失敗に対処する（他人のプロセスには触れない）
3. 新しい結果を集計し、分析・考察する
4. **非実験タスクを並行消化**（調査・小さな実装・記録など。実験ジョブ実行中でも可）
5. 結果・考察・次の実行内容を、研究の目的ごとの Notion ページ（`RESEARCH_OBJECTIVES.md` の台帳）の
   「実行ログ」と実験ディレクトリの `README.md` の「実行ログ」に追記し、Issue にもコメントする
6. ユーザー起票・既存手順の次実験を Issue にし、条件を満たせば起動する（GPU 空き・予算ありのとき）
7. **手が空いていれば自律探索**: ゴールと主仮説に効く次の 1 手を文献調査のうえで Issue 化し、
   ガードレール内なら起動する（1 tick で本番は高々 1 本）
8. `state/journal.md` に記録して終了する

## ガードレール（`PROMPT.md` に記載）

- 新しい実験を相談なしで起動してよいのは、`autopilot` ラベルの Issue の範囲内か既存実験の次の手順、
  または自律探索で作った `autopilot` 付き Issue で、見積もり GPU 時間が `config.env` の
  `MAX_NEW_GPU_HOURS`（既定 12 時間）以下、GPU に空きがあり、共通コードを変えない場合だけ
- 大きすぎる・方針が割れる案は `needs-decision` を付けて Issue だけ作り、小さい代替があればそちらを進める
- 自分が起動していないプロセスには触れない、データを消さない、`main` を書き換えない、秘密情報を書き出さない

## ファイル

| ファイル | 内容 |
|----------|------|
| `autopilot.sh` | ON/OFF・スケジューラ・1 回の実行 |
| `PROMPT.md` | 毎回エージェントに渡す指示書（ガードレール・自律探索含む） |
| `config.env` | 間隔・タイムアウト・GPU 予算・idle research・モデル |
| `../RESEARCH_GOAL.md` | 北極星と自律探索の優先順位 |
| `state/` | 有効フラグ、PID、`journal.md`（Git 管理外） |
| `logs/` | 各回の出力 `tick_<日時>.log`（Git 管理外） |
