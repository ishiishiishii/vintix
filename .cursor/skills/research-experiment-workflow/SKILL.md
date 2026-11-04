---
name: research-experiment-workflow
description: >-
  Branch-per-experiment git workflow and GitHub Issue tracking (mirrored to Notion) for this
  quadruped RL research repo. Use when starting, resuming or finishing an experiment, when the
  user raises a research task / question / experiment idea, when discussing what to try next,
  or when the autopilot (autopilot/) runs a scheduled tick.
---

# Research experiment workflow

ユーザーの方針（原文）:

- 実験ごとにGitでブランチを切って作業を進める。
- 私が行った課題やあなたと壁打ちした課題、おこなうべき実験などはGihubのイシューを使って記録してほしい。
- イシューはNotionとも連携してNotionから簡単に見れるようにしたい。

`AGENTS.md`（実行はコンテナ内、稼働中プロセスに触れない、共通コードは変更せず実験ディレクトリで差し替える等）は常に優先する。

## 1. ブランチ（実験ごと）

- 新しい実験に着手したら `main` から `exp/<実験ディレクトリ名のハイフン表記>` を切る
  （例: `vintix_go2/go1_body_scale_zeroshot/` → `exp/go1-body-scale-zeroshot`）。
- 作業ツリーには他の作業の未コミット変更があることが多い。`git checkout -b` で切り替えるだけにし、
  stash / reset / clean はしない。`git add` は自分が作った・変えたファイルだけ。
- 稼働中のジョブが読むファイルがブランチ間で異なる場合は切り替えない（実行中のコードが変わるため）。
- 区切りごとにこまめにコミットする。`main` への取り込みはユーザーの確認後（PR を作って `Closes #<番号>` を書く）。
- push は SSH が使える環境から行う（コンテナには `ssh` がない。`SSH_GITHUB_SETUP.md`）。

## 2. GitHub Issue（課題・壁打ち・実験の記録）

リポジトリ: `ishiishiishii/vintix`。`gh` は `~/.local/bin/gh`、認証は `.env` の `GH_TOKEN`
（`set -a; source /workspace/.env; set +a`）。

いつ作るか:

| 状況 | ラベル |
|------|--------|
| ユーザーが課題・タスクを出した | `task` |
| ユーザーと壁打ちした論点・決定事項・未解決の疑問 | `discussion` |
| やるべき実験・追加実験の案 | `experiment` |
| ユーザーの判断が必要で止めている | `needs-decision` |
| 自律実行（autopilot）が着手してよい | `autopilot` |

- 本文は [issue-templates.md](issue-templates.md) の型に従う。リポジトリの Web 用テンプレートは `.github/ISSUE_TEMPLATE/`。
- 1 実験 = 1 Issue を基本にし、Issue 本文に「実験ディレクトリ」「ブランチ」「Notion レポート」のリンクを書く。
  逆に `README.md` / `REPORT.md` と Notion レポートには Issue 番号（`#12`）を書く。
- 進捗・結果・考察・壁打ちの結論は Issue コメントに残す（数値と図のパス付き）。完了したら close する。
- 既存 Issue と重複する場合は新設せずコメントで追記する。
- ラベルが無ければ作る:
  `gh label create task -c 1f77b4; gh label create discussion -c 9467bd; gh label create experiment -c 2ca02c; gh label create needs-decision -c d62728; gh label create autopilot -c ff7f0e`

## 3. Notion との連携

- 方式: Notion 標準の GitHub 連携（同期データベース）。「研究進捗」配下のページ
  「研究イシュー（GitHub 同期）」に `https://github.com/ishiishiishii/vintix/issues` の同期データベースを置く
  （ユーザーが Notion 上で一度だけ GitHub と接続して作成する）。Issue を作れば自動で Notion に表示されるので、
  Notion 側へ Issue を手で複製しない。
- 実験レポートは従来どおり `REPORT_FORMAT.md` に従い Notion「研究進捗」に置き、Issue の URL を本文に書く。

## 4. 実験の 1 サイクル（チェックリスト）

```
- [ ] Issue を確認／作成（task / experiment）
- [ ] ブランチ exp/<name> を切る
- [ ] vintix_go2/<name>/ に README.md（目的・設定・実行方法、Issue 番号）
- [ ] 実装（共通コードは変更せず import／継承で差し替え）→ コミット
- [ ] スモーク → 本番（長時間は nohup / setsid で切り離し、再開可能に）
- [ ] 区切りごとに Issue コメント（進捗・数値）
- [ ] REPORT.md + Notion レポート（REPORT_FORMAT.md）、Issue にリンク
- [ ] 次にやるべきことは新しい experiment Issue に切り出す
- [ ] ユーザー確認後に PR → main、Issue close
```

## 5. 自律実行（autopilot）

`autopilot/README.md` を参照。ON/OFF は `autopilot/autopilot.sh on|off|status`。
自律実行中も本スキルと `autopilot/PROMPT.md` のガードレールに従う。
