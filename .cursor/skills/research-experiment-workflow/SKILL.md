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
- 実験の記録は研究の目的ごとの Notion ページに置き（§4）、Issue の URL を本文に書く。

## 4. 記録: 研究の目的ごとに Notion ページ 1 つ + 実験ディレクトリの README

ユーザーの方針（原文）: 自立実行の際はその都度結果、考察、次の実行内容などを実験内容ごとにページを用意してNotionに記録しつつ（各実行ごとにページを変える必要はなく、大まかな実験の目的ごとにページを作れば良い。今回のようにどれくらい体が違うとアルゴリズム蒸留だけではうまく適応できないかを調べるという一つの目的として複数の実験がある際は、1ページにどんどん記録していく）、実験ファイルにReadmeとしても残しておいてほしい

- 目的と記録先の対応はリポジトリ直下の `RESEARCH_OBJECTIVES.md`（台帳）。まずここで該当する目的を探す。
  無い目的なら台帳に行を足し、Notion「研究進捗」の M1 配下（`REPORT_FORMAT.md` §7.1 の位置）にページを 1 つ作る。
- 実験や実行が増えてもページは増やさない。同じ目的のページの末尾「## 実行ログ」に、新しいものが下になる順で
  1 エントリずつ追記する（Notion は `insert_content` の `position: end`）。エントリの形:

  ```markdown
  ### YYYY-MM-DD <短い見出し> {toggle="true"}
  	- 実行内容: …（何を、どの設定で、ジョブ数・GPU 時間）
  	- 結果: …（数値、図のパス）
  	- 考察: …（言えること／まだ言えないこと）
  	- 次の実行内容: …（Issue 番号）
  ```
- ページ冒頭の要約（目的・やったこと・これから・代表の表と図）は、結果が変わったら書き換えて最新に保つ。
  構成は `REPORT_FORMAT.md` に従う。
- 同じエントリを実験ディレクトリの `README.md` 末尾「## 実行ログ」にも追記する（Notion のトグルは Markdown の見出しと箇条書きにする）。
  `REPORT.md` は区切りのまとめとして更新する。

## 5. 実験の 1 サイクル（チェックリスト）

```
- [ ] Issue を確認／作成（task / experiment）、RESEARCH_OBJECTIVES.md で目的と記録先を確認
- [ ] ブランチ exp/<name> を切る
- [ ] vintix_go2/<name>/ に README.md（目的・設定・実行方法、Issue 番号、末尾に「実行ログ」）
- [ ] 実装（共通コードは変更せず import／継承で差し替え）→ コミット
- [ ] スモーク → 本番（長時間は nohup / setsid で切り離し、再開可能に）
- [ ] 区切りごとに: Issue コメント、目的の Notion ページと README の「実行ログ」に追記
- [ ] 区切りのまとめ: REPORT.md、Notion ページ冒頭の要約を更新
- [ ] 次にやるべきことは新しい experiment Issue に切り出す
- [ ] ユーザー確認後に PR → main、Issue close
```

## 6. 自律実行（autopilot）

`autopilot/README.md` を参照。ON/OFF は `autopilot/autopilot.sh on|off|status`。
自律実行中も本スキルと `autopilot/PROMPT.md` のガードレールに従う。
