---
name: research-experiment-workflow
description: >-
  Branch-per-experiment git workflow and GitHub Issue tracking (mirrored to Notion) for this
  quadruped RL research repo. Use when starting, resuming or finishing an experiment, when the
  user raises a research task / question / experiment idea, when discussing what to try next,
  when making or updating experiment summary figures / Notion reports, or when the autopilot
  (autopilot/) runs a scheduled tick. Before implementing experiment design, survey 5+ related
  papers (literature-design.md). Figure style: figure-style.md. Notion writing: notion-page-format.md.
---

# Research experiment workflow

ユーザーの方針（原文）:

- 実験ごとにGitでブランチを切って作業を進める。
- 私が行った課題やあなたと壁打ちした課題、おこなうべき実験などはGihubのイシューを使って記録してほしい。
- イシューはNotionとも連携してNotionから簡単に見れるようにしたい。
- 実験内容や方針を実装する際には、ネット上で論文（ロボット汎化やクロスロボット適応などテーマに近いものが望ましい）を
  5 つ以上調査し、変えるパラメータ・評価指標・正規化などの詳細設計を把握したうえで、妥当な設計を担保する。

`AGENTS.md`（実行はコンテナ内、稼働中プロセスに触れない、共通コードは変更せず実験ディレクトリで差し替える等）は常に優先する。

## 1. ブランチ（実験ごと）

- 新しい実験に着手したら `main` から `exp/<実験ディレクトリ名のハイフン表記>` を切る
  （例: `vintix_go2/go1_body_scale_zeroshot/` → `exp/go1-body-scale-zeroshot`）。
- 作業ツリーには他の作業の未コミット変更があることが多い。`git checkout -b` で切り替えるだけにし、
  stash / reset / clean はしない。`git add` は自分が作った・変えたファイルだけ。
- 稼働中のジョブが読むファイルがブランチ間で異なる場合は切り替えない（実行中のコードが変わるため）。
- 区切りごとにこまめにコミットする。`main` への取り込みはユーザーの確認後（PR を作って `Closes #<番号>` を書く）。
- push はコンテナから HTTPS で行う（コンテナには `ssh` がない。`origin` は SSH の URL のまま変えない）:
  `git push https://github.com/ishiishiishii/vintix.git <branch>`（`gh auth setup-git` 済みなので認証は `gh` のトークン）。
  force push はしない。

## 2. GitHub Issue（課題・壁打ち・実験の記録）

リポジトリ: `ishiishiishii/vintix`。`gh` は `~/.local/bin/gh`。認証は `vintix` だけに限定した
Fine-grained token（`gh` にログイン済み。同じ値が `/workspace/.env` の `GH_TOKEN`）。
**`vintix` 以外のリポジトリ（研究室の組織 `kawakera-lab` を含む）は操作しない**。トークンも `vintix` 以外には効かない。
トークンの期限切れで 401 になったら、ユーザーに作り直しを頼む（手順はユーザーに説明済み: Resource owner `ishiishiishii`、
`vintix` のみ、Contents / Issues / Pull requests を Read and write）。

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

- 方式: Notion「研究進捗」配下のページ「研究イシュー（GitHub 同期）」（page_id `40ed65e6-123f-81a9-80c3-ed7590337e8d`）に
  Issue の一覧表（# / 題名リンク / ラベル / 状態 / 更新日 / 要約）を置き、**Issue を作成・更新・close したら毎回作り直す**
  （`gh issue list -R ishiishiishii/vintix --state all --json number,title,state,labels,updatedAt,url` を元に
  `API-update-page-markdown` の `replace_content`、`allow_deleting_content: true`。冒頭の説明文・「最終同期」の日時・ラベル表も保つ）。
  **表は Markdown の pipe 表**にする（HTML `<table>` は途中で切れて行が消えることがある）。
  autopilot は毎回の最後にも同期する。Notion API ではデータベースを新規作成できないため表で代用している。
- 実験の記録は研究の目的ごとの Notion ページに置き（§4）、Issue の URL を本文に書く。

## 4. 記録: 研究の目的ごとに Notion ページ 1 つ + 実験ディレクトリの README

ユーザーの方針（原文）: 自立実行の際はその都度結果、考察、次の実行内容などを実験内容ごとにページを用意してNotionに記録しつつ（各実行ごとにページを変える必要はなく、大まかな実験の目的ごとにページを作れば良い。今回のようにどれくらい体が違うとアルゴリズム蒸留だけではうまく適応できないかを調べるという一つの目的として複数の実験がある際は、1ページにどんどん記録していく）、実験ファイルにReadmeとしても残しておいてほしい

- **ページの構成・書き方は [notion-page-format.md](notion-page-format.md) に従う**
  （結論のコールアウト → 図 → いま言えること → 次 → 詳細はトグル → 実行ログ。
  上部は箇条書き・短文で直感的に。長い説明は詳細トグルへ情報量を落とさず移す。
  表では強調したい数値・箇所を太字。図は GitHub の SHA 固定 URL で埋め込む）。
- **要約図・論文転載用図の見た目は [figure-style.md](figure-style.md) に従う**
  （図中タイトルなし／平均=実線・標準偏差=帯／文字大きめ／基準 1.0 を必ず目盛り表示／
  kg・cm は目盛りに書かない／系列の横軸範囲をそろえる／記述的ファイル名／Success rate 表記）。
- 目的と記録先の対応はリポジトリ直下の `RESEARCH_OBJECTIVES.md`（台帳）。まずここで該当する目的を探す。
  無い目的なら台帳に行を足し、Notion「研究進捗」の M1 配下（`REPORT_FORMAT.md` §7.1 の位置）にページを 1 つ作る。
- 実験や実行が増えてもページは増やさない。同じ目的のページ末尾の **「## 実行ログ {toggle="true"}」親トグルの中**に、
  新しいものが下になる順で 1 エントリずつ追記する（各エントリも入れ子トグル。ページ直下にログを並べない）。
- ページ上部（結論のコールアウト・主な結果の図と表・いま言えること・次）は結果が変わるたびに書き換えて最新に保つ。
- 同じエントリを実験ディレクトリの `README.md` 末尾「## 実行ログ」にも追記する（Notion のトグルは Markdown の見出しと箇条書きにする）。
  `REPORT.md` は区切りのまとめとして更新する。

## 5. 実験設計の文献調査（実装前・必須）

**詳細は [literature-design.md](literature-design.md)。** 要約:

- 新しい実験や方針変更を実装する前に、近いテーマの論文を **ネット上で 5 本以上**調べる
  （ロボット汎化・クロスロボット適応・形態／質量 DR・評価プロトコルなどが望ましい）。
- 把握する項目: **何を変えるか**、**評価指標**、**正規化**、試行数・シード、本実験との違い。
- 調査結果を Issue / README に残し、採否理由を付けてから設計を確定する。文献なしの憶測だけで本番に入らない。

## 6. 実験の 1 サイクル（チェックリスト）

```
- [ ] Issue を確認／作成（task / experiment）、RESEARCH_OBJECTIVES.md で目的と記録先を確認
- [ ] 文献調査 5 本以上（literature-design.md）→ 変える量・指標・正規化の根拠を Issue/README に書く
- [ ] ブランチ exp/<name> を切る
- [ ] vintix_go2/<name>/ に README.md（目的・設定・実行方法、Issue 番号、末尾に「実行ログ」）
- [ ] 実装（共通コードは変更せず import／継承で差し替え）→ コミット
- [ ] スモーク → 本番（長時間は nohup / setsid で切り離し、再開可能に）
- [ ] 区切りごとに: Issue コメント、目的の Notion ページと README の「実行ログ」に追記
- [ ] 区切りのまとめ: REPORT.md、Notion ページ冒頭の要約を更新
- [ ] 次にやるべきことは新しい experiment Issue に切り出す
- [ ] ユーザー確認後に PR → main、Issue close
```

## 7. 自律実行（autopilot）

`autopilot/README.md` を参照。ON/OFF は `autopilot/autopilot.sh on|off|status`。
自律実行中も本スキルと `autopilot/PROMPT.md` のガードレールに従う。
新しい実験方針を自分で決めて実装する場合も、§5 の文献調査を省略しない。
