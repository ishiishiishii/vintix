# 研究の目的と記録先の台帳

研究の大まかな目的ごとに、Notion の記録ページ 1 つと、それに属する実験ディレクトリ・Issue をまとめる。
実験（や自律実行の 1 回）が終わるたびに、該当する目的の Notion ページの「実行ログ」と、
実験ディレクトリの `README.md` の「実行ログ」に追記する（記録の書き方は
`.cursor/skills/research-experiment-workflow/SKILL.md` §4）。
新しい目的が生まれたらここに行を足し、Notion「研究進捗」の M1 配下にページを 1 つ作る。

直近の対外目標: **IROS（3月採択）**。Go2 実機は Issue [#19](https://github.com/ishiishiishii/vintix/issues/19)（案の蓄積。実行は確認後）。詳細は `RESEARCH_GOAL.md`。

| 目的 | Notion ページ | 実験ディレクトリ（ブランチ） | Issue |
|------|---------------|------------------------------|-------|
| 体の違いとゼロショット適応の限界: どれくらい体が違うとアルゴリズム蒸留（Vintix）だけではうまく適応できないかを調べる | **主ページ（#10 完了まとめ・次回以降の進捗報告）:** [4台体スケールのゼロショット限界](https://app.notion.com/p/41ad65e6123f819e9cabee1835c71c90)（`41ad65e6-123f-819e-9cab-ee1835c71c90`）。**Go1 前段:** [体の違いとゼロショット適応の限界](https://app.notion.com/p/40cd65e6123f81b3be04fe61479b7c70)（`40cd65e6-123f-81b3-be04-fe61479b7c70`）。キュー: **#15** → #18 → #13 → #11（#10 完了） | `vintix_go2/go1_body_scale_zeroshot/`（`exp/go1-body-scale-zeroshot`）、`vintix_go2/robot_body_scale_zeroshot/`（`exp/robot-body-scale-zeroshot`、Go2・A1・Mini Cheetah と密度一定の大きさ） | [#10](https://github.com/ishiishiishii/vintix/issues/10)（4 台 × 3 軸・**パイプライン完了**）、[#15](https://github.com/ishiishiishii/vintix/issues/15)（指標網羅保存・次）、[#18](https://github.com/ishiishiishii/vintix/issues/18)（同体のみ学習 Vintix・#15後）、[#1](https://github.com/ishiishiishii/vintix/issues/1)（目的）、[#2](https://github.com/ishiishiishii/vintix/issues/2)（刻み・10 シード・PPO）、[#3](https://github.com/ishiishiishii/vintix/issues/3)（設計の合意）、[#4](https://github.com/ishiishiishii/vintix/issues/4)（PPO・AD データ）、[#5](https://github.com/ishiishiishii/vintix/issues/5)（Go2 原因切り分け）、[#7](https://github.com/ishiishiishii/vintix/issues/7)（質量 × 脚の長さ仮説の検証）、[#8](https://github.com/ishiishiishii/vintix/issues/8)（Vintix の環境ごとの二極化）、[#9](https://github.com/ishiishiishii/vintix/issues/9)（大きさ 1.5 倍の PPO の学習失敗の切り分け）、[#13](https://github.com/ishiishiishii/vintix/issues/13)（下限追試） |
| DR ロボットの学習履歴データセット: Vintix の学習データとして、多様な体（DR で作った 200 体）の PPO 学習履歴を学習シードを変えて集める | [DR ロボットの学習履歴データセット](https://app.notion.com/p/416d65e6123f81f3a613f871abd3676c)（page_id `416d65e6-123f-81f3-a613-f871abd3676c`） | `vintix_go2/dr_robots_multiseed_history/`（`exp/robot-body-scale-zeroshot`）、元の DR 実験 `vintix_go2/dr_robots_ppo_history/` | [#11](https://github.com/ishiishiishii/vintix/issues/11) |
