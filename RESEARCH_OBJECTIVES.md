# 研究の目的と記録先の台帳

研究の大まかな目的ごとに、Notion の記録ページ 1 つと、それに属する実験ディレクトリ・Issue をまとめる。
実験（や自律実行の 1 回）が終わるたびに、該当する目的の Notion ページの「実行ログ」と、
実験ディレクトリの `README.md` の「実行ログ」に追記する（記録の書き方は
`.cursor/skills/research-experiment-workflow/SKILL.md` §4）。
新しい目的が生まれたらここに行を足し、Notion「研究進捗」の M1 配下にページを 1 つ作る。

直近の対外目標: **IROS（3月採択）**。Go2 実機は Issue [#19](https://github.com/ishiishiishii/vintix/issues/19)（**直近外・later**。案の蓄積可。実行は確認後）。詳細は `RESEARCH_GOAL.md`。

| 目的 | Notion ページ | 実験ディレクトリ（ブランチ） | Issue |
|------|---------------|------------------------------|-------|
| 体の違いとゼロショット適応の限界: どれくらい体が違うとアルゴリズム蒸留（Vintix）だけではうまく適応できないかを調べる | **主ページ:** [4台体スケールのゼロショット限界](https://app.notion.com/p/41ad65e6123f816093d0ebc4bb281f25)。**Go1のみ学習の限界:** [Go1のみで学習したモデルの限界](https://app.notion.com/p/421d65e6123f81c2ab9be103f343419e)（`421d65e6-123f-81c2-ab9b-e103f343419e`）。**多様な体で学習した場合との比較:** [多様な体のデータで学習したモデルの限界（Go1評価）](https://app.notion.com/p/41cd65e6123f81d3b85ddb5c838740d0)。**キュー（2026-11-23）:** #22 → #16（#20 closed）。A1 / Mini Cheetah / Go2 での同じ種類の評価は**いったん止める** | `vintix_go2/go1_body_scale_zeroshot/`、`robot_body_scale_zeroshot/`、`same_body_scale_zeroshot/`（#18 Go1のみ学習）、`dr_narrow_vintix_bodyscale/`（#21） | [#10](https://github.com/ishiishiishii/vintix/issues/10)（完了）、[#15](https://github.com/ishiishiishii/vintix/issues/15)（完了）、[#18](https://github.com/ishiishiishii/vintix/issues/18)（Go1 完了・他機種保留）、[#21](https://github.com/ishiishiishii/vintix/issues/21)（完了・closed） |
| DR ロボットの学習履歴データセット: Vintix の学習データとして、多様な体（DR）の PPO 学習履歴を学習シードを変えて集める（**narrow 100 体をシードごとに一通り → 次シード**。広い範囲は後） | [DR ロボットの学習履歴データセット](https://app.notion.com/p/416d65e6123f81f3a613f871abd3676c)（page_id `416d65e6-123f-81f3-a613-f871abd3676c`） | `vintix_go2/dr_robots_multiseed_history/`（`exp/robot-body-scale-zeroshot`）、元の DR 実験 `vintix_go2/dr_robots_ppo_history/` | [#11](https://github.com/ishiishiishii/vintix/issues/11) |
| 評価ロボット拡充: 他論文の評価機体を導入（URDF検証→Env。報酬・PPOは本線後） | （#16 調査は `robot_eval_expansion/ROBOT_SURVEY.md`。Notion は既存目的ページに追記予定） | `vintix_go2/robot_eval_expansion/`、URDF は `Genesis/genesis/assets/urdf/` | [#16](https://github.com/ishiishiishii/vintix/issues/16) |
| DR narrow 成功履歴で学習した Vintix の体スケール限界: 成功体の履歴だけで新規学習し、Go1 3 軸で崖を測り、学習分布を数直線で対応付ける | [DR narrow履歴Vintixの体スケール限界](https://app.notion.com/p/DR-narrow-Vintix-41cd65e6123f81d3b85ddb5c838740d0)（`41cd65e6-123f-81d3-b85d-db5c838740d0`） | `vintix_go2/dr_narrow_vintix_bodyscale/`（`exp/dr-narrow-vintix-bodyscale`） | [#21](https://github.com/ishiishiishii/vintix/issues/21)（Go1 3軸・数直線 **完了・closed**） |
| 評価ロボット拡充: 他論文で使われる四足を文献頻度で選定し、URDF 検証付きで導入（報酬・PPO は本線後） | [評価ロボット拡充（選定と導入）](https://app.notion.com/p/41cd65e6123f81d89e38e1d7ea285280)（`41cd65e6-123f-81d8-9e38-e1d7ea285280`） | 調査: `Genesis/examples/locomotion/robot_selection_literature.md`（Env は既存 `anymalc`/`laikago` + 今後 Aliengo） | [#16](https://github.com/ishiishiishii/vintix/issues/16)（タスク1 着手中。タスク2 は #18→#21 後） |
| 汎用歩行ポリシーの代表手法を比較実験用に導入（調査→OSSクローン／無ければ論文から実装） | [汎用歩行ベースライン導入](https://app.notion.com/p/41cd65e6123f81e7b047e70481eb0e6d)（`41cd65e6-123f-81e7-b047-e70481eb0e6d`） | `vintix_go2/general_locomotion_baselines/`（`exp/general-locomotion-baselines`） | [#22](https://github.com/ishiishiishii/vintix/issues/22)（調査表済・P0 クローン済。配線・学習は GPU 空き後） |
