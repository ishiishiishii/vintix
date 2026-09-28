# ISCIIA 2026 論文（採録後の査読対応）

- 原稿: `src/main.tex`（Overleaf からの Source ZIP `2026_ISCIIA_ishii.zip` を展開したもの。LLNCS 形式）
- 査読コメント: Notion「ISCIIAレビュー」（Review 167 / Review 168）
- コンテナに LaTeX は無いため、コンパイル確認は Overleaf で行う。

## 査読指摘と対応状況

| # | 指摘 | 原稿の該当箇所 | 状態 |
|---|------|----------------|------|
| 167-1 | 学習ハイパーパラメータ一式の追記 | 4.1 Experimental Setup | 未着手 |
| 167-2 | Sim-to-real の課題を Discussion に追記 | 4.4 Discussion | ローカル反映・Notion にコピペ用トグル追加済み（Overleaf 未反映） |
| 168-1 | Decoder 適応の効果と Encoder の寄与の区別 | Abstract / 1 / 4.4 | 未着手 |
| 168-2 | 10% のサンプル数と、PPO 専門家学習 / Decoder FT のコスト分離 | 4.1 / 4.4 | 未着手 |
| 168-3 | 評価リターンのばらつき（可能なら複数 seed） | Table 2 / 4.3 | 未着手 |
| 168-4 | Fig. 4 のプロトコル（履歴の持ち越し・context reset・平均と帯の計算） | 4.2 / 4.3 | 未着手 |
| 168-5 | 再現用詳細（context length・ハイパラ・noise distillation・観測/行動・報酬重み） | 4.1 | 未着手（167-1 と統合） |

## 保留タスク: Mini Cheetah の数値の出どころ

論文の Table 2 / Fig. 4 の Mini Cheetah「Decoder-only adaptation 19.6」は、10% ではなく
**70% データ・1 epoch** の run（`models/minicheetah_without/minicheetah_without/minicheetah_without_epoch1_0p7data_finetune`、19.646）から来ている。

| Mini Cheetah・10% データ | 平均 | 標準偏差 | ソース |
|---|---|---|---|
| 旧 run・1 epoch | 4.41 | 3.63 | `models/.../minicheetah_without_epoch1_0p1data_finetune` |
| スイープ・1 epoch | 8.26 | 2.54 | `eval_minicheetah_p10_epoch0/`（2026-09-28 に追加評価） |
| スイープ・2 epoch | 19.98 | 2.02 | `experience_decoder_ft_data_fraction_20260517/eval/minicheetah_without/p10` |

他 3 台の論文値（Go1 20.4 / Go2 20.0 / A1 20.7）は旧 run の 10%・1 epoch と一致。
ゼロショット値は `scripts/readable_eval_graphs/*_without_0001_epoch`、PPO 専門家値は `episode_reward_comparison.md` と一致。

方針候補（未決定）:
1. 4 台ともスイープ（10%・2 epoch）に揃える → 85.6–99.6%。Abstract は数値を含まないため修正不要。
2. Mini Cheetah のみスイープ 2 epoch 値に差し替え、Mini Cheetah だけ 2 epoch と明記。

修正対象行（main.tex）: 124, 420（"one epoch"）/ 126, 132, 598, 611（90.7--99.0%）/ 527, 530, 571–576（表と結果）/ Fig. 4 の Mini Cheetah 図。
また 557 行目の `figs/miniCheetah_...` はファイル名（小文字 `minicheetah_...`）と大文字小文字が不一致。
