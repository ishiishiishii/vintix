# ISCIIA 2026 論文（採録後の査読対応）

- 原稿: `src/main.tex`（Overleaf からの Source ZIP `2026_ISCIIA_ishii.zip` を展開したもの。LLNCS 形式）
- 査読コメント: Notion「ISCIIAレビュー」（Review 167 / Review 168）
- ローカルプレビュー: `~/.local/bin/tectonic`（musl 静的ビルド。コンテナの glibc が古く通常版は動かない）＋
  Cursor 拡張 LaTeX Workshop。設定は `~/.cursor-server/data/Machine/settings.json`（保存時に自動ビルド、
  出力は `/tmp/latex_build/main/`）。手動なら `tectonic -X compile main.tex --outdir /tmp/latex_build/main`。
  コンテナを作り直した場合は Tectonic の再インストールが必要。最終確認は Overleaf で行う。

## 査読指摘と対応状況

| # | 指摘 | 原稿の該当箇所 | 状態 |
|---|------|----------------|------|
| 167-1 | 学習ハイパーパラメータ一式の追記 | 4.1 Experimental Setup | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映）（表は 168-5 と共通） |
| 167-2 | Sim-to-real の課題を Discussion に追記 | 4.4 Discussion | 教授フィードバックを受けて「前提条件1つ＋主な課題1つ（頑健性→ドメインランダム化）＋結論」に改訂し、長すぎたため7文に短縮。ローカル反映・Notion 更新済み（Overleaf 未反映） |
| 168-1 | Decoder 適応の効果と Encoder の寄与の区別 | 1 / 4.4（Abstract は確定済みのため変更なし） | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映） |
| 168-2 | 10% のサンプル数と、PPO 専門家学習 / Decoder FT のコスト分離 | 4.1 / 4.4 | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映） |
| 168-3 | 評価リターンのばらつき（可能なら複数 seed） | Table 2 / 4.3 | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映）。複数 seed は未実施（単一 seed と明記） |
| 168-4 | Fig. 4 のプロトコル（履歴の持ち越し・context reset・平均と帯の計算） | 4.2 / 4.3 | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映） |
| 168-5 | 再現用詳細（context length・ハイパラ・noise distillation・観測/行動・報酬重み） | 4.1 | ローカル反映・Notion に方針コールアウトとコピペ用トグル追加済み（Overleaf 未反映）（167-1 と統合） |

## Overleaf 版との照合（2026-09-29）

Notion に添付された Overleaf の ZIP（08:08）を `src/main.tex` と比較した。査読 7 件の対応箇所はすべて入っているが、
Fig. 4 のファイル名（miniCheetah）、4.1 の「ten trajectories … one million time steps」の欠落、
ハイパーパラメータ表が旧版、などが未反映。一覧は Notion 冒頭の「照合結果」コールアウトと各項目の【照合 9/29】注記。

## 結果の差し替え（旧 保留タスク: Mini Cheetah の数値の出どころ）— ローカル反映済み

論文の Table 2 の Mini Cheetah「19.6」は、本文に書いた 10%・1 epoch ではなく **70% データ・1 epoch** の run
（`models/minicheetah_without/minicheetah_without/minicheetah_without_epoch1_0p7data_finetune`）だった。
10%・1 epoch では 8.26（`eval_minicheetah_p10_epoch0/`）。

**方針（2026-09-28 決定）:** 4 台とも `experience_decoder_ft_data_fraction_20260517`（10%・2 epoch・seed 5、
`--random_sample_frac 0.1`）の評価に統一する。ゼロショットも同じスイープの `p00` に揃えた。
Fig. 4 の PDF はもともとこのスイープの PNG（`vintix_without_vs_finetune/graphs/`）と画素単位で一致するため、図は差し替え不要。

| | Go1 | Go2 | A1 | Mini Cheetah |
|---|---|---|---|---|
| Zero-shot（`eval/*/p00`） | 15.17 ± 1.07 | 0.53 ± 0.13 | 15.19 ± 7.70 | −0.17 ± 0.04 |
| Decoder-only（`eval/*/p10`） | 19.27 ± 7.43 | 21.23 ± 0.07 | 20.70 ± 0.12 | 19.98 ± 2.02 |
| 専門家比（専門家 22.502 / 21.325 / 20.860 / 21.052） | 85.7% | 99.6% | 99.2% | 94.9% |

- 更新回数: `train_utils.py` は 8 ミニバッチごとに `optimizer.step()`、`step_num` はエポックごとにリセット
  → 975 // 8 = 121 回/epoch、2 epoch で 242 回（旧稿の「122 回/epoch」は誤り）。学習時間は 2 epoch で約 2.3 分。
- Abstract は確定済みのため確定版の文に戻した（168-1 の Abstract 変更は取り消し）。
- Notion: 各タスクのトグルを更新し、「追加対応（査読指摘外）」に差し替え項目を追加。

## 教授フィードバック対応（2026-09-30）— ローカル反映済み（図を除く）

Notion 冒頭「教授フィードバック対応（9/30）」にチェックボックスと貼り付け用コードあり。

- Optimizer は実装どおり AdamW に統一（`train_utils.py` / `train_vintix.py` とも `torch.optim.AdamW`。
  チェックポイントの `config.json` の `optimizer: "Adam"` は未使用の値）。`refs.bib` に Loshchilov & Hutter（キー `loshchilov2019decoupledweightdecayregularization`、他の ICLR 文献と同じ書式）を追加して引用。
- Table 4 は小数 2 桁に統一し、本文の差と専門家比は表の表示値から計算（差 4.10 / 20.70 / 5.51 / 20.15、
  専門家比 85.6 / 99.5 / 99.2 / 94.9%）。PPO 平均は `episode_reward_comparison.csv`。
- Fig. 3 は縦軸名を「Average episode return」にして再生成（`fig3_episode_return/make_fig3.py`、出力も同ディレクトリ、
  `src/figs/` に同名でコピー済み）。Go1・Go2・A1 は旧図と曲線が一致。Mini Cheetah の旧図は 1 月の古い収集データ由来
  （最終値 約 21.0）で、現行の `data/minicheetah_trajectories`（4/11 以降未変更、論文実験で使用）では 約 20.4。
- 評価の仕様（`save_vintix.py`）: 終了理由によらず `reset_idx` で既定姿勢・速度 0 に戻す（摂動は初回のみ）、
  履歴は保持し直近 2048 遷移のみ入力。
- **要判断:** 学習データは `step_num` がエピソードごとに 0 に戻り、エンコーダは step_num==0 の前行動・前報酬を
  padding に置換してエピソード先頭を示す。評価の `VintixHistoryBuffer` は step_num を 0 に戻さないため、
  評価ではエピソード境界がモデルに伝わっていない（学習と評価の不一致）。
- Fig. 2 は元ファイルがなく未対応（描き直しの項目は Notion に記載）。
