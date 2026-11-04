# go1_body_scale_zeroshot: Go1 の質量・大きさを変えたときのゼロショット性能

Go1 を基準に **質量だけ** / **大きさだけ** を段階的に変えた「改造 Go1」で、
Vintix のゼロショット（Go1 を学習していない `go1_without` を追加学習なしで使う）がどこまで歩けるかを調べる。
各倍率でエピソード報酬の平均と標準偏差を記録し、倍率を横軸にして並べる（成功／失敗の二値判定はしない）。

- 研究の目的: どれくらい体が違うとアルゴリズム蒸留（Vintix）だけではうまく適応できないかを調べる（`RESEARCH_OBJECTIVES.md`）
- 記録ページ（Notion）: [体の違いとゼロショット適応の限界](https://app.notion.com/p/40cd65e6123f81b3be04fe61479b7c70)（この目的の実験はすべてここに追記）
- 設計と議論: Notion「ISCIIA後のステップ」→「Go1 質量・サイズ変化のゼロショット境界（理解と設計書）」
- ブランチ: `exp/go1-body-scale-zeroshot`
- Issue: [#1](https://github.com/ishiishiishii/vintix/issues/1)（目的）、[#2](https://github.com/ishiishiishii/vintix/issues/2)（刻み・10 シード・PPO）、[#3](https://github.com/ishiishiishii/vintix/issues/3)（設計の合意）
- 区切りのまとめ: `REPORT.md`。経過は末尾の「実行ログ」

## 方針（Notion での合意）

| 項目 | 決定 |
|------|------|
| 質量の変え方 | 全リンクの質量と慣性を m 倍（体そのものが重い。寸法・重心位置は不変） |
| 大きさの変え方 | 全寸法を s 倍、**総質量は Go1 の 13.1 kg に固定**（慣性は s² 倍） |
| モデル | `models/go1_without/go1_without/0001_epoch`（ISCIIA のゼロショットと同じ） |
| 評価結果のまとめ方 | 倍率ごとの平均エピソード報酬 ± 標準偏差（二値判定なし） |
| PPO 専門家・AD データ作り | 本実験の範囲外（境界が分かった後、手が空いたら別途） |
| 既存コード | **変更しない**。この実験ディレクトリ内で既存クラスを継承・差し替えて使う |

## 固定するもの

制御器（kp=20、kd=0.5、行動スケール 0.25、既定関節角）、モータのトルク上限、報酬の重み、指令速度（前進 0.5 m/s）、
観測の正規化（既知 3 タスクの平均。`save_vintix.py` の未知タスク既定動作）は Go1 のまま。
大きさ軸だけ、初期高さ・報酬の目標高さ・足上げ高さを s 倍にする（大きい体が普通に立っているだけで罰されないように）。
ベースの設定は Go1 専門家の `Genesis/logs/go1-walking/cfgs.pkl`（初期高さ 0.40 m、目標高さ 0.32 m）。

## 掃引する倍率

| 軸 | 倍率 |
|----|------|
| 質量 m（Go1 13.1 kg 比） | 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0 |
| 大きさ s（太もも 0.213 m 比） | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0 |

報酬が急に落ちる区間は `--mass_scales` / `--size_scales` で点を足して細かくする。

## 評価する方策

| 名前 | 中身 |
|------|------|
| `vintix` | `go1_without/0001_epoch` のゼロショット（`scripts/save_vintix.py` の評価関数） |
| `expert` | 通常の Go1 で学習した PPO 専門家 `model_2000.pt` をそのまま改造 Go1 に使う（`scripts/save_expert.py` の評価関数） |

評価条件はどちらも ISCIIA と同じ 10 環境 × 各 10 エピソード、1 エピソード最大 1000 step。

## ファイル

| ファイル | 役割 |
|----------|------|
| `scaled_go1_env.py` | 共通の `go1_env.Go1Env` を継承した `ScaledGo1Env`。URDF の読み込み倍率と、ビルド後のリンク質量を変える |
| `body_scale.py` | 倍率の格子と、`cfgs.pkl` から改造 Go1 用の設定を作る関数 |
| `run_experiment.py` | オーケストレーション。1 点ごとに子プロセスを起動し、共通の評価関数を呼ぶ。子プロセス内でだけ `save_vintix.Go1Env` / `save_expert.Go1Env` を `ScaledGo1Env` に差し替える（共通スクリプトのファイルは無変更） |
| `results/<policy>/<axis>/s<倍率>_r<反復>/` | 各点の評価出力（エピソード CSV、要約 txt、図） |
| `results/probe/<axis>/` | 体の確認（シミュレータ上の総質量、ゼロ行動で立たせたときの胴体高さ、画像） |
| `results.csv` / `summary.json` | 全点の集計 |
| `graphs/{mass,size}_scale_zeroshot.png` | 倍率 vs 報酬・エピソード長（平均 ± 標準偏差） |
| `logs/` | 各点の実行ログ |

## 第 2 段階: 刻みの追加・10 シード・PPO（`pipeline.py`）

| 項目 | 内容 |
|------|------|
| 追加の刻み | 質量 0.55, 1.15, 1.25, 1.35 / 大きさ 0.55, 0.65, 1.15, 1.25, 1.35（`body_scale.py` の `*_REFINE`） |
| シード | 0〜9。評価前に python / numpy / torch / Genesis を固定し、各シード 10 env × 10 ep（1 点 1000 エピソード）。同じシードで結果が完全一致することを確認済み |
| 順番 | ① 体の確認 → ② 境界付近（`BOUNDARY_GRID`）を全シード → ③ PPO を体ごとにゼロから学習（`PPO_GRID`、500 反復、学習シード 1）して 10 シード評価 → ④ 残りの点を全シード |
| 並列 | 評価は最大 4 本同時。PPO 学習は 1 本（評価 2 本分として数える）で、② が終わってから ④ と並行 |
| PPO 学習 | `run_experiment.py --worker train_ppo`。Go1 専門家の `cfgs.pkl` と同じ PPO 設定・`OnPolicyRunner`、4096 env、`precision=64`。出力 `ppo/<axis>/s<倍率>/` |
| 集計 | `results.csv`（シード付きの結果のみ。全エピソードの平均 ± 標準偏差と、シード平均のばらつき `reward_seed_std`）。以前の 1 回だけの結果は `results_single_run.csv` / `graphs/single_run/` に分けて残す |
| 進捗 | `pipeline_status.json`、`logs/pipeline.log`。`STOP` ファイルを置くと新しいジョブを起動しない |

```bash
cd /workspace/vintix_go2
setsid nohup python -u go1_body_scale_zeroshot/pipeline.py > go1_body_scale_zeroshot/logs/pipeline.log 2>&1 < /dev/null &
python go1_body_scale_zeroshot/pipeline.py --dry_run   # 段階ごとの完了数
```

## 実行方法（第 1 段階、コンテナ内）

```bash
cd /workspace/vintix_go2
# 体の確認（質量・立ち高さ・画像）
python go1_body_scale_zeroshot/run_experiment.py --stage probe
# 評価（完了済みの点は飛ばすので中断後に再開できる）
nohup python -u go1_body_scale_zeroshot/run_experiment.py --stage eval --policies expert,vintix \
    --report_every_point > go1_body_scale_zeroshot/logs/eval_all.log 2>&1 &
# 集計と図の作り直しだけ
python go1_body_scale_zeroshot/run_experiment.py --stage report
# 点を足す例
python go1_body_scale_zeroshot/run_experiment.py --stage eval --axes mass --mass_scales 1.15,1.25
```

倍率は 1.0 から外側へ交互に（1.0 → 0.9 → 1.1 → …）評価するので、途中で止めても両側の様子が分かる。

## 結果

最新の集計は `results.csv` と `graphs/`、まとめは `REPORT.md`。経過は下の「実行ログ」。

## 実行ログ

Notion の記録ページの「実行ログ」と同じ内容。新しいものが下。

### 2026-11-01〜11-02 第 1 段階: 各点 1 回の粗い掃引

- 実行内容: 質量 17 点・大きさ 13 点 ×（Vintix ゼロショット、Go1 専門家の流用）= 60 点、各 10 env × 10 ep を 1 回（シード未固定）
- 結果: Vintix は重い側・大きい側とも 1.2 倍で 100 エピソード中 90 回完走、1.3 倍で完走 0 回（質量）・17 回（大きさ）。
  軽い側は質量 0.25 倍でも報酬 14.5、小さい側は 0.5 倍で 8.9。流用した専門家は 1.2 倍で転倒、
  軽い側は 0.5 倍以下、小さい側は 0.6 倍以下で転倒（`results_single_run.csv`、`graphs/single_run/`）
- 考察: 軽い・小さい側は Vintix の方が明らかに広い。重い・大きい側は両方とも約 1.2 倍で崩れるが、
  PD ゲインとトルク上限を固定しているので「体そのものが歩けない」可能性があり、まだ切り分けられていない。各点 1 回だけなのでばらつきも未確認
- 次の実行内容: 境界の刻み追加、10 シード評価、PPO を各体でゼロから学習して歩けるかを確認

### 2026-11-04 第 2 段階を開始: 刻み追加・10 シード・PPO

- 実行内容: 質量 0.55/1.15/1.25/1.35、大きさ 0.55/0.65/1.15/1.25/1.35 を追加。境界付近 26 点を先にシード 0〜9
  （各 10 env × 10 ep、1 点 1000 エピソード）で評価し、終わったら PPO を 16 点でゼロから学習（500 反復）して 10 シード評価、
  並行して残りの点も 10 シード評価。`pipeline.py` で自動実行（評価 4 並列）
- 確認済み: 同じシードなら結果が完全一致、シードを変えると変わる。PPO 学習とその評価は小さな設定で通る
- 結果: 実行中（11-04 12:47 時点で境界付近 520 ジョブ中 2 完了、失敗 0）
- 次の実行内容: 境界付近のシード評価がそろい次第、平均 ± 標準偏差とシード間のばらつきを集計して追記する
