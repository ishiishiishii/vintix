# robot_body_scale_zeroshot: 4 台のロボットで、体がどれだけ変わるとゼロショットで歩けなくなるか

`go1_body_scale_zeroshot`（Go1 の質量・大きさ）を、Go2・A1・Mini Cheetah にも広げ、
さらに「密度を変えずに大きさを変える（質量も s³ 倍）」軸を 4 台すべてに加えた実験。

- 研究の目的: どれくらい体が違うとアルゴリズム蒸留（Vintix）だけではうまく適応できないかを調べる（`RESEARCH_OBJECTIVES.md`）
- 記録ページ（Notion）: [体の違いとゼロショット適応の限界](https://app.notion.com/p/40cd65e6123f81b3be04fe61479b7c70)
- ブランチ: `exp/robot-body-scale-zeroshot`（`exp/go1-body-scale-zeroshot` から分岐）
- Issue: #10（この実験）、後続のデータ収集は #11（`vintix_go2/dr_robots_multiseed_history/`）

## ゴール

Go1 だけでなく Go2・A1・Mini Cheetah についても、シードを変えて複数回評価し、
質量・大きさがどれくらい変わると歩けなくなるのかを調べる。

## 設計

| 項目 | 内容 |
|------|------|
| 対象と軸 | Go2・A1・Mini Cheetah: 質量（大きさ固定）、大きさ（質量固定）、大きさ（密度一定 = 質量 s³ 倍）の 3 軸。Go1: 密度一定の大きさ軸だけ（質量・質量固定の大きさは `go1_body_scale_zeroshot` の結果を図に合わせて使う） |
| 倍率 | 質量 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0 / 大きさ（質量固定）0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0 / 大きさ（密度一定）0.5〜1.5（10 点、質量 0.125〜3.4 倍）。元の体（1.0）は各ロボット 1 回だけ評価して全軸で共有 |
| 方策 | ① Vintix ゼロショット: そのロボットを学習していない `<robot>_without/0001_epoch`（ISCIIA と同じ）② そのロボットの通常の PPO 専門家をそのまま流用（Go2 `model_300`、A1 `model_350`、Mini Cheetah `model_2990`、Go1 `model_2000`）③ その体でゼロから学習した PPO（各軸の両端 2 点、500 反復、学習シード 1） |
| 評価 | 各点 シード 0〜9 × 10 環境 × 10 エピソード（最大 1000 step）。シード固定で完全に再現できる |
| 固定するもの | PD ゲイン・トルク上限・報酬の重み・速度指令はそのロボットの元の設定のまま。大きさを変える 2 軸では初期高さと報酬の目標高さだけ s 倍 |
| 環境設定 | `save_vintix.py` と同じ取り方（`Genesis/logs/<exp>/cfgs.pkl`、無ければ `train.get_*_cfgs()`）。Mini Cheetah は既定値、Go2・A1・Go1 は学習時の `cfgs.pkl` |
| 指標 | 完走率（1000 step 転倒せずに終わったエピソードの割合）、エピソード報酬（各環境の先頭 10 本で集計） |

## 実装（共通コードは変更しない）

| ファイル | 役割 |
|----------|------|
| `robots.py` | ロボットごとの設定（環境クラス、URDF、設定・専門家・Vintix の場所）、倍率の格子、倍率から設定を作る関数 |
| `scaled_env.py` | 共通の環境クラスを継承し、URDF の読み込み倍率とリンク質量を変える（`scaled_env_class(base_cls, urdf_name)`） |
| `run_experiment.py` | 1 点の評価（Vintix・専門家・PPO）、PPO 学習、体の確認、集計・図。子プロセスの中でだけ `save_vintix` / `save_expert` の環境クラスを差し替える |
| `pipeline.py` | 体の確認 105 件 → 本番 2100 件（シード順）→ PPO 学習 22 本と評価 220 件。評価 4 本並列、PPO 学習は 1 本。完了済みは飛ばして再開できる |

## 実行

```bash
cd /workspace/vintix_go2
setsid nohup python -u robot_body_scale_zeroshot/pipeline.py > robot_body_scale_zeroshot/logs/pipeline.log 2>&1 < /dev/null &
python robot_body_scale_zeroshot/pipeline.py --dry_run          # 段階ごとの完了数
python robot_body_scale_zeroshot/run_experiment.py --stage report --tag 20261112   # 集計と図
```

進捗は `pipeline_status.json`、集計は `results.csv`、図は `figures/<robot>_body_scale_<tag>.png`。
`STOP` ファイルを置くと新しいジョブを起動しない。

## 見積もり

本番 2100 件のうち、歩ける点の Vintix 評価が 1 件 20〜60 分（4 並列）、すぐ転ぶ点は 1〜2 分、専門家は約 5 分。
ISCIIA では Go2・Mini Cheetah の元の体のゼロショットが転倒しているので、その 2 台は短く終わる点が多い見込み。
全体で 3〜5 日。終わったら `dr_robots_multiseed_history` のデータ収集が自動で始まる。

## 実行ログ

### 2026-11-12 開始: 4 台 × 3 軸の設計と実装、パイプライン起動

- 実行内容: 上の設計で実装し、各ロボット・各軸 1 点ずつ体の確認を手で実行して、シミュレータ上の質量が狙いどおりであることを確認
  （Mini Cheetah 密度一定 1.3 倍 18.13 kg = 8.25 × 1.3³、Go1 密度一定 0.6 倍 2.83 kg、A1 質量 1.5 倍 20.61 kg、Go2 大きさ 1.3 倍 15.02 kg）。
  パイプラインを 2026-11-12 05:51 UTC に起動
- 次: シード 0 が全点そろったら速報を記録（#10）
