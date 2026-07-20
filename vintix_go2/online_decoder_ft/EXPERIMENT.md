# EXPERIMENT: Online Decoder PPO Fine-Tuning

未知ロボット適応を、事前 AD データなしで行う。事前学習済み Vintix（`*_without`）を
シミュレーション上で動かしながら、**action decoder のみ**を PPO で更新する。

> 前提: Docker コンテナ内 `/workspace` で実行（AGENTS.md §2）。

---

## 1. 目的

| 現状の問題 | 本実験の解決 |
|---|---|
| オフライン decoder FT は対象ロボットの AD 軌跡が必要 | オンラインで対象ロボットを動かしながら FT |
| ゼロショットは勾配更新なしで性能が足りない場合がある | decoder を PPO で更新して性能を引き上げる |

比較対象（3 つ）:

1. **ゼロショット / in-context**（同一 `*_without`、勾配なし）
2. **オフライン decoder FT**（AD 10%、既存評価を参照）
3. **PPO スクラッチ / クロス FT**（効率比較は既存実験のオンライン step 数を引用）

主成果物: PPO 学習報酬曲線、`save_vintix.py` 評価、上記 1・2 との表・グラフ比較。

---

## 2. 対象モデル

| model_key | 事前学習 ckpt | 適応先ロボット |
|---|---|---|
| `go1_without` | `models/go1_without/go1_without/0001_epoch` | go1 |
| `go2_without` | `models/go2_without/go2_without/0001_epoch` | go2 |
| `a1_without` | `models/a1_without/a1_without/0001_epoch` | a1 |
| `minicheetah_without` | `models/minicheetah_without/minicheetah_without/0001_epoch` | minicheetah |

**実装順:** まず `go2_without` で動作確認 → 問題なければ 4 モデル。

---

## 3. 手法概要

- **Actor:** 事前学習 Vintix（encoder + transformer は freeze、`acs_decoders[quadruped_locomotion]` のみ更新）
- **Critic:** `train.py` と同型 MLP `[512,256,128]` + ELU（ゼロから学習）
- **Action noise:** 学習可能 `std`（Vintix 出力は mean）
- **履歴:** SAR 系列を env ごとに保持（既定: エピソード跨ぎ）。PPO update 時はロールアウトで保存した最終 token 特徴 → decoder のみ再計算
- **未知タスク正規化:** 既知タスク stats の平均（`save_vintix.py` と同趣旨）

共通実装:

- [`scripts/vintix_actor_critic.py`](../scripts/vintix_actor_critic.py)
- [`scripts/train_online_decoder_ppo.py`](../scripts/train_online_decoder_ppo.py)

本ディレクトリはオーケストレーションと結果保存のみ。

---

## 3.1 既定ハイパーパラメータ（事前学習と整合）

| 項目 | 既定値 | 備考 |
|------|--------|------|
| `num_envs` (`-B`) | **4096** | `train.py` と同じ |
| `context_len` | **2048** | ckpt `config.json` から自動読込（上書き可） |
| `trajectory_sparsity` | **128** | ckpt `config.json` から自動読込（上書き可） |
| `save_interval` | **5** | PPO **イテレーション**ごとに `model_N.pt` + Vintix export（最終 iter も必ず保存） |
| `max_iterations` | **100** | `run_experiment.py` / `train_online_decoder_ppo.py` 既定 |

`trajectory_sparsity` はオフライン学習時のウィンドウ開始間隔。オンライン PPO では履歴は **連続 step** を `context_len` 本保持し、`save_vintix.py` 評価と同様（ウィンドウ内はサブサンプルしない）。

---

## 4. 実行

出力は **`models/<model_key>/`（学習）** と **`result/<model_key>/`（評価・比較）** の 2 段のみ。

### 4.1 Go2 本番候補（64 env × 100 iter）

```bash
cd /workspace/vintix_go2
PYTHONPATH=/workspace/vintix_go2:/workspace/vintix_go2/scripts:/workspace/Genesis/examples/locomotion \
  python online_decoder_ft/run_experiment.py \
    --models go2_without --num_envs 64 --max_iterations 100 --context_len 2048
```

### 4.2 4 モデル本番（4096 env — GPU メモリに注意）

```bash
python online_decoder_ft/run_experiment.py \
  --models go1_without,go2_without,a1_without,minicheetah_without
```

### 4.3 学習のみ / 評価のみ

```bash
python online_decoder_ft/run_experiment.py --models go2_without --num_envs 64 --train_only
python online_decoder_ft/run_experiment.py --models go2_without --skip_train
```

比較グラフは `compare_results.py` 実行時に常に出力（`result/<model_key>/`）。

---

## 5. 出力レイアウト

```
online_decoder_ft/
  models/<model_key>/
    model_5.pt, model_100.pt     # PPO checkpoint
    model_5/, model_100/         # 同名 Vintix export
    run_config.json
  result/<model_key>/
    train_mean_reward.csv/.png   # 学習曲線
    go2_10envs_10episodes_*.csv  # 評価
    go2_without_zero_offline_online.png
    comparison_summary.csv
```

**Go2 本番候補:** `models/go2_without/` + `result/go2_without/`（64 env × 100 iter, ctx=2048）。

ゼロショット / オフライン FT の参照評価（再計算しない既定）:

`experience_decoder_ft_data_fraction_20260517/eval/<model_key>/p00` と `p10`

---

## 6. 設計メモ

- Critic は train.py と同型の独立 MLP（Vintix に value head は付けない）
- 起動は `.py` のみ（`.sh` は置かない）
- 稼働中の他実験プロセスには触れない

---

## 7. Go2 結果（`models/go2_without/` + `result/go2_without/`）

| 条件 | 値 |
|------|-----|
| num_envs | 64 |
| max_iterations | 100 |
| context_len | 2048 |
| 総 env steps | 153,600 |

| Zero-shot | Offline FT (10%) | Online PPO FT (100 iter) |
|---:|---:|---:|
| 0.53 | 21.23 | **0.36** |

学習曲線: train mean reward 約 -0.17 → +0.18。  
詳細: `runs/go2_without_B64_iter100/run_config.json` と `comparison_summary.md`。
