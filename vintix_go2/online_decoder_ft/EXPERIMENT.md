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
| `max_iterations` | **100** | ハード上限（1-based）。`-1` は `--saturate` 必須で飽和まで |
| `init_noise_std` | **0.2** | 非正規化行動空間の初期 Gaussian ノイズ。`train.py` の 1.0 より小さくし、序盤の即転倒ループを抑える |
| `--saturate` | **off** | 付けると報酬飽和で早期終了（閾値はコード内固定） |
| `--seed_retries` | **0** | 即死時に seed+1 で最大 N 回再試行（推奨 3） |

`trajectory_sparsity` はオフライン学習時のウィンドウ開始間隔。オンライン PPO では履歴は **連続 step** を `context_len` 本保持し、`save_vintix.py` 評価と同様（ウィンドウ内はサブサンプルしない）。

### 3.2 探索ノイズ（`init_noise_std`）

事前学習済み decoder FT では、ゼロからの PPO ほど大きな探索は不要。  
`init_noise_std=1.0` だと非正規化後の関節指令に対してノイズが過大になり、**全環境が即座に転倒 → 報酬信号消失 → 学習不能**になりやすい。既定は **0.2**。

- 新規学習: CLI `--init_noise_std` / `run_experiment.py --init_noise_std`
- resume: checkpoint 内の学習済み `std` が使われる（`init_noise_std` は新規初期化時のみ）

### 3.3 報酬飽和による自動終了（`--saturate`）

`--max_iterations` によるイテレーション指定はそのまま（ハード上限）。  
`--saturate` を付けると、平均エピソード報酬が頭打ちになった時点で早期終了する。

閾値は CLI に出さずコード内で固定（`train_online_decoder_ppo.py`）:

| 内部定数 | 値 | 意味 |
|--|--|--|
| `SATURATE_WINDOW` | 20 | 直近 20 iter の `mean_reward` で max−min を見る（＝複数 iter の平坦判定そのもの） |
| `SATURATE_EPS` | 0.25 | 絶対レンジ（max−min）≤ ε なら飽和候補 |
| `SATURATE_MIN_MEAN_REWARD` | 1.0 | 平均報酬がこれ未満の平坦は成功飽和とみなさない（即死と分離） |
| `SATURATE_MIN_RUN_ITERS` | 1000 | この `learn()` 内で最低 1000 iter 回してから判定（序盤の一時平坦の誤検知を抑える） |

判定は **当該学習呼び出しで得た報酬のみ**（resume 前の CSV は混ぜない）。

```bash
# 従来どおり: ちょうど 5000 iter
python scripts/train_online_decoder_ppo.py ... --max_iterations 5000

# 上限 10000 + 飽和したら早めに終了
python scripts/train_online_decoder_ppo.py ... --max_iterations 10000 --saturate

# イテレーション数に頼らず飽和まで（--saturate 必須）
python scripts/train_online_decoder_ppo.py ... --max_iterations -1 --saturate
```

### 3.4 即死ループ検知とシード再試行（`--seed_retries`）

短エピソードが続く失敗モード（即死）は、報酬平坦だけでは成功頭打ちと区別できない。  
**平均エピソード長**で即死を検知し、失敗時は成果物を退避して seed を変えてスクラッチ再試行する。

| 内部定数 | 値 | 意味 |
|--|--|--|
| `DEATH_WINDOW` | 20 | 直近 20 iter の `mean_ep_len` |
| `DEATH_MAX_MEAN_EP_LEN` | 30 | 平均 ep_len がこれ未満なら即死 |
| `DEATH_MIN_RUN_ITERS` | 1000 | 最低 1000 iter 後に判定（Go2 成功例の一時低迷 ~600 を超える） |

| 引数 | 既定 | 意味 |
|--|--|--|
| `--seed_retries N` | 0 | 即死時に `seed+1` で最大 N 回追加試行（推奨 **3**＝最大 4 試行） |

挙動:

1. 即死検知 → 最終 ckpt 保存 → `failed_seed{S}/` へ退避 → `seed_attempts.json` に記録
2. 残り retries があれば同一スクリプトを subprocess で再起動（Genesis env 再生成を避ける）
3. 成功した試行の成果物だけが `log_dir` / `result_dir` 直下に残る
4. `--resume_path` との併用不可

```bash
python scripts/train_online_decoder_ppo.py ... --max_iterations 10000 --saturate --seed 1 --seed_retries 3
```

`run_experiment.py` も `--saturate` / `--init_noise_std` / `--seed_retries` を転送する。

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

### 4.4 飽和停止つきスクラッチ学習（推奨）

```bash
python online_decoder_ft/run_experiment.py \
  --models go2_without --num_envs 64 --train_only \
  --max_iterations 10000 --saturate \
  --init_noise_std 0.2 \
  --save_interval 500
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

**成功 1000 iter（退避）:** `models/go2_without_1000_canonical/` + `result/go2_without_1000_canonical/`  
**canonical から +4000 resume:** `models/go2_without_resume5000/` + `result/go2_without_resume5000/`

ゼロショット / オフライン FT の参照評価（再計算しない既定）:

`experience_decoder_ft_data_fraction_20260517/eval/<model_key>/p00` と `p10`

---

## 6. 設計メモ

- Critic は train.py と同型の独立 MLP（Vintix に value head は付けない）
- 起動は `.py` のみ（`.sh` は置かない）
- 稼働中の他実験プロセスには触れない
- スクラッチ学習で序盤に `mean_ep_len` が ~4 付近で潰れたら即死ループ。`init_noise_std` 縮小や seed 再試行を検討

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

**補足（後続）:** 64×1000 の成功 run は `go2_without_1000_canonical`（train mean ~7.7、eval ~15.5）。  
同条件スクラッチ再実行は即死ループで失敗しやすい（`init_noise_std=1.0` 時代）。対策は §3.2–3.3。
