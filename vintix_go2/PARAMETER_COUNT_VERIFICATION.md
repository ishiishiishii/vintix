# パラメータ数 1,231,706 の根拠

## 結論

**総パラメータ数 1,231,706** は、表と同じ設定で `Vintix` を初期化し、`sum(p.numel() for p in model.parameters())` で得た値である。

## 算出方法

1. **スクリプト**: `scripts/count_vintix_params.py`
2. **設定**: 表「Vintix における Transformer モデルおよび学習設定」と同一  
   - `hidden_dim=256`, `transformer_depth=3`, `transformer_heads=4`, `intermediate_size=128`, `norm_class=LayerNorm`, `mlp_class=GptNeoxMLP`, その他 TrainConfig デフォルト
3. **モデル**: `Vintix()` を `init_model(config, metadata)` で初期化（1グループ・1タスクで代表）
4. **カウント**: `sum(p.numel() for p in model.parameters())`

## 再現手順

```bash
cd vintix_go2
python scripts/count_vintix_params.py
```

出力例:

```
=== 設定（表と一致） ===
  hidden_dim=256, layers=3, heads=4
  intermediate_size=128, norm_class=LayerNorm, mlp_class=GptNeoxMLP
  1グループ: obs_dim=33, action_dim=12

=== モジュール別パラメータ数 ===
  encoder (obs/act/rew encoders + padding): 39,630
  transformer:                             991,616
  head (action decoders):                   200,460
  内訳の合計:                               1,231,706

=== 総パラメータ数 ===
  total: 1,231,706
```

## モジュール別内訳（検証用）

| モジュール | パラメータ数 | 備考 |
|-----------|-------------|------|
| encoder | 39,630 | 観測・行動・報酬エンコーダ（1グループ）+ padding。obs_dim=33, action_dim=12 に依存 |
| transformer | 991,616 | 3層 Transformer（Attention + GptNeoxMLP + LayerNorm）。表の設定で一意 |
| head | 200,460 | 行動デコーダ（1グループ）。action_dim=12 に依存 |
| **合計** | **1,231,706** | 内訳の合計と total は一致（スクリプト内で assert） |

## 注意

- エンコーダ・ヘッドのパラメータ数は **1グループの代表タスクの obs_dim / action_dim** に依存する。  
  現在のスクリプトは `vintix/stats.json` の `go2_walking_ad`（obs_dim=33, action_dim=12）を用いている。
- **All one group** で先に読み込まれるタスクが Go1 等の別ロボットの場合、そのロボットの obs/action 次元が使われ、encoder + head のみ数値が若干変わりうる（Transformer は 991,616 で不変）。
- 表の「パラメータ数」は、上記のとおり **同じ設定で再実行可能な 1,231,706** を記載している。
