# 出力レイアウト

```
online_decoder_ft/
  models/<model_key>/     # model_N.pt + model_N/ (Vintix), run_config.json
  result/<model_key>/     # 学習曲線, 評価 CSV, 比較グラフ
```

チェックポイント命名（iter N 保存時）:

| ファイル | 内容 |
|---------|------|
| `model_25.pt` | PPO (rsl_rl) checkpoint |
| `model_25/` | Vintix export（`save_vintix.py` 用） |

例: `models/go2_without/model_100.pt` + `models/go2_without/model_100/`
