# 全ロボット専門家の再学習手順（#16 再開）

更新: 2026-11-25（日本時間）  
Issue: [#16](https://github.com/ishiishiishii/vintix/issues/16)

## なぜやり直すか

- 動画で Aliengo / B1 がうまく歩いて見えない
- 数値でも不十分（例: B1 指令比 ≈60% ぎりぎり）
- 既存手法リポジトリの設定を十分に移植できていない

## 参照（GenLoco は使わない）

| 機体 | 主参照 | 移植する主な量 |
|------|--------|----------------|
| Aliengo | MorFiC `aliengo_config` + 同系統 URDF | h/init、kp/kd、action_scale、指令帯の代表点 |
| B1 | MorFiC `b1_config` + **unitree URDF**（MorFiC URDF は沈む） | kp/kd、h、action_scale |
| ANYmal-B | MorFiC ANYmal-**C** config を B に寄せる | kp、h |
| Spot | **Isaac Lab SpotFlat** | init z≈0.5、足上げ等。報酬一式は丸コピーしない |
| Laikago | **RaisimGym** | init≈0.46、高さ≈0.44、kp40/kd1 |

詳細: `PARAMETER_DESIGN.md` / `MORFIC_EXPERT_PLAN.md` / `ITERATION_POLICY.md`

## 品質の合格線（2026-11-25・ユーザー合意）

- 完走率 **≥95%**
- 指令速度に対する平均前進 **≥80%**（指令比 95% は厳しすぎるため 80% に変更。旧 60% / 90% / 95% 案は撤回）
- **動画を見て**明らかに歩行していること（立位・すり足・すぐ転倒は不合格）

## イテレーション

1. 初回: `--max_iterations 301`、`save_interval 20`、**別ログディレクトリ**（旧を上書きしない）
2. 品質未達かつ終盤まで曲線上昇 → 601 → 最大 901
3. 途中崩壊 → iter を増やさず設定直し

## 実行順（GPU 1 本・直列）

いま #11 の DR 収集や待ちスクリプトが動いている場合は、**止める前にユーザー確認**。空いたら次の順:

1. **Aliengo**（MorFiC 揃え）→ 品質評価＋動画
2. **B1**（MorFiC 制御値 + unitree URDF）→ 評価＋動画
3. **ANYmal-B**（C 寄せ）→ 評価＋動画
4. **Spot**（Isaac Lab 高さ）→ 評価＋動画
5. **Laikago**（RaisimGym）→ 評価＋動画（失敗したら URDF/姿勢を直して再試行）

各機共通の流れ:

```bash
# 例: Aliengo（ログ名は機体ごとに変える。旧ディレクトリは触らない）
cd /workspace
python Genesis/examples/locomotion/train.py \
  -e aliengo-walking-retrain-morfic -r aliengo \
  --max_iterations 301 --seed 1

# 完了後
python vintix_go2/robot_eval_expansion/eval_expert_quality.py \
  -r aliengo --ckpt <model_xxx.pt> --cmd_vx <指令> --video
# 動画を目視確認してから次機体へ
```

## 記録

- 進捗・数値・動画パスは **#16 コメント**と Notion「評価ロボット拡充」ページへ
- 時刻は **日本時間（JST）**
- 採用した出典（どのリポジトリのどの値か）を機体ごとに残す
