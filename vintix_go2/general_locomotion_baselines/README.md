# 汎用歩行ベースライン導入（#22）

> Issue: [#22](https://github.com/ishiishiishii/vintix/issues/22)  
> ブランチ（予定）: `exp/general-locomotion-baselines`（本線 #18 評価中のため checkout は GPU 空き／安全時。先端 tip は作成済み）  
> 文献調査: [`LITERATURE_SURVEY.md`](LITERATURE_SURVEY.md)

## 目的

Vintix（＋デコーダ FT）を主張するとき、同条件で並べられる**汎用歩行ベースライン**を用意する。
代表手法と評価でよく選ばれる手法を調査し、OSS をクローン（無ければ論文準拠実装）し、Genesis 評価へ配線する。

## 採択（第1波）

| 優先 | 手法 | リポ | ライセンス |
|------|------|------|------------|
| P0 | GenLoco | HybridRobotics/GenLoco | Apache-2.0 |
| P0 | URMA | nico-bohlinger/one_policy_to_run_them_all | MIT |
| P0 | MorFiC | prakharmishra27/morfic-rl | MIT |
| P1 | McARL | prakharmishra27/McARL（**コード未収録・スタブ**） | MIT |

詳細・見送り理由は `LITERATURE_SURVEY.md`。

## ディレクトリ

```
general_locomotion_baselines/
  LITERATURE_SURVEY.md   # 手法／OSS／採否表
  README.md
  clone_third_party.sh   # shallow clone（third_party は gitignore）
  third_party/           # クローン先（コミットしない）
  wrappers/              # Genesis 評価への薄いラッパ（MorFiC / GenLoco / URMA）
  SOURCE_LICENSES.md     # 出典・ライセンス記録
```

## クローン（低 GPU）

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
bash clone_third_party.sh
```

`third_party/` は巨大なため **コミットしない**（ライセンス要約と clone スクリプトだけをリポに残す）。

## GPU 方針

- **調査・クローン・配線・次元対応表**: #18 と並行可
- **重い学習・大規模評価・pretrained の長時間再生**: #18→#21 と競合させない。GPU 空きを確認してから

## #16 との切り分け

| Issue | 対象 |
|-------|------|
| #16 | 評価**機体**（URDF / Env） |
| #22 | 比較用**アルゴリズム**（本ディレクトリ） |

## 実行ログ

### 2026-11-18 — 文献調査表と採択優先度を確定、P0/P1 クローン着手

- **実行内容:** 5 本以上＋追加の OSS 探索。調査表を `LITERATURE_SURVEY.md` に記載。P0/P1 を shallow clone
- **結果:** GenLoco / URMA / MorFiC を採用（pretrained あり）。McARL リポはスタブのみ。観測対応表を `wrappers/obs_action_map.md` に具体化
- **次:** MorFiC ラッパ実装の続き。pretrained スモークは GPU 空き時（現在 #18 が GPU 100%）

### 2026-11-18 — MorFiC ラッパ骨格＋CPU 次元スモーク（#22）

- **実行内容:** pretrained `ac_weights_last.pt` から入力次元を確定。`wrappers/morfic/` に morphology / obs_adapter / policy_loader を追加。`smoke_cpu_check.py` を実行（GPU・Isaac Gym 不要）
- **結果:** 単ステップ **70**・履歴 **30**（flat 2100）・形態 11→64・adaptation 2・actor 入力 **2166**・行動 12 を確認。合成 Genesis obs(45)→アダプタ→actor まで有限出力。旧対応表の 42×15 は誤りとして訂正
- **次:** Genesis Go2 Env 接続と評価入口（#18 GPU 空き後）。GenLoco/URMA ラッパは後続

### 2026-11-18 — GenLoco / URMA ラッパ骨格＋CPU スモーク（#22）

- **実行内容:** GenLoco pretrained zip から入力 **451**・MLP 451→1024→512→12 を確定し `wrappers/genloco/` を追加。URMA は Go2 観測内訳（desc23/10 → 合計 **380**）をコードから算出し `wrappers/urma/` を追加（actor は RL-X 依存のため deferred）
- **結果:** `smoke_cpu_check.py` 両手法 OK。pace/spin/inverse 3 zip とも入力 451。URMA ckpt magic 確認
- **次:** 低 GPU の配線骨格は一通り完了。Genesis 評価入口と公式再生比較は #18→#21 GPU 空き後。#16 立位レンダも同タイミング

### 2026-11-19 — MorFiC Genesis 評価入口の dry-run 骨格（#22）

- **実行内容:** `wrappers/morfic/run_morfic_genesis_eval.py` を追加。既定は `--dry_run`（Genesis / GPU 非起動）。checkpoint 寸法＋合成 obs→actor を再利用
- **結果:** dry-run OK（obs 70 / hist 2100 / action 12）。`--run` は #18/#21 解放まで明示拒否
- **次:** GPU 空き後に `--run` 本体（Go2 Env 接続）と公式 play 差分

### 2026-11-19 09:04（autopilot）: GenLoco 評価入口 dry-run + 公式評価手順書

- **実行内容:** `wrappers/genloco/run_genloco_genesis_eval.py`（`--dry_run` 既定）と `EVAL_PROTOCOL.md`（MorFiC/GenLoco/URMA の公式入口・本線指標対応）を追加。#18 GPU には非介入
- **結果:** GenLoco dry-run OK（obs 451 → action 12）。`--run` は明示拒否のまま
- **次:** GPU 空き後に MorFiC/GenLoco `--run`。URMA は RL-X 待ち

### 2026-11-23 05:14（autopilot）: MorFiC Genesis `--run` 初回配線

- **実行内容:** `MorficActorPolicy` + `run_morfic_genesis_eval.py --run`（Go2、4 env × 500 step）。観測は Genesis45→70 近似
- **結果:** `results/morfic_genesis/morfic_go2_ne4_s500.json` — completion **0%**、mean episode length **≈11.6**、mean reward/step **≈−0.033**（即転倒）。配線自体は通る
- **考察:** Isaac Gym pretrained を近似 obs でゼロショット再生しても歩けないのは想定内。比較表のベースラインとしては「未適応・近似配線」として記録。公式 play 差分と GenLoco `--run` が次
- **次:** GenLoco `--run` 実装。URMA は RL-X。必要なら obs 再構成の改善（gait cmd / clock）
