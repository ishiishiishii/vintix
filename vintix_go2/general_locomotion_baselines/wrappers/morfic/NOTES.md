# MorFiC → Genesis ラッパ

## 公式との差分

| 項目 | 公式 MorFiC | 本リポラッパ |
|------|-------------|--------------|
| Sim | Isaac Gym | Genesis（評価入口は未接続。CPU 次元チェックのみ） |
| 単ステップ obs | **70**（train.py / pretrained） | Genesis 45 → 近似再構成 70 |
| 履歴 | **30** → flat **2100** | 同寸法のリングバッファ |
| 形態 m | 11（質量×5 + 関節限界×6） | 同キー。Go2 は訓練範囲の中点プレースホルダ |
| 行動 | 12 | actor_body 出力 12 |

**訂正（2026-11-18）:** 初期対応表の「42 × history 15」は `config_shared` のデフォルト値。  
同梱 pretrained `ac_weights_last.pt` は **70 × 30**（adaptation 入力 2100、actor 入力 2166=2100+64+2）。

## 70 次元の内訳（train.py フラグ）

`gravity(3) + commands(15) + dof_pos(12) + dof_vel(12) + actions(12) + last_actions(12) + clock(4) = 70`

Genesis 側は速度コマンド 3 のみ → 残り gait コマンドは 0 埋め。clock は簡易位相（公式 gait index ではない）。

## 接続ステップ

1. ~~次元を checkpoint から確定~~（CPU スモーク通過）
2. ~~評価入口の dry-run 骨格~~（`run_morfic_genesis_eval.py`・Genesis 非起動）
3. ~~Genesis Go2 `--run`~~（2026-11-23: 4 env × 500 step → completion 0%、mean len ≈11.6。近似 obs のゼロショット即転倒）
4. 公式 `scripts/play.py`（Isaac Gym）での holdout 再生と比較し、差分を記録  
   - 2026-11-24: Isaac Gym Preview 4 を `isaacgym_py38` に導入（import OK）  
   - 入口: `run_morfic_official_eval.sh`（#16 ANYmal-B 履歴が GPU 占有中のため play は後続）
5. （任意）gait cmd / clock の再構成改善 → 再評価

## ファイル

| ファイル | 役割 |
|----------|------|
| `morphology.py` | MORPHOLOGY_KEYS / 正規化（third_party 非依存） |
| `obs_adapter.py` | Genesis 45 → MorFiC 70 + history |
| `policy_loader.py` | checkpoint 次元 assert + 最小 actor パス |
| `smoke_cpu_check.py` | GPU 不要スモーク |
| `run_morfic_genesis_eval.py` | Genesis 近似評価（参考のみ・完了判定に使わない） |
| `run_morfic_official_eval.sh` | **公式 Isaac Gym** `play.py` → JSON（完了判定用） |

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/morfic/smoke_cpu_check.py
python wrappers/morfic/run_morfic_genesis_eval.py --dry_run
# GPU 空き時（公式・完了判定）:
bash wrappers/morfic/run_morfic_official_eval.sh --robot go2 --steps 800 --vx 1.0
```
