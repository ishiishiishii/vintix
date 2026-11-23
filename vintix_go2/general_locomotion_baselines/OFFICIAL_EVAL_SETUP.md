# 公式シミュレータでの評価手順（Issue #22）

比較の主数値は、できるだけ各手法の公式シミュレータで取る。
Genesis への近似配線は参考列。

## GenLoco（できている）

- シミュレータ: PyBullet
- コマンド:
  ```bash
  cd /workspace/vintix_go2/general_locomotion_baselines
  python wrappers/genloco/run_genloco_pybullet_eval.py --robot go1 --policy pace --num_episodes 3
  ```
- 依存: `pybullet`, `gym`（TensorFlow は不要。学習済み zip を直接読む）
- 結果例: `results/genloco_pybullet/genloco_pace_go1_ep3_s300.json`

## URMA（短い公式再生までできた）

- シミュレータ: MuJoCo（RL-X）
- 隔離環境: conda `urma_eval`（Python 3.11.4）
- コマンド:
  ```bash
  bash wrappers/urma/run_urma_official_eval.sh --env multi_robot --num_episodes 1 --max_steps 100 --nr_envs 16
  ```
- 注意:
  - 公式の `PPO.test` は終了判定が無く止まらない → 本リポのスクリプトが有限ステップに直している
  - このマシンでは JAX の CUDA（cuDNN）が足りず **CPU** で動くことがある
- 結果例: `results/urma_official/urma_multi_robot_ep1_s100.json`（return ≈5.4、100/100 ステップ）

## MorFiC（いま止まっている理由）

- 公式再生: Isaac Gym + 同梱の `scripts/play.py`
- このコンテナには **Isaac Gym が入っていない**
- 入れる／別環境で回す／比較から外す、の判断が必要
- それまでは Genesis 近似配線の結果だけを「参考」と明記する

## McARL

- 公開リポに学習コードが無く、いまは対象外
