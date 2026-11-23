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

## MorFiC（いま止まっている理由）

- 公式再生: Isaac Gym + 同梱の `scripts/play.py`
- このコンテナには **Isaac Gym が入っていない**（`import isaacgym` 不可）
- 入れるときは公式の Isaac Gym Preview 手順に従い、隔離環境を推奨
- それまでは Genesis 近似配線の結果だけを「参考」と明記する

## URMA（いま止まっている理由）

- 公式再生: RL-X（Flax）+ `experiments/test.sh`
- 手順の骨子（公式 README）:
  1. `conda create -n urma python=3.11.4`
  2. 指定コミットの RL-X を入れる
  3. 本手法リポを `pip install -e .`
  4. `experiments/test.sh` で学習済み重みを再生
- 本リポ側は観測 380 次元の確認まで完了。行動の実行は上記環境が必要

## McARL

- 公開リポに学習コードが無く、いまは対象外
