# ベースライン比較とシミュレータ方針（#22）

更新: 2026-11-24。ユーザー指示: シミュレータが手法ごとに違ってよいか文献で判断し、
必要なら公式シミュレータを入れ、ダメなら Genesis へ寄せる。
完了条件: **各ベースラインが、その論文で動かしていたロボットで安定歩行できること**。

## 結論（判断）

**「手法が歩けることの検証」と「公式数値の再現」には、論文どおりのシミュレータを使うのが正しい。**
絶対 return をシミュレータ横断で並べる公平比較は、文献上も危険なので主張しない。

| 目的 | 方針 |
|------|------|
| ベースラインが安定歩行できることの証明 | **公式シミュレータで再生**（GenLoco→PyBullet、URMA→MuJoCo、MorFiC→Isaac Gym） |
| 本線（Genesis + Vintix）との同条件ヘッドツーヘッド | 将来オプション。いまは「公式再生で歩行確認」を完了条件の主軸にする |
| Genesis 薄い配線 | 参考列のみ。完走 0% でも手法の失格にはしない（観測・接触・タスク差） |

したがって **MorFiC 用に Isaac Gym を導入する**（既存の PyBullet / MuJoCo は既に利用可）。
Genesis への本格移植は「公式再生がどうしても不可能」な場合のフォールバック。

## 文献調査（5 本以上）

| 論文・資料 | 変える量／設定 | 指標 | シミュレータに関する知見 | 本方針への採否 |
|------------|----------------|------|--------------------------|----------------|
| **GenLoco** (CoRL 2022) | 形態 DR | 模倣・ゼロショット | 公式は **PyBullet** + pretrained | **採用**: PyBullet で Go1 等を再生 |
| **URMA** (CoRL 2024) | 多 URDF MTRL | return・実機 | 公式は **MuJoCo / RL-X** | **採用**: MuJoCo で Go2 等を再生 |
| **MorFiC** (CoRL 2026) | 形態 DR + FiLM critic | ゼロショット・追跡 | 公式は **Isaac Gym**（Go2 学習） | **採用**: Isaac Gym を入れて公式 `play.py` |
| **Rama et al. 2020** [2010.14444](https://arxiv.org/abs/2010.14444) | 同一制御タスクを複数エンジンへ | 成功・報酬 | **物理エンジン間で方策は一般化しない**（特に PyBullet 学習） | **採用**: 横断の生 return 比較を禁止 |
| **Physics engines review 2024** [2407.08590](https://arxiv.org/html/2407.08590v2) | 9 エンジン比較 | 転送性能 | エンジン差が評価を歪める | **採用**: 上記と同趣旨 |
| **Sim2Sim quadruped (IsaacLab→MuJoCo)** | 同一方策の転送 | 歩容・接触力 | 歩けても関節・接触は系統的にずれる | **採用**: 「公式で歩く」≠「Genesis でも同じ数値」 |
| **Cross-engine humanoid benchmark** | 同一指令の MuJoCo vs PyBullet | 力・追従 RMSE | 同一 URDF でも力が大きく違う | **参考** |

## いまの環境

| シミュレータ | 状態 | 担当手法 |
|--------------|------|----------|
| PyBullet | **利用可**（コンテナ内） | GenLoco — Go1 pace で完走 3/3 済 |
| MuJoCo 3.3.4 | **利用可** | URMA — Go2 1000 step 完走 3/3 済 |
| Isaac Gym Preview | **未導入**（`ModuleNotFoundError`） | MorFiC — 必須。NVIDIA 配布の Preview 4 を別 conda 環境へ |
| Genesis | 本線 | Vintix／専門家 PPO。ベースラインの公式検証には使わない |

## Isaac Gym 導入手順（予定）

1. NVIDIA Developer から **Isaac Gym Preview 4** の tarball を取得（要アカウント・ライセンス同意）
2. 本線の conda（PyTorch 2.5 / Genesis）を壊さないよう **別 env**（例: `isaacgym_py38`、MorFiC 推奨 Py3.8 + torch 1.10）に入れる
3. `third_party/MorFiC` で `scripts/play.py --robot go2`（論文どおり）→ 生存・速度を JSON 化
4. 成功したら主表に MorFiC 行を戻す

※ tarball がリポに無い場合はユーザーに配置パスを依頼する（勝手な再配布は不可）。

## 完了条件（#22・更新）

各 P0 手法について:

1. **公式（または論文指定）シミュレータ**で学習済み方策を再生できる
2. 論文で動かしていたロボット（例: GenLoco→Go1、URMA→Go2、MorFiC→Go2）で **安定歩行**
   - 定量: エピソード上限までの生存率、前進速度（指令があれば追従誤差）
   - 定性: 転倒しない歩容（必要なら動画）
3. 結果を `results/MAIN_TABLE.md` と Notion に記録

Genesis 近似列は参考のまま残してよいが、完了判定には使わない。
