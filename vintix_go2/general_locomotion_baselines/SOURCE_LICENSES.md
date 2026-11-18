# 第三者コードの出典・ライセンス（#22）

`third_party/` 配下は **再配布のために本リポへコミットしない**。ローカルで `clone_third_party.sh` により取得する。
利用・改変・公開時は各 upstream の LICENSE に従う。

| ディレクトリ名 | Upstream | SPDX / 表記 | メモ |
|----------------|----------|-------------|------|
| `GenLoco` | https://github.com/HybridRobotics/GenLoco | Apache-2.0 | pretrained policies 同梱。PyBullet / motion imitation 系 |
| `URMA_one_policy_to_run_them_all` | https://github.com/nico-bohlinger/one_policy_to_run_them_all | MIT | 依存 RL-X は別リポ（README の pin commit） |
| `MorFiC` | https://github.com/prakharmishra27/morfic-rl | MIT | Go2 学習 checkpoint 同梱。Isaac Gym 前提 |
| `McARL` | https://github.com/prakharmishra27/McARL | MIT | **コード未収録**（README/docs のみ）。再取得 or 論文準拠実装が必要 |

## Clone tip SHA（2026-11-18）

| 名前 | SHA |
|------|-----|
| GenLoco | `98b2dd77bb53ac32f04a15806ff4c6d0e7a1b70e` |
| URMA | `7c1819444028c529b0a964fe3f32878333b1310a` |
| MorFiC | `bba9efebfd5916f01aca9f925e929e2582ed7281` |
| McARL | `e0703ee77f7b3d37df97c46fbe56f9291109a6cc` |

## Pretrained の有無（clone 時点）

| 手法 | Pretrained |
|------|------------|
| GenLoco | あり（`motion_imitation/data/policies/morphology_generator_*.zip`） |
| URMA | あり（`experiments/pre_trained_model`） |
| MorFiC | あり（`runs/.../ac_weights_last.pt`） |
| McARL | なし（コード自体なし） |

## クローン時に記録する情報

各 `third_party/<name>/.clone_sha` に取得時の tip SHA を書く（スクリプトが自動生成）。

## 本リポ側の改変方針

- upstream 本体を直接編集せず、`wrappers/` で観測・行動・評価プロトコルを合わせる
- ライセンス無視の再配布はしない
- Genesis 移植時の論文との差分は各手法の `wrappers/<method>/NOTES.md` に明記する
