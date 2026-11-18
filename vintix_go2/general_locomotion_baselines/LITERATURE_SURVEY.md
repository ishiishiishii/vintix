# 汎用歩行ポリシー手法の文献調査（#22）

日付: 2026-11-18。テーマ: クロス身体／形態汎化の**代表手法**と、評価でよく選ばれる比較ベースライン。
実装前必須調査（`literature-design.md`）。論文サーベイ（MorFiC / Rapid）＋指標調査（`go1_body_scale_zeroshot/metrics_literature_survey.md`）＋追加 OSS 探索。

## 調査の問い

1. コミュニティが「汎用歩行ポリシー」として引用・比較する代表手法は何か
2. OSS（公式／第三者）はあるか。ライセンスは再配布可能か
3. 本リポ（Genesis + Vintix 評価格子）で比較実験のベースラインとして採否するか

## 採否サマリ（再現優先度）

| 優先 | 手法 | 採否 | 理由（1行） |
|------|------|------|-------------|
| **P0** | GenLoco | **採用（第1波クローン）** | 形態DRの古典代表。OSS+pretrained。他論文の比較表に頻出 |
| **P0** | URMA（One Policy to Run Them All） | **採用（第1波クローン）** | 多身体単一方策の代表。OSS+pretrained。MorFiC コスト表でも引用 |
| **P0** | MorFiC | **採用（第1波クローン）** | ゼロショット形態転移の最新代表。OSS+Go2 pretrained。問いが本線に近い |
| **P1** | McARL | **採用候補（コード未収録）** | GitHub はあるが現状 LICENSE/README/docs のみ。実装コード待ち／論文からの再実装が必要 |
| **P2** | Body Transformer (BoT) | **参考（第2波）** | MorFiC 比較に登場するが主眼は身体グラフ inductive bias。多機種汎化の本命ではない |
| **P2** | MetaMorph | **参考** | モジュラ／Unimal。商用四足クロス身体の比較軸としては弱い |
| **—** | ManyQuadrupeds | **参考（OSS 未公開）** | 代表だが公式「Code Coming Soon」。DeepTransition の CPG-RL は別論文 |
| **—** | MetaLoco | **見送り（OSS なし）** | メタRL万能歩行として重要だが公式コード未確認 |
| **—** | Rapid Embodiment Adaptation | **見送り（OSS なし）** | オンライン身体適応。比較対象になり得るがコード未公開 |
| **—** | RMA | **参考（別軸）** | 地形・動力学適応の古典。形態汎化の直接ベースラインではない |
| **—** | Walk These Ways / DreamWaQ | **見送り（本課題外）** | 歩容条件・地形頑健。汎用「クロス身体」主張の主軸ではない |

## 手法一覧表

| 手法 | 論文 | 「汎用」の定義 | OSS | よく比較されるか | 採否 |
|------|------|----------------|-----|------------------|------|
| **GenLoco** | CoRL 2022 / [2209.05309](https://arxiv.org/abs/2209.05309) | 手続き的形態DRで単一方策→未見四足ゼロショット | **あり** [HybridRobotics/GenLoco](https://github.com/HybridRobotics/GenLoco)（Apache-2.0、pretrained） | **高**（McARL/MorFiC/指標調査で頻出） | **採用 P0** |
| **ManyQuadrupeds** | ICRA 2024 / [2310.10486](https://arxiv.org/abs/2310.10486) | CPG+PF で 2–200 kg・多形態を単一方策 | **公式未公開**（[project](https://miladshafiee.github.io/ManyQuadrupeds/) “Coming Soon”）。関連: [DeepTransition](https://github.com/MiladShafiee/DeepTransition) は別論文 | **高**（比較表の定番） | **参考**（再実装は要検討・後回し） |
| **URMA** | CoRL 2024 / [2409.06366](https://arxiv.org/abs/2409.06366) | 形態非依存 encoder/decoder で多形態 MTRL 単一方策 | **あり** [nico-bohlinger/one_policy_to_run_them_all](https://github.com/nico-bohlinger/one_policy_to_run_them_all)（MIT、pretrained） | **高** | **採用 P0** |
| **MorFiC** | CoRL 2026 / [2603.14554](https://arxiv.org/abs/2603.14554) | 単一ソース+形態DR。critic を FiLM で形態条件付け | **あり** [prakharmishra27/morfic-rl](https://github.com/prakharmishra27/morfic-rl)（MIT、Go2 pretrained） | **高**（最新・本線近傍） | **採用 P0** |
| **McARL** | arXiv 2025 / [2505.18418](https://arxiv.org/abs/2505.18418) | 形態・制御ベクトルを actor/critic に条件付け、単機学習→他機ゼロショット | **リポはあるがコード未収録** [prakharmishra27/McARL](https://github.com/prakharmishra27/McARL)（MIT・README のみ） | **中〜高**（自前比較表で GenLoco/URMA/BoT/ManyQuad と並記） | **P1・実装待ち** |
| **Body Transformer** | CoRL 2024 / [2408.06316](https://arxiv.org/abs/2408.06316) | 身体グラフ+masked attention の方策アーキテクチャ | **あり** [carlosferrazza/BodyTransformer](https://github.com/carlosferrazza/BodyTransformer)（LICENSE=Other/NOASSERTION） | **中**（MorFiC 比較） | **参考 P2** |
| **MetaMorph** | ICLR 2022 | Transformer 条件付けのユニバーサル制御（モジュラ形態） | **あり** [agrimgupta92/metamorph](https://github.com/agrimgupta92/metamorph)（license 未設定） | **中**（一般ユニバーサル制御。四足商用機比較では薄い） | **参考** |
| **MetaLoco** | arXiv 2024 / [2407.17502](https://arxiv.org/abs/2407.17502) | 少数手続き生成体+メタRL でゼロショット実機 | **公式未確認**（第三者別題あり） | **中** | **見送り**（OSS なし） |
| **Rapid Embodiment Adaptation** | arXiv 2026 / [2608.01506](https://arxiv.org/abs/2608.01506) | 大規模身体DR + 短時間オンライン同定 | **公式未公開**（[project](https://embodiment-adaptation.github.io/) は paper/video のみ） | **中**（ゼミ読了） | **見送り**（OSS なし） |
| **RMA** | RSS 2021 / [2107.04034](https://arxiv.org/abs/2107.04034) | 隠れ環境因子のオンライン適応（地形・荷重） | 参考実装は RaisimGymTorch 系（[antonilo/rl_locomotion](https://github.com/antonilo/rl_locomotion) は派生・GPL-3） | **高**（適応モジュールの古典。形態汎化とは軸が違う） | **参考（別軸）** |
| **URMAv2 Extreme ER** | arXiv 2025 / [2509.02815](https://arxiv.org/abs/2509.02815) | URMA 拡張・極端形態ER | 要確認（URMA 派生） | **中** | **後追い** |
| **Multi-Loco / QWM 等** | 2025–2026 | 多形態・世界モデル条件 | 個別に再探索 | 低〜中 | 本課題の第1波外 |

## 設計メモ（比較実験に使うとき）

| 項目 | 文献からの読み | 本リポへの方針（暫定） |
|------|----------------|------------------------|
| 変える量 | GenLoco/MorFiC: リンク寸法・質量・摩擦・PD。URMA: 多 URDF 同時。McARL: 形態ベクトル乱択 | まず既存 4 台＋#16 導入機で**ゼロショット／同条件評価**。学習時 DR 範囲は各手法の論文設定を優先 |
| 指標 | Survival/Success + 速度が主流（MorFiC 警告: Survival だけでは不十分）。URMA/GenLoco は return 寄り | 既存 #14/#15 指標（完走・速度・追従・return）で揃える |
| 正規化 | 論文ごとに return 正規化や列内正規化が異なる | **生値＋本リポ正規化**の両方を保存し、論文再現値は別カラム |
| シミュレータ | 多くが Isaac Gym / PyBullet。本環境は Genesis | **ラッパで観測・行動・PD を合わせる**。完全バイナリ互換は目指さない。差分を README に明記 |
| 学習コスト | MorFiC Table 16: URMA ~311 GPU-h、GenLoco 53–213 h 等 | **重い学習は #18→#21 後**。第1波はクローン＋配線＋（あれば）pretrained 評価スモーク |

## OSS 探索ログ（「無い」判定の根拠）

| 手法 | 探索した場所 | 結果 |
|------|--------------|------|
| ManyQuadrupeds | 著者ページ・project・GitHub `MiladShafiee` | “Code (Coming Soon)”。DeepTransition は Nature Comms 別論文 |
| MetaLoco | arXiv HTML・alphaXiv・`MetaLoco`/`metarl quadruped` 検索 | 公式リポなし。`chenci107/metarl_for_quadruped` は別題（モータ固着） |
| Rapid Embodiment | [embodiment-adaptation.github.io](https://embodiment-adaptation.github.io/)・arXiv | Paper/Video のみ。コード URL なし |
| McARL | project “View Code” → `prakharmishra27/McARL` | リポあり（MIT）だが **2026-11-18 clone 時点で訓練コード無し**（docs/画像のみ） |
| MorFiC | project → `prakharmishra27/morfic-rl` | **あり（MIT）** ※同作者の `MorFiC` はページ用リポの可能性 |

## 次アクション

1. ~~調査表~~（本ファイル）
2. P0/P1 を `third_party/` に shallow clone（`.gitignore`。ライセンス記録）
3. Genesis 評価への薄いラッパ骨格（観測次元・行動スケールの対応表）
4. pretrained がある手法から**低 GPU スモーク**（#18 空き時）
5. ManyQuadrupeds は OSS 公開待ち。必要なら CPG+PF の論文準拠再実装 Issue を切り出す
