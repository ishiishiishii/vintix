# 評価ロボット選定の文献調査（#16）

日付: 2026-11-18。テーマ近傍（クロス身体・形態汎化・ゼロショット歩行・多機種単一方策）の主要論文から、
学習・評価・実機に使われた四足を洗い出し、本リポジトリへの導入優先度を決める。

決め打ち禁止（Issue #16）。頻度が高い機体を優先し、URDF 公開・ライセンス・既存 Env の有無で採否する。

## 調査した主要論文（8 本以上）

| 論文 | arXiv / 出典 | 使った主なロボット | 役割 |
|------|--------------|-------------------|------|
| GenLoco (2022) | 2209.05309 / HybridRobotics/GenLoco | A1, Go1, Laikago, Aliengo, Sirius, Mini Cheetah, ANYmal-B/C, Spot, SpotMicro | 単一ポリシーの多機種ゼロショット（sim） |
| ManyQuadrupeds (2023) | 2310.10486 | A1, Go1, Aliengo, Laikago, B1, Spot, ANYmal-B/C, Mini Cheetah, LittleDog, SpotMicro, Solo, HYQ 等 16 機 | 多形態同時学習。実機は Go1/A1 |
| MetaLoco (2024) | 2407.17502 | 学習=手続き生成（Go1/Aliengo 由来）。実機=Go1, Go2, Aliengo | メタ RL のゼロショット実機 |
| Sampling Strategies / Universal Quad Policies (2025) | 2510.07094 | sim=A1/Laikago/ANYmal 系。実機=ANYmal（+13 kg） | 万能方策のサンプリングと sim2real |
| QWM Morphology-Conditioned World Model (2026) | 2604.08780 | 学習コホート＋holdout。実機=ANYmal-D, Go1。言及=B2 | 形態条件付き世界モデルのゼロショット |
| MorFiC (2026) | 2603.14554 | 学習=Go2+形態 DR。実機=Go1/Go2 | 形態転移・Survival+速度 |
| Zero-Shot Retargeting + HKD-MPC (2022) | 2209.14123 | A1 / Laikago → Mini Cheetah | 方策のゼロショットリターゲット |
| ANYmal Parkour (2023) | 2306.14874 | ANYmal | 実機スキル学習（形態横断ではないが工業級基準） |

補助: `vintix_go2/go1_body_scale_zeroshot/metrics_literature_survey.md`（指標調査 40 本）もロボット名の出現を参照。

## 頻度表（上記＋指標調査からの再掲）

「登場」= 学習・sim 評価・実機のいずれかに明示。複数論文で繰り返し出るほど優先。

| ロボット | 登場論文（代表） | 頻度帯 | 本リポ現状 | URDF・ライセンス目安 | 採否 |
|----------|------------------|--------|------------|----------------------|------|
| Go1 | GenLoco, ManyQuad, MetaLoco, MorFiC, QWM, … | 高 | 本線（#10/#18） | あり | **維持（本線）** |
| Go2 | MetaLoco, MorFiC, … | 中〜高 | 本線 | あり | **維持（本線）** |
| A1 | GenLoco, ManyQuad, Sampling, Retargeting, … | 高 | 本線 | あり | **維持（本線）** |
| Mini Cheetah | GenLoco, ManyQuad, Retargeting | 中〜高 | 本線 | あり | **維持（本線）** |
| ANYmal (B/C/D) | GenLoco, ManyQuad, Sampling, QWM, Parkour | **高** | C の Env/URDF あり（`anymalc_env.py`） | ANYmal C: BSD 系（`ANYmal_c_license.txt`） | **採用（検証優先）** — まず C。B/D は後回し |
| Aliengo | GenLoco, ManyQuad, MetaLoco | **高** | **未導入** | `unitree_ros` BSD-3-Clause | **採用（新規導入 1 番）** |
| Laikago | GenLoco, ManyQuad, Sampling, Retargeting | **高** | Env/URDF あり（`laikago_env.py`） | Unitree 系（公式 ros にも同梱） | **採用（検証優先）** |
| Spot (BD) | GenLoco, ManyQuad | 中 | なし | 商用・再配布制限の可能性大 | **見送り**（ライセンス） |
| SpotMicro | GenLoco, ManyQuad | 中（オープン玩具級） | Env/URDF あり | オープン | **任意**（軽量形態の予備。優先度低） |
| Solo / HYQ / LittleDog | ManyQuad 中心 | 低〜中 | なし | 論文・OSS 依存 | **見送り**（頻度とコスト） |
| B1 / B2 | ManyQuad / QWM | 低〜中 | なし | Unitree BSD 系の見込み | **保留**（大型 Unitree。Aliengo 後に検討） |
| Sirius / Honey / Stoch 等 | 単発寄り | 低 | なし | 公開状況まちまち | **見送り** |

## 導入一覧（確定）

タスク1 で扱う順（報酬・PPO はタスク2・本線後）:

1. **Aliengo**（新規）— 文献頻度高・本線 4 台に無い Unitree 中型・URDF 公開（BSD-3）
2. **ANYmal C**（既存資産の検証）— 工業級・質量スケールが本線と大きく違う。URDF 合計質量 ≈52.1 kg（仕様表と一致）。摩擦は URDF 内未指定 → Genesis 側デフォルト／論文値を要確認
3. **Laikago**（既存資産の検証）— 古い Unitree・OOD 形態として文献で多用。URDF 合計質量 ≈25.2 kg（公称に近い）。摩擦は URDF 内未指定 → 要確認
4. （任意）SpotMicro — 既にあるが科学的優先度は Spot 代替としては弱い。余力があれば姿勢確認のみ

見送り: Spot (BD)、Solo/HYQ/LittleDog、Sirius 等。B1/B2 は Aliengo 完了後に再検討。

## 既存 URDF の一次チェック（2026-11-18）

| ロボット | 合計質量 (URDF) | 怪しい質量 | 摩擦ブロック | メモ |
|----------|-----------------|------------|--------------|------|
| ANYmal C | 52.14 kg | なし | なし | ライセンスファイルあり。導入時に脚長・立位高を論文値と突合 |
| Laikago | 25.17 kg | なし | なし | chassis 13.7 kg。公式公称〜25 kg 帯と整合 |
| SpotMicro | 4.52 kg | なし | なし | 任意枠 |
| Go2 | 15.02 kg | imu/radar=0（無視可） | なし | 本線 |

※「摩擦なし」は不正確定ではない（シミュレータ既定に依存）。導入時に脚先摩擦を明示設定するか、既存 Go1 系と同じ扱いをするかを決める。

## 次の実装ステップ（タスク1）

1. ~~Aliengo: `unitree_ros` から URDF+メッシュを取り込み（ライセンス全文を同居）、質量・慣性チェック~~ **済（2026-11-18）**
   - 配置: `Genesis/genesis/assets/urdf/aliengo/`（`LICENSE` + `SOURCE.md` + `urdf/` + `meshes/`）
   - 質量 ≈24.94 kg、脚長 calf/foot z=0.25 m。`package://` メッシュパスは `../meshes/` に置換済み
   - 残り: `aliengo_env.py`（既存 Env を import／継承）→ 立位レンダ確認
2. ANYmal C / Laikago: 既存 Env の立位・観測次元・PD を本線 4 台と比較表にまとめ、仕様表へ追記
3. 仕様表 `robot_specifications_table.md` を更新（Aliengo/Laikago 列を追加済み。base 寸法は後日）
4. タスク2（報酬+PPO）は #18→#21 一段落後

## 出典メモ（参考文献ページ）

Notion「参考文献」単独ページは API 検索で未検出のため、本ファイルと Issue #16 を正本とする。必要なら研究進捗配下に後から作成して転記する。
