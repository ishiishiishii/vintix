# 4 台体スケールのゼロショット限界 — 実験レポート

> やったこと: Go1・Go2・A1・Mini Cheetah で質量／大きさ（質量固定）／大きさ（密度一定）を変え、leave-one-out Vintix・通常体 Expert 流用・端点 Body PPO をシード 0–9 で評価した（#10）。
> これから: #15（評価指標の網羅保存）→ #18（同体のみ学習 Vintix）→ #13（下限）→ #11（DR 履歴）。

---

### 概要

パイプライン（probe 105 + main 2100 + PPO 220）は **2026-11-16 18:07 に完了**（failed なし）。
Go1 以外では「通常体で高完走」が成り立たない。ゼロショット失敗の多くは Body PPO が歩ける＝**適応失敗**だが、一部端点は PPO も失敗＝**物理／学習限界**。

### 目的

どれくらい体が違うと、アルゴリズム蒸留（Vintix）のゼロショットだけでは歩けなくなるかを、4 台・3 軸で定量する。

### 方法

| 項目 | 内容 |
|------|------|
| 方策 | Vintix（`<robot>_without`）／通常体 Expert 流用／端点 Body PPO（500 iter） |
| 評価 | シード 0–9 × 10 env × 10 ep（最大 1000 step） |
| 指標 | Success rate（完走率）、episode return |
| 実験 dir | `vintix_go2/robot_body_scale_zeroshot/`（ブランチ `exp/robot-body-scale-zeroshot`） |
| Issue | [#10](https://github.com/ishiishiishii/vintix/issues/10) |
| Notion | [4台体スケールのゼロショット限界](https://app.notion.com/p/41ad65e6123f819e9cabee1835c71c90)（`41ad65e6-123f-819e-9cab-ee1835c71c90`） |

### 結果

完走率 ≥80% の帯（シード 0–9）:

| ロボット | 軸 | V@1.0 | V帯 | E@1.0 | E帯 |
|---|---|---|---|---|---|
| Go1 | mass | **100** | **0.25–1.25** | 100 | 0.6–1.1 |
| Go1 | size | **100** | **0.55–1.25** | 100 | 0.65–1.1 |
| Go1 | size_density | **100** | **0.8–1.0** | 100 | 0.9–1.0 |
| Go2 | mass | **0** | **0.4–0.7** | 100 | 0.25–1.1 |
| Go2 | size | **0** | **0.5–0.7** | 100 | 0.6–1.1 |
| Go2 | size_density | **0** | **0.8–0.9** | 100 | 0.8–1.0 |
| A1 | mass | **59±8** | **0.4–0.9** | 100 | 0.4–1.2 |
| A1 | size | 59±8 | 0.6–0.9 | 100 | 0.6–1.2 |
| A1 | size_density | 59±8 | **0.9 のみ** | 100 | 0.8–1.0 |
| Mini Cheetah | 各軸 | ~0 | なし※ | ~99 | 質量 0.7–1.5 等 |

※ Mini Cheetah の大サイズで完走率が上がる点があるが return は負 → 歩行ではない。

Body PPO 端点の切り分け（完走率 % / return）:

| 点 | Vintix | Expert | Body PPO | 解釈 |
|---|---|---|---|---|
| Go2 mass 2.0 | 0 | 0 | **100** / 19.3 | 適応失敗 |
| Go2 size 1.3 | 0 | 0 | **99** / 21.2 | 適応失敗 |
| Go2 size_density 1.3 | 0 | 0 | **0** | 物理限界 |
| A1 全 6 端点 | 低〜0 | 低〜0 | **≈100**（密度1.3は r≈10.6） | 適応失敗 |
| Go1 size 1.5/2.0 | 0 | 0 | **0** | 物理限界 |
| Go1 size_density 1.3 | 0 | 0 | **0** | 物理限界 |
| MC mass 2.0 / size 1.3 / dens 1.3 | 低〜偽完走 | 混在 | **0** | 物理限界寄り |
| MC size 0.6 / mass 0.5 / dens 0.6 | ~0 | ~0 | **≈96–100** | 適応失敗 |

図: `figures/<robot>_{mass,size,size_density}_scale_completion_return_mean_std_20261117.png`

### 考察・いま言えること

- Go1 型（通常体 OK・重い側崖）は他台に一般化できない。
- Go2 は軽い／小さい側にゼロショットの島があり、1.0× は 0%。
- Expert 流用の方が Vintix より帯が広いことが多い。
- ゼロショット失敗の多くは「体が歩けない」ではなく適応失敗（Body PPO 可）。
- 物理限界として残るのは Go1 大サイズ系・Go2 密度一定 1.3・MC の重い／大きい端点。
- MC size 1.3 は Expert≈99% なのに Body PPO=0% → 500 iter の学習失敗の可能性あり（#9 系と近い論点）。

### 次の予定

1. **#15** 評価指標の網羅保存
2. **#18** 同体のみ学習 Vintix
3. **#13** 軽い／小さい側の下限
4. **#11** DR 履歴収集

---

<details>
<summary><strong>詳細</strong></summary>

## 1. 成果物

- `results.csv`（339 点）
- `pipeline_status.json`（probe/main/ppo 全 done、failed 空）
- `figures/*_20261117.png`（12 枚）
- Notion: `41ad65e6-123f-819e-9cab-ee1835c71c90`

## 2. パイプライン完了時刻

2026-11-16 18:07:35（`pipeline finished`）

</details>
