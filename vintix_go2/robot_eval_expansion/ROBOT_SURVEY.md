# 評価ロボット拡充 — 文献調査と導入計画（#16）

Issue: [#16](https://github.com/ishiishiishii/vintix/issues/16)  
方針: タスク1（導入・URDF検証）を先に。タスク2（報酬+PPO）は #18→#21 など本線後。

## A. 文献での登場頻度（主要論文）

| ロボット | GenLoco | ManyQuadrupeds | MorFiC | McARL | その他 | 本リポ現状 | 採否（暫定） |
|----------|---------|----------------|--------|-------|--------|------------|--------------|
| Unitree Go1 | ○ | ○ | ○ | 学習元 | 多数 | Env+URDF 済 | **既存・維持** |
| Unitree Go2 | ○ | — | 学習元 | ○ | Rapid Embod. 実機 | Env+URDF 済 | **既存・維持** |
| Unitree A1 | ○ | ○ | ○ | ○ | retarget 等 | Env+URDF 済 | **既存・維持** |
| MIT Mini Cheetah | ○ | ○ | ○ | ○ | | Env+URDF 済 | **既存・維持** |
| Unitree Laikago | ○ | ○ | — | — | A1→MC retarget 源 | Env+URDF あり | **導入優先（検証）** |
| ANYmal-C | ○ | ○ | ○ | — | | Env+URDF あり | **導入優先（検証）** |
| ANYmal-B | ○ | ○ | — | — | simple_description | なし | **導入優先** |
| Unitree Aliengo | ○ | ○ | ○ | — | unitree_ros | なし | **導入優先** |
| Unitree B1 | — | ○ | ○（失敗例） | — | unitree_ros | なし | **導入優先** |
| BD Spot | ○ | ○ | — | — | 第三者 URDF | なし | **導入（ライセンス確認）** |
| SpotMicro | ○ | ○ | — | — | | Env+URDF あり | 参考・小型（任意） |
| Sirius | ○ | — | — | — | GenLoco のみ | なし | 後回し（1論文） |
| HYQ / Solo / LittleDog | — | ○ | — | — | 形態差大 | なし | 後回し |

出典メモ:

- GenLoco (CoRL 2022): 10 体ゼロショット（A1, Go1, Laikago, Aliengo, Sirius, MC, ANYmal-B/C, Spot, SpotMicro）
- ManyQuadrupeds (arXiv:2310.10486): 16 体（上表＋カスタム）。質量・高さ・Kp/Kd 表あり
- MorFiC (arXiv:2603.14554): Go2 学習 → Go1/A1/Aliengo/MC/ANYmal/B1
- McARL (arXiv:2505.18418): Go1 学習 → Go2/MC/A1

## B. タスク1 導入順（早く終わりそうな順）

1. **既存資産の検証**: Laikago / ANYmal-C / SpotMicro（URDF 質量・摩擦・Env の default pose）
2. **Unitree 公式系統**: Aliengo → B1（`unitree_ros`）
3. **ANYmal-B**（`anymal_b_simple_description`）
4. **Spot**（公開 description・ライセンス確認後）
5. その他（Sirius 等）は必要なら後続

各機体の完了条件（タスク1）: URDF 検証メモ、Genesis 読み込み、初期姿勢の視覚確認、仕様表追記。  
**報酬の本詰め・PPO 専門家はタスク2（本線後）。**

## C. URDF 一次チェック（リポ内・2026-11-18）

| 資産 | total mass (URDF Σ) | 文献目安 | 摩擦タグ | 所見 |
|------|---------------------|----------|----------|------|
| `anymal_c/urdf/anymal_c.urdf` | 52.14 kg | ~52.1 kg (ManyQ) | なし（シム既定） | 質量は文献と一致。リンク数多い。摩擦は Genesis 側設定を要確認 |
| `laikago/.../laikago_toes*.urdf` | 25.57 kg | ~25 kg | lateral_friction=1 | 質量おおむね妥当。どの URDF を正とするか Env と照合 |
| `spotmicro/.../spotmicro.urdf` | 4.52 kg | ~4.8 kg | なし | 小型。優先度は低め |
| `go2/urdf/go2.urdf`（参考） | 15.02 kg | ~15 kg | なし | ゼロ質量リンク 2（既存既知） |

Env 参照 URDF:

- Laikago → `laikago/urdf/laikago.urdf`（Σmass ≈ **25.17 kg**, ManyQ 目安 25 kg）✓
- ANYmal-C → `anymal_c/urdf/anymal_c.urdf`（Σmass ≈ **52.14 kg**, ManyQ 52.1 kg）✓

## D. Env（リポ）

- 既存: `laikago_env.py` / `anymalc_env.py` / `spotmicro_env.py` 他
- 新規（#16）: `aliengo_env.py` / `b1_env.py` / `anymalb_env.py`（`train.py -r` 対応済）

→ 「ゼロから書く」より、**検証・パラメータの文献照合・足りない機体の追加**が主作業。立位レンダは GPU 空き待ち。

## E. 新規取り込み（2026-11-18）

| ロボット | 置き場 | Σmass (URDF) | 文献目安 | メモ |
|----------|--------|--------------|----------|------|
| Aliengo | `Genesis/genesis/assets/urdf/aliengo/` | 24.94 kg | ManyQ 20.6 kg / 公式~21.5 | **差あり** → [`MASS_QA.md`](MASS_QA.md)。`aliengo_env.py` + `train.py -r aliengo` 追加済 |
| B1 | `.../urdf/b1/xacro/b1.urdf` | 62.58 kg | ManyQ 52.7 / 公式~50 | **差あり**。`b1_env.py` + `-r b1` 追加済 |
| ANYmal-B | `.../urdf/anymal_b/urdf/anymal.urdf` | **30.42 kg** | ManyQ 30 kg | 一致。`anymalb_env.py` + `-r anymalb` 追加済 |
| Spot | `.../urdf/spot/urdf/spot_simple.urdf` | 50.34 kg | 公式~32.5 | RAI `spot_simple`（MIT+BSD-3）。`spot_env.py` + `-r spot`。Clearpath 素 URDF は inertial 無しで不採用 |

次: (1) GPU 空き時に立位レンダ（#18 と競合させない） (2) 仕様表の base 寸法埋め (3) タスク2（報酬+PPO）は本線後
