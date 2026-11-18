# Robot Specifications Table

URDFファイルから抽出した各ロボットの物理パラメータ（長さ・質量）．

| Parameter | Go1 | Go2 | A1 | Mini Cheetah | ANYmalC | Laikago | Aliengo | SpotMicro |
|-----------|--------|--------|--------|--------|--------|--------|--------|--------|
| Total weight (kg) | 13.1 | 15.0 | 13.7 | 8.3 | 52.1 | 25.2 | 24.9 | 4.5 |
| Base length (m) | 0.38 | 0.38 | 0.27 | 0.28 | 0.20 | — | — | 0.14 |
| Base width (m) | 0.09 | 0.09 | 0.19 | 0.20 | 0.20 | — | — | 0.11 |
| Base height (m) | 0.11 | 0.11 | 0.11 | 0.10 | 0.75 | — | — | 0.07 |
| Thigh Length (m) | 0.21 | 0.21 | 0.20 | 0.21 | 0.28 | — | 0.25 | 0.12 |
| Calf Length (m) | 0.21 | 0.21 | 0.20 | 0.21 | 0.34 | — | 0.25 | 0.12 |

注（#16・2026-11-18）: Aliengo は `unitree_ros` の URDF を `genesis/assets/urdf/aliengo/` に取り込み（BSD-3）。質量はリンク合計 ≈24.94 kg。脚長は calf/foot の origin z（各 0.25 m）。Laikago・Aliengo の base 寸法はメッシュ依存のため後日レンダ確認後に埋める。
