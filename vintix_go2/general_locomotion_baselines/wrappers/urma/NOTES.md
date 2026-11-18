# URMA → Genesis ラッパ

## 公式との差分

| 項目 | 公式 URMA | 本リポラッパ |
|------|-----------|--------------|
| Sim | MuJoCo + RL-X | Genesis（観測寸法のみ。actor は未接続） |
| Go2 観測 | 可変 DoF 記述 + 動的量 + 文脈 = **380**（パディング無し） | 同寸法。関節／足 description・接触・身長はプレースホルダ |
| 行動 | 12（multi-robot 時は max にパディング） | 12 |
| Pretrained | `experiments/pre_trained_model`（RL-X Flax/msgpack） | ファイル存在＋ magic 確認のみ。forward は RL-X 待ち |

## Go2 観測内訳（コードから算出）

| ブロック | 計算 | 次元 |
|----------|------|------|
| 関節 ×12 | description **23** + pos/vel/prev_act **3** | 312 |
| 足 ×4 | description **10** + contact/cycle **2** | 48 |
| 一般 | trunk6 + goal3 + gravity3 + h1 + pd3 + mass1 + dim3 | 20 |
| **合計** | | **380** |

関節 description の並びは `environment.py` の `get_name_to_description_vector` を正本とする（`joint_description.py` に転記）。

## 接続ステップ

1. ~~Go2 観測寸法をコードから確定・CPU スモーク~~（本 tick）
2. MuJoCo／URDF から本物の description ベクトルを生成（#16 機体整備と共有可）
3. GPU 空き後: 公式 `experiments/test.sh` で pretrained 再生、または RL-X を pin して actor を読む
4. Genesis 評価入口は description が揃ってから

## ファイル

| ファイル | 役割 |
|----------|------|
| `joint_description.py` | 23 / 10 / 380 の内訳定数 |
| `obs_adapter.py` | Genesis 45 → URMA 380（プレースホルダ付き） |
| `policy_loader.py` | ckpt 検査 + 観測スモーク（actor は deferred） |
| `smoke_cpu_check.py` | GPU 不要スモーク |

```bash
cd /workspace/vintix_go2/general_locomotion_baselines
python wrappers/urma/smoke_cpu_check.py
```
