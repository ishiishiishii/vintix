# Genesisシミュレータでの回転の仕組み

## 1. クォータニオンの形式

Genesisでは、クォータニオンは **`[w, x, y, z]`** 形式で指定します。

```python
# 回転なし
quat = [1.0, 0.0, 0.0, 0.0]

# Y軸周りに90度回転
quat = [0.7071068, 0.0, 0.7071068, 0.0]
```

## 2. URDFへの回転の適用方法

### 2.1 適用の流れ

1. URDFファイルからベースリンクのクォータニオンを読み込む（通常は `[1.0, 0.0, 0.0, 0.0]`）
2. `gs.morphs.URDF` の `quat` パラメータで指定したクォータニオンを取得
3. これらを `transform_pos_quat_by_trans_quat` 関数で合成

### 2.2 合成方法

```python
# rigid_entity.py の491-505行目
if morph.pos is not None or morph.quat is not None:
    pos = np.asarray(l_info.get("pos", (0.0, 0.0, 0.0)))
    quat = np.asarray(l_info.get("quat", (1.0, 0.0, 0.0, 0.0)))
    pos_offset = np.asarray(morph.pos) if morph.pos is not None else np.zeros((3,))
    quat_offset = np.asarray(morph.quat) if morph.quat is not None else np.array((1.0, 0.0, 0.0, 0.0))
    
    l_info["pos"], l_info["quat"] = gu.transform_pos_quat_by_trans_quat(
        pos, quat, pos_offset, quat_offset
    )
```

### 2.3 transform_pos_quat_by_trans_quat の動作

```python
# geom.py の1039-1042行目
def transform_pos_quat_by_trans_quat(pos, quat, t_trans, t_quat):
    new_pos = t_trans + transform_by_quat(pos, t_quat)
    new_quat = transform_quat_by_quat(quat, t_quat)
    return new_pos, new_quat
```

### 2.4 transform_quat_by_quat の動作

```python
# geom.py の927-943行目
def transform_quat_by_quat(v, u):
    """
    This method transforms quat_v by quat_u.
    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    quat = quat_mul(u, v)  # u * v の順序
    return normalize(quat)
```

**重要なポイント**: `transform_quat_by_quat(v, u)` は `R_u @ R_v` を計算します。
つまり、**固定座標系での回転**を適用しています。

## 3. 回転の順序

### 3.1 固定座標系での回転

固定座標系では、回転を順番に適用します：
- まず `R_urdf` を適用（URDFのベースリンクの回転）
- 次に `R_morph` を適用（morphで指定した回転）

結果: `R_final = R_morph @ R_urdf`

### 3.2 ローカル座標系での回転

ローカル座標系では、回転を逆順に適用します：
- まず `R_morph` を適用
- 次に `R_urdf` を適用

結果: `R_final = R_urdf @ R_morph`

**Genesisでは固定座標系を使用しているため、`R_morph @ R_urdf` の順序です。**

## 4. Laikagoの回転の問題

### 4.1 理論的な計算

初期状態（回転なし）: 正面=Z軸、足=-Y軸
目標状態: 正面=X軸、足=-Z軸

固定座標系での回転:
1. Y軸周りに90度回転: 正面 Z軸 → X軸
2. X軸周りに90度回転: 足 -Y軸 → -Z軸

この回転を表すクォータニオン: `[0.5, 0.5, 0.5, 0.5]`

### 4.2 実際の問題

計算上は正しいはずですが、実際の表示では斜めになっています。

**考えられる原因**:
1. URDFの座標系が想定と異なる
2. 回転の適用方法が異なる
3. クォータニオンの解釈が異なる

### 4.3 解決策

段階的に回転を試して、実際の表示を確認しながら適切な回転を見つける必要があります。

`check_laikago_rotation.py` スクリプトを使用して、複数の回転を試すことができます。

## 5. 参考: 他のロボットの設定

- **Go2, MiniCheetah, Go1, AnymalC, UnitreeA1**: `[1.0, 0.0, 0.0, 0.0]` (回転なし)
- **Spotmicro**: `[0.0, 0.0, 0.0, 1.0]` (Z軸周りに180度回転)

## 6. まとめ

1. Genesisではクォータニオンは `[w, x, y, z]` 形式
2. URDFのベースリンクのクォータニオンとmorphで指定したクォータニオンを合成
3. 合成は固定座標系で行われる（`R_morph @ R_urdf`）
4. 理論的な計算が正しくても、実際の表示が異なる場合は、URDFの座標系を確認する必要がある

