# check_laikago_rotation.py の使い方

このスクリプトは、Laikagoロボットの回転を段階的に試して、適切な向きを見つけるためのツールです。

## 基本的な使い方

### 1. デフォルトの回転で実行（Y軸90度）

```bash
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py"
```

引数を指定しない場合、デフォルトで「Y軸90度」の回転が適用されます。

### 2. 特定の回転を指定して実行

```bash
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py \"回転名\""
```

## 使用可能な回転

スクリプトには以下の回転が定義されています：

1. **回転なし**
   ```bash
   python check_laikago_rotation.py "回転なし"
   ```
   - クォータニオン: `[1.0, 0.0, 0.0, 0.0]`
   - URDFの元の向きを確認できます

2. **Y軸90度**
   ```bash
   python check_laikago_rotation.py "Y軸90度"
   ```
   - クォータニオン: `[0.7071068, 0.0, 0.7071068, 0.0]`
   - Y軸周りに90度回転

3. **Y軸90度→X軸90度（固定座標系）**
   ```bash
   python check_laikago_rotation.py "Y軸90度→X軸90度（固定座標系）"
   ```
   - クォータニオン: `[0.5, 0.5, 0.5, 0.5]`
   - 理論的には正しいはずの回転

4. **Y軸90度→X軸90度（符号反転）**
   ```bash
   python check_laikago_rotation.py "Y軸90度→X軸90度（符号反転）"
   ```
   - クォータニオン: `[-0.5, -0.5, -0.5, -0.5]`
   - 符号を反転させたバージョン

5. **Z軸90度**
   ```bash
   python check_laikago_rotation.py "Z軸90度"
   ```
   - クォータニオン: `[0.7071068, 0.0, 0.0, 0.7071068]`
   - Z軸周りに90度回転

6. **X軸90度**
   ```bash
   python check_laikago_rotation.py "X軸90度"
   ```
   - クォータニオン: `[0.7071068, 0.7071068, 0.0, 0.0]`
   - X軸周りに90度回転

## 実行の流れ

1. **スクリプトを実行**
   - 指定した回転でLaikagoが表示されます
   - ビューアウィンドウが開きます

2. **ビューアで確認**
   - ロボットの向きを確認します
   - 正面がX軸方向（赤い線）を向いているか
   - 足が-Z軸方向（下）を向いているか
   - 斜めになっていないか

3. **終了**
   - `Ctrl+C` で終了します

4. **別の回転を試す**
   - 別の回転名を指定して再度実行します

## 使用例

### 例1: 回転なしの状態を確認

```bash
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py \"回転なし\""
```

これで、URDFの元の向きを確認できます。

### 例2: 段階的に回転を試す

```bash
# 1. 回転なしを確認
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py \"回転なし\""

# 2. Y軸90度を試す
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py \"Y軸90度\""

# 3. 理論的に正しい回転を試す
docker exec genesis_tensorboard bash -c "cd /workspace/Genesis/examples/locomotion && python check_laikago_rotation.py \"Y軸90度→X軸90度（固定座標系）\""
```

## 新しい回転を追加する

スクリプトの `test_rotations` 辞書に新しい回転を追加できます：

```python
test_rotations = {
    "回転なし": [1.0, 0.0, 0.0, 0.0],
    "Y軸90度": [0.7071068, 0.0, 0.7071068, 0.0],
    # 新しい回転を追加
    "カスタム回転": [w, x, y, z],  # クォータニオン [w, x, y, z] を指定
}
```

## 注意事項

- ビューアウィンドウが開いている間は、スクリプトは実行され続けます
- 別の回転を試す場合は、前のスクリプトを `Ctrl+C` で終了してから実行してください
- GPUメモリを使用するため、複数のインスタンスを同時に実行しないでください

## トラブルシューティング

### ビューアが表示されない場合

- X11転送が有効になっているか確認してください
- Dockerコンテナが正しく起動しているか確認してください

### エラーが発生する場合

- GPUメモリが不足している可能性があります
- 他のGenesisプロセスが実行中でないか確認してください



