# データローダ比較: Vintix と Vintix-Go2

## 1. 期待するデータ形式（HDF5）

**両方とも同じ形式を期待しています。**

| キー | 内容 | Vintix | Vintix-Go2 |
|------|------|--------|------------|
| `proprio_observation` | 状態（観測）の配列 | ✓ | ✓ |
| `action` | 行動の配列 | ✓ | ✓ |
| `reward` | 報酬の配列 | ✓ | ✓ |
| `step_num` | エピソード内ステップ番号（0始まり） | ✓ | ✓ |

- **次の状態**: どちらも別キーでは持たず、`proprio_observation` の次の行が次の状態。
- **終了判定 (done)**: どちらも専用キーはなく、`step_num == 0` でエピソード境界を判定。

---

## 2. サンプルとして返す辞書（`__getitem__`）

**両方とも同じキー・同じスライスで返します。**

| キー | 中身 | Vintix | Vintix-Go2 |
|------|------|--------|------------|
| `observation` | `obs[1:]`（時刻 t の状態） | ✓ | ✓ |
| `prev_action` | `acs[0:-1]`（1つ前の行動） | ✓ | ✓ |
| `prev_reward` | `rew[:-1]`（1つ前の報酬） | ✓ | ✓ |
| `action` | `acs[1:]`（予測ターゲットの行動） | ✓ | ✓ |
| `step_num` | `stp_num[1:]`（エピソード内ステップ） | ✓ | ✓ |

**結論: 期待するデータとサンプル構造は Vintix と Vintix-Go2 で同一。**

---

## 3. データローダの違い（オプション・機能）

### 3.1 FoundationMapDataset（シングルタスク・マップデータセット）

| 項目 | Vintix | Vintix-Go2 |
|------|--------|------------|
| HDF5 キー | proprio_observation, action, reward, step_num | 同じ |
| `context_len` | 使用（`context_len + 1` で内部長） | 同じ |
| `traj_sparsity` | 使用 | 同じ |
| `ep_sparsity` | 使用 | 同じ |
| `last_frac` | 使用 | 同じ |
| **`episode_range`** | **なし** | **あり**。`[start_frac, end_frac]` でデータの使用区間を指定可能（例: 先頭10%のみ）。指定時は `last_frac` は無視。 |
| サンプル構築 | `observation`, `prev_action`, `prev_reward`, `action`, `step_num` | 同じ |

### 3.2 FoundationRandomMapDataset（ランダムマップ・シングルタスク）

| 項目 | Vintix | Vintix-Go2 |
|------|--------|------------|
| 期待する HDF5 / 返すサンプル | 上と同じ | 同じ |
| エピソード境界の扱い | `ep_lens`, `transes` でスプリット計算 | 同じ |
| `episode_range` | なし | FoundationMapDataset 側のみ対応（このクラスにはなし） |

### 3.3 MultiTaskMapDataset（マルチタスク）

| 項目 | Vintix | Vintix-Go2 |
|------|--------|------------|
| `data_dir`, `datasets_info` | 使用 | 同じ |
| `trajectory_len`, `trajectory_sparsity` | 使用 | 同じ |
| `ep_sparsity` | リスト or 整数 | 同じ |
| `last_frac`, `preload`, `randomized` | 使用 | 同じ |
| **`episode_range`** | **引数なし** | **あり**。`Optional[Union[List[Tuple[float, float]], Tuple[float, float]]]`。タスクごと or 共通で使用区間を指定。 |
| 各タスクの Dataset | FoundationMapDataset / FoundationRandomMapDataset | 同じ。Go2 では FoundationMapDataset に `episode_range` を渡す。 |

---

## 4. 訓練スクリプトから渡す引数の違い

| 引数 | Vintix `train.py` | Vintix-Go2 `train_vintix.py` |
|------|-------------------|------------------------------|
| `data_dir` | ✓ | ✓ |
| `datasets_info` / `dataset_names` | ✓ | ✓ |
| `trajectory_len` (= context_len) | ✓ | ✓ |
| `trajectory_sparsity` | ✓ | ✓ |
| `ep_sparsity` (= episode_sparsity) | ✓ | ✓ |
| `last_frac` | 渡さない（デフォルト None） | 渡さない |
| `preload` | ✓ | ✓ |
| **`episode_range`** | **渡していない（データローダにも引数なし）** | **config から読み、渡す**（`config.episode_range`）。 |

---

## 5. まとめ

- **どんなデータを期待するか**: 両方とも **（状態, 行動, 報酬, ステップ番号）** の 4 種を HDF5 に持つ形式。次の状態・終了判定は「次の行の状態」と `step_num` で扱う点も同じ。
- **サンプルの中身**: **observation, prev_action, prev_reward, action, step_num** の 5 キーで、スライスも同じ。
- **主な違い**: Vintix-Go2 だけが **`episode_range`** をサポートしており、データの使用区間（例: 先頭 10% だけ使う）を指定できる。Vintix 側のデータローダには `episode_range` は存在しない。

そのため、**同じ HDF5 形式（proprio_observation, action, reward, step_num）のデータであれば、Vintix 用データは Vintix-Go2 でもそのまま使える。** 逆も同じ形式を期待するので、Go2 用に保存したデータを Vintix 側で読むことも可能（Vintix 側では `episode_range` は使えないだけ）。
