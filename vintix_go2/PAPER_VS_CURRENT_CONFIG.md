# Vintix 先行研究（論文）と当プログラムの設定比較

## 比較表

| パラメータ | 先行研究（論文） | 当プログラム（TrainConfig） | 備考 |
|------------|------------------|----------------------------|------|
| **Learning Rate** | 0.0003 | 0.0003 | **同じ** |
| **Optimizer** | Adam | 設定上 "Adam"／実装は **AdamW** | 実装では `configure_optimizers` が AdamW + weight decay を使用 |
| **Beta 1** | 0.9 | 0.9 | **同じ** |
| **Beta 2** | 0.99 | 0.99 | **同じ** |
| **Batch Size** | 64 | 8 | **違う**（下記の実効バッチで調整） |
| **Gradient Accumulation Steps** | 2 | 8 | **違う** |
| **実効バッチサイズ** | 64 | 8 × 8 = 64 | **同じ**（64） |
| **Transformer Layers** | 20（本文では24層と記載） | 3 | **違う** |
| **Transformer Heads** | 16 | 4 | **違う** |
| **Context Length** | 8192 | 2048 | **違う** |
| **Transformer Hidden Dim** | 1024 | 256 | **違う** |
| **FF Hidden Size (intermediate_size)** | 4096 | 128 | **違う** |
| **MLP Type** | GptNeoxMLP | GptNeoxMLP | **同じ** |
| **Normalization Type** | LayerNorm | LayerNorm | **同じ** |
| **Training Precision** | bf16 | bf16 | **同じ** |
| **Parameters** | 332,100,768（約3.3億） | 約100万オーダー（Transformer のみ約99万） | **違う**（モデルが小型） |
| **Weight Decay** | （付録Dに記載なし） | 0.1 | 論文表にはなし／当プログラムで使用 |
| **Dropout** | （付録Dに記載なし） | attn 0.1, residual 0.1 | 論文表にはなし／当プログラムで追加 |
| **warmup_ratio** | （付録Dに記載なし） | 0.005 | 論文表にはなし |

---

## 同じパラメータ

- Learning Rate: **0.0003**
- Betas: **(0.9, 0.99)**
- 実効バッチサイズ: **64**
- MLP Type: **GptNeoxMLP**
- Normalization Type: **LayerNorm**
- Training Precision: **bf16**

---

## 違うパラメータ（要約）

| 項目 | 論文 | 当プログラム |
|------|------|--------------|
| オプティマイザ実装 | Adam | AdamW（weight decay 付き） |
| バッチサイズ／勾配蓄積 | 64 / 2 | 8 / 8 |
| Transformer 層数 | 20（本文24の記述あり） | 3 |
| アテンションヘッド数 | 16 | 4 |
| コンテキスト長 | 8192 | 2048 |
| Hidden Dim | 1024 | 256 |
| FFN 中間次元 | 4096 | 128 |
| 総パラメータ数 | 約 3.32億 | 約 100万オーダー |

---

## アテンション・位置エンコーディング・エンコーダまわりの違い

論文では「FlashAttention (Dao et al., 2022) を用い、絶対位置エンコーディングは KV-cache を用いたスライディング窓では位置のずれで相性が悪いため、ALiBi (Press et al., 2022) を用いた」とある。当プログラムとの対応は以下の通り。

| 項目 | 先行研究（論文） | 当プログラム | 備考 |
|------|------------------|--------------|------|
| **アテンション実装** | **FlashAttention**（Dao et al., 2022） | **標準 PyTorch アテンション** | **違う**。当プログラムは `flash_attn` を import するが、forward では「Standard PyTorch attention (replacing FlashAttention)」とコメントされ、`torch.matmul` + causal mask で計算している。FlashAttention は実行時には未使用。 |
| **位置エンコーディング（Transformer）** | **ALiBi**（相対的なバイアス） | **ALiBi**（`get_alibi_slopes` で slopes を計算し、attention のスコアに加算） | **同じ**。絶対位置は使わず、ALiBi により推論時の KV-cache・スライディング窓と整合。 |
| **絶対位置エンコーディング** | 使用しない（ALiBi が相対なので採用） | Transformer 側には絶対位置なし | **同じ**。 |
| **KV-cache / スライディング窓** | 推論でスライディングアテンション窓と KV-cache を想定 | **KV-cache 対応あり**（`k_cache`, `v_cache`, `cache_seqlens` で推論モード） | **同じ**（推論時のキャッシュの考え方は一致）。 |
| **エンコーダ側の位置情報** | 論文では言及なし | **オプション** `inner_ep_pos_enc`（エピソード内ステップの Embedding）。デフォルトは **False** | 論文に相当する記述はなし。当プログラムではオフにすればエンコーダに追加の位置埋め込みは入れない。 |
| **QK 正規化** | （付録に記載なし） | **使用**（`normalize_qk=True`、LayerNorm で Q/K を正規化） | 論文表にはなし／当プログラムで使用。 |

### まとめ（アテンション・位置まわり）

- **同じ**: ALiBi、絶対位置を使わない方針、KV-cache 対応の考え方。
- **違う**: アテンションの**実装が論文は FlashAttention、当プログラムは標準 PyTorch**。また当プログラムは QK 正規化とエンコーダ用のオプション `inner_ep_pos_enc` を持つ。

---

## 補足

- 論文本文では「24層のトランスフォーマー」とあるが、付録Dのハイパーパラメータ表では **Transformer Layers 20** となっている。表の値（20）で比較した。
- 当プログラムは **小規模設定**（層数・次元・コンテキスト長を抑え、メモリ・計算量を削減）であり、パラメータ数が論文より約 300 分の 1 程度になっている。
