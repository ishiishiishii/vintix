# Aliengo 見た目（メッシュ）調査メモ

更新: 2026-11-25（日本時間）

## 結論

- **メッシュ本体は壊れていない**（MorFiC と unitree の `*.dae` は MD5 一致）
- 歩行動画でロボットが「白い塊」に見える主因は、**MorFiC URDF の下腿（calf）visual scale が未補正**だったこと
- `calf.dae` は COLLADA 単位が **インチ（meter=0.0254）**。Genesis はこれを無視するため、URDF 側で `scale=0.0254` が必要
- unitree 版 URDF は既に補正済み。**学習で使っている MorFiC 版は scale=1 のままだった** → 見た目だけ約 39 倍
- 衝突形状は box なので、歩行の物理・定量は通っても動画では脚が巨大な白い板に見える

## 実施した修正

- MorFiC `aliengo.urdf` の calf visual を `scale="0.0254 0.0254 0.0254"` に変更（4 脚）
- 記録: `third_party/MorFiC/.../aliengo_description/SOURCE_VISUAL_FIX.md`
- バックアップ: `aliengo.urdf.bak_calf_scale_1`

## 立位 PNG との関係

- `figures/standing/aliengo_standing.png` は **unitree URDF（補正済み）** 側で撮っており、四足として認識できる
- 学習・評価動画は **MorFiC URDF** を読むため、修正前は白い塊に見えた

## 動画について（今後）

カメラ距離・画角は評価時に意識する（今回の調査対象外）。メッシュ修正後の再録画は、次の品質評価のときに行う。
