# Laikago 見た目（メッシュ）調査メモ

更新: 2026-11-26（日本時間）

## 結論

- メッシュ本体は壊れていない（MorFiC と GenLoco/unitree の `*.dae` は MD5 一致）
- 歩行動画で胴体が巨大な灰色の板に見える主因は、**URDF の visual scale 未補正**
- `trunk.dae` / `calf.dae` は COLLADA `<unit meter="0.025400"/>`（インチ）。Genesis はこれを無視する
- 学習で使う MorFiC `laikago.urdf` は両メッシュとも `scale="1 1 1"` だった → 見た目だけ約 39 倍
- 衝突は box / cylinder なので、歩行の物理・定量は通っても動画ではおかしく見える
- `hip.dae` / `thigh*.dae` は頂点がメートル相当で、今回の巨体の主因ではない
- 立位 PNG（`figures/standing/laikago_standing.png`）は **別 URDF**（Genesis stock `laikago.urdf`）で撮っており、採用 ckpt の見た目とは一致しない

## 実施した修正

- MorFiC `laikago_description/urdf/laikago.urdf` の **trunk / calf** visual を `scale="0.0254 0.0254 0.0254"` に変更
- 同内容を GenLoco 側 `unitree_ros/.../laikago.urdf` にも反映（バックアップ `*.bak_trunk_calf_scale_1`）
- 衝突・慣性・関節は未変更（物理は変えない）

## 動画の再録

- 完了: 2026-11-26 18:16 JST
- 対象: `figures/videos/laikago_unitree_walk_model300.mp4`（寄りカメラ、指令 0.4、`model_300`）
- trunk/calf visual scale 0.0254 適用後に撮り直し。物理は未変更
- #11 は現ジョブ完了後に STOP → 再録 → collect 再開

