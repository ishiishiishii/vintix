# 実験用コンテナの起動と接続

`docker-compose.yml` に GPU・共有メモリ・マウント・ポートなどをまとめています。ここに書いた設定が `docker compose up -d` でそのまま適用されます。

## 前提

- リポジトリのルート: `genesis_project`（このファイルと同じ階層）
- コンテナ名: **`genesis_exp`**（常駐用）
- コンテナ内ユーザー: **`kawa37`**（ビルド時の `HOST_UID` / `HOST_GID` とホストを合わせること）

初回または `Dockerfile` を変えたあと:

```bash
cd /path/to/genesis_project/Genesis
docker build -t genesis:dev -f docker/Dockerfile docker \
  --build-arg HOST_UID="$(id -u)" \
  --build-arg HOST_GID="$(id -g)" \
  --build-arg CONTAINER_USER=kawa37
```

ホストの UID/GID が変わったら、上記をやり直してください。

## 起動（常駐）

```bash
cd /path/to/genesis_project
xhost +local:docker
docker compose up -d
```

`docker compose ps` で `genesis_exp` が `Up` になっていれば起動済みです。`--rm` は使っていません。停止だけする場合は `docker compose stop`、コンテナを消す場合は `docker compose down` です。

## 途中から入る（別ターミナルでシェル）

```bash
cd /path/to/genesis_project
docker compose exec genesis bash
```

コンテナ名で直接入る場合:

```bash
docker exec -it genesis_exp bash
```

作業ディレクトリの目安: ホストのリポジトリ全体が **`/workspace`** にマウントされています。

## TensorBoard のポート

ホストの **6006** などとぶつかった場合、Compose では **`16006` → コンテナ内 6006**、**`18888` → 8888** にマッピングしています。TensorBoard はコンテナ内で例えば次のようにし、ブラウザでは `http://localhost:16006` を開きます。

```bash
tensorboard --logdir /workspace/vintix_go2/logs --host 0.0.0.0 --port 6006
```

## 動作確認: 四足 `eval.py`（go2-walking, 300 イテレーション）

チェックポイントは **`Genesis/logs/go2-walking/model_300.pt`**（スクリプトは小文字の `logs` を参照します）。`cfgs.pkl` も同じディレクトリに必要です。

コンテナ内で:

```bash
cd /workspace/Genesis/examples/locomotion
python eval.py -e go2-walking -r go2 --ckpt 300
```

**注意:** このスクリプトは学習済みポリシーで環境を回し続ける **`while True` ループ**のため、止めるときは **Ctrl+C** で中断します。ビューアが開く設定（`show_viewer=True`）のため、表示まわりで失敗する場合はホストの `DISPLAY` と `xhost +local:docker` を確認してください。
