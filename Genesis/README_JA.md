![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genesis-world)](https://pypi.org/project/genesis-world/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)
[![Discord](https://img.shields.io/discord/1322086972302430269?logo=discord)](https://discord.gg/nukCuhB47p)
<a href="https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" height="20" style="display:inline"></a>

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![README en Français](https://img.shields.io/badge/Francais-d9d9d9)](./README_FR.md)
[![한국어 README](https://img.shields.io/badge/한국어-d9d9d9)](./README_KR.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
[![日本語版 README](https://img.shields.io/badge/日本語-d9d9d9)](./README_JA.md)

# Genesis

## 🔥 最新情報

- [2024-12-25] [レイトレーシングレンダラー](#docker)をサポートするDockerを追加しました。
- [2024-12-24] [Genesisへの貢献方法](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/.github/CONTRIBUTING.md)に関するガイドラインを追加しました。

## 目次

1. [Genesisとは？](#genesisとは)
2. [主な機能](#主な機能)
3. [インストール](#インストール)
4. [Docker](#docker)
5. [ドキュメント](#ドキュメント)
6. [Genesisへの貢献](#genesisへの貢献)
7. [サポート](#サポート)
8. [ライセンスと謝辞](#ライセンスと謝辞)
9. [関連論文](#関連論文)
10. [引用](#引用)

## Genesisとは？

Genesisは、汎用的な*ロボティクス/身体性を持ったAI*アプリケーション向けに設計された物理シミュレーションプラットフォームです。このプラットフォームは以下のような特徴があります：

1. あらゆる種類の材料や物理現象をシミュレート可能な**汎用物理エンジン**。
2. **軽量**、**超高速**、**Python的**、そして**ユーザーフレンドリー**なロボティクスシミュレーションプラットフォーム。
3. 高速で強力な**フォトリアリスティックなレンダリングシステム**。
4. ユーザーの自然言語による指示をもとに様々なデータモダリティを生成する**生成型データエンジン**。

Genesisの目指すところ：

- **物理シミュレーションのハードルを下げ**、ロボティクス研究を誰でもアクセス可能にすること。詳細は[ミッションステートメント](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html)をご覧ください。
- **多様な物理ソルバーを統合**し、最高の忠実度で物理世界を再現すること。
- **データ生成を自動化**し、人間の労力を削減し、データ生成の効率を最大化すること。

プロジェクトページ: <https://genesis-embodied-ai.github.io/>

## 主な機能

- **速度**: RTX 4090単体でフランカロボットアームを4300万FPS（リアルタイムの43万倍速）でシミュレーション可能。
- **クロスプラットフォーム**: Linux、macOS、Windowsで動作し、CPU、Nvidia/AMD GPU、Apple Metalをサポート。
- **多様な物理ソルバーの統合**: 剛体、MPM、SPH、FEM、PBD、安定流体シミュレーション。
- **幅広い材料モデル**: 剛体、液体、気体、変形体、薄膜オブジェクト、粒状材料などをシミュレーション可能。
- **様々なロボットへの対応**: ロボットアーム、脚付きロボット、ドローン、*ソフトロボット*など。また、`MJCF (.xml)`、`URDF`、`.obj`、`.glb`、`.ply`、`.stl`などの形式をサポート。
- **フォトリアルなレンダリング**: レイトレーシングベースのレンダリングをネイティブでサポート。
- **微分可能性**: 完全な微分可能性を備えた設計。現時点では、MPMソルバーとツールソルバーが対応しており、将来的には他のソルバーも対応予定（まず剛体および連結体ソルバーから開始）。
- **物理ベースの触覚シミュレーション**: 微分可能な[触覚センサーシミュレーション](https://github.com/Genesis-Embodied-AI/DiffTactile)が近日公開予定（バージョン0.3.0を予定）。
- **ユーザーフレンドリー**: シンプルで直感的なインストールとAPI設計。

## インストール

GenesisはPyPIで利用可能です：

```bash
pip install genesis-world  # Python>=3.10,<3.13 が必要です;
```

また、**PyTorch**を[公式手順](https://pytorch.org/get-started/locally/)に従ってインストールする必要があります。

最新版を利用する場合はこのレポジトリから直接インストールしてください:

```bash
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

このパッケージは定期的に手動で最新版と同期する必要があることに注意してください。

ソースコードを変更したいユーザはGenesisを開発者モードでインストールすることを推奨します。まず、pip経由の `genesis-world` がアンインストールされていることを確認してください。そして、このレポジトリーを以下のようにクローンしてインストールしてください。

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
```

## Docker

DockerからGenesisを利用する場合は、まずDockerイメージをビルドします：

```bash
docker build -t genesis -f docker/Dockerfile docker \
  --build-arg HOST_UID=$(id -u) --build-arg HOST_GID=$(id -g) --build-arg CONTAINER_USER=kawa37
```

イメージ内では **root には入らず**、ログイン名 `kawa37` の非 root ユーザーのみで動作します（`CONTAINER_USER` で変更可）。`HOST_UID` / `HOST_GID` はホストの `id -u` / `id -g` に合わせてください（`/workspace` をマウントしたときの生成物の所有者がホストの kawa37 と一致します）。ホストの UID/GID が変わったら再ビルドしてください。

その後、Dockerイメージ内で例を実行できます（`/workspace` にマウント）：

```bash
xhost +local:docker # コンテナユーザーがディスプレイにアクセスできるようにする

docker run --gpus all --shm-size=8g --rm -it \
-e DISPLAY=$DISPLAY \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-p 6006:6006 -p 8888:8888 \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis
```

TensorBoard の例（コンテナ内）:

```bash
tensorboard --logdir /path/to/logs --host 0.0.0.0 --port 6006
```

ホストのブラウザで `http://localhost:6006` を開きます。Weights & Biases を使う場合は `WANDB_API_KEY` などを環境変数で渡してください。

親ディレクトリで Genesis 以外のコードもまとめてマウントする場合は、リポジトリ直下の `docker-compose.yml` を使う方法もあります（`.env.example` を `.env` にコピーして `HOST_UID` / `HOST_GID` と必要なら `CONTAINER_USER` を調整し、`docker compose build` → `docker compose run --rm genesis`）。

## ドキュメント

包括的なドキュメントは現時点では[英語](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)、[中国語](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html)、および[日本語](https://genesis-world.readthedocs.io/ja/latest/user_guide/index.html)で提供されています。詳細なインストール手順、チュートリアル、APIリファレンスが含まれています。

## Genesisへの貢献

Genesisプロジェクトはオープンで協力的な取り組みです。以下を含む、コミュニティからのあらゆる貢献を歓迎します：

- 新機能やバグ修正のための**プルリクエスト**。
- GitHub Issuesを通じた**バグ報告**。
- Genesisの使いやすさを向上させるための**提案**。

詳細は[貢献ガイド](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/.github/CONTRIBUTING.md)をご参照ください。

## サポート

- バグ報告や機能リクエストはGitHubの[Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues)をご利用ください。
- 議論や質問はGitHubの[Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions)で行えます。

## ライセンスと謝辞

GenesisのソースコードはApache 2.0ライセンスで提供されています。

Genesisの開発は以下のオープンソースプロジェクトのおかげで可能になりました：

- [Taichi](https://github.com/taichi-dev/taichi): 高性能でクロスプラットフォーム対応の計算バックエンド。Taichiチームの技術サポートに感謝します！
- [FluidLab](https://github.com/zhouxian/FluidLab): 参照用のMPMソルバー実装。
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): 参照用のSPHソルバー実装。
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) と [PBF3D](https://github.com/WASD4959/PBF3D): 参照用のPBD（粒子ベースの物理）ソルバー実装。
- [MuJoCo](https://github.com/google-deepmind/mujoco): 剛体ダイナミクスの参照用実装。
- [libccd](https://github.com/danfis/libccd): 衝突検出の参照用実装。
- [PyRender](https://github.com/mmatl/pyrender): ラスタライズベースのレンダラー。
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) と [LuisaRender](https://github.com/LuisaGroup/LuisaRender): レイトレーシングDSL。

## 関連論文

Genesisプロジェクトに関与した主要な研究論文の一覧：

- Xian, Zhou, et al. "Fluidlab: A differentiable environment for benchmarking complex fluid manipulation." arXiv preprint arXiv:2303.02346 (2023).
- Xu, Zhenjia, et al. "Roboninja: Learning an adaptive cutting policy for multi-material objects." arXiv preprint arXiv:2302.11553 (2023).
- Wang, Yufei, et al. "Robogen: Towards unleashing infinite data for automated robot learning via generative simulation." arXiv preprint arXiv:2311.01455 (2023).
- Wang, Tsun-Hsuan, et al. "Softzoo: A soft robot co-design benchmark for locomotion in diverse environments." arXiv preprint arXiv:2303.09555 (2023).
- Wang, Tsun-Hsuan Johnson, et al. "Diffusebot: Breeding soft robots with physics-augmented generative diffusion models." Advances in Neural Information Processing Systems 36 (2023): 44398-44423.
- Katara, Pushkal, Zhou Xian, and Katerina Fragkiadaki. "Gen2sim: Scaling up robot learning in simulation with generative models." 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
- Si, Zilin, et al. "DiffTactile: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation." arXiv preprint arXiv:2403.08716 (2024).
- Wang, Yian, et al. "Thin-Shell Object Manipulations With Differentiable Physics Simulations." arXiv preprint arXiv:2404.00451 (2024).
- Lin, Chunru, et al. "UBSoft: A Simulation Platform for Robotic Skill Learning in Unbounded Soft Environments." arXiv preprint arXiv:2411.12711 (2024).
- Zhou, Wenyang, et al. "EMDM: Efficient motion diffusion model for fast and high-quality motion generation." European Conference on Computer Vision. Springer, Cham, 2025.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Scalable differentiable physics for learning and control." International Conference on Machine Learning. PMLR, 2020.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Efficient differentiable simulation of articulated bodies." In International Conference on Machine Learning, PMLR, 2021.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming Lin. "Differentiable simulation of soft multi-body systems." Advances in Neural Information Processing Systems 34 (2021).
- Wan, Weilin, et al. "Tlcontrol: Trajectory and language control for human motion synthesis." arXiv preprint arXiv:2311.17135 (2023).
- Wang, Yian, et al. "Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting." arXiv preprint arXiv:2411.09823 (2024).
- Zheng, Shaokun, et al. "LuisaRender: A high-performance rendering framework with layered and unified interfaces on stream architectures." ACM Transactions on Graphics (TOG) 41.6 (2022): 1-19.
- Fan, Yingruo, et al. "Faceformer: Speech-driven 3d facial animation with transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Wu, Sichun, Kazi Injamamul Haque, and Zerrin Yumak. "ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE." Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games. 2024.
- Dou, Zhiyang, et al. "C· ase: Learning conditional adversarial skill embeddings for physics-based characters." SIGGRAPH Asia 2023 Conference Papers. 2023.

さらに多数の現在進行形のプロジェクトがあります。

## 引用

研究でGenesisを使用する場合、以下を引用してください：

```bibtex
@misc{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```