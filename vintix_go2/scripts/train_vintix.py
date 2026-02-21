import datetime
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import pyrallis
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

from vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.training.utils.misc import set_seed
from vintix.training.utils.schedule import cosine_annealing_with_warmup
from vintix.training.utils.train_utils import (compute_stats,
                                               configure_optimizers,
                                               initialize_model, train_loop)
from vintix.nn.individual_task_head import AcsDecoder
from pathlib import Path


class RandomSampledDataset:
    """Wrapper class to randomly sample a fraction of the dataset after loading"""
    
    def __init__(self, dataset, sample_frac: float, seed: int = 5):
        """
        Args:
            dataset: The original dataset
            sample_frac: Fraction of data to keep (e.g., 0.5 for half)
            seed: Random seed for reproducibility
        """
        if sample_frac <= 0 or sample_frac > 1:
            raise ValueError(f"sample_frac must be between 0 and 1, got {sample_frac}")
        
        self.original_dataset = dataset
        original_size = len(dataset)
        
        if original_size == 0:
            raise ValueError("Cannot sample from an empty dataset")
        
        sample_size = max(1, int(original_size * sample_frac))  # Ensure at least 1 sample
        
        # Randomly sample indices
        rng = np.random.RandomState(seed=seed)
        self.sampled_indices = rng.permutation(original_size)[:sample_size]
        self.sampled_indices = np.sort(self.sampled_indices)  # Sort for efficiency
        
        print(f"Randomly sampled {sample_size} / {original_size} samples ({sample_frac*100:.1f}%)")
        print(f"  Actual sampling ratio: {sample_size/original_size*100:.2f}%")
    
    def __len__(self):
        return len(self.sampled_indices)
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.sampled_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.sampled_indices)})")
        # Map the new index to the original dataset index
        original_idx = self.sampled_indices[idx]
        return self.original_dataset[original_idx]
    
    @property
    def metadata(self):
        """Forward metadata from original dataset"""
        return self.original_dataset.metadata


@dataclass
class TrainConfig:
    # dataloader config
    data_dir: str = "data/go2_trajectories"
    context_len: int = 2048
    trajectory_sparsity: int = 128
    preload: bool = False
    last_frac: Optional[float] = None
    random_sample_frac: Optional[float] = None  # Randomly sample this fraction of the dataset (e.g., 0.5 for half)

    # model config
    action_emb_dim: int = 127
    observation_emb_dim: int = 127
    reward_emb_dim: int = 2
    hidden_dim: int = 256  # 固定値 (action_emb + obs_emb + reward_emb)
    transformer_depth: int = 3
    transformer_heads: int = 4
    attn_dropout: float = 0.1  # 過学習防止のため追加
    residual_dropout: float = 0.1  # 過学習防止のため追加
    normalize_qk: bool = True
    bias: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_class: str = "LayerNorm"
    mlp_class: str = "GptNeoxMLP"
    intermediate_size: int = 128  # hidden_dim × 0.5
    inner_ep_pos_enc: bool = False
    norm_acs: bool = False
    norm_obs: bool = True

    # optimizer config
    optimizer: str = "Adam"
    lr: float = 0.0003
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    precision: str = "bf16"
    clip_grad: Optional[float] = None  # 元の設定に合わせる
    grad_accum_steps: int = 8  # 実効バッチサイズ64を実現（8 × 8 = 64）
    warmup_ratio: float = 0.005  # 元の設定に合わせる

    # training config
    local_rank: int = 0
    epochs: int = 1
    batch_size: int = 8
    save_every: int = 1
    save_every_steps: int = 1000
    save_dir: str = "models/vintix_go2"
    stats_path: str = "vintix/stats.json"
    load_ckpt: Optional[str] = None
    start_epoch: int = 0
    seed: int = 5

    # Dataset config
    dataset_config_paths: List[str] = None

    # wandb config
    project: str = "Vintix_Go2"
    group: str = "go2_walking"
    name: str = "Trajectory128"  # trajectory_sparsity=128での訓練

    # finetune config (decoder-only, based on an existing multitask checkpoint)
    # Intended for cases like "All_onegroup" -> create robot-specific decoder and finetune it
    finetune_decoder_only: bool = False
    finetune_robot: Optional[str] = None  # e.g. "minicheetah"
    finetune_task_name: Optional[str] = None  # e.g. "minicheetah_walking_ad" (auto if finetune_robot is set)
    finetune_new_group: Optional[str] = None  # e.g. "minicheetah_finetune" (auto if finetune_robot is set)
    finetune_reference_group: Optional[str] = None  # e.g. "quadruped_locomotion" (auto from ckpt task2group)
    finetune_output_subdir: Optional[str] = None  # saved under parent of load_ckpt (auto: "<robot>_finetune")
    finetune_freeze_transformer: bool = True
    finetune_freeze_encoder: bool = True

    def __post_init__(self):
        if self.dataset_config_paths is None:
            self.dataset_config_paths = ["configs/go2_dataset_config.yaml"]
        
        # 相対パスを絶対パスに変換（スクリプトの親ディレクトリ（vintix_go2）を基準）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)  # vintix_go2ディレクトリ
        
        # data_dirを絶対パスに変換
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(parent_dir, self.data_dir)
        
        # stats_pathを絶対パスに変換
        if not os.path.isabs(self.stats_path):
            self.stats_path = os.path.join(parent_dir, self.stats_path)
        
        # stats_pathのディレクトリが存在しない場合は作成
        stats_dir = os.path.dirname(self.stats_path)
        os.makedirs(stats_dir, exist_ok=True)
        
        # 保存ディレクトリにモデル名を含める
        self.save_dir = os.path.join(self.save_dir, self.name)
        
        self.dataset_config = {}
        for dcp in self.dataset_config_paths:
            # 相対パスの場合、スクリプトの親ディレクトリ（vintix_go2）を基準にする
            if not os.path.isabs(dcp):
                dcp = os.path.join(parent_dir, dcp)
            dc = OmegaConf.load(dcp)
            self.dataset_config = {**self.dataset_config, **dc}
        self.dataset_names = {
            v.path: v.group
                for k, v in self.dataset_config.items() if v.type == "default"
        }
        self.reward_scales = [
            v.reward_scale
                for v in self.dataset_config.values() if v.type == "default"
        ]
        self.episode_sparsity = [
            v.episode_sparsity
                for v in self.dataset_config.values() if v.type == "default"
        ]
        # episode_rangeを読み込む（オプション）
        self.episode_range = []
        for v in self.dataset_config.values():
            if v.type == "default":
                if hasattr(v, 'episode_range') and v.episode_range is not None:
                    # OmegaConf ListConfigをlist/tupleに変換
                    try:
                        if hasattr(v.episode_range, '__iter__'):
                            ep_range_list = list(v.episode_range)
                            self.episode_range.append(tuple(ep_range_list))
                        else:
                            self.episode_range.append(None)
                    except Exception:
                        self.episode_range.append(None)
                else:
                    self.episode_range.append(None)


HALF_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}

_ROBOT_TASK_DEFAULT = {
    "go1": "go1_walking_ad",
    "go2": "go2_walking_ad",
    "minicheetah": "minicheetah_walking_ad",
    "unitreea1": "unitreea1_walking_ad",
    "a1": "unitreea1_walking_ad",
}


def _prepare_decoder_only_finetune(
    config: TrainConfig,
    model,
    stats: dict,
) -> None:
    """
    手順1-2: モデルを読み込んでパラメータをコピー（既にinitialize_modelで完了）
    手順2: デコーダ以外をフリーズする
    手順3: 指定したロボットのdecoderのみを訓練可能にする
    
    シンプルなアプローチ：
    - 既存のdecoder group（例：quadruped_locomotion）のdecoderのみを訓練可能にする
    - TransformerとEncoderは完全にフリーズ
    - 他のdecoder groupもフリーズ
    
    未知タスクの場合：
    - 統計情報を使ってタスクを動的に追加
    """
    if not config.finetune_decoder_only:
        return

    if not config.load_ckpt:
        raise ValueError("finetune_decoder_only=True requires load_ckpt to be set (path to an epoch folder).")

    if not config.finetune_robot and not config.finetune_task_name:
        raise ValueError("Specify finetune_robot (e.g. minicheetah) or finetune_task_name (e.g. minicheetah_walking_ad).")

    task_name = config.finetune_task_name or _ROBOT_TASK_DEFAULT.get(config.finetune_robot)
    if not task_name:
        raise ValueError(f"Could not infer finetune_task_name for finetune_robot={config.finetune_robot!r}.")

    # タスクがモデルに存在しない場合、統計情報を使って追加
    if task_name not in model.head.task2group:
        print(f"[decoder_only_finetune] Unknown task detected: {task_name}")
        print(f"[decoder_only_finetune] Adding task to model using computed statistics...")
        
        # statsのキーを確認（dataset_configのキーと一致する可能性がある）
        if task_name not in stats:
            # statsのキーを表示してデバッグ
            available_keys = list(stats.keys())
            print(f"[decoder_only_finetune] Available stats keys: {available_keys[:10]}")
            raise ValueError(
                f"Task {task_name!r} not found in loaded checkpoint and statistics not available. "
                f"Available tasks in model: {sorted(list(model.head.task2group.keys()))[:20]} ... "
                f"Available stats keys: {available_keys[:10]} ..."
            )
        
        # 統計情報を確認（stats[task_name]が存在することを確認）
        if task_name not in stats:
            available_keys = list(stats.keys())
            print(f"[decoder_only_finetune] Available stats keys: {available_keys[:10]}")
            raise ValueError(
                f"Task {task_name!r} not found in computed statistics. "
                f"Available stats keys: {available_keys[:10]} ..."
            )
        
        group_name = config.dataset_config.get(task_name, {}).get("group", "quadruped_locomotion")
        rew_scale = config.dataset_config.get(task_name, {}).get("reward_scale", 1.0)
        
        # タスクを追加（stats全体を渡す。add_taskは内部でstats[task_name]にアクセス）
        model.add_task(
            task_name=task_name,
            group_name=group_name,
            stats=stats,
            rew_scale=rew_scale,
        )
        print(f"[decoder_only_finetune] Added task {task_name} to group {group_name}")

    # タスクが使用するdecoder groupを取得（例：quadruped_locomotion）
    decoder_group = model.head.task2group[task_name]
    
    if decoder_group not in model.head.acs_decoders:
        raise ValueError(
            f"Decoder group {decoder_group!r} not found for task {task_name!r}. "
            f"Available decoder groups: {list(model.head.acs_decoders.keys())}"
        )

    print(f"[decoder_only_finetune] Task: {task_name}")
    print(f"[decoder_only_finetune] Decoder group to finetune: {decoder_group}")

    # 手順2: デコーダ以外をフリーズする
    print("[decoder_only_finetune] Freezing Transformer...")
    for p in model.transformer.parameters():
        p.requires_grad = False
    
    print("[decoder_only_finetune] Freezing Encoder...")
    for p in model.encoder.parameters():
        p.requires_grad = False

    # すべてのdecoderを一旦フリーズ
    print("[decoder_only_finetune] Freezing all decoders...")
    for gn, dec in model.head.acs_decoders.items():
        for p in dec.parameters():
            p.requires_grad = False

    # 指定したdecoder groupのみを訓練可能にする
    print(f"[decoder_only_finetune] Unfreezing decoder group: {decoder_group}")
    for p in model.head.acs_decoders[decoder_group].parameters():
        p.requires_grad = True

    # 保存ディレクトリの設定
    ckpt_parent = Path(config.load_ckpt).resolve().parent
    out_subdir = config.finetune_output_subdir or f"{config.finetune_robot}_finetune" if config.finetune_robot else f"{task_name}_finetune"
    config.save_dir = str(ckpt_parent / out_subdir)
    os.makedirs(config.save_dir, exist_ok=True)
    config.name = out_subdir

    # 統計情報を出力
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[decoder_only_finetune] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed)
    
    # シングルGPUの場合はDDPを使わない
    use_distributed = torch.cuda.device_count() > 1
    
    if use_distributed:
        dist.init_process_group(backend='nccl',
                                timeout=datetime.timedelta(minutes=30))
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        config.local_rank = device_id
        num_gpus = int(os.getenv("WORLD_SIZE", "1"))
        print(f"NCCL Init on rank {rank}")
    else:
        rank = 0
        config.local_rank = 0
        num_gpus = 1
        print("Running on single GPU (no DDP)")

    stats = {}
    if config.local_rank == 0:
        config.save_dir = os.path.join(config.save_dir, config.name)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)

        _ = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            save_code=True,
        )
        print("COMPUTING STATS")
        stats = compute_stats(config.data_dir, config.dataset_names)
        with open(config.stats_path, 'w') as f:
            json.dump(stats, f)
    else:
        # 他のランクは待機
        if use_distributed:
            dist.barrier()
        # 統計情報を読み込み
        if os.path.exists(config.stats_path):
            with open(config.stats_path, 'r') as f:
                stats = json.load(f)

    print("PREPARING DATASET")
    dataset = MultiTaskMapDataset(
        data_dir=config.data_dir,
        datasets_info=config.dataset_names,
        trajectory_len=config.context_len,
        trajectory_sparsity=config.trajectory_sparsity,
        ep_sparsity=config.episode_sparsity,
        episode_range=config.episode_range if config.episode_range else None,
        preload=config.preload,
    )
    
    # Apply random sampling if specified - sample the dataset itself before creating DataLoader
    original_dataset_size = len(dataset)
    if config.random_sample_frac is not None:
        if config.random_sample_frac <= 0 or config.random_sample_frac > 1:
            raise ValueError(f"random_sample_frac must be between 0 and 1, got {config.random_sample_frac}")
        print(f"Applying random sampling: keeping {config.random_sample_frac*100:.1f}% of the dataset")
        dataset = RandomSampledDataset(dataset, config.random_sample_frac, seed=config.seed)
        sampled_dataset_size = len(dataset)
        print(f"Dataset size: {original_dataset_size} -> {sampled_dataset_size} samples")
    else:
        sampled_dataset_size = original_dataset_size
    
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=int(os.getenv("WORLD_SIZE", "1")),
            rank=int(config.local_rank),
            seed=0
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            sampler=sampler,
            drop_last=True,
            num_workers=12
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            drop_last=True,
            num_workers=2
        )

    print("INITIALIZING MODEL")
    scaler = torch.cuda.amp.GradScaler()
    torch_dtype = HALF_DTYPES[config.precision]
    
    if use_distributed:
        dist.barrier()

    device = torch.device("cuda", config.local_rank)
    model, _ = initialize_model(config,
                                dataset.metadata,
                                device)

    # 手順1-3: デコーダのみのファインチューニング設定
    # 手順1: モデルは既にinitialize_modelで読み込まれている（パラメータコピー済み）
    # 手順2-3: デコーダ以外をフリーズし、指定decoderのみを訓練可能にする
    # 未知タスクの場合は統計情報を使ってタスクを追加
    _prepare_decoder_only_finetune(config, model, stats)
    
    # モデルを訓練モードに設定
    model.train()
    
    if use_distributed:
        model = DDP(model,
                    device_ids=[config.local_rank],
                    find_unused_parameters=True)
        model_to_save = model.module
    else:
        model_to_save = model
        
    # Optimizer: in decoder-only finetune, only optimize trainable params for speed/clarity
    if config.finetune_decoder_only:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )
        # Don't load optimizer state for decoder-only finetune (different parameter groups)
        load_optimizer_state = False
    else:
        optimizer = configure_optimizers(config, model)
        load_optimizer_state = True
    if config.load_ckpt and load_optimizer_state:
        opt_dct = torch.load(os.path.join(config.load_ckpt, "state.pth"))
        optimizer.load_state_dict(opt_dct["optimizer_state"])

    if config.local_rank == 0:
        wandb.watch(model, log_freq=500)
    total_updates = len(dataloader) * config.epochs
    if config.local_rank == 0:
        print(f"Training configuration:")
        print(f"  Dataset size: {sampled_dataset_size} samples")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Steps per epoch: {len(dataloader)}")
        print(f"  Total epochs: {config.epochs}")
        print(f"  Total updates: {total_updates}")
    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=int(total_updates * config.warmup_ratio),
        total_steps=total_updates,
    )
    if config.load_ckpt and load_optimizer_state:
        opt_dct = torch.load(os.path.join(config.load_ckpt, "state.pth"))
        scheduler.load_state_dict(opt_dct["scheduler_state"])

    print("MODEL INIT COMPLETE")
    print(f"PARAMETERS: {sum(p.numel() for p in model.parameters())}")
    print(f"Device {config.local_rank}")
    
    if use_distributed:
        dist.barrier()

    if config.local_rank == 0:
        folder_name = "0".zfill(4) + "_epoch"
        model_to_save.save_model(
            os.path.join(config.save_dir, folder_name)
        )
        torch.save(
            {"optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict()},
            os.path.join(config.save_dir, folder_name, "state.pth")
        )

    for ep in range(config.start_epoch, config.epochs):
        if use_distributed:
            sampler.set_epoch(ep)
        train_loop(config,
                   ep,
                   num_gpus,
                   model,
                   optimizer,
                   scheduler,
                   scaler,
                   dataloader,
                   device,
                   torch_dtype,
                   distributed=use_distributed)
        # 定期保存 + 最終エポックも必ず保存
        if (ep % config.save_every == 0 or ep == config.epochs - 1) and config.local_rank == 0:
            folder_name = f"{ep}".zfill(4) + "_epoch"
            model_to_save.save_model(
                os.path.join(config.save_dir, folder_name)
            )
            torch.save(
                {"optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict()},
                os.path.join(config.save_dir, folder_name, "state.pth")
            )
        if use_distributed:
            dist.barrier()


if __name__ == '__main__':
    train()

