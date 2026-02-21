"""
Minicheetah用の追加訓練スクリプト

既存のMinicheetah_without_separategroupモデルをロードし、
Transformerパラメータを凍結して、Minicheetahタスク用の
エンコーダー/デコーダーのみを追加訓練する。

使用方法:
    python3 scripts/finetune_minicheetah.py \
        --pretrained_path models/vintix_go2/Minicheetah_without_separategroup/Minicheetah_without_separategroup/0015_epoch \
        --dataset_config_paths configs/minicheetah_finetune_config.yaml \
        --name Minicheetah_without_separategroup_finetuned \
        --epochs 10 \
        --data_dir data \
        --freeze_transformer
"""

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

from vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.training.utils.misc import set_seed
from vintix.training.utils.schedule import cosine_annealing_with_warmup
from vintix.training.utils.train_utils import (compute_stats,
                                               configure_optimizers,
                                               initialize_model, train_loop)
from vintix.vintix import Vintix
from vintix.nn.individual_task_encoder import IndividualTaskEncoderNew
from vintix.nn.individual_task_head import IndividualTaskHeadNew


@dataclass
class FinetuneConfig:
    # dataloader config
    data_dir: str = "data"
    context_len: int = 2048
    trajectory_sparsity: int = 128
    preload: bool = False
    last_frac: Optional[float] = None

    # model config
    action_emb_dim: int = 127
    observation_emb_dim: int = 127
    reward_emb_dim: int = 2
    hidden_dim: int = 256
    transformer_depth: int = 3
    transformer_heads: int = 4
    attn_dropout: float = 0.1
    residual_dropout: float = 0.1
    normalize_qk: bool = True
    bias: bool = True
    parallel_residual: bool = False
    shared_attention_norm: bool = False
    norm_class: str = "LayerNorm"
    mlp_class: str = "GptNeoxMLP"
    intermediate_size: int = 128
    inner_ep_pos_enc: bool = False
    norm_acs: bool = False
    norm_obs: bool = True

    # optimizer config
    optimizer: str = "Adam"
    lr: float = 0.0001  # 追加訓練では少し低めの学習率
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.1
    precision: str = "bf16"
    clip_grad: Optional[float] = None
    grad_accum_steps: int = 8
    warmup_ratio: float = 0.005

    # training config
    local_rank: int = 0
    epochs: int = 10
    batch_size: int = 8
    save_every: int = 1
    save_every_steps: int = 1000
    save_dir: str = "models/vintix_go2"
    stats_path: str = "vintix/stats.json"
    pretrained_path: str = None  # 事前訓練済みモデルのパス
    start_epoch: int = 0
    seed: int = 5
    freeze_transformer: bool = True  # Transformerを凍結するかどうか

    # Dataset config
    dataset_config_paths: List[str] = None

    # wandb config
    project: str = "Vintix_Go2"
    group: str = "minicheetah_finetune"
    name: str = "minicheetah_finetune"

    def __post_init__(self):
        if self.dataset_config_paths is None:
            self.dataset_config_paths = ["configs/minicheetah_finetune_config.yaml"]
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(parent_dir, self.data_dir)
        
        if not os.path.isabs(self.stats_path):
            self.stats_path = os.path.join(parent_dir, self.stats_path)
        
        stats_dir = os.path.dirname(self.stats_path)
        os.makedirs(stats_dir, exist_ok=True)
        
        self.save_dir = os.path.join(self.save_dir, self.name)
        
        self.dataset_config = {}
        for dcp in self.dataset_config_paths:
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
        self.episode_range = []
        for v in self.dataset_config.values():
            if v.type == "default":
                if hasattr(v, 'episode_range') and v.episode_range is not None:
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


def add_new_group_to_model(model: Vintix, new_task_name: str, new_group_name: str,
                           group_metadata: dict, stats: dict) -> None:
    """既存モデルに新しいグループのエンコーダー/デコーダーを追加
    
    Args:
        model: 既存のVintixモデル
        new_task_name: 新しいタスク名（例: "minicheetah_walking_ad"）
        new_group_name: 新しいグループ名（例: "minicheetah_locomotion"）
        group_metadata: グループのメタデータ
        stats: タスクの統計情報
    """
    from torch import nn
    
    # メタデータを更新
    model.metadata[new_task_name] = group_metadata.copy()
    model.metadata[new_task_name].update(stats)
    
    # エンコーダーに新しいグループを追加
    from vintix.nn.individual_task_encoder import (
        get_obs_encoder, get_acs_encoder, RewardEncoder
    )
    
    # 観測エンコーダーを追加
    if new_group_name not in model.encoder.obs_encoders:
        model.encoder.obs_encoders[new_group_name] = get_obs_encoder(
            group_metadata=group_metadata,
            emb_dim=model.encoder.observation_emb_dim,
            emb_activation=nn.LeakyReLU(),
            image_encoder=None
        )
    
    # 行動エンコーダーを追加
    if new_group_name not in model.encoder.acs_encoders:
        model.encoder.acs_encoders[new_group_name] = get_acs_encoder(
            group_metadata=group_metadata,
            emb_dim=model.encoder.action_emb_dim,
            emb_activation=nn.LeakyReLU()
        )
    
    # 報酬エンコーダーを追加
    if new_group_name not in model.encoder.rews_encoders:
        model.encoder.rews_encoders[new_group_name] = RewardEncoder(
            emb_dim=model.encoder.reward_emb_dim,
            emb_activation=nn.LeakyReLU()
        )
    
    # task2groupとgroup_metadataを更新
    model.encoder.task2group[new_task_name] = new_group_name
    model.encoder.task_metadata[new_task_name] = model.metadata[new_task_name]
    model.encoder.group_metadata[new_group_name] = group_metadata
    
    # デコーダーに新しいグループを追加
    from vintix.nn.individual_task_head import AcsDecoder
    
    if new_group_name not in model.head.acs_decoders:
        model.head.acs_decoders[new_group_name] = AcsDecoder(
            task_metadata=group_metadata,
            hidden_dim=model.conf['hidden_dim'],
            emb_activation=nn.LeakyReLU(),
            out_activation=nn.Identity()
        )
    
    # headのメタデータを更新
    model.head.task2group[new_task_name] = new_group_name
    model.head.task_metadata[new_task_name] = model.metadata[new_task_name]
    model.head.group_metadata[new_group_name] = group_metadata


def freeze_transformer_parameters(model: Vintix) -> None:
    """Transformerのパラメータを凍結"""
    for param in model.transformer.parameters():
        param.requires_grad = False
    print("Transformer parameters frozen.")


def get_trainable_parameters(model: Vintix, new_group_name: str) -> List[torch.nn.Parameter]:
    """訓練可能なパラメータを取得（新しいグループのエンコーダー/デコーダーのみ）"""
    trainable_params = []
    
    # 新しいグループのエンコーダーのみ
    if new_group_name in model.encoder.obs_encoders:
        trainable_params.extend(model.encoder.obs_encoders[new_group_name].parameters())
    if new_group_name in model.encoder.acs_encoders:
        trainable_params.extend(model.encoder.acs_encoders[new_group_name].parameters())
    if new_group_name in model.encoder.rews_encoders:
        trainable_params.extend(model.encoder.rews_encoders[new_group_name].parameters())
    
    # 新しいグループのデコーダーのみ
    if new_group_name in model.head.acs_decoders:
        trainable_params.extend(model.head.acs_decoders[new_group_name].parameters())
    
    return trainable_params


@pyrallis.wrap()
def finetune(config: FinetuneConfig):
    set_seed(config.seed)
    
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

    print("LOADING PRETRAINED MODEL")
    scaler = torch.cuda.amp.GradScaler()
    torch_dtype = HALF_DTYPES[config.precision]
    
    if use_distributed:
        dist.barrier()

    device = torch.device("cuda", config.local_rank)
    
    # 事前訓練済みモデルをロード
    if config.pretrained_path is None:
        raise ValueError("pretrained_path must be specified for finetuning")
    
    model = Vintix()
    model.load_model(config.pretrained_path)
    model = model.to(device)
    
    # 新しいグループのメタデータを取得
    new_task_name = list(config.dataset_names.keys())[0]  # 最初のタスク名を取得
    new_group_name = config.dataset_names[new_task_name]
    
    # 新しいグループのエンコーダー/デコーダーを追加
    with open(config.stats_path, 'r') as f:
        stats = json.load(f)
    
    # グループメタデータを構築（dataset.metadataから取得）
    if new_task_name not in dataset.metadata:
        raise ValueError(f"Task {new_task_name} not found in dataset metadata")
    
    group_metadata = dataset.metadata[new_task_name].copy()
    # 統計情報を追加
    task_stats = stats.get(new_task_name, {})
    for key in task_stats.keys():
        group_metadata[key] = task_stats[key]
    
    # reward_scaleを追加
    reward_scale = config.dataset_config.get(new_task_name, {}).get("reward_scale", 1.0)
    group_metadata["reward_scale"] = reward_scale
    
    add_new_group_to_model(model, new_task_name, new_group_name, group_metadata, stats)
    
    # Transformerを凍結
    if config.freeze_transformer:
        freeze_transformer_parameters(model)
    
    if use_distributed:
        model = DDP(model,
                    device_ids=[config.local_rank],
                    find_unused_parameters=True)
        model_to_save = model.module
    else:
        model_to_save = model
    
    # 訓練可能なパラメータのみを最適化器に追加
    if config.freeze_transformer:
        trainable_params = get_trainable_parameters(model_to_save, new_group_name)
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        print(f"Training {sum(p.numel() for p in trainable_params)} parameters")
        print(f"Frozen {sum(p.numel() for p in model_to_save.transformer.parameters())} transformer parameters")
    else:
        optimizer = configure_optimizers(config, model)

    if config.local_rank == 0:
        wandb.watch(model, log_freq=500)
    
    total_updates = len(dataloader) * config.epochs
    scheduler = cosine_annealing_with_warmup(
        optimizer=optimizer,
        warmup_steps=int(total_updates * config.warmup_ratio),
        total_steps=total_updates,
    )

    print("MODEL INIT COMPLETE")
    print(f"TRAINABLE PARAMETERS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"FROZEN PARAMETERS: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}")
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
    finetune()
