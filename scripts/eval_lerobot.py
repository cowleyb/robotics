import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from scripts.act_utils import split_episode_indices
from sim.policy_observation import validate_features
from sim.stages import find_latest_checkpoint, get_stage_config


def evaluate_checkpoint(
    checkpoint: Path,
    stage: int,
    train_fraction: float = 0.9,
    split_seed: int = 0,
    batch_size: int = 16,
    num_workers: int = 4,
    max_batches: int | None = None,
) -> dict[str, int | float]:
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("--train_fraction must be in (0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least 1")

    stage_config = get_stage_config(stage)
    if not stage_config.dataset_root.exists():
        raise FileNotFoundError(f"LeRobot dataset not found at {stage_config.dataset_root}")

    metadata = LeRobotDatasetMetadata(
        repo_id=stage_config.repo_id,
        root=stage_config.dataset_root,
        force_cache_sync=False,
    )
    validate_features(
        metadata.features,
        source=f"Dataset at {stage_config.dataset_root}",
    )
    _, validation_episodes = split_episode_indices(
        total_episodes=metadata.total_episodes,
        train_fraction=train_fraction,
        seed=split_seed,
    )
    if not validation_episodes:
        raise ValueError(
            "No validation episodes available for the requested split. "
            "Record at least 2 episodes or lower --train_fraction."
        )

    checkpoint = checkpoint.resolve()
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    validate_features(
        policy_config.input_features,
        source=f"Checkpoint at {checkpoint}",
    )
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    device = torch.device(policy.config.device)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=stage_config.repo_id,
            root=str(stage_config.dataset_root),
            episodes=validation_episodes,
        ),
        policy=policy_config,
        output_dir=str(stage_config.train_output_root),
        batch_size=batch_size,
        num_workers=num_workers,
        steps=1,
        eval_freq=0,
        save_freq=0,
        save_checkpoint=False,
    )
    dataset = make_dataset(cfg)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    total_loss = 0.0
    total_l1_loss = 0.0
    total_kld_loss = 0.0
    total_items = 0
    batch_count = 0
    saw_kld_loss = False

    policy.train()
    with torch.no_grad():
        for batch in dataloader:
            processed_batch = preprocessor(batch)
            loss, metrics = policy.forward(processed_batch)

            batch_items = int(processed_batch["action"].shape[0])
            total_items += batch_items
            batch_count += 1
            total_loss += float(loss.item()) * batch_items
            total_l1_loss += float(metrics["l1_loss"]) * batch_items
            if "kld_loss" in metrics:
                saw_kld_loss = True
                total_kld_loss += float(metrics["kld_loss"]) * batch_items

            if max_batches is not None and batch_count >= max_batches:
                break

    if total_items == 0:
        raise RuntimeError("Validation dataset is empty after applying the requested split.")

    results: dict[str, int | float] = {
        "validation_batches": batch_count,
        "validation_items": total_items,
        "mean_validation_loss": total_loss / total_items,
        "mean_validation_l1_loss": total_l1_loss / total_items,
    }
    if saw_kld_loss:
        results["mean_validation_kld_loss"] = total_kld_loss / total_items
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    args = parser.parse_args()

    stage_config = get_stage_config(args.stage)
    checkpoint = args.checkpoint or find_latest_checkpoint(stage_config)
    results = evaluate_checkpoint(
        checkpoint=checkpoint,
        stage=args.stage,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )

    print(f"{stage_config.label}")
    print(f"checkpoint: {checkpoint}")
    print(f"validation batches: {results['validation_batches']}")
    print(f"validation items: {results['validation_items']}")
    print(f"mean validation loss: {results['mean_validation_loss']:.6f}")
    print(f"mean validation l1_loss: {results['mean_validation_l1_loss']:.6f}")
    if "mean_validation_kld_loss" in results:
        print(f"mean validation kld_loss: {results['mean_validation_kld_loss']:.6f}")


if __name__ == "__main__":
    main()
