import argparse
from pathlib import Path

import torch
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

from scripts.train_act import split_episode_indices
from sim.stages import find_latest_checkpoint, get_stage_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    stage_config = get_stage_config(args.stage)
    if not stage_config.dataset_root.exists():
        raise FileNotFoundError(f"LeRobot dataset not found at {stage_config.dataset_root}")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train_fraction must be in (0, 1]")

    checkpoint = args.checkpoint or find_latest_checkpoint(stage_config)
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    device = torch.device(policy.config.device)

    metadata = LeRobotDatasetMetadata(
        repo_id=stage_config.repo_id,
        root=stage_config.dataset_root,
        force_cache_sync=False,
    )
    _, validation_episodes = split_episode_indices(
        total_episodes=metadata.total_episodes,
        train_fraction=args.train_fraction,
    )
    if not validation_episodes:
        raise ValueError("No validation episodes available. Record at least 2 episodes.")

    dataset = LeRobotDataset(
        repo_id=stage_config.repo_id,
        root=stage_config.dataset_root,
        episodes=validation_episodes,
        delta_timestamps=resolve_delta_timestamps(policy.config, metadata),
        tolerance_s=1e-4,
    )
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

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
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    policy.eval()
    total_loss = 0.0
    total_l1_loss = 0.0
    total_kld_loss = 0.0
    batch_count = 0

    with torch.inference_mode():
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, metrics = policy.forward(batch)
            total_loss += float(loss.item())
            total_l1_loss += float(metrics["l1_loss"])
            total_kld_loss += float(metrics.get("kld_loss", 0.0))
            batch_count += 1
            if args.max_batches is not None and batch_count >= args.max_batches:
                break

    if batch_count == 0:
        raise ValueError("Validation dataloader produced no batches.")

    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    print(f"checkpoint: {checkpoint}")
    print(f"validation episodes: {validation_episodes}")
    print(f"validation batches: {batch_count}")
    print(f"mean validation loss: {total_loss / batch_count:.6f}")
    print(f"mean validation l1_loss: {total_l1_loss / batch_count:.6f}")
    if total_kld_loss > 0.0:
        print(f"mean validation kld_loss: {total_kld_loss / batch_count:.6f}")


if __name__ == "__main__":
    main()
