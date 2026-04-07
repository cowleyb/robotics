import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.scripts.lerobot_train import train

from sim.stages import get_stage_config


def split_episode_indices(total_episodes: int, train_fraction: float) -> tuple[list[int], list[int]]:
    if total_episodes < 2:
        return list(range(total_episodes)), []

    train_count = int(total_episodes * train_fraction)
    train_count = max(1, min(total_episodes - 1, train_count))
    train_episodes = list(range(train_count))
    validation_episodes = list(range(train_count, total_episodes))
    return train_episodes, validation_episodes


def build_policy_config(chunk_size: int, checkpoint: Path | None) -> ACTConfig:
    if checkpoint is None:
        return ACTConfig(
            chunk_size=chunk_size,
            n_action_steps=1,
            push_to_hub=False,
        )

    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, ACTConfig):
        raise ValueError(f"Checkpoint at {checkpoint} is not an ACT policy.")
    policy_config.pretrained_path = checkpoint
    policy_config.push_to_hub = False
    return policy_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    stage_config = get_stage_config(args.stage)
    if not stage_config.dataset_root.exists():
        raise FileNotFoundError(f"LeRobot dataset not found at {stage_config.dataset_root}")
    if not 0.0 < args.train_fraction <= 1.0:
        raise ValueError("--train_fraction must be in (0, 1]")

    metadata = LeRobotDatasetMetadata(
        repo_id=stage_config.repo_id,
        root=stage_config.dataset_root,
        force_cache_sync=False,
    )
    train_episodes, validation_episodes = split_episode_indices(
        total_episodes=metadata.total_episodes,
        train_fraction=args.train_fraction,
    )
    output_dir = args.output_dir or stage_config.train_output_root
    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    print(f"training episodes: {train_episodes}")
    print(f"validation episodes: {validation_episodes}")
    print(f"training outputs: {output_dir}")
    if args.checkpoint is not None:
        print(f"warm-start checkpoint: {args.checkpoint}")

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=stage_config.repo_id,
            root=str(stage_config.dataset_root),
            episodes=train_episodes,
        ),
        policy=build_policy_config(
            chunk_size=args.chunk_size,
            checkpoint=args.checkpoint,
        ),
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        eval_freq=0,
        log_freq=100,
        save_freq=5_000,
        save_checkpoint=True,
        wandb=WandBConfig(enable=False),
    )
    train(cfg)


if __name__ == "__main__":
    main()
