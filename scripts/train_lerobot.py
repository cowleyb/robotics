import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.scripts.lerobot_train import train
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

from scripts.act_utils import build_policy_config, split_episode_indices
from scripts.eval_lerobot import evaluate_checkpoint
from sim.policy_observation import validate_features
from sim.stages import get_stage_config


def find_latest_checkpoint_under(output_dir: Path) -> Path:
    checkpoints = [
        path
        for path in output_dir.rglob(PRETRAINED_MODEL_DIR)
        if path.is_dir() and path.parent.parent.name == "checkpoints"
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No trained checkpoint found under {output_dir}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--train_fraction", type=float, default=0.9)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=0.01)
    parser.add_argument("--disable_image_transforms", action="store_true")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_num_workers", type=int, default=4)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--skip_eval", action="store_true")
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
    validate_features(
        metadata.features,
        source=f"Dataset at {stage_config.dataset_root}",
    )
    train_episodes, validation_episodes = split_episode_indices(
        total_episodes=metadata.total_episodes,
        train_fraction=args.train_fraction,
        seed=args.split_seed,
    )
    if not train_episodes:
        raise ValueError("No training episodes available. Record at least 1 episode.")
    if not args.skip_eval and not validation_episodes:
        raise ValueError(
            "No validation episodes available for the requested split. "
            "Record at least 2 episodes or pass --skip_eval."
        )

    output_dir = args.output_dir or stage_config.train_output_root
    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    print(f"training episodes: {train_episodes}")
    print(f"validation episodes: {validation_episodes}")
    print(f"training outputs: {output_dir}")
    print(f"split seed: {args.split_seed}")
    print(f"temporal ensemble coeff: {args.temporal_ensemble_coeff}")
    print(f"image transforms enabled: {not args.disable_image_transforms}")
    if args.checkpoint is not None:
        print(f"warm-start checkpoint: {args.checkpoint}")

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=stage_config.repo_id,
            root=str(stage_config.dataset_root),
            episodes=train_episodes,
            image_transforms=ImageTransformsConfig(
                enable=not args.disable_image_transforms,
                max_num_transforms=2,
            ),
        ),
        policy=build_policy_config(
            chunk_size=args.chunk_size,
            checkpoint=args.checkpoint,
            temporal_ensemble_coeff=args.temporal_ensemble_coeff,
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

    trained_checkpoint = find_latest_checkpoint_under(output_dir)
    print(f"trained checkpoint: {trained_checkpoint}")
    if args.skip_eval:
        return

    results = evaluate_checkpoint(
        checkpoint=trained_checkpoint,
        stage=args.stage,
        train_fraction=args.train_fraction,
        split_seed=args.split_seed,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        max_batches=args.eval_max_batches,
    )
    print(f"validation batches: {results['validation_batches']}")
    print(f"mean validation loss: {results['mean_validation_loss']:.6f}")
    print(f"mean validation l1_loss: {results['mean_validation_l1_loss']:.6f}")
    if "mean_validation_kld_loss" in results:
        print(f"mean validation kld_loss: {results['mean_validation_kld_loss']:.6f}")


if __name__ == "__main__":
    main()
