from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


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
