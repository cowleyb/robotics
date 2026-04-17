import random
from pathlib import Path

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from sim.policy_observation import get_policy_input_features, validate_features


def split_episode_indices(
    total_episodes: int,
    train_fraction: float,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    if total_episodes < 2:
        return list(range(total_episodes)), []

    train_count = int(total_episodes * train_fraction)
    train_count = max(1, min(total_episodes - 1, train_count))
    episode_indices = list(range(total_episodes))
    random.Random(seed).shuffle(episode_indices)
    train_episodes = sorted(episode_indices[:train_count])
    validation_episodes = sorted(episode_indices[train_count:])
    return train_episodes, validation_episodes


def build_policy_config(
    chunk_size: int,
    checkpoint: Path | None,
    temporal_ensemble_coeff: float | None = 0.01,
) -> ACTConfig:
    if checkpoint is None:
        return ACTConfig(
            chunk_size=chunk_size,
            n_action_steps=1,
            input_features=get_policy_input_features(),
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
            },
            temporal_ensemble_coeff=temporal_ensemble_coeff,
            push_to_hub=False,
        )

    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, ACTConfig):
        raise ValueError(f"Checkpoint at {checkpoint} is not an ACT policy.")
    validate_features(
        policy_config.input_features,
        source=f"Checkpoint at {checkpoint}",
    )
    policy_config.pretrained_path = checkpoint
    if temporal_ensemble_coeff is not None:
        policy_config.temporal_ensemble_coeff = temporal_ensemble_coeff
    policy_config.push_to_hub = False
    return policy_config
