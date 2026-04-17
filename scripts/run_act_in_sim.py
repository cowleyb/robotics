import argparse
from pathlib import Path
import secrets

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from sim.policy_observation import build_lerobot_observation, validate_features
from sim.stages import find_latest_checkpoint, get_stage_config
from sim.world import DRIVE_LIMITS, MAX_STEERING_ANGLE, World

DEFAULT_ACT_TEMPORAL_ENSEMBLE_COEFF = 0.01


def unnormalize_action(action: np.ndarray) -> tuple[float, float]:
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steering = float(np.clip(action[1], -1.0, 1.0))
    if throttle >= 0.0:
        throttle *= DRIVE_LIMITS.max_forward_wheel_speed
    else:
        throttle *= DRIVE_LIMITS.max_reverse_wheel_speed
    steering *= MAX_STEERING_ANGLE
    return throttle, steering


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    stage_config = get_stage_config(args.stage)
    checkpoint = args.checkpoint or find_latest_checkpoint(stage_config)
    instruction = args.instruction or stage_config.instruction
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    validate_features(
        policy_config.input_features,
        source=f"Checkpoint at {checkpoint}",
    )
    if (
        policy_config.type == "act"
        and getattr(policy_config, "chunk_size", 1) > 1
        and getattr(policy_config, "temporal_ensemble_coeff", None) is None
    ):
        policy_config.temporal_ensemble_coeff = DEFAULT_ACT_TEMPORAL_ENSEMBLE_COEFF
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    if getattr(policy.config, "temporal_ensemble_coeff", None) is None:
        policy.config.temporal_ensemble_coeff = getattr(
            policy_config, "temporal_ensemble_coeff", None
        )
    device = torch.device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )

    base_seed = args.seed
    initial_seed = base_seed if base_seed is not None else int(secrets.randbelow(2**31 - 1))
    world = World(
        seed=initial_seed,
        instruction=instruction,
        show_viewer=not args.headless,
        obstacle_count=stage_config.obstacle_count,
        gps_sensor_config=stage_config.gps_sensor_config,
    )
    successes = 0
    collisions = 0
    timeouts = 0

    print(f"{stage_config.label}")
    print(f"checkpoint: {checkpoint}")
    print(f"temporal ensemble coeff: {policy.config.temporal_ensemble_coeff}")
    try:
        for episode_idx in range(args.episodes):
            if base_seed is not None:
                seed = base_seed + episode_idx
            elif episode_idx == 0:
                seed = initial_seed
            else:
                seed = int(secrets.randbelow(2**31 - 1))

            policy.reset()
            if episode_idx == 0:
                observation = world.get_observation()
            else:
                observation = world.reset(seed=seed)
            print(f"episode {episode_idx + 1}/{args.episodes} seed: {world.seed}")

            for step_idx in range(1, args.max_steps + 1):
                model_observation = build_lerobot_observation(
                    observation=observation,
                )
                action = predict_action(
                    observation=model_observation,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=world.instruction,
                )
                normalized_action = np.asarray(
                    action.squeeze(0).detach().cpu().numpy(),
                    dtype=np.float32,
                )
                world.last_action = normalized_action.copy()
                throttle, steering = unnormalize_action(normalized_action)
                world.move_car(throttle=throttle, steering=steering)
                observation = world.step()

                if world.goal_reached():
                    successes += 1
                    print(f"episode {episode_idx + 1}: goal reached at step {step_idx}")
                    break
                if world.hit_obstacle():
                    collisions += 1
                    print(f"episode {episode_idx + 1}: hit obstacle at step {step_idx}")
                    break
            else:
                timeouts += 1
                print(f"episode {episode_idx + 1}: timeout at step {args.max_steps}")
    finally:
        world.close()

    print(f"successes: {successes}/{args.episodes}")
    print(f"collisions: {collisions}")
    print(f"timeouts: {timeouts}")


if __name__ == "__main__":
    main()
