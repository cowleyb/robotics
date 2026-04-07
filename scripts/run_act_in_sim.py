import argparse
from pathlib import Path
import secrets

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from sim.test import build_lerobot_observation
from sim.stages import find_latest_checkpoint, get_stage_config
from sim.world import DRIVE_LIMITS, MAX_STEERING_ANGLE, World


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
    parser.add_argument(
        "--instruction",
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=5000)
    args = parser.parse_args()

    stage_config = get_stage_config(args.stage)
    checkpoint = args.checkpoint or find_latest_checkpoint(stage_config)
    instruction = args.instruction or stage_config.instruction
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(checkpoint, config=policy_config)
    device = torch.device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
        },
    )

    seed = args.seed if args.seed is not None else int(secrets.randbelow(2**31 - 1))
    world = World(
        seed=seed,
        instruction=instruction,
        show_viewer=True,
        obstacle_count=stage_config.obstacle_count,
    )
    policy.reset()
    observation = world.get_observation()
    print(f"{stage_config.label}")
    print(f"checkpoint: {checkpoint}")
    print(f"seed: {world.seed}")

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
            print(f"goal reached at step {step_idx}")
            return
        if world.hit_obstacle():
            print(f"hit obstacle at step {step_idx}")
            return

    print(f"timeout at step {args.max_steps}")


if __name__ == "__main__":
    main()
