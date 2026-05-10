import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import genesis as gs
import torch

from sim_mentor_pi.car_env import TestEnv
from sim_mentor_pi.car_train import get_cfgs


def simple_entry_controller(obs_buf: torch.Tensor) -> torch.Tensor:
    target_x = obs_buf[:, 0]
    target_y = obs_buf[:, 1]

    actions = torch.zeros((obs_buf.shape[0], 2), device=gs.device, dtype=gs.tc_float)
    actions[:, 0] = torch.where(target_x > 0.05, 0.7, 0.15)
    actions[:, 1] = torch.clamp(target_y * 3.0, -1.0, 1.0)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    gs.init(
        backend=gs.cpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=False,
    )

    env_cfg, obs_cfg, reward_cfg, target_cfg = get_cfgs()
    env_cfg["visualize_target"] = True
    env_cfg["max_visualize_FPS"] = 60
    reward_cfg["reward_scales"] = {}

    env = TestEnv(
        base_seed=args.seed,
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
        show_viewer=True,
    )

    obs_dict = env.reset()
    while True:
        actions = simple_entry_controller(obs_dict["policy"])
        obs_dict, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
