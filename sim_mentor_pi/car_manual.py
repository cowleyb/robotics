import argparse

import genesis as gs

from sim_mentor_pi.car_train import get_cfgs
from sim_mentor_pi.car_env import TestEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    gs.init(
        backend=gs.cpu,
        precision="32",
        # logging_level="warning",
        seed=args.seed,
        performance_mode=False,
    )

    env_cfg, obs_cfg, reward_cfg, target_cfg = get_cfgs()
    reward_cfg["reward_scales"] = {}
    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.cam
    env_cfg["max_visualize_FPS"] = 60

    env = TestEnv(
        base_seed=args.seed,
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
        show_viewer=True,
        manual=True,
    )

    print("Arrow keys: up/down throttle, left/right steering. Esc to exit.")

    step_count = 0
    while env.manual_is_running:
        env.step(env.manual_action)
        step_count += 1
        if args.debug and step_count % 60 == 0:
            pos = env.car.get_pos()[0].detach().cpu().numpy()
            wheel_vel = (
                env.car.get_dofs_velocity(dofs_idx_local=env.car.drive_dofs_idx)[0]
                .detach()
                .cpu()
                .numpy()
            )
            pitch = float(env.base_euler[0, 1].detach().cpu())
            print(
                f"pos={pos.round(3)} pitch={pitch:.2f} wheel_vel={wheel_vel.round(2)}"
            )


if __name__ == "__main__":
    main()
