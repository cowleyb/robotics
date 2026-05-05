import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from sim_mentor_pi.car_env import TestEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="rc-car-yolo-rl")
    parser.add_argument("--ckpt", type=int, default=300)
    parser.add_argument("--record", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(
        backend=gs.cpu,
        precision="32",
        logging_level="warning",
        performance_mode=False,
    )

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, target_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env_cfg["visualize_target"] = True
    env_cfg["visualize_camera"] = args.record
    env_cfg["max_visualize_FPS"] = 60
    env_cfg["target_cfg"] = {
        "num_commands": 3,
        "box_size": [0.3, 0.3, 0.3],
        "pos_x_range": [-4.0, 4.0],
        "pos_y_range": [-4.0, 4.0],
        "pos_z_range": [0.1, 0.1],
    }

    env = TestEnv(
        base_seed=1,
        num_envs=5,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
        show_viewer=True,
        manual=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs_dict = env.reset()
    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])

    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs_dict)
                obs_dict, _, _, _ = env.step(actions)
            env.cam.stop_recording(
                save_to_filename="video.mp4",
                fps=env_cfg["max_visualize_FPS"],
            )
        else:
            try:
                while env.manual_is_running:
                    actions = policy(obs_dict)
                    obs_dict, _, _, _ = env.step(actions)
            except KeyboardInterrupt:
                print("stopped")


if __name__ == "__main__":
    main()
