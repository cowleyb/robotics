import argparse
import os
import pickle
import shutil
from importlib import metadata

import genesis as gs

from sim_mentor_pi.car_env import TestEnv

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.004,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 100,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
    env_cfg = {
        "num_actions": 2,
        "termination_if_x_greater_than": 5.0,
        "termination_if_y_greater_than": 5.0,
        "termination_if_roll_greater_than": 90.0,
        "termination_if_pitch_greater_than": 45.0,
        "base_init_pos": [0.0, 0.0, 0.0335],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "at_target_threshold": 0.15,
        "episode_length_s": 20.0,
        "clip_actions": 1.0,
        "cam_resolution": (160, 120),
        "cam_fov": 90,
        "cam_pos": [0.064015, -0.00013463, 0.051155],
        "cam_lookat": [1.064015, -0.00013463, 0.051155],
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }
    obs_cfg = {
        "state_dim": 15,
        "image_shape": (3, 120, 160),
        "obs_scales": {
            "rel_pos": 0.25,
            "lin_vel": 0.5,
            "ang_vel": 0.25,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "progress": 20.0,
            "heading": 1.0,
            "reverse": -1.0,
            "smooth": -0.02,
            "success": 200.0,
            "crash": -100.0,
            "timeout": -50.0,
        }
    }
    target_cfg = {
        "num_commands": 3,
        "box_size": [0.3, 0.3, 0.3],
        "pos_x_range": [-1.0, 1.0],
        "pos_y_range": [-1.0, 1.0],
        "pos_z_range": [0.1, 0.1],
    }
    return env_cfg, obs_cfg, reward_cfg, target_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="rc-car-yolo-rl")
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--cam", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    backend = gs.cpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.vis,
    )

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, target_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True
    if args.cam:
        env_cfg["visualize_camera"] = True

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, target_cfg, train_cfg], f)

    env = TestEnv(
        base_seed=args.seed,
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )
    # while True:
    #     env.step()


if __name__ == "__main__":
    main()
