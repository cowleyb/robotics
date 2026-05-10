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

BASE_POLICY_STATE_DIM = 15
WALL_POSITION_DIM = 2


def update_policy_state_dim(env_cfg, obs_cfg):
    wall_count = env_cfg["goal_walls"]["count"]
    if not env_cfg["goal_walls"]["enabled"]:
        wall_count = 0
    obs_cfg["state_dim"] = (
        BASE_POLICY_STATE_DIM
        + wall_count * WALL_POSITION_DIM
    )


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
        "at_target_threshold": 0.10,
        "episode_length_s": 20.0,
        "clip_actions": 1.0,
        "cam_resolution": (160, 120),
        "cam_fov": 90,
        "cam_pos": [0.064015, -0.00013463, 0.051155],
        "cam_lookat": [1.064015, -0.00013463, 0.051155],
        "visualize_target": False,
        "visualize_camera": False,
        "max_visualize_FPS": 60,
        "goal_walls": {
            "enabled": True,
            "count": 3,
            "depth": 0.7,
            "half_width": 0.35,
            "thickness": 0.06,
            "height": 0.18,
            "car_radius": 0.16,
            "near_distance": 0.5,
            "entry_offset": 0.25,
            "entry_threshold": 0.25,
        },
    }
    obs_cfg = {
        "state_dim": 0,
        "image_shape": (3, 120, 160),
        "obs_scales": {
            "rel_pos": 0.25,
            "lin_vel": 0.5,
            "ang_vel": 0.25,
            "wall_pos": 0.5,
        },
    }
    update_policy_state_dim(env_cfg, obs_cfg)
    reward_cfg = {
        "reward_scales": {
            "entry_progress": 30.0,
            "progress": 20.0,
            "heading": 0.2,
            "reverse": -0.2,
            "steering": -0.1,
            "smooth": -0.02,
            "near_wall": -1.0,
            "success": 200.0,
            "crash": -300.0,
            "timeout": -50.0,
        }
    }
    target_cfg = {
        "num_commands": 3,
        "box_size": [0.3, 0.3, 0.3],
        "pos_x_range": [-1.2, 1.2],
        "pos_y_range": [-1.2, 1.2],
        "pos_z_range": [0.1, 0.1],
        "min_start_distance": 0.8,
    }
    return env_cfg, obs_cfg, reward_cfg, target_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="room-goal-privileged-rl")
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--cam", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--no_walls", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", action="store_true", default=False)
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
    if args.no_walls:
        env_cfg["goal_walls"]["enabled"] = False
        update_policy_state_dim(env_cfg, obs_cfg)

    if os.path.exists(log_dir) and not args.resume:
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.vis:
        env_cfg["visualize_target"] = True
    if args.cam:
        env_cfg["visualize_camera"] = True

    print(f"training run: {args.exp_name}")
    print(f"parallel envs: {args.num_envs}")
    print(f"goal walls enabled: {env_cfg['goal_walls']['enabled']}")
    print(f"policy state dim: {obs_cfg['state_dim']}")
    print("policy inputs: goal direction, car state, last action, wall positions")

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
    if args.resume:
        checkpoints = [
            int(path.removeprefix("model_").removesuffix(".pt"))
            for path in os.listdir(log_dir)
            if path.startswith("model_") and path.endswith(".pt")
        ]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}")
        ckpt = max(checkpoints)
        runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )
    # while True:
    #     env.step()


if __name__ == "__main__":
    main()
