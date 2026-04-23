import argparse
import os
import genesis as gs
from huggingface_hub import snapshot_download
import numpy as np

from importlib import metadata
from sim2.car_env import RoomEnv

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name):
    train_cfg_dict = {
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

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        # --- CONTROLS ---
        "num_actions": 2,  # RC cars only need [Throttle, Steering]
        # --- TERMINATION (When to auto-reset the sim) ---
        "termination_if_x_greater_than": 5.0,  # Don't let it drive too far away
        "termination_if_y_greater_than": 5.0,
        "termination_if_roll_greater_than": 90,  # If the car flips over (90 deg), end episode
        "termination_if_pitch_greater_than": 45,  # If it goes up a crazy ramp, end episode
        # --- INITIAL POSE ---
        "base_init_pos": [
            0.0,
            0.0,
            0.1,
        ],  # Start slightly above ground so it drops down
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],  # Flat on the ground
        "at_target_threshold": 0.1,
        # --- TIMING ---
        "episode_length_s": 30.0,  # Give yourself 30 seconds to teleop to the box
        "ctrl_dt": 0.02,  # 50 Hz control loop (standard for RC)
        # --- CAMERA (CRITICAL FOR LEROBOT) ---
        "cam_resolution": (160, 120),  # Keep it small! Pi needs this to run fast later
        "cam_fov": 90,  # Wide angle lens
        "cam_pos": [
            0.2,
            0.0,
            0.15,
        ],  # Position relative to car center (front, center, height)
        "cam_lookat": [1.0, 0.0, 0.0],  # Look straight forward along the X axis
        # --- VISUALIZATION ---
        "visualize_target": True,  # Turn this ON so you can see the yellow box in Genesis
        "visualize_camera": False,
        "max_visualize_FPS": 60,
    }

    # --- OBSERVATION (For LeRobot State Vector) ---
    # We define the dimension of the state vector LeRobot will receive.
    # [error_x, error_y, box_size, is_visible, last_error_x, last_box_size]
    obs_cfg = {
        "state_dim": 6,
        "image_shape": (
            3,
            120,
            160,
        ),  # (Channels, Height, Width) - standard PyTorch format
    }

    # --- REWARD CONFIG ---
    # DELETE THIS ENTIRELY. You are not doing RL. You do not need rewards.

    # --- TARGET/YELLOW BOX CONFIG ---
    # Instead of "commands", we just define where the yellow box spawns
    target_cfg = {
        "num_commands": 3,
        "box_size": [0.3, 0.3, 0.3],  # 30cm yellow box
        # Randomly spawn the box in a 4x4 meter area around the car
        "pos_x_range": [-5.0, 5.0],
        "pos_y_range": [-5.0, 5.0],
        "pos_z_range": [0.1, 0.1],  # Always on the ground (half the box size)
    }

    # Note: We return None for reward_cfg
    return env_cfg, obs_cfg, None, target_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true", help="Drive car with wasd")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--vis", action="store_true", default=False, help="Visualize the environment"
    )
    parser.add_argument(
        "--cam", action="store_true", default=False, help="Visualize the camera view"
    )

    args = parser.parse_args()

    instruction = "Drive to A."

    # How much we mess with the car in the simulation
    recovery_data = {
        "perturb_probability": 0,
        "throttle_std": 0.08,
        "steering_std": 0.25,
        "burst_length_range_steps": (2, 5),
        "recovery_length_range_steps": (20, 45),
    }

    seed = int(args.seed) if args.seed is not None else None
    rng = np.random.default_rng(seed)
    world_seed = rng.integers(0, 2**32 - 1)

    # Download InteriorAgent scene
    # asset_path = snapshot_download(
    #     repo_id="spatialverse/InteriorAgent",
    #     repo_type="dataset",
    #     allow_patterns="kujiale_0003/*",
    #     max_workers=4,
    # )

    env_cfg, obs_cfg, reward_cfg, target_cfg = get_cfgs()
    if args.cam:
        env_cfg["visualize_camera"] = True

    gs.init(
        backend=gs.cpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        # performance_mode=True,
    )

    env = RoomEnv(
        base_seed=seed,
        num_envs=4,
        show_viewer=args.vis,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
    )

    while True:
        env.step()


if __name__ == "__main__":
    main()
