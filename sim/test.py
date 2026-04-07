import argparse
import secrets

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from genesis.vis.keybindings import Key, KeyAction, Keybind
from sim.stages import StageConfig, get_stage_config
from sim.world import DRIVE_LIMITS, MAX_STEERING_ANGLE, World


def set_control(control_state: dict[str, float], key: str, value: float) -> None:
    control_state[key] = value


def normalize_action(throttle: float, steering: float) -> dict[str, float]:
    if throttle >= 0.0:
        normalized_throttle = throttle / DRIVE_LIMITS.max_forward_wheel_speed
    else:
        normalized_throttle = throttle / DRIVE_LIMITS.max_reverse_wheel_speed
    return {
        "throttle": max(-1.0, min(1.0, normalized_throttle)),
        "steering": max(-1.0, min(1.0, steering / MAX_STEERING_ANGLE)),
    }


def rotate_world_vector_to_car_frame(
    vector_xy: np.ndarray, car_quaternion: np.ndarray
) -> np.ndarray:
    w, x, y, z = car_quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray(
        [
            cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
            -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
        ],
        dtype=np.float32,
    )


def save_episode(
    trajectory: list[dict[str, object]],
    stage_config: StageConfig,
) -> Path:
    if not trajectory:
        raise ValueError("trajectory must contain at least one frame")

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": trajectory[0]["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": trajectory[0]["action"].shape,
            "names": ["throttle", "steering"],
        },
        "observation.images.front": {
            "dtype": "image",
            "shape": trajectory[0]["observation.images.front"].shape,
            "names": ["height", "width", "channels"],
        },
    }

    dataset = open_lerobot_dataset(
        features=features,
        stage_config=stage_config,
    )

    for frame in trajectory:
        dataset.add_frame(dict(frame))
    dataset.save_episode()
    dataset.finalize()
    return dataset.root


def open_lerobot_dataset(
    features: dict[str, dict[str, object]],
    stage_config: StageConfig,
) -> LeRobotDataset:
    if not stage_config.dataset_root.exists():
        return LeRobotDataset.create(
            repo_id=stage_config.repo_id,
            fps=100,
            features=features,
            root=stage_config.dataset_root,
            use_videos=False,
        )

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.repo_id = stage_config.repo_id
    dataset.root = stage_config.dataset_root
    dataset.revision = None
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.episodes = None
    dataset.tolerance_s = 1e-4
    dataset.video_backend = None
    dataset.delta_indices = None
    dataset.batch_encoding_size = 1
    dataset.episodes_since_last_encoding = 0
    dataset.vcodec = "libsvtav1"
    dataset._encoder_threads = None
    dataset.image_writer = None
    dataset.episode_buffer = None
    dataset.writer = None
    dataset.latest_episode = None
    dataset._current_file_start_frame = None
    dataset._streaming_encoder = None
    dataset.meta = LeRobotDatasetMetadata(
        repo_id=stage_config.repo_id,
        root=stage_config.dataset_root,
        force_cache_sync=False,
    )
    dataset._lazy_loading = False
    dataset._recorded_frames = dataset.meta.total_frames
    dataset._writer_closed_for_reading = False
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset._absolute_to_relative_idx = None
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def build_lerobot_observation(
    observation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    goal_delta_world = np.asarray(
        observation["goal_position"][:2] - observation["car_position"][:2],
        dtype=np.float32,
    )
    goal_delta = rotate_world_vector_to_car_frame(
        goal_delta_world,
        np.asarray(observation["car_quaternion"], dtype=np.float32),
    )
    car_velocity = rotate_world_vector_to_car_frame(
        np.asarray(observation["car_linear_velocity"][:2], dtype=np.float32),
        np.asarray(observation["car_quaternion"], dtype=np.float32),
    )
    observation_state = np.concatenate(
        (
            goal_delta,
            car_velocity,
            np.asarray(observation["steering_position"], dtype=np.float32),
            np.asarray(observation["last_action"], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    return {
        "observation.state": np.ascontiguousarray(observation_state),
        "observation.images.front": np.ascontiguousarray(
            observation["image"],
            dtype=np.uint8,
        ),
    }


def build_lerobot_frame(
    observation: dict[str, np.ndarray],
    action: dict[str, float],
    instruction: str,
) -> dict[str, object]:
    frame = build_lerobot_observation(
        observation=observation,
    )
    frame["task"] = instruction
    frame["action"] = np.asarray(
        [action["throttle"], action["steering"]],
        dtype=np.float32,
    )
    return frame


def register_keyboard_controls(world: World, control_state: dict[str, float]) -> None:
    world.scene.viewer.register_keybinds(
        Keybind(
            "drive_forward_press",
            Key.W,
            KeyAction.PRESS,
            callback=set_control,
            args=(control_state, "forward", 1.0),
        ),
        Keybind(
            "drive_forward_release",
            Key.W,
            KeyAction.RELEASE,
            callback=set_control,
            args=(control_state, "forward", 0.0),
        ),
        Keybind(
            "drive_reverse_press",
            Key.S,
            KeyAction.PRESS,
            callback=set_control,
            args=(control_state, "reverse", 1.0),
        ),
        Keybind(
            "drive_reverse_release",
            Key.S,
            KeyAction.RELEASE,
            callback=set_control,
            args=(control_state, "reverse", 0.0),
        ),
        Keybind(
            "steer_left_press",
            Key.A,
            KeyAction.PRESS,
            callback=set_control,
            args=(control_state, "left", 1.0),
        ),
        Keybind(
            "steer_left_release",
            Key.A,
            KeyAction.RELEASE,
            callback=set_control,
            args=(control_state, "left", 0.0),
        ),
        Keybind(
            "steer_right_press",
            Key.D,
            KeyAction.PRESS,
            callback=set_control,
            args=(control_state, "right", 1.0),
        ),
        Keybind(
            "steer_right_release",
            Key.D,
            KeyAction.RELEASE,
            callback=set_control,
            args=(control_state, "right", 0.0),
        ),
        overwrite=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual",
        action="store_true",
    )
    parser.add_argument(
        "--show_viewer",
        action="store_true",
    )
    parser.add_argument(
        "--instruction",
        default=None,
    )
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--seed",
        default=None,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    stage_config = get_stage_config(args.stage)
    instruction = args.instruction or stage_config.instruction
    base_seed = int(args.seed) if args.seed is not None else None
    output_path = None
    saved_episodes = 0
    failed_seeds = []
    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    for episode_idx in range(args.episodes):
        if base_seed is not None:
            seed = base_seed + episode_idx
        else:
            seed = int(secrets.randbelow(2**31 - 1))

        world = World(
            seed=seed,
            instruction=instruction,
            show_viewer=args.show_viewer,
            obstacle_count=stage_config.obstacle_count,
        )

        observation = world.get_observation()
        print(f"episode {episode_idx + 1}/{args.episodes} seed: {world.seed}")

        step_count = 0
        trajectory = []
        control_state = {
            "forward": 0.0,
            "reverse": 0.0,
            "left": 0.0,
            "right": 0.0,
        }

        if args.manual:
            register_keyboard_controls(world, control_state)

        reached_goal = False
        hit_obstacle = False
        timed_out = False
        planning_error: RuntimeError | None = None
        while True:
            if args.manual:
                throttle = (
                    DRIVE_LIMITS.max_forward_wheel_speed * control_state["forward"]
                    - DRIVE_LIMITS.max_reverse_wheel_speed * control_state["reverse"]
                )
                steering = MAX_STEERING_ANGLE * (
                    control_state["left"] - control_state["right"]
                )
            else:
                try:
                    throttle, steering = world.heuristic_action()
                except RuntimeError as exc:
                    planning_error = exc
                    print(
                        f"episode {episode_idx + 1}: planner failed at step {step_count} "
                        f"for seed {world.seed}: {exc}"
                    )
                    break
            normalized_action = normalize_action(throttle=throttle, steering=steering)
            world.last_action = np.array(
                [normalized_action["throttle"], normalized_action["steering"]],
                dtype=np.float32,
            )
            world.move_car(throttle=throttle, steering=steering)
            next_observation = world.step()
            step_count += 1
            reached_goal = world.goal_reached()
            hit_obstacle = world.hit_obstacle()
            timed_out = step_count >= 5000
            trajectory.append(
                build_lerobot_frame(
                    observation=observation,
                    action=normalized_action,
                    instruction=world.instruction,
                )
            )
            observation = next_observation
            if reached_goal:
                print(f"episode {episode_idx + 1}: goal reached at step {step_count}")
                break
            if hit_obstacle:
                print(f"episode {episode_idx + 1}: hit obstacle at step {step_count}")
                break
            if timed_out:
                print(f"episode {episode_idx + 1}: timeout at step {step_count}")
                break

        print(f"episode {episode_idx + 1}: collected {len(trajectory)} samples")
        if planning_error is not None:
            failed_seeds.append(world.seed)
            print(f"episode {episode_idx + 1}: skipped saving planner failure")
            continue
        if not reached_goal:
            failed_seeds.append(world.seed)
            print(f"episode {episode_idx + 1}: skipped saving unsuccessful rollout")
            continue

        output_path = save_episode(
            trajectory=trajectory,
            stage_config=stage_config,
        )
        saved_episodes += 1
        print(f"episode {episode_idx + 1}: saved LeRobot dataset: {output_path}")

    print(f"saved {saved_episodes}/{args.episodes} successful episodes")
    if failed_seeds:
        print(f"failed seeds: {failed_seeds}")
    if output_path is not None:
        print(f"saved LeRobot dataset: {output_path}")


if __name__ == "__main__":
    main()
