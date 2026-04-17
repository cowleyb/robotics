import argparse
from pathlib import Path
import secrets

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.utils import DEFAULT_FEATURES

from genesis.vis.keybindings import Key, KeyAction, Keybind
from sim.policy_observation import build_lerobot_observation
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


def unnormalize_action(action: np.ndarray) -> tuple[float, float]:
    throttle = float(np.clip(action[0], -1.0, 1.0))
    steering = float(np.clip(action[1], -1.0, 1.0))
    if throttle >= 0.0:
        throttle *= DRIVE_LIMITS.max_forward_wheel_speed
    else:
        throttle *= DRIVE_LIMITS.max_reverse_wheel_speed
    steering *= MAX_STEERING_ANGLE
    return throttle, steering


def maybe_perturb_action(
    action: np.ndarray,
    rng: np.random.Generator,
    perturb_prob: float,
    throttle_std: float,
    steering_std: float,
) -> tuple[np.ndarray, bool]:
    executed_action = action.copy()
    if perturb_prob <= 0.0 or rng.random() >= perturb_prob:
        return executed_action, False

    executed_action[0] = float(
        np.clip(executed_action[0] + rng.normal(0.0, throttle_std), -1.0, 1.0)
    )
    executed_action[1] = float(
        np.clip(executed_action[1] + rng.normal(0.0, steering_std), -1.0, 1.0)
    )
    return executed_action, True


def goal_distance(observation: dict[str, np.ndarray]) -> float:
    return float(
        np.linalg.norm(observation["goal_position"][:2] - observation["car_position"][:2])
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
    expected_features = {**features, **DEFAULT_FEATURES}
    metadata_path = stage_config.dataset_root / "meta" / "info.json"
    if not metadata_path.exists():
        return LeRobotDataset.create(
            repo_id=stage_config.repo_id,
            fps=100,
            features=features,
            root=stage_config.dataset_root,
            use_videos=False,
        )

    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = stage_config.repo_id
    meta._requested_root = stage_config.dataset_root
    meta.root = stage_config.dataset_root
    meta.revision = CODEBASE_VERSION
    meta._pq_writer = None
    meta.latest_episode = None
    meta._metadata_buffer = []
    meta._metadata_buffer_size = 10
    meta._finalized = False
    try:
        meta._load_metadata()
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise FileNotFoundError(
            f"Existing dataset at {stage_config.dataset_root} is incomplete. "
            "Delete the partial dataset and record again, or recreate it from scratch."
        ) from exc
    if meta.features != expected_features:
        raise ValueError(
            f"Existing dataset schema at {stage_config.dataset_root} does not match "
            "the current GPS+vision recorder schema. Record into a fresh dataset "
            "directory or remove the old dataset before collecting again."
        )

    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.repo_id = stage_config.repo_id
    dataset._requested_root = stage_config.dataset_root
    dataset.root = meta.root
    dataset.revision = meta.revision
    dataset.tolerance_s = 1e-4
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.episodes = None
    dataset._video_backend = "pyav"
    dataset._batch_encoding_size = 1
    dataset._vcodec = "libsvtav1"
    dataset._encoder_threads = None
    dataset.reader = None
    dataset.meta = meta
    dataset.writer = DatasetWriter(
        meta=meta,
        root=meta.root,
        vcodec=dataset._vcodec,
        encoder_threads=dataset._encoder_threads,
        batch_encoding_size=dataset._batch_encoding_size,
        initial_frames=meta.total_frames,
    )
    dataset._is_finalized = False
    return dataset


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
    parser.add_argument("--action_noise_prob", type=float, default=0.05)
    parser.add_argument("--action_noise_throttle_std", type=float, default=0.05)
    parser.add_argument("--action_noise_steering_std", type=float, default=0.15)
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument("--min_failure_progress", type=float, default=0.75)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if not 0.0 <= args.action_noise_prob <= 1.0:
        raise ValueError("--action_noise_prob must be in [0, 1]")
    if args.action_noise_throttle_std < 0.0:
        raise ValueError("--action_noise_throttle_std must be non-negative")
    if args.action_noise_steering_std < 0.0:
        raise ValueError("--action_noise_steering_std must be non-negative")
    if not 0.0 <= args.min_failure_progress <= 1.0:
        raise ValueError("--min_failure_progress must be in [0, 1]")

    stage_config = get_stage_config(args.stage)
    instruction = args.instruction or stage_config.instruction
    base_seed = int(args.seed) if args.seed is not None else None
    output_path = None
    saved_episodes = 0
    failed_seeds = []
    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    initial_seed = base_seed if base_seed is not None else int(secrets.randbelow(2**31 - 1))
    world = World(
        seed=initial_seed,
        instruction=instruction,
        show_viewer=args.show_viewer,
        obstacle_count=stage_config.obstacle_count,
        gps_sensor_config=stage_config.gps_sensor_config,
    )
    try:
        for episode_idx in range(args.episodes):
            if base_seed is not None:
                seed = base_seed + episode_idx
            elif episode_idx == 0:
                seed = initial_seed
            else:
                seed = int(secrets.randbelow(2**31 - 1))

            if episode_idx == 0:
                observation = world.get_observation()
            else:
                observation = world.reset(seed=seed)
            print(f"episode {episode_idx + 1}/{args.episodes} seed: {world.seed}")
            episode_noise_rng = np.random.default_rng(world.seed)

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
            perturbed_steps = 0
            planning_error: RuntimeError | None = None
            initial_goal_distance = goal_distance(observation)
            while True:
                if args.manual:
                    throttle = (
                        DRIVE_LIMITS.max_forward_wheel_speed * control_state["forward"]
                        - DRIVE_LIMITS.max_reverse_wheel_speed
                        * control_state["reverse"]
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
                normalized_action = normalize_action(
                    throttle=throttle, steering=steering
                )
                teacher_action = np.array(
                    [normalized_action["throttle"], normalized_action["steering"]],
                    dtype=np.float32,
                )
                executed_action = teacher_action.copy()
                if not args.manual:
                    executed_action, was_perturbed = maybe_perturb_action(
                        action=executed_action,
                        rng=episode_noise_rng,
                        perturb_prob=args.action_noise_prob,
                        throttle_std=args.action_noise_throttle_std,
                        steering_std=args.action_noise_steering_std,
                    )
                    perturbed_steps += int(was_perturbed)
                world.last_action = executed_action
                executed_throttle, executed_steering = unnormalize_action(executed_action)
                world.move_car(
                    throttle=executed_throttle,
                    steering=executed_steering,
                )
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
                    print(
                        f"episode {episode_idx + 1}: goal reached at step {step_count}"
                    )
                    break
                if hit_obstacle:
                    print(
                        f"episode {episode_idx + 1}: hit obstacle at step {step_count}"
                    )
                    break
                if timed_out:
                    print(f"episode {episode_idx + 1}: timeout at step {step_count}")
                    break

            print(f"episode {episode_idx + 1}: collected {len(trajectory)} samples")
            if planning_error is not None:
                failed_seeds.append(world.seed)
                print(f"episode {episode_idx + 1}: skipped saving planner failure")
                continue

            final_goal_distance = goal_distance(observation)
            progress = max(
                0.0,
                min(
                    1.0,
                    1.0 - final_goal_distance / max(initial_goal_distance, 1e-6),
                ),
            )
            print(
                f"episode {episode_idx + 1}: perturbed steps={perturbed_steps}, "
                f"progress={progress:.3f}"
            )

            if not reached_goal and not (
                args.save_failures and progress >= args.min_failure_progress
            ):
                failed_seeds.append(world.seed)
                print(f"episode {episode_idx + 1}: skipped saving unsuccessful rollout")
                continue
            if not reached_goal:
                print(
                    f"episode {episode_idx + 1}: saving failure rollout with "
                    f"progress {progress:.3f}"
                )

            output_path = save_episode(
                trajectory=trajectory,
                stage_config=stage_config,
            )
            saved_episodes += 1
            print(f"episode {episode_idx + 1}: saved LeRobot dataset: {output_path}")
    finally:
        world.close()

    print(f"saved {saved_episodes}/{args.episodes} successful episodes")
    if failed_seeds:
        print(f"failed seeds: {failed_seeds}")
    if output_path is not None:
        print(f"saved LeRobot dataset: {output_path}")


if __name__ == "__main__":
    main()
