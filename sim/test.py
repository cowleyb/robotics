import argparse
from dataclasses import dataclass
from pathlib import Path
import secrets

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.dataset_writer import DatasetWriter
from lerobot.datasets.utils import DEFAULT_FEATURES

from genesis.vis.keybindings import Key, KeyAction, Keybind
from sim.policy_observation import build_lerobot_observation
from sim.stages import RecoveryDataConfig, StageConfig, get_stage_config
from sim.world import DRIVE_LIMITS, MAX_STEERING_ANGLE, World

DEFAULT_EPISODE_MAX_STEPS = 600


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


@dataclass
class RecoveryPerturbationController:
    rng: np.random.Generator
    recovery_data_config: RecoveryDataConfig
    active_burst_steps_remaining: int = 0
    recovery_steps_remaining: int = 0
    burst_offset: np.ndarray | None = None

    def _sample_step_count(self, step_range: tuple[int, int]) -> int:
        min_steps, max_steps = step_range
        if max_steps <= 0:
            return 0
        return int(self.rng.integers(min_steps, max_steps + 1))

    def _start_burst(self) -> bool:
        self.active_burst_steps_remaining = self._sample_step_count(
            self.recovery_data_config.burst_length_range_steps
        )
        if self.active_burst_steps_remaining <= 0:
            self.burst_offset = None
            return False
        self.burst_offset = np.asarray(
            [
                self.rng.normal(0.0, self.recovery_data_config.throttle_std),
                self.rng.normal(0.0, self.recovery_data_config.steering_std),
            ],
            dtype=np.float32,
        )
        return True

    def sample_action(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        executed_action = action.copy()
        if self.active_burst_steps_remaining > 0:
            if self.burst_offset is None:
                raise RuntimeError("burst_offset must be set while a burst is active")
            executed_action = np.clip(executed_action + self.burst_offset, -1.0, 1.0)
            self.active_burst_steps_remaining -= 1
            if self.active_burst_steps_remaining == 0:
                self.recovery_steps_remaining = self._sample_step_count(
                    self.recovery_data_config.recovery_length_range_steps
                )
                self.burst_offset = None
            return executed_action.astype(np.float32), True, False

        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            return executed_action, False, True

        if (
            self.recovery_data_config.perturb_prob <= 0.0
            or self.rng.random() >= self.recovery_data_config.perturb_prob
        ):
            return executed_action, False, False

        if not self._start_burst():
            return executed_action, False, False
        return self.sample_action(action)


def resolve_step_range(
    min_steps_override: int | None,
    max_steps_override: int | None,
    default_range: tuple[int, int],
    flag_prefix: str,
) -> tuple[int, int]:
    min_steps = default_range[0] if min_steps_override is None else min_steps_override
    max_steps = default_range[1] if max_steps_override is None else max_steps_override
    if min_steps < 0 or max_steps < 0:
        raise ValueError(f"{flag_prefix} values must be non-negative")
    if min_steps > max_steps:
        raise ValueError(f"{flag_prefix} min must be <= max")
    return min_steps, max_steps


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
    parser.add_argument("--max_steps", type=int, default=DEFAULT_EPISODE_MAX_STEPS)
    parser.add_argument("--action_noise_prob", type=float, default=None)
    parser.add_argument("--action_noise_throttle_std", type=float, default=None)
    parser.add_argument("--action_noise_steering_std", type=float, default=None)
    parser.add_argument("--action_noise_burst_min_steps", type=int, default=None)
    parser.add_argument("--action_noise_burst_max_steps", type=int, default=None)
    parser.add_argument("--action_recovery_min_steps", type=int, default=None)
    parser.add_argument("--action_recovery_max_steps", type=int, default=None)
    parser.add_argument("--save_failures", action="store_true")
    parser.add_argument("--min_failure_progress", type=float, default=0.75)
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if not 0.0 <= args.min_failure_progress <= 1.0:
        raise ValueError("--min_failure_progress must be in [0, 1]")

    stage_config = get_stage_config(args.stage)
    recovery_data_defaults = stage_config.recovery_data_config
    action_noise_prob = (
        recovery_data_defaults.perturb_prob
        if args.action_noise_prob is None
        else args.action_noise_prob
    )
    action_noise_throttle_std = (
        recovery_data_defaults.throttle_std
        if args.action_noise_throttle_std is None
        else args.action_noise_throttle_std
    )
    action_noise_steering_std = (
        recovery_data_defaults.steering_std
        if args.action_noise_steering_std is None
        else args.action_noise_steering_std
    )
    action_noise_burst_steps = resolve_step_range(
        min_steps_override=args.action_noise_burst_min_steps,
        max_steps_override=args.action_noise_burst_max_steps,
        default_range=recovery_data_defaults.burst_length_range_steps,
        flag_prefix="--action_noise_burst",
    )
    action_recovery_steps = resolve_step_range(
        min_steps_override=args.action_recovery_min_steps,
        max_steps_override=args.action_recovery_max_steps,
        default_range=recovery_data_defaults.recovery_length_range_steps,
        flag_prefix="--action_recovery",
    )
    if not 0.0 <= action_noise_prob <= 1.0:
        raise ValueError("--action_noise_prob must be in [0, 1]")
    if action_noise_throttle_std < 0.0:
        raise ValueError("--action_noise_throttle_std must be non-negative")
    if action_noise_steering_std < 0.0:
        raise ValueError("--action_noise_steering_std must be non-negative")
    recovery_data_config = RecoveryDataConfig(
        perturb_prob=action_noise_prob,
        throttle_std=action_noise_throttle_std,
        steering_std=action_noise_steering_std,
        burst_length_range_steps=action_noise_burst_steps,
        recovery_length_range_steps=action_recovery_steps,
    )
    instruction = args.instruction or stage_config.instruction
    base_seed = int(args.seed) if args.seed is not None else None
    output_path = None
    saved_episodes = 0
    failed_seeds = []
    print(f"{stage_config.label} dataset: {stage_config.dataset_root}")
    print(
        "recovery perturbations: "
        f"prob={recovery_data_config.perturb_prob:.2f}, "
        f"throttle_std={recovery_data_config.throttle_std:.2f}, "
        f"steering_std={recovery_data_config.steering_std:.2f}, "
        f"burst_steps={recovery_data_config.burst_length_range_steps}, "
        f"recovery_steps={recovery_data_config.recovery_length_range_steps}"
    )
    print(f"max steps per episode: {args.max_steps}")
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
            perturbation_controller = RecoveryPerturbationController(
                rng=episode_noise_rng,
                recovery_data_config=recovery_data_config,
            )

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
            recovery_steps = 0
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
                    executed_action, was_perturbed, was_recovery = (
                        perturbation_controller.sample_action(
                            action=executed_action,
                        )
                    )
                    perturbed_steps += int(was_perturbed)
                    recovery_steps += int(was_recovery)
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
                timed_out = step_count >= args.max_steps
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
                f"recovery steps={recovery_steps}, progress={progress:.3f}"
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
