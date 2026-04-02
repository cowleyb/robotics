import argparse
from pathlib import Path
import secrets

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_REPO_ID = "local/robotics-sim-car"
LEROBOT_ROOT = PROJECT_ROOT / "data" / "lerobot" / "robotics-sim-car"

from genesis.vis.keybindings import Key, KeyAction, Keybind
from sim.world import DRIVE_LIMITS, MAX_STEERING_ANGLE, World


def set_control(control_state: dict[str, float], key: str, value: float) -> None:
    control_state[key] = value


def normalize_action(throttle: float, steering: float) -> dict[str, float]:
    if throttle >= 0.0:
        normalized_throttle = throttle / DRIVE_LIMITS.max_forward_wheel_speed
    else:
        normalized_throttle = throttle / DRIVE_LIMITS.max_reverse_wheel_speed
    return {
        "throttle": float(max(-1.0, min(1.0, normalized_throttle))),
        "steering": float(max(-1.0, min(1.0, steering / MAX_STEERING_ANGLE))),
    }


def save_episode(
    trajectory: list[dict[str, object]],
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

    if LEROBOT_ROOT.exists():
        dataset = LeRobotDataset(
            repo_id=LEROBOT_REPO_ID,
            root=LEROBOT_ROOT,
            force_cache_sync=False,
            download_videos=False,
        )
        if "observation.images.front" not in dataset.features:
            raise ValueError(
                f"Existing dataset at {LEROBOT_ROOT} was created without camera images. "
                "Remove it before recording with mandatory images."
            )
    else:
        dataset = LeRobotDataset.create(
            repo_id=LEROBOT_REPO_ID,
            fps=100,
            features=features,
            root=LEROBOT_ROOT,
            use_videos=False,
        )

    for frame in trajectory:
        dataset.add_frame(dict(frame))
    dataset.save_episode()
    dataset.finalize()
    return dataset.root


def build_lerobot_frame(
    observation: dict[str, np.ndarray],
    action: dict[str, float],
    instruction: str,
) -> dict[str, object]:
    goal_delta = np.asarray(
        observation["goal_position"][:2] - observation["car_position"][:2],
        dtype=np.float32,
    )
    observation_state = np.concatenate(
        (
            goal_delta,
            np.asarray(observation["car_linear_velocity"][:2], dtype=np.float32),
            np.asarray(observation["steering_position"], dtype=np.float32),
            np.asarray(observation["last_action"], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    frame = {
        "task": instruction,
        "observation.state": observation_state,
        "action": np.asarray(
            [action["throttle"], action["steering"]],
            dtype=np.float32,
        ),
        "observation.images.front": np.asarray(
            observation["image"],
            dtype=np.uint8,
        ),
    }
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
        "--instruction",
        default="drive to the goal without hitting obstacles",
    )
    args = parser.parse_args()

    seed = int(secrets.randbelow(2**31 - 1))
    world = World(
        seed=seed,
        instruction=args.instruction,
    )

    observation = world.get_observation()
    print(observation)
    print(f"seed: {world.seed}")

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
            throttle, steering = world.heuristic_action()
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
            print(f"goal reached at step {step_count}")
            break
        if hit_obstacle:
            print(f"hit obstacle at step {step_count}")
            break
        if timed_out:
            print(f"timeout at step {step_count}")
            break

    print(f"collected {len(trajectory)} samples")
    output_path = save_episode(
        trajectory=trajectory,
    )
    print(f"saved LeRobot dataset: {output_path}")
    print(f"seed: {world.seed}")


if __name__ == "__main__":
    main()
