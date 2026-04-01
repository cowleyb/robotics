import argparse
import os
from pathlib import Path
import secrets
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
        "--camera",
        action="store_true",
        help="Enable the observation camera. Defaults to off in manual mode and on in teacher mode.",
    )
    args = parser.parse_args()

    seed = int(secrets.randbelow(2**31 - 1))
    enable_camera = args.camera or not args.manual
    world = World(seed=seed, enable_camera=enable_camera)

    observation = world.get_observation()
    print(observation)

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
        world.move_car(throttle=throttle, steering=steering)
        next_observation = world.step()
        normalized_action = normalize_action(throttle=throttle, steering=steering)
        step_count += 1
        reached_goal = world.goal_reached()
        hit_obstacle = world.hit_obstacle()
        timed_out = step_count >= 5000
        done = reached_goal or hit_obstacle or timed_out
        trajectory.append(
            {
                "observation": observation,
                "action": normalized_action,
                "next_observation": next_observation,
                "done": done,
            }
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
    print(f"seed: {world.seed}")


if __name__ == "__main__":
    main()
