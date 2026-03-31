import os
from pathlib import Path
import secrets
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genesis.vis.keybindings import Key, KeyAction, Keybind

from sim.world import DEFAULT_FORWARD_THROTTLE, MAX_STEERING_ANGLE, World


def set_control(control_state: dict[str, float], key: str, value: float) -> None:
    control_state[key] = value


def main() -> None:
    seed = int(secrets.randbelow(2**31 - 1))
    world = World(seed=seed, enable_camera=True)
    observation = world.get_observation()
    print(f"seed: {world.seed}")
    print(observation)
    step_count = 0
    trajectory = []
    control_state = {
        "forward": 0.0,
        "reverse": 0.0,
        "left": 0.0,
        "right": 0.0,
    }

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

    while True:
        throttle = DEFAULT_FORWARD_THROTTLE * (
            control_state["forward"] - control_state["reverse"]
        )
        steering = MAX_STEERING_ANGLE * (control_state["left"] - control_state["right"])
        world.move_car(throttle=throttle, steering=steering)
        next_observation = world.step()
        step_count += 1
        reached_goal = world.goal_reached()
        hit_obstacle = world.hit_obstacle()
        timed_out = step_count >= 5000
        done = reached_goal or hit_obstacle or timed_out
        trajectory.append(
            {
                "observation": observation,
                "action": {"throttle": throttle, "steering": steering},
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


if __name__ == "__main__":
    main()
