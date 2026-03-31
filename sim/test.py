from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.world import World


def main() -> None:
    world = World(seed=1)
    observation = world.get_observation()
    print(observation)

    while True:
        world.move_car(throttle=10.0, steering=0.0)
        observation = world.step()


if __name__ == "__main__":
    main()
