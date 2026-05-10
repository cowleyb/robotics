from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sim_mentor_pi.car_config import CarExtractor


def main() -> None:
    car_path = REPO_ROOT / "assets" / "mentorpi_car.xacro"

    print(f"Inspecting model: {car_path}")
    car_geom = CarExtractor(str(car_path)).get_geom()

    print("\nController geometry")
    print(f"wheel radius: {car_geom.wheel_radius:.4f} m")
    print(f"track width: {car_geom.track_width:.4f} m")
    print(f"wheelbase: {car_geom.wheelbase:.4f} m")
    print(
        "front steering limit: "
        f"{car_geom.front_steering_limit[0]:.4f} to "
        f"{car_geom.front_steering_limit[1]:.4f} rad"
    )


if __name__ == "__main__":
    main()
