import xml.etree.ElementTree as ET
import math

from pathlib import Path

from dataclasses import dataclass

STEERING_JOINT_NAMES = ("base_to_left_hinge", "base_to_right_hinge")
DRIVING_JOINT_NAMES = ("base_to_left_back_wheel", "base_to_right_back_wheel")


@dataclass(frozen=True)
class CarGeom:
    track_width: float
    wheelbase: float
    wheel_radius: float
    front_steering_limit: tuple[float, float]


@dataclass(frozen=True)
class CarConfig:
    geom: CarGeom
    steering_joint_names: tuple[str, ...] = STEERING_JOINT_NAMES
    driving_joint_names: tuple[str, ...] = DRIVING_JOINT_NAMES


class CarExtractor:
    def __init__(self, path: str):
        tree = ET.parse(path)
        self.root = tree.getroot()

    def _get_joint_xyz(self, joint_name) -> list[float]:
        joint = self.root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Joint not found: {joint_name}")

        origin = joint.find("origin")
        xyz = origin.attrib.get("xyz", "0 0 0").split()
        return list(map(float, xyz))

    def _get_wheel_radius(self) -> float:
        # assumes all wheels use same cylinder geometry
        wheel_link = self.root.find(".//link[@name='left_front_wheel']")
        if wheel_link is None:
            raise ValueError("Wheel link not found: left_front_wheel")
        cylinder = wheel_link.find(".//cylinder")
        r = float(cylinder.get("radius"))
        return r

    def _get_steering_limit(self, joint_name) -> tuple[float, float]:
        joint = self.root.find(f".//joint[@name='{joint_name}']")
        limit = joint.find("limit")

        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])

        return lower, upper

    def get_geom(self) -> CarGeom:
        front_left_pos = self._get_joint_xyz("base_to_left_hinge")
        front_right_pos = self._get_joint_xyz("base_to_right_hinge")
        back_left_pos = self._get_joint_xyz("base_to_left_back_wheel")
        back_right_pos = self._get_joint_xyz("base_to_right_back_wheel")

        print("Front Left Position:", front_left_pos)
        print("Front Right Position:", front_right_pos)
        print("Back Left Position:", back_left_pos)
        print("Back Right Position:", back_right_pos)

        wheel_radius = self._get_wheel_radius()
        print("Wheel Radius:", wheel_radius)

        # Distance between left and right wheels (track width)
        track_width = abs(front_left_pos[1] - front_right_pos[1])
        print("Track Width:", track_width)

        # Distance between front and back wheels (wheelbase)
        wheelbase = abs(front_left_pos[0] - back_left_pos[0])
        print("Wheelbase:", wheelbase)

        front_steering_limit = self._get_steering_limit("base_to_left_hinge")
        print("Front Steering Limit (radians):", front_steering_limit)

        return CarGeom(
            wheel_radius=wheel_radius,
            track_width=track_width,
            wheelbase=wheelbase,
            front_steering_limit=front_steering_limit,
        )
