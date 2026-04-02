from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def min_turning_radius(wheelbase: float, max_steering_angle: float) -> float:
    """minimum turning radius  using bicycle model"""
    denom = math.tan(float(max_steering_angle))
    if abs(denom) < 1e-9:
        return float("inf")
    return float(wheelbase) / abs(denom)


@dataclass(frozen=True)
class CarGeometry:
    max_steering_angle: float
    wheelbase: float
    min_turning_radius: float
    front_track: float
    rear_track: float
    base_size: tuple[float, float, float]
    wheel_radius: float
    wheel_width: float


def _parse_xyz(value: str) -> np.ndarray:
    return np.asarray([float(x) for x in value.split()], dtype=np.float32)


def _load_urdf_root(urdf_path: Path) -> ET.Element:
    return ET.fromstring(urdf_path.read_text(encoding="utf-8"))


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"./joint[@name='{name}']")
    if joint is None:
        raise ValueError(f"URDF joint not found: {name}")
    return joint


def _joint_origin_xyz(root: ET.Element, name: str) -> np.ndarray:
    joint = _find_joint(root, name)
    origin = joint.find("origin")
    if origin is None or "xyz" not in origin.attrib:
        raise ValueError(f"URDF joint missing origin xyz: {name}")
    return _parse_xyz(origin.attrib["xyz"])


def _joint_limit_range(root: ET.Element, name: str) -> tuple[float, float]:
    joint = _find_joint(root, name)
    limit = joint.find("limit")
    if limit is None:
        raise ValueError(f"URDF joint missing limit: {name}")
    return float(limit.attrib["lower"]), float(limit.attrib["upper"])


def _link_box_size(root: ET.Element, link_name: str) -> np.ndarray:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"URDF link not found: {link_name}")
    box = link.find("./visual/geometry/box")
    if box is None or "size" not in box.attrib:
        raise ValueError(f"URDF link missing visual box size: {link_name}")
    return _parse_xyz(box.attrib["size"])


def _wheel_dimensions(root: ET.Element, wheel_link_name: str) -> tuple[float, float]:
    link = root.find(f"./link[@name='{wheel_link_name}']")
    if link is None:
        raise ValueError(f"URDF link not found: {wheel_link_name}")
    cyl = link.find("./collision/geometry/cylinder") or link.find(
        "./visual/geometry/cylinder"
    )
    if cyl is None or "radius" not in cyl.attrib:
        raise ValueError(f"URDF wheel missing cylinder radius: {wheel_link_name}")
    if "length" not in cyl.attrib:
        raise ValueError(f"URDF wheel missing cylinder length: {wheel_link_name}")
    return float(cyl.attrib["radius"]), float(cyl.attrib["length"])


def load_simplecar_geometry(urdf_path: Path) -> CarGeometry:
    root = _load_urdf_root(urdf_path)

    lower, upper = _joint_limit_range(root, "base_to_left_hinge")
    max_steer = max(abs(lower), abs(upper))

    left_front = _joint_origin_xyz(root, "base_to_left_hinge")
    right_front = _joint_origin_xyz(root, "base_to_right_hinge")
    left_rear = _joint_origin_xyz(root, "base_to_left_back_wheel")
    right_rear = _joint_origin_xyz(root, "base_to_right_back_wheel")

    front_x = 0.5 * (left_front[0] + right_front[0])
    rear_x = 0.5 * (left_rear[0] + right_rear[0])
    wheelbase = float(abs(front_x - rear_x))
    front_track = float(abs(left_front[1] - right_front[1]))
    rear_track = float(abs(left_rear[1] - right_rear[1]))

    base_box = _link_box_size(root, "base_link")
    wheel_r, wheel_w = _wheel_dimensions(root, "left_front_wheel")

    return CarGeometry(
        max_steering_angle=float(max_steer),
        wheelbase=float(wheelbase),
        min_turning_radius=min_turning_radius(
            wheelbase=wheelbase, max_steering_angle=max_steer
        ),
        front_track=float(front_track),
        rear_track=float(rear_track),
        base_size=(float(base_box[0]), float(base_box[1]), float(base_box[2])),
        wheel_radius=float(wheel_r),
        wheel_width=float(wheel_w),
    )
