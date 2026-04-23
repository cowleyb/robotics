import xml.etree.ElementTree as ET
import math

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CAR_PATH = str(BASE_DIR / "assets" / "simplecar.urdf")

tree = ET.parse(CAR_PATH)
root = tree.getroot()
print(root.tag)  # Should print 'robot' if the URDF is correct


def get_joint_xyz(joint_name):
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is None:
        raise ValueError(f"Joint not found: {joint_name}")

    origin = joint.find("origin")
    xyz = origin.attrib.get("xyz", "0 0 0").split()
    return list(map(float, xyz))


def get_wheel_radius():
    # assumes all wheels use same cylinder geometry
    wheel_link = root.find(".//link[@name='left_front_wheel']")
    if wheel_link is None:
        raise ValueError("Wheel link not found: left_front_wheel")
    cylinder = wheel_link.find(".//cylinder")
    r = float(cylinder.get("radius"))
    return r


def get_steering_limit(joint_name):
    joint = root.find(f".//joint[@name='{joint_name}']")
    limit = joint.find("limit")

    lower = float(limit.attrib["lower"])
    upper = float(limit.attrib["upper"])

    return lower, upper


front_left_pos = get_joint_xyz("base_to_left_hinge")
front_right_pos = get_joint_xyz("base_to_right_hinge")
back_left_pos = get_joint_xyz("base_to_left_back_wheel")
back_right_pos = get_joint_xyz("base_to_right_back_wheel")

print("Front Left Position:", front_left_pos)
print("Front Right Position:", front_right_pos)
print("Back Left Position:", back_left_pos)
print("Back Right Position:", back_right_pos)

wheel_radius = get_wheel_radius()
print("Wheel Radius:", wheel_radius)

# Distance between left and right wheels (track width)
track_width = abs(front_left_pos[1] - front_right_pos[1])
print("Track Width:", track_width)

# Distance between front and back wheels (wheelbase)
wheelbase = abs(front_left_pos[0] - back_left_pos[0])
print("Wheelbase:", wheelbase)

front_steering_limit = get_steering_limit("base_to_left_hinge")
print("Front Steering Limit (radians):", front_steering_limit)
