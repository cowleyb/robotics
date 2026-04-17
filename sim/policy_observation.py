import numpy as np
from lerobot.configs.types import FeatureType, PolicyFeature

GPS_STALE_AGE_CLIP_S = 5.0
GPS_HORIZONTAL_ACCURACY_CLIP_M = 5.0
GPS_STATE_FEATURE_NAMES = (
    "gps_goal_delta_world_x",
    "gps_goal_delta_world_y",
    "gps_valid",
    "gps_age_s",
    "gps_horizontal_accuracy",
    "car_velocity_x",
    "car_velocity_y",
    "steering_left",
    "steering_right",
    "last_action_throttle",
    "last_action_steering",
)
GPS_STATE_SHAPE = (len(GPS_STATE_FEATURE_NAMES),)


def rotate_world_vector_to_car_frame(
    vector_xy: np.ndarray, car_quaternion: np.ndarray
) -> np.ndarray:
    w, x, y, z = car_quaternion
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray(
        [
            cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
            -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
        ],
        dtype=np.float32,
    )


def build_gps_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    gps_position = np.asarray(observation["gps_position"][:2], dtype=np.float32)
    goal_delta_world = np.asarray(
        observation["goal_position"][:2] - gps_position,
        dtype=np.float32,
    )
    gps_valid = float(np.asarray(observation["gps_valid"], dtype=np.float32).reshape(-1)[0])
    gps_age_s = float(np.asarray(observation["gps_age_s"], dtype=np.float32).reshape(-1)[0])
    gps_horizontal_accuracy = float(
        np.asarray(observation["gps_horizontal_accuracy"], dtype=np.float32).reshape(-1)[0]
    )
    car_velocity = rotate_world_vector_to_car_frame(
        np.asarray(observation["car_linear_velocity"][:2], dtype=np.float32),
        np.asarray(observation["car_quaternion"], dtype=np.float32),
    )
    state = np.concatenate(
        (
            goal_delta_world,
            np.asarray(
                [
                    np.clip(gps_valid, 0.0, 1.0),
                    np.clip(gps_age_s, 0.0, GPS_STALE_AGE_CLIP_S),
                    np.clip(
                        gps_horizontal_accuracy,
                        0.0,
                        GPS_HORIZONTAL_ACCURACY_CLIP_M,
                    ),
                ],
                dtype=np.float32,
            ),
            car_velocity,
            np.asarray(observation["steering_position"], dtype=np.float32),
            np.asarray(observation["last_action"], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    validate_gps_state_shape(
        state.shape,
        source="Built observation state",
    )
    return np.ascontiguousarray(state)


def build_lerobot_observation(
    observation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "observation.state": build_gps_state(observation),
        "observation.images.front": np.ascontiguousarray(
            observation["image"],
            dtype=np.uint8,
        ),
    }


def get_policy_input_features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 128, 128),
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=GPS_STATE_SHAPE,
        ),
    }


def validate_features(features: dict[str, object], source: str) -> None:
    if "observation.images.front" not in features:
        raise ValueError(f"{source} is missing observation.images.front.")
    if "observation.state" not in features:
        raise ValueError(f"{source} is missing observation.state.")
    validate_gps_state_shape(features["observation.state"], source=source)


def validate_gps_state_shape(actual_shape: object, source: str) -> None:
    if isinstance(actual_shape, dict):
        shape = tuple(int(dim) for dim in actual_shape["shape"])
    elif hasattr(actual_shape, "shape"):
        shape = tuple(int(dim) for dim in getattr(actual_shape, "shape"))
    else:
        shape = tuple(int(dim) for dim in actual_shape)
    if shape != GPS_STATE_SHAPE:
        raise ValueError(
            f"{source} observation.state shape {shape} does not match the current "
            f"GPS policy shape {GPS_STATE_SHAPE}. Re-record datasets and retrain "
            "checkpoints with the updated GPS observation pipeline."
        )
