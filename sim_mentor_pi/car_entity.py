import torch
import genesis as gs
import quadrants as qd

from sim_mentor_pi.car_config import CarConfig, CarGeom

MAX_DRIVE_VELOCITY_MPS = 0.2
MAX_ANGULAR_VELOCITY_RADPS = 0.5
MAX_DRIVE_ACCEL_MPS2 = 1.2

EPSILON = 1e-8


class CarEntity:
    def __init__(self, car_entity, car_config: CarConfig, dt: float):
        self._entity = car_entity
        self.car_config = car_config
        self.dt = dt

        steer_dofs = [
            self._entity.get_joint(name).dofs_idx_local
            for name in car_config.steering_joint_names
        ]
        drive_dofs = [
            self._entity.get_joint(name).dofs_idx_local
            for name in car_config.driving_joint_names
        ]

        self.steer_dofs_idx = torch.tensor(
            steer_dofs, device=gs.device, dtype=gs.tc_int
        ).reshape(-1)
        self.drive_dofs_idx = torch.tensor(
            drive_dofs, device=gs.device, dtype=gs.tc_int
        ).reshape(-1)
        self.num_steer_dofs = len(self.steer_dofs_idx)
        self.num_drive_dofs = len(self.drive_dofs_idx)
        self._steering_limit = torch.tensor(
            car_config.geom.front_steering_limit, device=gs.device, dtype=gs.tc_float
        )
        self._drive_velocity = None

    def __getattr__(self, item):
        return getattr(self._entity, item)

    def move_car(self, actions: qd.types.ndarray) -> None:
        # normalize controls
        throttle_input = torch.clamp(actions[:, 0], -1.0, 1.0)
        steering_input = torch.clamp(actions[:, 1], -1.0, 1.0)

        angular_speed = steering_input * MAX_ANGULAR_VELOCITY_RADPS
        linear_speed = throttle_input * MAX_DRIVE_VELOCITY_MPS

        # The real car receives one servo command, then the mechanical linkage
        # gives each front wheel its own Ackermann angle. Genesis has separate
        # left/right steering joints, so apply those two angles directly.
        # Target steering angles for Genesis: [left_front_angle, right_front_angle].
        steering_targets = torch.zeros(
            (actions.shape[0], self.num_steer_dofs),
            device=gs.device,
            dtype=gs.tc_float,
        )
        turning_envs = (torch.abs(linear_speed) > EPSILON) & (
            torch.abs(angular_speed) > EPSILON
        )

        turn_radius = linear_speed[turning_envs] / angular_speed[turning_envs]
        half_track = self.car_config.geom.track_width / 2
        left_steering = torch.atan(
            self.car_config.geom.wheelbase / (turn_radius - half_track)
        )
        right_steering = torch.atan(
            self.car_config.geom.wheelbase / (turn_radius + half_track)
        )
        steering_targets[turning_envs] = torch.column_stack(
            (left_steering, right_steering)
        )
        steering_targets = torch.clamp(
            steering_targets,
            self._steering_limit[0],
            self._steering_limit[1],
        )
        self._entity.control_dofs_position(
            position=steering_targets,
            dofs_idx_local=self.steer_dofs_idx,
        )

        # into left/right rear wheel linear speeds, then into wheel angular speed.
        drive_targets = torch.zeros(
            (actions.shape[0], self.num_drive_dofs),
            device=gs.device,
            dtype=gs.tc_float,
        )
        moving_envs = torch.abs(linear_speed) > EPSILON
        left_speed = linear_speed - angular_speed * self.car_config.geom.track_width / 2
        right_speed = (
            linear_speed + angular_speed * self.car_config.geom.track_width / 2
        )
        drive_targets[moving_envs] = torch.column_stack(
            (left_speed[moving_envs], right_speed[moving_envs])
        )
        drive_targets = self.smooth_drive_velocity(drive_targets)
        self._entity.control_dofs_velocity(
            velocity=self.speed_convert(drive_targets),
            dofs_idx_local=self.drive_dofs_idx,
        )

    def speed_convert(self, speed):
        return speed / self.car_config.geom.wheel_radius

    def smooth_drive_velocity(self, target_velocity):
        # Real motors, tires, and gearboxes do not jump to full speed instantly.
        # This small ramp keeps rigid sim contacts from jolting at launch.
        if (
            self._drive_velocity is None
            or self._drive_velocity.shape != target_velocity.shape
        ):
            self._drive_velocity = torch.zeros_like(target_velocity)

        max_delta = MAX_DRIVE_ACCEL_MPS2 * self.dt
        delta = torch.clamp(
            target_velocity - self._drive_velocity,
            -max_delta,
            max_delta,
        )
        self._drive_velocity = self._drive_velocity + delta
        return self._drive_velocity

    def reset_drive_velocity(self, envs_idx) -> None:
        if self._drive_velocity is not None:
            self._drive_velocity[envs_idx] = 0.0
