import torch
import genesis as gs
import quadrants as qd

from sim2.car_geom import CarConfig, CarGeom

MAX_DRIVE_VELOCITY = 20.0


class CarEntity:
    def __init__(self, car_entity, car_config: CarConfig):
        self._entity = car_entity

        self.car_config = car_config
        steer_dofs = [
            self._entity.get_joint(name).dofs_idx_local
            for name in car_config.steering_joint_names
        ]
        print(car_config.steering_joint_names)
        print(steer_dofs)
        drive_dofs = [
            self._entity.get_joint(name).dofs_idx_local
            for name in car_config.driving_joint_names
        ]

        self.steer_dofs_idx = torch.tensor(
            steer_dofs, device=gs.device, dtype=gs.tc_int
        )
        self.drive_dofs_idx = torch.tensor(
            drive_dofs, device=gs.device, dtype=gs.tc_int
        )

    def __getattr__(self, item):
        return getattr(self._entity, item)

    def move_car(self, actions: qd.types.ndarray()):
        throttle_input = torch.clamp(actions[:, 0], -1.0, 1.0)
        steering_input = torch.clamp(actions[:, 1], -1.0, 1.0)

        steering_limit = max(
            abs(self.car_config.geom.front_steering_limit[0]),
            abs(self.car_config.geom.front_steering_limit[1]),
        )
        steering_target = steering_input * steering_limit

        steering_targets = torch.stack((steering_target, steering_target), dim=1)
        drive_target = throttle_input * MAX_DRIVE_VELOCITY
        drive_velocity = torch.stack((drive_target, drive_target), dim=1)

        self._entity.control_dofs_position(
            position=steering_targets,
            dofs_idx_local=self.steer_dofs_idx,
        )
        self._entity.control_dofs_velocity(
            velocity=drive_velocity,
            dofs_idx_local=self.drive_dofs_idx,
        )
