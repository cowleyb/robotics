from pathlib import Path
import genesis as gs
import numpy as np
import random

from sim.car_geometry import load_simplecar_geometry

CAR_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "simplecar.urdf"
DEFAULT_FORWARD_THROTTLE = 15.0
DEFAULT_REVERSE_THROTTLE = -5.0


_CAR_GEOM = load_simplecar_geometry(CAR_URDF_PATH)
MAX_STEERING_ANGLE = _CAR_GEOM.max_steering_angle
WHEELBASE = _CAR_GEOM.wheelbase
FRONT_TRACK = _CAR_GEOM.front_track
REAR_TRACK = _CAR_GEOM.rear_track


class World:
    """simple test world for genesis"""

    _gs_initialized = False

    def __init__(
        self,
        seed: int = 1,
        show_viewer: bool = True,
        enable_camera: bool = False,
        obstacle_count: int = 10,
        backend=gs.gpu,
    ) -> None:
        self.show_viewer = show_viewer
        self.enable_camera = enable_camera
        self.obstacle_count = obstacle_count
        self.backend = backend
        self._build_world(seed)

    @staticmethod
    def _check_overlap(pos1, size1, pos2, size2):
        ## AABB check overlapping
        min1 = [pos1[i] - size1[i] / 2 for i in range(3)]
        max1 = [pos1[i] + size1[i] / 2 for i in range(3)]

        min2 = [pos2[i] - size2[i] / 2 for i in range(3)]
        max2 = [pos2[i] + size2[i] / 2 for i in range(3)]

        overlap_x = min1[0] < max2[0] and max1[0] > min2[0]
        overlap_y = min1[1] < max2[1] and max1[1] > min2[1]
        overlap_z = min1[2] < max2[2] and max1[2] > min2[2]

        return overlap_x and overlap_y and overlap_z

    def _build_world(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        if not World._gs_initialized:
            gs.init(backend=self.backend)
            World._gs_initialized = True

        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(5.0, -5.0, 4.0),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=45,
            ),
            # vis_options=gs.options.VisOptions(
        )

        self.scene.add_entity(
            gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.18, 0.24, 0.18))
        )

        self.spawned_objects = []
        self.car_size = _CAR_GEOM.base_size
        self.car_pos = (
            self.rng.uniform(-3.0, 0.0),
            self.rng.uniform(-3.0, 0.0),
            _CAR_GEOM.wheel_radius + 0.02,
        )

        self.car = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(CAR_URDF_PATH),
                pos=self.car_pos,
                collision=True,
            ),
            name="car",
        )
        self.steering_dofs = [
            self.car.get_joint("base_to_left_hinge").dofs_idx_local[0],
            self.car.get_joint("base_to_right_hinge").dofs_idx_local[0],
        ]
        self.front_drive_dofs = [
            self.car.get_joint("left_hinge_to_left_front_wheel").dofs_idx_local[0],
            self.car.get_joint("right_hinge_to_right_front_wheel").dofs_idx_local[0],
        ]
        self.rear_drive_dofs = [
            self.car.get_joint("base_to_left_back_wheel").dofs_idx_local[0],
            self.car.get_joint("base_to_right_back_wheel").dofs_idx_local[0],
        ]
        self.drive_dofs = self.front_drive_dofs + self.rear_drive_dofs
        self.fixed_body_height = 0.16
        self.kinematic_xy = np.array(self.car_pos[:2], dtype=np.float32)
        self.kinematic_yaw = 0.0
        self.kinematic_speed = 0.0
        self.commanded_throttle = 0.0
        self.commanded_steering = 0.0
        self.reverse_steps_remaining = 0
        self.camera = None
        if self.enable_camera:
            self.camera = self.scene.add_camera(
                res=(128, 128),
                pos=(self.car_pos[0] - 0.6, self.car_pos[1], 0.45),
                lookat=(self.car_pos[0] + 1.0, self.car_pos[1], 0.15),
                fov=90,
                GUI=False,
            )
        self.spawned_objects.append((self.car_pos, self.car_size))

        self.goal_size = (0.5, 0.5, 0.1)
        while True:
            self.goal_pos = (
                self.rng.uniform(3.0, 0.0),
                self.rng.uniform(3.0, 0.0),
                0.05,
            )
            overlap = any(
                self._check_overlap(self.goal_pos, self.goal_size, p, s)
                for p, s in self.spawned_objects
            )
            if not overlap:
                break

        self.goal_zone = self.scene.add_entity(
            gs.morphs.Box(
                pos=self.goal_pos,
                size=self.goal_size,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
            name="goal_zone",
        )
        self.spawned_objects.append((self.goal_pos, self.goal_size))

        self.obstacles = []
        self.obstacle_size = (0.4, 0.4, 0.5)
        self.obstacle_positions = []
        for i in range(self.obstacle_count):
            while True:
                obs_pos = (
                    self.rng.uniform(-3.0, 3.0),
                    self.rng.uniform(-3.0, 3.0),
                    0.25,
                )
                overlap = any(
                    self._check_overlap(obs_pos, self.obstacle_size, p, s)
                    for p, s in self.spawned_objects
                )
                if not overlap:
                    break

            obstacle = self.scene.add_entity(
                gs.morphs.Box(
                    pos=obs_pos,
                    size=self.obstacle_size,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(color=(0.5, 0.5, 0.5)),
                name=f"obstacle_{i}",
            )
            self.spawned_objects.append((obs_pos, self.obstacle_size))
            self.obstacle_positions.append(obs_pos)
            self.obstacles.append(obstacle)

        self.scene.build()

    def _yaw_from_quat(self, quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def get_observation(self) -> dict[str, np.ndarray]:
        observation = {
            "car_position": np.asarray(self.car.get_pos(), dtype=np.float32),
            "car_quaternion": np.asarray(self.car.get_quat(), dtype=np.float32),
            "car_linear_velocity": np.asarray(self.car.get_vel(), dtype=np.float32),
            "car_angular_velocity": np.asarray(self.car.get_ang(), dtype=np.float32),
            "car_size": np.asarray(self.car_size, dtype=np.float32),
            "steering_position": np.asarray(
                self.car.get_dofs_position(self.steering_dofs), dtype=np.float32
            ),
            "wheel_velocity": np.asarray(
                self.car.get_dofs_velocity(self.drive_dofs), dtype=np.float32
            ),
            "goal_position": np.asarray(self.goal_pos, dtype=np.float32),
            "goal_size": np.asarray(self.goal_size, dtype=np.float32),
            "obstacle_positions": np.asarray(self.obstacle_positions, dtype=np.float32),
            "obstacle_size": np.asarray(self.obstacle_size, dtype=np.float32),
        }
        if self.camera is not None:
            observation["image"] = self.camera.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )[0]
        return observation

    def move_car(self, throttle: float, steering: float) -> None:
        steering = float(np.clip(steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))
        self.commanded_throttle += 0.2 * (throttle - self.commanded_throttle)
        self.commanded_steering += 0.3 * (steering - self.commanded_steering)
        steering_targets = np.array(
            [self.commanded_steering, self.commanded_steering], dtype=np.float32
        )
        drive_velocity = np.full(4, self.commanded_throttle, dtype=np.float32)

        if abs(self.commanded_steering) > 1e-4 and abs(self.commanded_throttle) > 1e-4:
            steering_sign = float(np.sign(self.commanded_steering))
            turn_radius = WHEELBASE / np.tan(abs(self.commanded_steering))

            inner_steer = np.arctan(
                WHEELBASE / max(turn_radius - FRONT_TRACK / 2.0, 1e-3)
            )
            outer_steer = np.arctan(WHEELBASE / (turn_radius + FRONT_TRACK / 2.0))

            rear_inner_radius = max(turn_radius - REAR_TRACK / 2.0, 1e-3)
            rear_outer_radius = turn_radius + REAR_TRACK / 2.0
            front_inner_radius = np.hypot(WHEELBASE, rear_inner_radius)
            front_outer_radius = np.hypot(WHEELBASE, rear_outer_radius)

            if steering_sign > 0.0:
                steering_targets = np.array(
                    [inner_steer, outer_steer], dtype=np.float32
                )
                wheel_radius_scale = np.array(
                    [
                        front_inner_radius,
                        front_outer_radius,
                        rear_inner_radius,
                        rear_outer_radius,
                    ],
                    dtype=np.float32,
                )
            else:
                steering_targets = np.array(
                    [-outer_steer, -inner_steer], dtype=np.float32
                )
                wheel_radius_scale = np.array(
                    [
                        front_outer_radius,
                        front_inner_radius,
                        rear_outer_radius,
                        rear_inner_radius,
                    ],
                    dtype=np.float32,
                )

            drive_velocity = (
                self.commanded_throttle * wheel_radius_scale / turn_radius
            ).astype(np.float32)

        self.car.control_dofs_position(
            position=steering_targets,
            dofs_idx_local=self.steering_dofs,
        )
        self.car.control_dofs_velocity(
            velocity=drive_velocity,
            dofs_idx_local=self.drive_dofs,
        )

    def goal_reached(self) -> bool:
        car_position = np.asarray(self.car.get_pos(), dtype=np.float32)
        goal_position = np.asarray(self.goal_pos, dtype=np.float32)
        return np.linalg.norm(car_position[:2] - goal_position[:2]) < 0.3

    def hit_obstacle(self) -> bool:
        car_position = np.asarray(self.car.get_pos(), dtype=np.float32)
        car_quat = np.asarray(self.car.get_quat(), dtype=np.float32)
        car_yaw = self._yaw_from_quat(car_quat)

        car_half = 0.5 * np.asarray(self.car_size[:2], dtype=np.float32)
        obs_half = 0.5 * np.asarray(self.obstacle_size[:2], dtype=np.float32)
        safety_margin = 0.05
        car_half = car_half + safety_margin
        obs_half = obs_half + safety_margin

        c = car_position[:2].astype(np.float32)
        o = np.asarray(self.obstacle_positions, dtype=np.float32)[:, :2]

        cy, sy = float(np.cos(car_yaw)), float(np.sin(car_yaw))
        car_axes = (
            np.array([cy, sy], dtype=np.float32),
            np.array([-sy, cy], dtype=np.float32),
        )
        world_axes = (
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )

        def _overlap_on_axis(
            center_delta: np.ndarray,
            axis: np.ndarray,
            a_axes: tuple[np.ndarray, np.ndarray],
            a_half: np.ndarray,
            b_axes: tuple[np.ndarray, np.ndarray],
            b_half: np.ndarray,
        ) -> bool:
            axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
            proj_center = float(abs(np.dot(center_delta, axis)))
            proj_a = float(
                abs(np.dot(a_axes[0] * a_half[0], axis))
                + abs(np.dot(a_axes[1] * a_half[1], axis))
            )
            proj_b = float(
                abs(np.dot(b_axes[0] * b_half[0], axis))
                + abs(np.dot(b_axes[1] * b_half[1], axis))
            )
            return proj_center <= (proj_a + proj_b)

        for obs_center in o:
            d = (obs_center - c).astype(np.float32)
            axes_to_test = (car_axes[0], car_axes[1], world_axes[0], world_axes[1])
            if all(
                _overlap_on_axis(
                    d,
                    axis,
                    car_axes,
                    car_half,
                    world_axes,
                    obs_half,
                )
                for axis in axes_to_test
            ):
                return True
        return False

    def heuristic_action(self) -> tuple[float, float]:
        """should act as a teacher method in the simulations to return the correct values to get to the goal"""
        """teacher is priveledged, can see all objects and goal"""
        return (0, 0)

    def step(self) -> dict[str, np.ndarray]:
        self.scene.step()
        return self.get_observation()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        next_seed = self.seed if seed is None else seed
        self._build_world(next_seed)
        return self.get_observation()
