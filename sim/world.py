from pathlib import Path
import genesis as gs
import numpy as np
import random

from sim.car_geometry import load_simplecar_geometry
from sim.control_limits import DEFAULT_DRIVE_LIMITS
from sim.path_planning import (
    EnvironmentBounds,
    ObstacleBox,
    TeacherPlanner,
    VehicleGeometry,
    VehicleState,
)

CAR_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "simplecar.urdf"
DRIVE_LIMITS = DEFAULT_DRIVE_LIMITS

SPAWN_RANGE = 3.0
WORLD_MARGIN = 0.25
CAR_SPAWN_HEIGHT_OFFSET = 0.02
CAMERA_OFFSET = (-0.6, 0.0, 0.45)
CAMERA_LOOKAHEAD = (1.0, 0.0, 0.15)
GOAL_SIZE = (0.5, 0.5, 0.1)
GOAL_HEIGHT = 0.05
OBSTACLE_SIZE = (0.4, 0.4, 0.5)
OBSTACLE_HEIGHT = 0.25
GOAL_REACHED_DISTANCE = 0.3
STEERING_EPSILON = 1e-4
THROTTLE_EPSILON = 1e-4


_CAR_GEOM = load_simplecar_geometry(CAR_URDF_PATH)
MAX_STEERING_ANGLE = _CAR_GEOM.max_steering_angle
WHEELBASE = _CAR_GEOM.wheelbase
MIN_TURNING_RADIUS = _CAR_GEOM.min_turning_radius
FRONT_TRACK = _CAR_GEOM.front_track
REAR_TRACK = _CAR_GEOM.rear_track
REAR_AXLE_OFFSET = abs(_CAR_GEOM.rear_axle_x)
PLANNER_COLLISION_CENTER_OFFSET = 0.5 * WHEELBASE


class World:
    """simple test world for genesis"""

    def __init__(
        self,
        seed: int = 1,
        instruction: str = "drive to the goal without hitting obstacles",
        show_viewer: bool = True,
        obstacle_count: int = 10,
        backend=gs.gpu,
    ) -> None:
        self.instruction = instruction
        self.show_viewer = show_viewer
        self.obstacle_count = obstacle_count
        self.backend = backend
        self._build_world(seed)

    @staticmethod
    def _check_overlap(
        pos1: tuple[float, float, float],
        size1: tuple[float, float, float],
        pos2: tuple[float, float, float],
        size2: tuple[float, float, float],
    ) -> bool:
        return all(
            abs(float(center1) - float(center2))
            < 0.5 * (float(extent1) + float(extent2))
            for center1, extent1, center2, extent2 in zip(pos1, size1, pos2, size2)
        )

    def _sample_spawn_position(
        self,
        size: tuple[float, float, float],
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z: float,
        spawned_objects: list[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ]
        | None = None,
    ) -> tuple[float, float, float]:
        existing_objects = (
            self.spawned_objects if spawned_objects is None else spawned_objects
        )
        while True:
            position = (
                self.rng.uniform(*x_range),
                self.rng.uniform(*y_range),
                z,
            )
            overlap = any(
                self._check_overlap(position, size, existing_pos, existing_size)
                for existing_pos, existing_size in existing_objects
            )
            if not overlap:
                return position

    def _make_teacher_planner(self) -> TeacherPlanner:
        return TeacherPlanner(
            geometry=VehicleGeometry(
                wheelbase=WHEELBASE,
                max_steering_angle=MAX_STEERING_ANGLE,
                length=float(self.spawn_car_size[0]),
                width=float(self.spawn_car_size[1]),
                wheel_radius=float(_CAR_GEOM.wheel_radius),
                collision_center_offset=float(PLANNER_COLLISION_CENTER_OFFSET),
                rear_axle_offset=float(REAR_AXLE_OFFSET),
            ),
            bounds=EnvironmentBounds(
                min_x=-(SPAWN_RANGE + WORLD_MARGIN),
                max_x=SPAWN_RANGE + WORLD_MARGIN,
                min_y=-(SPAWN_RANGE + WORLD_MARGIN),
                max_y=SPAWN_RANGE + WORLD_MARGIN,
            ),
        )

    def _sample_layout(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        list[tuple[float, float, float]],
        list[ObstacleBox],
    ]:
        spawned_objects: list[
            tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = []
        car_pos = (
            self.rng.uniform(-SPAWN_RANGE, 0.0),
            self.rng.uniform(-SPAWN_RANGE, 0.0),
            _CAR_GEOM.wheel_radius + CAR_SPAWN_HEIGHT_OFFSET,
        )
        spawned_objects.append((car_pos, self.spawn_car_size))

        goal_pos = self._sample_spawn_position(
            size=self.goal_size,
            x_range=(0.0, SPAWN_RANGE),
            y_range=(0.0, SPAWN_RANGE),
            z=GOAL_HEIGHT,
            spawned_objects=spawned_objects,
        )
        spawned_objects.append((goal_pos, self.goal_size))

        obstacle_positions = []
        for _ in range(self.obstacle_count):
            obstacle_pos = self._sample_spawn_position(
                size=self.obstacle_size,
                x_range=(-SPAWN_RANGE, SPAWN_RANGE),
                y_range=(-SPAWN_RANGE, SPAWN_RANGE),
                z=OBSTACLE_HEIGHT,
                spawned_objects=spawned_objects,
            )
            spawned_objects.append((obstacle_pos, self.obstacle_size))
            obstacle_positions.append(obstacle_pos)

        teacher_obstacles = [
            ObstacleBox(
                center_x=float(position[0]),
                center_y=float(position[1]),
                size_x=float(self.obstacle_size[0]),
                size_y=float(self.obstacle_size[1]),
            )
            for position in obstacle_positions
        ]
        return (
            car_pos,
            goal_pos,
            obstacle_positions,
            teacher_obstacles,
        )

    def _build_scene_from_layout(
        self,
        car_pos: tuple[float, float, float],
        goal_pos: tuple[float, float, float],
        obstacle_positions: list[tuple[float, float, float]],
        teacher_obstacles: list[ObstacleBox],
        teacher_planner: TeacherPlanner,
    ) -> None:
        self.car_pos = car_pos
        self.goal_pos = goal_pos
        self.obstacle_positions = list(obstacle_positions)
        self.teacher_obstacles = list(teacher_obstacles)
        self.teacher_planner = teacher_planner
        self.spawned_objects = [(self.car_pos, self.spawn_car_size), (self.goal_pos, self.goal_size)]

        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.0, 0.0, 10.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=60,
            ),
            # vis_options=gs.options.VisOptions(
        )

        self.scene.add_entity(
            gs.morphs.Plane(), surface=gs.surfaces.Default(color=(0.18, 0.24, 0.18))
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
        self.kinematic_xy = np.array(self.car_pos[:2], dtype=np.float32)
        self.kinematic_yaw = 0.0
        self.kinematic_speed = 0.0
        self.commanded_throttle = 0.0
        self.commanded_steering = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.camera = self.scene.add_camera(
            res=(128, 128),
            pos=(
                self.car_pos[0] + CAMERA_OFFSET[0],
                self.car_pos[1] + CAMERA_OFFSET[1],
                CAMERA_OFFSET[2],
            ),
            lookat=(
                self.car_pos[0] + CAMERA_LOOKAHEAD[0],
                self.car_pos[1] + CAMERA_LOOKAHEAD[1],
                CAMERA_LOOKAHEAD[2],
            ),
            fov=90,
            GUI=False,
        )

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

        self.obstacles = []
        for i, obs_pos in enumerate(self.obstacle_positions):
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
            self.obstacles.append(obstacle)

        self.scene.build()

    def _build_world(self, seed: int) -> None:
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        if gs.backend is None:
            gs.init(backend=self.backend)
        elif self.backend != gs.gpu and gs.backend != self.backend:
            raise RuntimeError(
                f"Genesis already initialized with backend {gs.backend}, "
                f"cannot create World with backend {self.backend}."
            )

        self.car_size = _CAR_GEOM.base_size
        self.spawn_car_size = (
            max(
                float(self.car_size[0]), float(WHEELBASE + 2.0 * _CAR_GEOM.wheel_radius)
            ),
            max(
                float(self.car_size[1]),
                float(max(FRONT_TRACK, REAR_TRACK) + _CAR_GEOM.wheel_width),
            ),
            float(self.car_size[2]),
        )
        self.goal_size = GOAL_SIZE
        self.obstacle_size = OBSTACLE_SIZE
        car_pos, goal_pos, obstacle_positions, teacher_obstacles = self._sample_layout()
        self._build_scene_from_layout(
            car_pos=car_pos,
            goal_pos=goal_pos,
            obstacle_positions=obstacle_positions,
            teacher_obstacles=teacher_obstacles,
            teacher_planner=self._make_teacher_planner(),
        )

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
            "last_action": self.last_action.copy(),
            "image": self.camera.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )[0],
        }
        return observation

    def move_car(self, throttle: float, steering: float) -> None:
        steering = float(np.clip(steering, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE))
        self.commanded_throttle += 0.2 * (throttle - self.commanded_throttle)
        self.commanded_steering = steering
        steering_targets = np.array(
            [self.commanded_steering, self.commanded_steering], dtype=np.float32
        )
        drive_velocity = np.full(4, self.commanded_throttle, dtype=np.float32)

        if (
            abs(self.commanded_steering) > STEERING_EPSILON
            and abs(self.commanded_throttle) > THROTTLE_EPSILON
        ):
            steering_sign = float(np.sign(self.commanded_steering))
            turn_radius = WHEELBASE / np.tan(abs(self.commanded_steering))
            front_inner_radius = turn_radius - FRONT_TRACK / 2.0
            rear_inner_radius = turn_radius - REAR_TRACK / 2.0
            if front_inner_radius > 0.0 and rear_inner_radius > 0.0:
                inner_steer = np.arctan(WHEELBASE / front_inner_radius)
                outer_steer = np.arctan(WHEELBASE / (turn_radius + FRONT_TRACK / 2.0))

                rear_outer_radius = turn_radius + REAR_TRACK / 2.0
                front_inner_wheel_radius = np.hypot(WHEELBASE, rear_inner_radius)
                front_outer_wheel_radius = np.hypot(WHEELBASE, rear_outer_radius)

                if steering_sign > 0.0:
                    steering_targets = np.array(
                        [inner_steer, outer_steer], dtype=np.float32
                    )
                    wheel_radius_scale = np.array(
                        [
                            front_inner_wheel_radius,
                            front_outer_wheel_radius,
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
                            front_outer_wheel_radius,
                            front_inner_wheel_radius,
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
        return np.linalg.norm(car_position[:2] - goal_position[:2]) < GOAL_REACHED_DISTANCE

    def hit_obstacle(self) -> bool:
        for obstacle in self.obstacles:
            contact_info = self.car.get_contacts(with_entity=obstacle)
            if int(contact_info["geom_a"].numel()) > 0:
                return True
        return False

    def heuristic_action(self) -> tuple[float, float]:
        """should act as a teacher method in the simulations to return the correct values to get to the goal"""
        """teacher is priveledged, can see all objects and goal"""
        car_position = np.asarray(self.car.get_pos(), dtype=np.float32)
        car_quat = np.asarray(self.car.get_quat(), dtype=np.float32)
        car_velocity = np.asarray(self.car.get_vel(), dtype=np.float32)
        car_yaw = self._yaw_from_quat(car_quat)

        heading = np.array([np.cos(car_yaw), np.sin(car_yaw)], dtype=np.float32)
        signed_speed = float(np.linalg.norm(car_velocity[:2]))
        if float(np.dot(car_velocity[:2], heading)) < 0.0:
            signed_speed *= -1.0

        throttle, steering = self.teacher_planner.compute_action(
            state=VehicleState(
                x=float(car_position[0]),
                y=float(car_position[1]),
                yaw=car_yaw,
                speed=signed_speed,
            ),
            goal_xy=np.asarray(self.goal_pos[:2], dtype=np.float32),
            obstacles=self.teacher_obstacles,
        )
        return (float(throttle), float(steering))

    def step(self) -> dict[str, np.ndarray]:
        self.scene.step()
        return self.get_observation()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        next_seed = self.seed if seed is None else seed
        self._build_world(next_seed)
        return self.get_observation()
