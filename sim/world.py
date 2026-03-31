from pathlib import Path
import genesis as gs
import numpy as np
import random

CAR_URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "simplecar.urdf"


class World:
    """simple test world for genesis"""

    _gs_initialized = False

    def __init__(
        self,
        seed: int = 1,
        show_viewer: bool = True,
        obstacle_count: int = 10,
        backend=gs.gpu,
    ) -> None:
        self.show_viewer = show_viewer
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
        self.seed = seed
        self.rng = random.Random(seed)
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
        self.car_size = (0.4, 0.2, 0.1)
        self.car_pos = (
            self.rng.uniform(-3.0, 0.0),
            self.rng.uniform(-3.0, 0.0),
            0.12,
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
        self.drive_dofs = [
            self.car.get_joint("left_hinge_to_left_front_wheel").dofs_idx_local[0],
            self.car.get_joint("right_hinge_to_right_front_wheel").dofs_idx_local[0],
            self.car.get_joint("base_to_left_back_wheel").dofs_idx_local[0],
            self.car.get_joint("base_to_right_back_wheel").dofs_idx_local[0],
        ]
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

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
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

    def move_car(self, throttle: float, steering: float) -> None:
        self.car.control_dofs_position(
            position=np.array([steering, steering], dtype=np.float32),
            dofs_idx_local=self.steering_dofs,
        )
        self.car.control_dofs_velocity(
            velocity=np.array(
                [throttle, throttle, throttle, throttle], dtype=np.float32
            ),
            dofs_idx_local=self.drive_dofs,
        )

    def step(self) -> dict[str, np.ndarray]:
        self.scene.step()
        return self.get_observation()

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        next_seed = self.seed if seed is None else seed
        self._build_world(next_seed)
        return self.get_observation()
