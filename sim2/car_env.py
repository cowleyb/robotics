import genesis as gs

import copy
import torch
import math

from pathlib import Path

from genesis.utils.geom import pos_lookat_up_to_T

from sim2.car_entity import CarEntity
from sim2.car_geom import CarConfig, CarExtractor

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


BASE_DIR = Path(__file__).resolve().parents[1]
CAR_PATH = BASE_DIR / "assets" / "simplecar.urdf"

MAX_STEER_DEGREES = 35
MAX_STEER = math.radians(MAX_STEER_DEGREES)
MAX_RPM = 1000.0


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class RoomEnv:
    def __init__(
        self,
        base_seed: int,
        num_envs: int,
        env_cfg,
        obs_cfg,
        reward_cfg,
        target_cfg,
        show_viewer=False,
        manual=False,
    ):
        print(f"is gs_madrona available: {_ENABLE_MADRONA}")
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = target_cfg["num_commands"]
        self.cfg = env_cfg
        self.device = gs.device
        self.cam = None
        self.cams = []

        self.dt = 0.01  # run in 100hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.target_cfg = target_cfg

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=env_cfg["max_visualize_FPS"],
                camera_pos=(3.0, 0.0, 3.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add target
        if self.env_cfg["visualize_target"]:
            print("helooooooo")
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=f"{BASE_DIR}/assets/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
            )
        else:
            self.target = None

        # add car
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )

        if self.env_cfg["visualize_camera"]:
            # use one camera per env, only env 0 opens a GUI window.
            for env_idx in range(self.num_envs):
                self.cams.append(
                    self.scene.add_camera(
                        res=self.env_cfg["cam_resolution"],
                        pos=(
                            self.env_cfg["base_init_pos"][0]
                            + self.env_cfg["cam_pos"][0],
                            self.env_cfg["base_init_pos"][1]
                            + self.env_cfg["cam_pos"][1],
                            self.env_cfg["base_init_pos"][2]
                            + self.env_cfg["cam_pos"][2],
                        ),
                        lookat=(
                            self.env_cfg["base_init_pos"][0]
                            + self.env_cfg["cam_lookat"][0],
                            self.env_cfg["base_init_pos"][1]
                            + self.env_cfg["cam_lookat"][1],
                            self.env_cfg["base_init_pos"][2]
                            + self.env_cfg["cam_lookat"][2],
                        ),
                        fov=self.env_cfg["cam_fov"],
                        GUI=env_idx == 0,
                        env_idx=env_idx,
                    )
                )

        raw_car = self.scene.add_entity(gs.morphs.URDF(file=str(CAR_PATH)))
        car_geom = CarExtractor(str(CAR_PATH)).get_geom()
        car_config = CarConfig(geom=car_geom)
        self.car = CarEntity(car_entity=raw_car, car_config=car_config)

        # build scene
        self.scene.build(n_envs=num_envs)

        # attach cameras to the cars
        self.cam = self.cams[0] if self.cams else None
        cam_mount_T = pos_lookat_up_to_T(
            pos=torch.tensor(
                self.env_cfg["cam_pos"], dtype=gs.tc_float, device=gs.device
            ),
            lookat=torch.tensor(
                self.env_cfg["cam_lookat"], dtype=gs.tc_float, device=gs.device
            ),
            up=torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device),
        )
        for cam in self.cams:
            cam.attach(self.car.base_link, cam_mount_T)
            cam.move_to_attach()

        # initilize buffers
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.rel_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.obs_buf = torch.zeros(
            (self.num_envs, self.obs_cfg["state_dim"]),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.image_obs_buf = torch.zeros(
            (self.num_envs, *self.obs_cfg["image_shape"]),
            device=gs.device,
            dtype=torch.uint8,
        )

        envs_idx = torch.arange(self.num_envs, device=gs.device, dtype=gs.tc_int)
        self._resample_commands(envs_idx)
        self.base_pos[:] = self.car.get_pos()
        self.base_quat[:] = self.car.get_quat()
        self.rel_pos[:] = self.commands - self.base_pos
        self.obs_buf[:] = self.simulate_yolo_box()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(
            *self.target_cfg["pos_x_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 1] = gs_rand_float(
            *self.target_cfg["pos_y_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 2] = gs_rand_float(
            *self.target_cfg["pos_z_range"], (len(envs_idx),), gs.device
        )

    def _at_target(self):
        return (
            (torch.norm(self.rel_pos, dim=1) < self.env_cfg["at_target_threshold"])
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )

    def simulate_yolo_box(self) -> torch.Tensor:
        yaw = torch.atan2(
            2.0
            * (
                self.base_quat[:, 0] * self.base_quat[:, 3]
                + self.base_quat[:, 1] * self.base_quat[:, 2]
            ),
            1.0
            - 2.0
            * (
                self.base_quat[:, 2] * self.base_quat[:, 2]
                + self.base_quat[:, 3] * self.base_quat[:, 3]
            ),
        )
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        half_fov_x = math.radians(self.env_cfg["cam_fov"]) * 0.5
        half_fov_y = math.atan(
            math.tan(half_fov_x)
            * (self.obs_cfg["image_shape"][1] / self.obs_cfg["image_shape"][2])
        )
        depth = torch.clamp(
            cos_yaw * self.rel_pos[:, 0] + sin_yaw * self.rel_pos[:, 1], min=1e-6
        )

        # Project the target into a simple YOLO-like box signal for RL training.
        error_x = (
            torch.atan2(
                -sin_yaw * self.rel_pos[:, 0] + cos_yaw * self.rel_pos[:, 1],
                depth,
            )
            / half_fov_x
        )
        error_y = (
            torch.atan2(
                self.rel_pos[:, 2]
                + float(self.target_cfg["box_size"][2]) * 0.5
                - float(self.env_cfg["cam_pos"][2]),
                depth,
            )
            / half_fov_y
        )
        box_size = float(self.target_cfg["box_size"][0]) / (
            2.0 * torch.norm(self.rel_pos, dim=1).clamp_min(1e-6) * math.tan(half_fov_x)
        )
        is_visible = (
            (cos_yaw * self.rel_pos[:, 0] + sin_yaw * self.rel_pos[:, 1] > 0.0)
            & (torch.abs(error_x) <= 1.0)
            & (torch.abs(error_y) <= 1.0)
        )

        return torch.stack(
            (
                torch.where(is_visible, torch.clamp(error_x, -1.0, 1.0), 0.0),
                torch.where(is_visible, torch.clamp(error_y, -1.0, 1.0), 0.0),
                torch.where(is_visible, torch.clamp(box_size, 0.0, 1.0), 0.0),
                is_visible.to(gs.tc_float),
            ),
            dim=1,
        )

    def get_observation(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "observation": {
                "images": {"cam": self.image_obs_buf},
                "state": self.obs_buf,
            }
        }

    def step(self) -> int:
        self.actions[0] = torch.tensor(
            [6, 7.0], device=gs.device
        )  # throttle, steering_input
        self.car.move_car(self.actions)

        # update target position
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.car.get_pos()
        self.base_quat[:] = self.car.get_quat()
        self.rel_pos[:] = self.commands - self.base_pos
        self.obs_buf[:] = self.simulate_yolo_box()

        # Refresh attached camera poses after the car moves.
        for cam in self.cams:
            cam.move_to_attach()

        if self.cam is not None:
            self.cam.render()

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        return 0
