import genesis as gs

import copy
import torch
import math

from pathlib import Path

from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

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
    ):
        print(f"is gs_madrona available: {_ENABLE_MADRONA}")
        self.num_envs = num_envs
        self.rendered_env_num = min(10, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = target_cfg["num_commands"]
        self.cfg = env_cfg
        self.device = gs.device
        self.cam = None

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

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True,
            )

        # add drone
        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )
        self.car = self.scene.add_entity(gs.morphs.URDF(file=str(CAR_PATH)))

        # build scene
        self.scene.build(n_envs=num_envs)

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

    def step(self) -> int:
        # update target position
        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.car.get_pos()
        self.rel_pos = self.commands - self.base_pos

        if self.cam is not None:
            self.cam.render()

        envs_idx = self._at_target()
        self._resample_commands(envs_idx)

        return 0
