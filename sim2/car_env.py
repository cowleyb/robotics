import copy
import math
from pathlib import Path

import genesis as gs
import torch
from tensordict import TensorDict

from genesis.utils.geom import pos_lookat_up_to_T, quat_to_xyz

from sim2.car_entity import CarEntity
from sim2.car_geom import CarConfig, CarExtractor

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


BASE_DIR = Path(__file__).resolve().parents[1]
CAR_PATH = BASE_DIR / "assets" / "simplecar.urdf"


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
        self.rendered_env_num = min(5, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = target_cfg["num_commands"]
        self.cfg = env_cfg
        self.device = gs.device
        self.cam = None
        self.cams = []

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.target_cfg = target_cfg
        self.reward_scales = copy.deepcopy(reward_cfg["reward_scales"])

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

        self.scene.add_entity(gs.morphs.Plane())

        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=f"{BASE_DIR}/assets/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.5, 0.5))
                ),
            )
        else:
            self.target = None

        self.base_init_pos = torch.tensor(
            self.env_cfg["base_init_pos"], device=gs.device, dtype=gs.tc_float
        )
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device, dtype=gs.tc_float
        )

        if self.env_cfg["visualize_camera"]:
            # Use one camera per env. Only env 0 opens a GUI window.
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

        self.scene.build(n_envs=num_envs)

        self.cam = self.cams[0] if self.cams else None
        camera_pose_in_car_frame = pos_lookat_up_to_T(
            pos=torch.tensor(
                self.env_cfg["cam_pos"], dtype=gs.tc_float, device=gs.device
            ),
            lookat=torch.tensor(
                self.env_cfg["cam_lookat"], dtype=gs.tc_float, device=gs.device
            ),
            up=torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device),
        )
        for cam in self.cams:
            cam.attach(self.car.base_link, camera_pose_in_car_frame)
            cam.move_to_attach()

        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.base_euler = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.rel_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
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
        self.success_condition = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self.crash_condition = torch.zeros_like(self.success_condition)
        self.extras = {}

        self.reset()

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
            2.0
            * torch.norm(self.rel_pos[:, :2], dim=1).clamp_min(1e-6)
            * math.tan(half_fov_x)
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

    def _update_observation(self):
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
        self.obs_buf[:, 0] = (
            torch.cos(yaw) * self.rel_pos[:, 0] + torch.sin(yaw) * self.rel_pos[:, 1]
        )
        self.obs_buf[:, 1] = (
            -torch.sin(yaw) * self.rel_pos[:, 0] + torch.cos(yaw) * self.rel_pos[:, 1]
        )
        self.obs_buf[:, 2] = self.rel_pos[:, 2]
        self.obs_buf[:, 3] = yaw

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def get_observation(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "observation": {
                "images": {"cam": self.image_obs_buf},
                "state": self.obs_buf,
            }
        }

    def step(self, actions):
        self.actions[:] = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        self.car.move_car(self.actions)

        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos[:] = self.car.get_pos()
        self.base_quat[:] = self.car.get_quat()
        self.base_euler[:] = quat_to_xyz(self.base_quat, rpy=True, degrees=True)
        self.rel_pos[:] = self.commands - self.base_pos
        self._update_observation()

        self.success_condition = (
            torch.norm(self.rel_pos[:, :2], dim=1) < self.env_cfg["at_target_threshold"]
        )
        self.crash_condition = (
            (
                torch.abs(self.base_euler[:, 0])
                > self.env_cfg["termination_if_roll_greater_than"]
            )
            | (
                torch.abs(self.base_euler[:, 1])
                > self.env_cfg["termination_if_pitch_greater_than"]
            )
            | (
                torch.abs(self.rel_pos[:, 0])
                > self.env_cfg["termination_if_x_greater_than"]
            )
            | (
                torch.abs(self.rel_pos[:, 1])
                > self.env_cfg["termination_if_y_greater_than"]
            )
        )
        timeout_condition = self.episode_length_buf > self.max_episode_length
        self.reset_buf[:] = (
            self.success_condition | self.crash_condition | timeout_condition
        ).to(gs.tc_int)

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.extras["time_outs"] = timeout_condition.to(gs.tc_float)

        self.last_actions[:] = self.actions
        self.last_rel_pos[:] = self.rel_pos

        done_envs = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        self.reset_idx(done_envs)

        for cam in self.cams:
            cam.move_to_attach()

        if self.cam is not None:
            self.cam.render()

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.car.set_pos(
            self.base_init_pos.repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self.car.set_quat(
            self.base_init_quat.repeat(len(envs_idx), 1),
            envs_idx=envs_idx,
            zero_velocity=True,
            relative=False,
        )
        self.car.set_dofs_position(
            torch.zeros(
                (len(envs_idx), len(self.car.steer_dofs_idx)),
                device=gs.device,
                dtype=gs.tc_float,
            ),
            dofs_idx_local=self.car.steer_dofs_idx,
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        self.car.zero_all_dofs_velocity(envs_idx=envs_idx)

        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat
        self.base_euler[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)
        self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]

        if self.target is not None:
            self.target.set_pos(self.commands, zero_velocity=True)

        self._update_observation()

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.get_observations()

    def _reward_progress(self):
        return torch.norm(self.last_rel_pos[:, :2], dim=1) - torch.norm(
            self.rel_pos[:, :2], dim=1
        )

    def _reward_visible(self):
        return self.obs_buf[:, 3]

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_success(self):
        return self.success_condition.to(gs.tc_float)

    def _reward_crash(self):
        return self.crash_condition.to(gs.tc_float)
