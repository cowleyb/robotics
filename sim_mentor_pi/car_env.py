import copy
import math
from pathlib import Path

import genesis as gs
import genesis.vis.keybindings as kb
import torch
from tensordict import TensorDict

from genesis.utils.geom import pos_lookat_up_to_T, quat_to_xyz

from sim_mentor_pi.car_config import CarConfig, CarExtractor
from sim_mentor_pi.car_entity import CarEntity

try:
    import gs_madrona

    _ENABLE_MADRONA = True
except ImportError:
    _ENABLE_MADRONA = False


BASE_DIR = Path(__file__).resolve().parents[1]
CAR_PATH = BASE_DIR / "assets" / "mentorpi_car.xacro"
CAMERA_LINK_NAME = "depth_cam"


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class TestEnv:
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
        # TODO over here
        self.rendered_env_num = min(5, self.num_envs)
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = target_cfg["num_commands"]
        self.cfg = env_cfg
        self.device = gs.device
        self.cam = None
        self.cams = []
        self.manual = manual
        self.manual_is_running = True
        self.manual_action = torch.zeros(
            (num_envs, 2), device=gs.device, dtype=gs.tc_float
        )
        self.manual_forward = False
        self.manual_reverse = False
        self.manual_left = False
        self.manual_right = False

        self.dt = 0.01
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.obs_scales = obs_cfg["obs_scales"]
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
                enable_self_collision=False,
                noslip_iterations=4,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=1.2),
        )

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
            # Cameras are only for visual debugging here, so only bind them to
            # the envs Genesis is already rendering.
            for env_idx in range(self.rendered_env_num):
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

        raw_car = self.scene.add_entity(
            gs.morphs.URDF(
                file=str(CAR_PATH),
                links_to_keep=(CAMERA_LINK_NAME,),
                default_armature=0.001,
            ),
            material=gs.materials.Rigid(friction=2.2),
        )
        car_geom = CarExtractor(str(CAR_PATH)).get_geom()
        car_config = CarConfig(
            geom=car_geom,
            steering_joint_names=("base_to_left_hinge", "base_to_right_hinge"),
            driving_joint_names=("base_to_left_back_wheel", "base_to_right_back_wheel"),
        )
        self.car = CarEntity(car_entity=raw_car, car_config=car_config, dt=self.dt)

        self.scene.build(n_envs=num_envs)
        if self.manual and show_viewer:
            self.register_manual_keybinds()

        self.cam = self.cams[0] if self.cams else None
        camera_pose_in_camera_frame = pos_lookat_up_to_T(
            pos=torch.tensor((0.0, 0.0, 0.0), dtype=gs.tc_float, device=gs.device),
            lookat=torch.tensor((1.0, 0.0, 0.0), dtype=gs.tc_float, device=gs.device),
            up=torch.tensor((0.0, 0.0, 1.0), dtype=gs.tc_float, device=gs.device),
        )
        camera_link = self.car.get_link(CAMERA_LINK_NAME)
        for cam in self.cams:
            cam.attach(camera_link, camera_pose_in_camera_frame)
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
        self.base_lin_vel = torch.zeros_like(self.base_pos)
        self.base_ang_vel = torch.zeros_like(self.base_pos)
        self.rel_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.last_rel_pos = torch.zeros_like(self.rel_pos)
        self.base_rel_pos = torch.zeros_like(self.rel_pos)
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
        self.timeout_condition = torch.zeros_like(self.success_condition)
        self.extras = {}

        self.reset()

    def set_manual_forward(self, is_pressed: bool) -> None:
        self.manual_forward = is_pressed
        self.update_manual_action()

    def set_manual_reverse(self, is_pressed: bool) -> None:
        self.manual_reverse = is_pressed
        self.update_manual_action()

    def set_manual_left(self, is_pressed: bool) -> None:
        self.manual_left = is_pressed
        self.update_manual_action()

    def set_manual_right(self, is_pressed: bool) -> None:
        self.manual_right = is_pressed
        self.update_manual_action()

    def update_manual_action(self) -> None:
        self.manual_action[:, 0] = float(self.manual_forward) - float(
            self.manual_reverse
        )
        self.manual_action[:, 1] = float(self.manual_left) - float(self.manual_right)

    def stop_manual(self) -> None:
        self.manual_is_running = False

    def register_manual_keybinds(self) -> None:
        self.scene.viewer.register_keybinds(
            kb.Keybind(
                "forward",
                kb.Key.UP,
                kb.KeyAction.PRESS,
                callback=lambda: self.set_manual_forward(True),
            ),
            kb.Keybind(
                "stop forward",
                kb.Key.UP,
                kb.KeyAction.RELEASE,
                callback=lambda: self.set_manual_forward(False),
            ),
            kb.Keybind(
                "reverse",
                kb.Key.DOWN,
                kb.KeyAction.PRESS,
                callback=lambda: self.set_manual_reverse(True),
            ),
            kb.Keybind(
                "stop reverse",
                kb.Key.DOWN,
                kb.KeyAction.RELEASE,
                callback=lambda: self.set_manual_reverse(False),
            ),
            kb.Keybind(
                "left",
                kb.Key.LEFT,
                kb.KeyAction.PRESS,
                callback=lambda: self.set_manual_left(True),
            ),
            kb.Keybind(
                "stop left",
                kb.Key.LEFT,
                kb.KeyAction.RELEASE,
                callback=lambda: self.set_manual_left(False),
            ),
            kb.Keybind(
                "right",
                kb.Key.RIGHT,
                kb.KeyAction.PRESS,
                callback=lambda: self.set_manual_right(True),
            ),
            kb.Keybind(
                "stop right",
                kb.Key.RIGHT,
                kb.KeyAction.RELEASE,
                callback=lambda: self.set_manual_right(False),
            ),
            kb.Keybind(
                "quit",
                kb.Key.ESCAPE,
                kb.KeyAction.PRESS,
                callback=self.stop_manual,
            ),
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

    def _yaw_from_base_quat(self) -> torch.Tensor:
        return torch.atan2(
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

    def _target_pos_in_base_frame(self) -> torch.Tensor:
        yaw = self._yaw_from_base_quat()
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        return torch.stack(
            (
                cos_yaw * self.rel_pos[:, 0] + sin_yaw * self.rel_pos[:, 1],
                -sin_yaw * self.rel_pos[:, 0] + cos_yaw * self.rel_pos[:, 1],
                self.rel_pos[:, 2],
            ),
            dim=1,
        )

    def simulate_yolo_box(self) -> torch.Tensor:
        self.base_rel_pos[:] = self._target_pos_in_base_frame()
        depth = torch.clamp(self.base_rel_pos[:, 0], min=1e-6)
        lateral = self.base_rel_pos[:, 1]
        half_fov_x = math.radians(self.env_cfg["cam_fov"]) * 0.5
        half_fov_y = math.atan(
            math.tan(half_fov_x)
            * (self.obs_cfg["image_shape"][1] / self.obs_cfg["image_shape"][2])
        )

        error_x = (
            torch.atan2(
                lateral,
                depth,
            )
            / half_fov_x
        )
        error_y = (
            torch.atan2(
                self.base_rel_pos[:, 2]
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
            (self.base_rel_pos[:, 0] > 0.0)
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
        self.base_rel_pos[:] = self._target_pos_in_base_frame()
        self.obs_buf = torch.cat(
            [
                torch.clip(self.base_rel_pos * self.obs_scales["rel_pos"], -1.0, 1.0),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1.0, 1.0),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1.0, 1.0),
                self.last_actions,
            ],
            dim=1,
        )

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

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
        self.base_lin_vel[:] = self.car.get_vel()
        self.base_ang_vel[:] = self.car.get_ang()
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
        self.timeout_condition = self.episode_length_buf > self.max_episode_length
        self.reset_buf[:] = (
            self.success_condition | self.crash_condition | self.timeout_condition
        ).to(gs.tc_int)

        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.extras["time_outs"] = self.timeout_condition.to(gs.tc_float)

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

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        # TODO last_pos??
        self.base_quat[envs_idx] = self.base_init_quat
        self.car.set_pos(
            self.base_pos[envs_idx],
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.car.set_quat(
            self.base_quat[envs_idx],
            zero_velocity=True,
            envs_idx=envs_idx,
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
        self.car.reset_drive_velocity(envs_idx)

        # Reset buffers
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0
        self.extras["episode"]["success_rate"] = torch.mean(
            self.success_condition[envs_idx].to(gs.tc_float)
        ).item()
        self.extras["episode"]["crash_rate"] = torch.mean(
            self.crash_condition[envs_idx].to(gs.tc_float)
        ).item()
        self.extras["episode"]["timeout_rate"] = torch.mean(
            self.timeout_condition[envs_idx].to(gs.tc_float)
        ).item()

        self._resample_commands(envs_idx)
        self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
        self.success_condition[envs_idx] = False
        self.crash_condition[envs_idx] = False
        self.timeout_condition[envs_idx] = False
        self._update_observation()

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.get_observations()

    def _reward_progress(self):
        return torch.norm(self.last_rel_pos[:, :2], dim=1) - torch.norm(
            self.rel_pos[:, :2], dim=1
        )

    def _reward_heading(self):
        target_distance = torch.norm(self.base_rel_pos[:, :2], dim=1).clamp_min(1e-6)
        target_forward_alignment = self.base_rel_pos[:, 0] / target_distance
        return target_forward_alignment * self.actions[:, 0]

    def _reward_visible(self):
        return self.obs_buf[:, 3]

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_reverse(self):
        return torch.clamp(-self.actions[:, 0], min=0.0)

    def _reward_success(self):
        return self.success_condition.to(gs.tc_float)

    def _reward_crash(self):
        return self.crash_condition.to(gs.tc_float)

    def _reward_timeout(self):
        return self.timeout_condition.to(gs.tc_float)
