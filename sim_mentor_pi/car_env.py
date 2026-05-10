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
        self.wall_cfg = env_cfg["goal_walls"]
        self.wall_count = int(self.wall_cfg["count"]) if self.wall_cfg["enabled"] else 0
        if self.wall_count not in (0, 3):
            raise ValueError("goal_walls count must be 3 when walls are enabled")
        self.wall_depth = float(self.wall_cfg["depth"])
        self.wall_half_width = float(self.wall_cfg["half_width"])
        self.wall_thickness = float(self.wall_cfg["thickness"])
        self.wall_height = float(self.wall_cfg["height"])
        self.wall_entry_offset = float(self.wall_cfg["entry_offset"])
        self.wall_entry_threshold = float(self.wall_cfg["entry_threshold"])
        side_wall_size = [self.wall_depth, self.wall_thickness, self.wall_height]
        back_wall_size = [
            self.wall_half_width * 2.0 + self.wall_thickness,
            self.wall_thickness,
            self.wall_height,
        ]
        self.wall_sizes = torch.tensor(
            [side_wall_size, side_wall_size, back_wall_size],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.wall_z = self.wall_height * 0.5

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

        self.walls = self._add_goal_walls()

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
        terminal_rewards = {"success", "crash", "timeout"}
        for name in self.reward_scales:
            if name not in terminal_rewards:
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
        self.nav_rel_pos = torch.zeros_like(self.rel_pos)
        self.last_nav_rel_pos = torch.zeros_like(self.rel_pos)
        self.base_rel_pos = torch.zeros_like(self.rel_pos)
        self.entry_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.entry_rel_pos = torch.zeros_like(self.entry_pos)
        self.last_entry_rel_pos = torch.zeros_like(self.entry_pos)
        self.wall_pos = torch.zeros(
            (self.num_envs, self.wall_count, 3),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.wall_yaw = torch.zeros(
            (self.num_envs, self.wall_count),
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.wall_rel_pos = torch.zeros(
            (self.num_envs, self.wall_count, 2),
            device=gs.device,
            dtype=gs.tc_float,
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
        env_count = len(envs_idx)
        min_start_distance = float(self.target_cfg["min_start_distance"])
        base_xy = self.base_pos[envs_idx, :2]

        candidates = torch.zeros((env_count, 3), device=gs.device, dtype=gs.tc_float)
        for _ in range(30):
            too_close = (
                torch.norm(candidates[:, :2] - base_xy, dim=1) <= min_start_distance
            )
            if not torch.any(too_close):
                break

            sample_count = int(torch.sum(too_close).item())
            candidates[too_close, 0] = gs_rand_float(
                *self.target_cfg["pos_x_range"], (sample_count,), gs.device
            )
            candidates[too_close, 1] = gs_rand_float(
                *self.target_cfg["pos_y_range"], (sample_count,), gs.device
            )
            candidates[too_close, 2] = gs_rand_float(
                *self.target_cfg["pos_z_range"], (sample_count,), gs.device
            )

        self.commands[envs_idx] = candidates

    # The goal is surrounded by a simple U-shaped wall. The side facing the spawn
    # is left open so the first version has a clear, teachable path to learn.
    def _add_goal_walls(self):
        walls = []
        for wall_idx in range(self.wall_count):
            walls.append(
                self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(0.0, 0.0, self.wall_z),
                        size=tuple(float(v) for v in self.wall_sizes[wall_idx]),
                        fixed=True,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(
                            color=(0.45, 0.45, 0.45)
                        )
                    ),
                    name=f"goal_wall_{wall_idx}",
                )
            )
        return walls

    def _resample_goal_walls(self, envs_idx):
        if self.wall_count == 0:
            return

        env_count = len(envs_idx)
        target_xy = self.commands[envs_idx, :2]
        base_xy = self.base_init_pos[:2].expand(env_count, -1)
        to_spawn = base_xy - target_xy
        open_along_x = torch.abs(to_spawn[:, 0]) >= torch.abs(to_spawn[:, 1])

        x_sign = torch.where(
            to_spawn[:, 0] >= 0.0,
            torch.ones((env_count,), device=gs.device, dtype=gs.tc_float),
            -torch.ones((env_count,), device=gs.device, dtype=gs.tc_float),
        )
        y_sign = torch.where(
            to_spawn[:, 1] >= 0.0,
            torch.ones((env_count,), device=gs.device, dtype=gs.tc_float),
            -torch.ones((env_count,), device=gs.device, dtype=gs.tc_float),
        )
        open_dir = torch.zeros((env_count, 2), device=gs.device, dtype=gs.tc_float)
        open_dir[:, 0] = torch.where(
            open_along_x,
            x_sign,
            torch.zeros((env_count,), device=gs.device, dtype=gs.tc_float),
        )
        open_dir[:, 1] = torch.where(
            open_along_x,
            torch.zeros((env_count,), device=gs.device, dtype=gs.tc_float),
            y_sign,
        )
        closed_dir = -open_dir
        side_dir = torch.stack((-closed_dir[:, 1], closed_dir[:, 0]), dim=1)

        wall_pos = torch.zeros(
            (env_count, self.wall_count, 3),
            device=gs.device,
            dtype=gs.tc_float,
        )
        wall_yaw = torch.zeros(
            (env_count, self.wall_count),
            device=gs.device,
            dtype=gs.tc_float,
        )

        wall_pos[:, 0, :2] = target_xy + side_dir * self.wall_half_width
        wall_pos[:, 1, :2] = target_xy - side_dir * self.wall_half_width
        wall_pos[:, 2, :2] = target_xy + closed_dir * (self.wall_depth * 0.5)
        wall_pos[:, :, 2] = self.wall_z

        entry_pos = torch.zeros((env_count, 3), device=gs.device, dtype=gs.tc_float)
        entry_pos[:, :2] = target_xy + open_dir * (
            self.wall_depth * 0.5 + self.wall_entry_offset
        )
        entry_pos[:, 2] = self.commands[envs_idx, 2]

        side_yaw = torch.atan2(closed_dir[:, 1], closed_dir[:, 0])
        back_yaw = torch.atan2(side_dir[:, 1], side_dir[:, 0])
        wall_yaw[:, 0] = side_yaw
        wall_yaw[:, 1] = side_yaw
        wall_yaw[:, 2] = back_yaw

        self.wall_pos[envs_idx] = wall_pos
        self.wall_yaw[envs_idx] = wall_yaw
        self.entry_pos[envs_idx] = entry_pos
        for wall_idx, wall in enumerate(self.walls):
            wall.set_pos(
                self.wall_pos[envs_idx, wall_idx],
                zero_velocity=True,
                envs_idx=envs_idx,
            )
            yaw = self.wall_yaw[envs_idx, wall_idx]
            wall.set_quat(
                torch.stack(
                    (
                        torch.cos(yaw * 0.5),
                        torch.zeros_like(yaw),
                        torch.zeros_like(yaw),
                        torch.sin(yaw * 0.5),
                    ),
                    dim=1,
                ),
                zero_velocity=True,
                envs_idx=envs_idx,
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
                cos_yaw * self.nav_rel_pos[:, 0] + sin_yaw * self.nav_rel_pos[:, 1],
                -sin_yaw * self.nav_rel_pos[:, 0] + cos_yaw * self.nav_rel_pos[:, 1],
                self.nav_rel_pos[:, 2],
            ),
            dim=1,
        )

    def _update_navigation_target(self, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=gs.device)

        if self.wall_count == 0:
            self.nav_rel_pos[envs_idx] = self.rel_pos[envs_idx]
            return

        entry_distance = torch.norm(self.entry_rel_pos[envs_idx, :2], dim=1)
        use_entry = entry_distance > self.wall_entry_threshold
        self.nav_rel_pos[envs_idx] = torch.where(
            use_entry[:, None],
            self.entry_rel_pos[envs_idx],
            self.rel_pos[envs_idx],
        )

    def _update_wall_rel_pos(self) -> None:
        if self.wall_count == 0:
            return

        yaw = self._yaw_from_base_quat()
        cos_yaw = torch.cos(yaw)[:, None]
        sin_yaw = torch.sin(yaw)[:, None]
        rel = self.wall_pos[:, :, :2] - self.base_pos[:, None, :2]
        base_x = cos_yaw * rel[:, :, 0] + sin_yaw * rel[:, :, 1]
        base_y = -sin_yaw * rel[:, :, 0] + cos_yaw * rel[:, :, 1]
        wall_rel_pos = torch.stack((base_x, base_y), dim=2)
        wall_distances = torch.norm(wall_rel_pos, dim=2)
        nearest_first = torch.argsort(wall_distances, dim=1)
        self.wall_rel_pos[:] = torch.gather(
            wall_rel_pos,
            1,
            nearest_first[:, :, None].expand(-1, -1, 2),
        )

    def _wall_hit(self) -> torch.Tensor:
        # Use a small circular car footprint against the wall boxes. Physics
        # contacts still exist visually, but this is faster and stable for PPO.
        if self.wall_count == 0:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=torch.bool)

        rel = self.base_pos[:, None, :2] - self.wall_pos[:, :, :2]
        cos_yaw = torch.cos(self.wall_yaw)
        sin_yaw = torch.sin(self.wall_yaw)
        local_x = cos_yaw * rel[:, :, 0] + sin_yaw * rel[:, :, 1]
        local_y = -sin_yaw * rel[:, :, 0] + cos_yaw * rel[:, :, 1]
        half_size = self.wall_sizes[: self.wall_count, :2] * 0.5 + float(
            self.wall_cfg["car_radius"]
        )
        in_x = torch.abs(local_x) < half_size[None, :, 0]
        in_y = torch.abs(local_y) < half_size[None, :, 1]
        return torch.any(in_x & in_y, dim=1)

    def _update_observation(self):
        self.base_rel_pos[:] = self._target_pos_in_base_frame()
        self._update_wall_rel_pos()
        self.obs_buf = torch.cat(
            [
                torch.clip(self.base_rel_pos * self.obs_scales["rel_pos"], -1.0, 1.0),
                self.base_quat,
                torch.clip(self.base_lin_vel * self.obs_scales["lin_vel"], -1.0, 1.0),
                torch.clip(self.base_ang_vel * self.obs_scales["ang_vel"], -1.0, 1.0),
                self.last_actions,
                torch.clip(
                    self.wall_rel_pos.reshape(self.num_envs, -1)
                    * self.obs_scales["wall_pos"],
                    -1.0,
                    1.0,
                ),
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
        self.entry_rel_pos[:] = self.entry_pos - self.base_pos
        self._update_navigation_target()
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
            | self._wall_hit()
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
        self.last_entry_rel_pos[:] = self.entry_rel_pos
        self.last_nav_rel_pos[:] = self.nav_rel_pos

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
        self._resample_goal_walls(envs_idx)
        self.rel_pos[envs_idx] = self.commands[envs_idx] - self.base_pos[envs_idx]
        self.last_rel_pos[envs_idx] = self.rel_pos[envs_idx]
        self.entry_rel_pos[envs_idx] = self.entry_pos[envs_idx] - self.base_pos[envs_idx]
        self.last_entry_rel_pos[envs_idx] = self.entry_rel_pos[envs_idx]
        self._update_navigation_target(envs_idx)
        self.last_nav_rel_pos[envs_idx] = self.nav_rel_pos[envs_idx]
        self.success_condition[envs_idx] = False
        self.crash_condition[envs_idx] = False
        self.timeout_condition[envs_idx] = False
        self._update_observation()

    def reset(self):
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.get_observations()

    def _reward_progress(self):
        return torch.norm(self.last_nav_rel_pos[:, :2], dim=1) - torch.norm(
            self.nav_rel_pos[:, :2], dim=1
        )

    def _reward_entry_progress(self):
        if self.wall_count == 0:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        entry_distance = torch.norm(self.entry_rel_pos[:, :2], dim=1)
        last_entry_distance = torch.norm(self.last_entry_rel_pos[:, :2], dim=1)
        needs_entry = entry_distance > 0.20
        return torch.where(
            needs_entry,
            last_entry_distance - entry_distance,
            torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float),
        )

    def _reward_heading(self):
        target_distance = torch.norm(self.base_rel_pos[:, :2], dim=1).clamp_min(1e-6)
        target_forward_alignment = self.base_rel_pos[:, 0] / target_distance
        return target_forward_alignment * self.actions[:, 0]

    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_near_wall(self):
        if self.wall_count == 0:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        rel = self.base_pos[:, None, :2] - self.wall_pos[:, :, :2]
        cos_yaw = torch.cos(self.wall_yaw)
        sin_yaw = torch.sin(self.wall_yaw)
        local_x = cos_yaw * rel[:, :, 0] + sin_yaw * rel[:, :, 1]
        local_y = -sin_yaw * rel[:, :, 0] + cos_yaw * rel[:, :, 1]
        half_size = self.wall_sizes[: self.wall_count, :2] * 0.5
        dx = torch.clamp(torch.abs(local_x) - half_size[None, :, 0], min=0.0)
        dy = torch.clamp(torch.abs(local_y) - half_size[None, :, 1], min=0.0)
        nearest_distance = torch.min(torch.sqrt(dx * dx + dy * dy), dim=1).values
        clear_distance = torch.clamp(
            nearest_distance - float(self.wall_cfg["car_radius"]),
            min=0.0,
        )
        near_distance = float(self.wall_cfg["near_distance"])
        return torch.clamp(1.0 - clear_distance / near_distance, 0.0, 1.0)

    def _reward_steering(self):
        return torch.square(self.actions[:, 1])

    def _reward_reverse(self):
        return torch.clamp(-self.actions[:, 0], min=0.0)

    def _reward_success(self):
        return self.success_condition.to(gs.tc_float)

    def _reward_crash(self):
        return self.crash_condition.to(gs.tc_float)

    def _reward_timeout(self):
        return self.timeout_condition.to(gs.tc_float)
