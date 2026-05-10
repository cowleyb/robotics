# MentorPi Genesis RL Tutorial

This tutorial builds a small Genesis reinforcement learning project around the
Hiwonder MentorPi-style RC car model in `sim_mentor_pi`.

The goal is not to build a perfect autonomous car in one jump. The goal is to
learn the Genesis simulation workflow step by step:

1. Load a car model.
2. Control the car with simple throttle and steering actions.
3. Wrap the car in a reinforcement learning environment.
4. Add a destination task.
5. Add simple goal walls as obstacles.
6. Train a PPO policy.
7. Watch the trained policy drive in simulation.

This tutorial uses only the `sim_mentor_pi` package. The older `sim` package is
not part of this walkthrough.

## What You Will Build

You will train a lightweight policy that drives a simulated MentorPi car toward
a goal area. The policy receives privileged training observations:

- the goal or entry direction in the car frame
- the car orientation
- the car linear and angular velocity
- the previous action
- the nearby goal wall positions in the car frame

This is intentionally simple. It teaches Genesis, vehicle control, reward
design, vectorized environments, and PPO before moving toward camera-based
perception or real hardware.

## Project Files

The tutorial is centered on these files:

| File | Purpose |
|---|---|
| `assets/mentorpi_car.xacro` | The MentorPi-style car model loaded into Genesis. |
| `sim_mentor_pi/car_config.py` | Reads wheelbase, track width, wheel radius, and steering limits from the model. |
| `sim_mentor_pi/car_entity.py` | Converts policy actions into Ackermann steering and wheel velocity commands. |
| `sim_mentor_pi/car_env.py` | Builds the Genesis scene, goal task, observations, rewards, resets, and stepping logic. |
| `sim_mentor_pi/car_train.py` | Configures PPO and starts training. |
| `sim_mentor_pi/car_eval.py` | Loads a checkpoint and watches the trained policy. |
| `sim_mentor_pi/car_manual.py` | Lets you drive the simulated car with the keyboard. |

## Setup

Activate the environment:

```bash
conda activate robot
```

Install dependencies if this is a fresh environment:

```bash
pip install -r requirements.txt
```

For viewer-based runs on Linux, use:

```bash
PYOPENGL_PLATFORM=glx
```

You can place it before each command, as shown in the examples below.

## Stage 1: Load The Car Model

The car model lives at:

```text
assets/mentorpi_car.xacro
```

The environment loads it in `sim_mentor_pi/car_env.py`:

```python
raw_car = self.scene.add_entity(
    gs.morphs.URDF(
        file=str(CAR_PATH),
        links_to_keep=(CAMERA_LINK_NAME,),
        default_armature=0.001,
    ),
    material=gs.materials.Rigid(friction=2.2),
)
```

Genesis treats the Xacro/URDF model as a rigid articulated robot. The tutorial
keeps the camera link because later stages can attach a simulated front camera
to that link.

The car geometry is extracted from the model instead of being typed by hand:

```python
car_geom = CarExtractor(str(CAR_PATH)).get_geom()
```

That happens in `sim_mentor_pi/car_config.py`. It reads:

- front wheel positions
- rear wheel positions
- wheel radius
- steering joint limits

This matters because the controller needs the real wheelbase and track width to
compute Ackermann steering.

## Stage 2: Drive The Car Manually

Before training RL, confirm the simulated car can drive.

Run:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_manual
```

Controls:

| Key | Action |
|---|---|
| Up | throttle forward |
| Down | reverse |
| Left | steer left |
| Right | steer right |
| Escape | exit |

For basic debug output:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_manual --debug
```

For camera visualization:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_manual --cam
```

At this stage, check simple behavior:

- The car moves forward when you press Up.
- The car reverses when you press Down.
- The front wheels steer when you press Left or Right.
- The car does not immediately flip or jitter.

If manual control does not work, fix that before training. RL will not learn a
good policy from broken physics or broken controls.

## Stage 3: Understand The Action Space

The policy outputs two numbers:

```text
[throttle, steering]
```

Both values are clipped to `[-1.0, 1.0]`.

In `sim_mentor_pi/car_entity.py`, those two actions become real vehicle
commands:

```python
throttle_input = torch.clamp(actions[:, 0], -1.0, 1.0)
steering_input = torch.clamp(actions[:, 1], -1.0, 1.0)
```

The controller then converts them into:

- a target linear speed
- a target angular speed
- left and right front steering angles
- left and right drive wheel velocities

The steering uses Ackermann geometry. This is important for MentorPi-style cars
because they steer with front wheels instead of turning like a differential
drive robot.

The tutorial also smooths drive velocity:

```python
max_delta = MAX_DRIVE_ACCEL_MPS2 * self.dt
```

That small ramp keeps the simulated wheels from jumping instantly to full speed.
It makes the physics more stable and closer to a real small RC car.

## Stage 4: Build The Genesis Environment

The RL environment is `TestEnv` in `sim_mentor_pi/car_env.py`.

It creates:

- a Genesis scene
- a ground plane
- optional goal visualization
- optional goal walls
- the MentorPi car
- optional camera visualization
- observation buffers
- reward buffers
- reset buffers

The scene uses a small physics timestep:

```python
self.dt = 0.01
```

That means the simulation steps at 100 Hz. A 20 second episode becomes about
2000 simulation steps:

```python
self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
```

The environment is vectorized. During training, many copies of the car run at
the same time:

```bash
--num_envs 1024
```

Vectorized training is one of the main reasons Genesis is useful for RL. The
policy gets much more experience per training iteration than it would from a
single simulated car.

## Stage 5: Add A Destination Task

Each episode samples a destination command:

```python
self.commands[envs_idx] = candidates
```

The target position is sampled inside these ranges from `car_train.py`:

```python
"pos_x_range": [-1.2, 1.2],
"pos_y_range": [-1.2, 1.2],
"pos_z_range": [0.1, 0.1],
```

The environment avoids placing the target too close to the starting position:

```python
"min_start_distance": 0.8
```

The policy does not receive the world position directly. It receives the target
direction in the car frame. That makes the learning problem easier and teaches a
useful robotics pattern:

```text
world goal -> relative goal -> robot-frame observation -> action
```

The success condition is simple:

```python
torch.norm(self.rel_pos[:, :2], dim=1) < self.env_cfg["at_target_threshold"]
```

By default, the car succeeds when it gets within `0.10` meters of the target.

## Stage 6: Add Goal Walls

The current task uses a simple U-shaped wall around the goal. The open side
faces the car spawn area, so there is a clear entry path.

This is not meant to be a final obstacle avoidance system. It is a teaching
stage. The policy must learn that driving straight to the goal is not always
enough. Sometimes it must first drive toward the opening, then enter the goal
area.

The wall settings are in `get_cfgs()` inside `sim_mentor_pi/car_train.py`:

```python
"goal_walls": {
    "enabled": True,
    "count": 3,
    "depth": 0.7,
    "half_width": 0.35,
    "thickness": 0.06,
    "height": 0.18,
    "car_radius": 0.16,
    "near_distance": 0.5,
    "entry_offset": 0.25,
    "entry_threshold": 0.25,
}
```

The policy receives wall positions in the car frame:

```python
self.wall_rel_pos.reshape(self.num_envs, -1)
```

This is privileged information. Later, this can be replaced with camera-based
perception, AprilTags, ArUco markers, or local obstacle detections. For this
tutorial, privileged observations keep the RL lesson focused and debuggable.

## Stage 7: Understand The Observation

The policy observation is built in `_update_observation()`:

```python
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
```

The base observation has 15 values:

- 3 values for target direction in the car frame
- 4 values for car orientation quaternion
- 3 values for linear velocity
- 3 values for angular velocity
- 2 values for the previous action

When walls are enabled, each wall adds 2 more values:

```python
WALL_POSITION_DIM = 2
```

With 3 walls, the policy state has:

```text
15 + 3 * 2 = 21 values
```

You can confirm this when training starts. The script prints:

```text
policy state dim: 21
```

## Stage 8: Design The Rewards

The reward terms are configured in `car_train.py`:

```python
"reward_scales": {
    "entry_progress": 30.0,
    "progress": 20.0,
    "heading": 0.2,
    "reverse": -0.2,
    "steering": -0.1,
    "smooth": -0.02,
    "near_wall": -1.0,
    "success": 200.0,
    "crash": -300.0,
    "timeout": -50.0,
}
```

The main lesson is reward shaping:

- `progress` rewards getting closer to the active navigation target.
- `entry_progress` rewards moving toward the wall opening before trying to enter.
- `heading` rewards facing and moving toward the target.
- `reverse` discourages unnecessary reversing.
- `steering` discourages excessive steering.
- `smooth` discourages sudden action changes.
- `near_wall` discourages scraping walls.
- `success` strongly rewards reaching the goal.
- `crash` strongly penalizes flipping, leaving bounds, or hitting a wall.
- `timeout` penalizes failing to finish in time.

The environment changes the navigation target when walls are enabled:

```text
far from entry -> drive to the entry point
near entry -> drive to the real goal
```

That staged target makes learning easier and keeps the behavior understandable.

## Stage 9: Train PPO

Start a normal training run:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_train --exp_name room-goal-privileged-rl --num_envs 1024 --max_iterations 500
```

For a quick smoke test, use fewer environments and fewer iterations:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_train --exp_name smoke-test --num_envs 16 --max_iterations 2
```

Training writes to:

```text
logs/<exp_name>/
```

That folder contains:

- `cfgs.pkl` with the saved environment and PPO config
- TensorBoard event files
- checkpoint files like `model_0.pt`, `model_100.pt`, and `model_500.pt`

To resume an existing run:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_train --exp_name room-goal-privileged-rl --num_envs 1024 --max_iterations 500 --resume
```

Without `--resume`, the training script removes the old log folder for the same
experiment name before starting again.

## Stage 10: Watch The Trained Policy

Evaluate the latest checkpoint:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_eval -e room-goal-privileged-rl
```

Evaluate a specific checkpoint:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_eval -e room-goal-privileged-rl --ckpt 100
```

Record a camera video:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_eval -e room-goal-privileged-rl --record
```

The recorded file is written as:

```text
video.mp4
```

When watching, look for these behaviors:

- The car turns toward the entry point instead of blindly driving at the goal.
- The car slows or steers before hitting the wall.
- The car enters the U-shaped goal area.
- The car reaches the goal threshold before the episode times out.

## Stage 11: Train Without Walls

It is useful to compare the harder wall task with a simpler goal-reaching task.

Run:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_train --exp_name goal-only-rl --num_envs 1024 --max_iterations 300 --no_walls
```

Then watch it:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_eval -e goal-only-rl
```

This version removes wall observations and wall collision penalties. The policy
only needs to learn basic point-to-point navigation.

Use this stage when debugging:

- If goal-only works but walls fail, the car control is probably fine and the
  wall reward or observations need work.
- If goal-only also fails, debug the action space, resets, target sampling, and
  basic rewards before changing the wall task.

## Stage 12: How This Connects To The Real Robot

This tutorial is deliberately privileged. The policy is given clean state and
wall positions. A real camera-based robot will not get those values directly.

The intended long-term direction is:

```text
spoken/text command -> destination name
destination name -> landmark ID or goal region
camera -> landmark detection and obstacle cues
policy/controller -> throttle and steering
```

The Genesis RL tutorial teaches the driving task first. Later stages can replace
privileged wall and goal observations with perception:

- AprilTags or ArUco markers for global cues
- front camera detections for local obstacles
- simple command mapping from names to destination IDs
- lightweight policy deployment on Raspberry Pi

Do not optimize this tutorial for hidden-goal, image-only navigation. The target
project direction is command-to-destination driving with landmarks and local
obstacle avoidance.

## Recommended Learning Order

Follow this order when writing or teaching the blog version:

1. Show the MentorPi model in Genesis.
2. Drive it manually.
3. Explain the two-action control space.
4. Explain vectorized RL environments.
5. Add a simple goal.
6. Add U-shaped goal walls.
7. Explain observations and rewards.
8. Train PPO.
9. Evaluate checkpoints.
10. Explain how privileged simulation can later become camera-based real-world control.

## Common Problems

### The viewer does not open

Use:

```bash
PYOPENGL_PLATFORM=glx
```

If you are using a remote machine, make sure display forwarding or local GPU
display access is configured.

### Training is slow

Lower the number of environments for testing:

```bash
--num_envs 16
```

Use larger values like `1024` for real training once the environment works.

### The policy crashes into walls

First check whether the goal-only task works:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_train --exp_name goal-only-rl --num_envs 1024 --max_iterations 300 --no_walls
```

If that works, inspect:

- `entry_progress`
- `near_wall`
- `crash`
- wall positions in the observation
- whether the entry point is on the open side of the U-shaped wall

### The car jitters or flips

Check manual driving first:

```bash
PYOPENGL_PLATFORM=glx python -m sim_mentor_pi.car_manual --debug
```

If manual driving is unstable, inspect:

- wheel radius
- wheelbase
- steering limits
- friction
- maximum drive speed
- acceleration smoothing

## Summary

You now have a staged Genesis RL tutorial for a MentorPi-style car:

- Stage 1 loads the car model.
- Stage 2 confirms manual driving.
- Stage 3 explains throttle and steering actions.
- Stage 4 builds a Genesis RL environment.
- Stage 5 adds destination commands.
- Stage 6 adds simple goal walls.
- Stage 7 explains the policy observation.
- Stage 8 explains reward shaping.
- Stage 9 trains PPO.
- Stage 10 evaluates the policy.
- Stage 11 compares against a no-wall task.
- Stage 12 connects the simulator to the future real robot direction.

This is a practical base for a blog tutorial and for later work on camera-based
landmarks, local obstacle avoidance, and Raspberry Pi deployment.
