# Stage 3: Goal Task Without Walls

## 3.1 Lesson Goal

In this lesson, you will run the simplest navigation task:

```text
start position -> sampled goal position
```

The wall task is disabled. This lets you confirm that target sampling,
observations, resets, and vehicle movement work before adding obstacles.

## 3.2 Program Logic

The script creates the normal RL environment but changes one setting:

```python
env_cfg["goal_walls"]["enabled"] = False
```

Then it updates the observation size:

```python
update_policy_state_dim(env_cfg, obs_cfg)
```

This matters because wall positions are part of the policy observation when
walls are enabled. When walls are disabled, those observation values are
removed.

The script uses a simple hand-written controller:

```python
actions[:, 0] = torch.where(target_x > 0.05, 0.8, 0.2)
actions[:, 1] = torch.clamp(target_y * 3.0, -1.0, 1.0)
```

This controller is not meant to be perfect. It is just a readable way to show
how the observation can become throttle and steering.

## 3.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_03_goal_task/run_goal_task.py
```

Main implementation:

```text
sim_mentor_pi/car_env.py
sim_mentor_pi/car_train.py
```

Important environment code:

```python
self.commands[envs_idx] = candidates
```

That line stores the sampled goal position for each environment.

## 3.4 Run The Program

From the repo root:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_03_goal_task/run_goal_task.py
```

Run more parallel environments:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_03_goal_task/run_goal_task.py --num_envs 5
```

## 3.5 Observation Explanation

The first values in the policy observation are the target position in the car
frame. In simple terms:

| Value | Meaning |
|---|---|
| `target_x` | how far the target is in front of or behind the car |
| `target_y` | how far the target is to the left or right of the car |

The hand-written controller uses `target_y` to steer toward the target.

## 3.6 What To Observe

Look for:

- the red target marker appearing in different positions
- the car steering toward the marker
- the episode resetting after success, crash, or timeout

## 3.7 Note

This stage is not training. If the simple controller misses some targets, that
is acceptable. The goal is to verify the environment is understandable before
adding PPO.
