# Stage 4: Goal Walls

## 4.1 Lesson Goal

In this lesson, you will enable a simple U-shaped wall around the goal.

This stage teaches why direct goal driving is not enough. With walls enabled,
the useful behavior is:

```text
drive to entry opening -> enter goal area -> reach target
```

## 4.2 Program Logic

The wall task is enabled by default in `get_cfgs()`.

The wall configuration is:

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

The environment places three boxes around the target:

- left side wall
- right side wall
- back wall

The side facing the spawn area is left open.

The environment also creates an entry point outside the opening. When the car is
far from the entry, the navigation target is the entry point. When the car is
near the entry, the navigation target becomes the true goal.

## 4.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_04_goal_walls/run_goal_walls.py
```

Main implementation:

```text
sim_mentor_pi/car_env.py
sim_mentor_pi/car_train.py
```

Important methods:

| Method | Purpose |
|---|---|
| `_add_goal_walls()` | Adds the wall entities to the Genesis scene. |
| `_resample_goal_walls()` | Moves the wall boxes around each new target. |
| `_update_navigation_target()` | Chooses entry point or true target. |
| `_wall_hit()` | Detects wall collision for RL termination. |

## 4.4 Run The Program

From the repo root:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_04_goal_walls/run_goal_walls.py
```

Run more rendered environments:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_04_goal_walls/run_goal_walls.py --num_envs 5
```

## 4.5 Parameter Explanation

| Parameter | Meaning |
|---|---|
| `count` | Number of walls. This tutorial uses 3. |
| `depth` | Length of the side walls. |
| `half_width` | Half of the opening/goal area width. |
| `thickness` | Wall thickness. |
| `height` | Wall height in the simulator. |
| `car_radius` | Simple circular footprint used for wall collision checks. |
| `near_distance` | Distance used by the wall penalty reward. |
| `entry_offset` | How far outside the opening the entry point is placed. |
| `entry_threshold` | Distance at which the task switches from entry point to true goal. |

## 4.6 What To Observe

The simple controller may fail in this stage. That is useful. Watch for:

- cases where the car drives directly into a wall
- cases where the car approaches the opening
- how the wall layout changes when the goal resets

## 4.7 Note

This stage explains the need for reward shaping. A direct controller can solve
some easy cases, but the RL policy will learn from many randomized examples.
