# Stage 2: Manual Driving

## 2.1 Lesson Goal

In this lesson, you will drive the simulated MentorPi car with the keyboard.

The purpose is to confirm the basic physics and control code before using RL.
If manual driving is unstable, the policy will also learn unstable behavior.

## 2.2 Program Logic

The script starts a Genesis scene, loads the car, registers keyboard controls,
and sends the current manual action into the environment every simulation step.

The action has two values:

```text
[throttle, steering]
```

The keyboard changes those two values:

| Key | Action value changed |
|---|---|
| Up | increases throttle |
| Down | decreases throttle |
| Left | steers left |
| Right | steers right |

The environment then calls:

```python
env.step(env.manual_action)
```

That line is useful because RL will later use the same `env.step(actions)`
interface. Manual driving and policy driving share the same vehicle physics.

## 2.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_02_manual_drive/manual_drive.py
```

Main implementation:

```text
sim_mentor_pi/car_manual.py
sim_mentor_pi/car_env.py
sim_mentor_pi/car_entity.py
```

The stage script is intentionally short:

```python
from sim_mentor_pi.car_manual import main
```

That keeps the tutorial stage runnable while keeping the real simulator logic in
one maintained place.

## 2.4 Run The Program

From the repo root:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_02_manual_drive/manual_drive.py
```

Controls:

| Key | Action |
|---|---|
| Up | throttle forward |
| Down | reverse |
| Left | steer left |
| Right | steer right |
| Escape | exit |

## 2.5 Optional Debug Output

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_02_manual_drive/manual_drive.py --debug
```

The debug mode prints values such as:

- car position
- pitch angle
- wheel velocity

Use this when the car appears to move incorrectly.

## 2.6 Optional Camera View

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_02_manual_drive/manual_drive.py --cam
```

The camera is attached to the car's camera link. In this tutorial it is only for
visual debugging. Later, this can become the input for visual navigation.

## 2.7 What To Observe

Check these behaviors:

- pressing Up moves the car forward
- pressing Down moves the car backward
- pressing Left and Right steer the front wheels
- the car does not immediately flip
- the car does not jitter heavily when starting

## 2.8 Note

If the viewer does not open, check that `PYOPENGL_PLATFORM=glx` is set. If the
car jitters, inspect the drive speed and acceleration smoothing in
`sim_mentor_pi/car_entity.py`.
