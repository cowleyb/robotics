# MentorPi Genesis RL Tutorial

This folder contains the code-backed stages for learning Genesis simulation and
reinforcement learning with the MentorPi-style Ackermann car.

The layout follows the same teaching style as the Hiwonder MentorPi lessons:

- start with the lesson goal
- explain the program logic
- show the source code location
- give numbered running steps
- explain the important parameters
- add notes for common mistakes

Each stage has its own folder with a runnable script and a README. The scripts
reuse the maintained simulator code in `sim_mentor_pi` instead of copying the
full environment into every stage. That keeps the tutorial clear without
creating several stale versions of the same simulator.

## Course Target

By the end of the stages, you will have:

- loaded the MentorPi car model into Genesis
- driven the car manually
- connected throttle and steering actions to Ackermann control
- created a goal-reaching RL environment
- added simple goal walls
- trained a PPO policy
- watched a trained checkpoint in the simulator

This is a simulation-first course. The later real robot direction is:

```text
command -> destination
camera -> landmarks and obstacle cues
policy/controller -> throttle and steering
```

## Stage List

| Stage | Folder | Teaches |
|---|---|---|
| 1 | `stage_01_model` | Load the MentorPi model and inspect its geometry. |
| 2 | `stage_02_manual_drive` | Drive the simulated car with keyboard controls. |
| 3 | `stage_03_goal_task` | Run a simple goal-reaching task without walls. |
| 4 | `stage_04_goal_walls` | Add U-shaped goal walls and staged navigation. |
| 5 | `stage_05_train_ppo` | Train PPO on the MentorPi environment. |
| 6 | `stage_06_evaluate_policy` | Load and watch a trained checkpoint. |

## Preparation

From the repo root:

```bash
conda activate robot
pip install -r requirements.txt
```

Viewer commands on Linux should use:

```bash
PYOPENGL_PLATFORM=glx
```

## Learning Order

Run the stages in order. Do not start training until manual driving and the
goal task are working.

1. Confirm the model geometry.
2. Confirm manual control.
3. Run a no-wall goal task.
4. Run the goal-wall task.
5. Train PPO.
6. Evaluate the trained checkpoint.

## Note

Commands are case-sensitive and paths are relative to the repo root. If a script
cannot import `sim_mentor_pi`, make sure you are running it from this project
checkout.
