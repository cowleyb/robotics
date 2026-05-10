# Stage 6: Evaluate A Trained Policy

## 6.1 Lesson Goal

In this lesson, you will load a trained PPO checkpoint and watch the policy
drive in Genesis.

Evaluation is separate from training. Training runs many environments quickly.
Evaluation opens the viewer and renders a small number of environments so you
can inspect behavior.

## 6.2 Program Logic

The evaluation script:

1. Loads `cfgs.pkl` from the experiment folder.
2. Turns off training rewards.
3. Enables target visualization.
4. Creates a small rendered environment.
5. Loads the selected checkpoint.
6. Runs the policy in a loop.

Important code:

```python
runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
policy = runner.get_inference_policy(device=gs.device)
```

Then each frame:

```python
actions = policy(obs_dict)
obs_dict, _, _, _ = env.step(actions)
```

This is the same environment step interface used in the earlier stages.

## 6.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_06_evaluate_policy/evaluate_policy.py
```

Main implementation:

```text
sim_mentor_pi/car_eval.py
```

## 6.4 Evaluate The Latest Checkpoint

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_06_evaluate_policy/evaluate_policy.py -e room-goal-privileged-rl
```

If `--ckpt` is not provided, the script finds the largest checkpoint number in
the experiment folder.

## 6.5 Evaluate A Specific Checkpoint

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_06_evaluate_policy/evaluate_policy.py -e room-goal-privileged-rl --ckpt 100
```

Use this to compare early and late training behavior.

## 6.6 Record A Video

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_06_evaluate_policy/evaluate_policy.py -e room-goal-privileged-rl --record
```

The video is saved as:

```text
video.mp4
```

## 6.7 What To Observe

A useful policy should:

- turn toward the entry side of the wall
- avoid direct wall collisions
- enter the U-shaped goal area
- reach the target before timeout
- avoid excessive reversing or steering jitter

## 6.8 Troubleshooting

| Problem | Check |
|---|---|
| No checkpoint found | Confirm the experiment name under `logs/`. |
| Policy drives into walls | Compare with an earlier/later checkpoint and inspect reward balance. |
| Viewer fails | Use `PYOPENGL_PLATFORM=glx`. |
| Behavior differs from training | Confirm `cfgs.pkl` belongs to the same experiment. |

## 6.9 Note

Evaluation does not improve the policy. If behavior is poor, go back to Stage 5
and train longer, tune rewards, or compare against the no-wall task.
