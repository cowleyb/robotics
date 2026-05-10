# Stage 5: Train PPO

## 5.1 Lesson Goal

In this lesson, you will train the MentorPi policy with PPO.

PPO means Proximal Policy Optimization. In this project it learns a small neural
network that maps the policy observation to:

```text
[throttle, steering]
```

## 5.2 Program Logic

The training script does five main things:

1. Initializes Genesis.
2. Loads environment, observation, reward, target, and PPO configs.
3. Creates many parallel copies of the car environment.
4. Creates an `OnPolicyRunner` from `rsl_rl`.
5. Runs PPO learning and saves checkpoints.

Important code:

```python
env = TestEnv(
    base_seed=args.seed,
    num_envs=args.num_envs,
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    target_cfg=target_cfg,
    show_viewer=args.vis,
)
```

This creates the vectorized Genesis environment.

Then:

```python
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.learn(num_learning_iterations=args.max_iterations)
```

This starts PPO training.

## 5.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_05_train_ppo/train_ppo.py
```

Main implementation:

```text
sim_mentor_pi/car_train.py
sim_mentor_pi/car_env.py
```

The stage script calls the maintained training entry point:

```python
from sim_mentor_pi.car_train import main
```

## 5.4 PPO Configuration

The training config is built in `get_train_cfg()`.

Important settings:

| Setting | Meaning |
|---|---|
| `num_steps_per_env` | Number of simulation steps collected before each PPO update. |
| `save_interval` | How often checkpoints are saved. |
| `hidden_dims` | Neural network layer sizes. |
| `learning_rate` | PPO optimizer learning rate. |
| `entropy_coef` | Encourages exploration. |
| `desired_kl` | Helps control update size. |

The actor network is small:

```python
"hidden_dims": [128, 128]
```

That is intentional. The project goal is a lightweight policy that can later be
adapted toward Raspberry Pi deployment.

## 5.5 Reward Configuration

The reward scales are:

```python
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
```

The largest positive reward is success. The largest negative reward is crash.
That makes the lesson clear:

```text
reach the goal, but do not hit walls or fail the episode
```

## 5.6 Run A Smoke Test

Use this first:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_05_train_ppo/train_ppo.py --exp_name smoke-test --num_envs 16 --max_iterations 2
```

This confirms imports, environment creation, PPO setup, and checkpoint writing.

## 5.7 Run Normal Training

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_05_train_ppo/train_ppo.py --exp_name room-goal-privileged-rl --num_envs 1024 --max_iterations 500
```

## 5.8 Train Without Walls

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_05_train_ppo/train_ppo.py --exp_name goal-only-rl --num_envs 1024 --max_iterations 300 --no_walls
```

Use this if the wall task is too hard. The no-wall task is a cleaner check of
basic goal navigation.

## 5.9 Output Files

Checkpoints are written to:

```text
logs/<exp_name>/
```

The folder contains:

| File | Purpose |
|---|---|
| `cfgs.pkl` | Saved environment and training configs. |
| `model_*.pt` | PPO checkpoints. |
| `events.out.tfevents...` | TensorBoard logs. |

## 5.10 Resume Training

To continue an existing experiment:

```bash
PYOPENGL_PLATFORM=glx python tutorials/mentorpi_genesis_rl/stage_05_train_ppo/train_ppo.py --exp_name room-goal-privileged-rl --num_envs 1024 --max_iterations 500 --resume
```

## 5.11 Note

Without `--resume`, the script removes the old log folder for the same
experiment name. Use a new `--exp_name` if you want to keep previous results.
