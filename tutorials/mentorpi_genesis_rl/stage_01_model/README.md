# Stage 1: Load And Inspect The Model

## 1.1 Lesson Goal

In this lesson, you will check that the MentorPi-style car model can be read by
the tutorial code before opening Genesis or training RL.

This is the first stage because the controller depends on real geometry from
the model. If the wheelbase, track width, wheel radius, or steering joints are
wrong, later stages will be difficult to debug.

## 1.2 Program Logic

The script loads:

```text
assets/mentorpi_car.xacro
```

Then it uses `CarExtractor` to read the model and print the values used by the
Ackermann controller:

- wheel radius
- track width
- wheelbase
- steering limits

These values are not guessed. They are taken from the Xacro/URDF joints and
wheel geometry.

## 1.3 Source Code

Stage script:

```text
tutorials/mentorpi_genesis_rl/stage_01_model/inspect_model.py
```

Main implementation:

```text
sim_mentor_pi/car_config.py
```

Important code:

```python
car_geom = CarExtractor(str(car_path)).get_geom()
```

`CarExtractor` reads joint positions from the XML model. For example, it
computes track width from the distance between the left and right front wheel
joints.

## 1.4 Run The Program

From the repo root, enter:

```bash
python tutorials/mentorpi_genesis_rl/stage_01_model/inspect_model.py
```

## 1.5 Expected Result

You should see output similar to:

```text
Controller geometry
wheel radius: 0.0335 m
track width: 0.1330 m
wheelbase: 0.1450 m
front steering limit: -0.5061 to 0.5061 rad
```

The exact values should match the current model file.

## 1.6 Parameter Explanation

| Parameter | Meaning |
|---|---|
| wheel radius | Used to convert wheel linear speed into wheel angular speed. |
| track width | Distance between left and right wheels. Used for turning calculations. |
| wheelbase | Distance between front and rear axles. Used for Ackermann steering. |
| steering limit | Maximum left and right steering angle allowed by the model. |

## 1.7 Note

If this stage fails, do not continue to RL training. First check:

- the model path
- the joint names in `sim_mentor_pi/car_config.py`
- whether `assets/mentorpi_car.xacro` exists

The later controller assumes these geometry values are correct.
Run from the repo root:

```bash
python tutorials/mentorpi_genesis_rl/stage_01_model/inspect_model.py
```

If this stage fails, fix the model path or joint names before moving on.
