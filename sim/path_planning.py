from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline


def wrap_to_pi(angle: float) -> float:
    # angles repeat every full turn so fold them into the range from minus pi to plus pi
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class VehicleGeometry:
    wheelbase: float
    max_steering_angle: float
    length: float
    width: float
    wheel_radius: float


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    yaw: float
    speed: float


@dataclass(frozen=True)
class ObstacleBox:
    center_x: float
    center_y: float
    size_x: float
    size_y: float


@dataclass(frozen=True)
class EnvironmentBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass(frozen=True)
class PlannedPath:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    directions: np.ndarray
    cost: float

    def distance_to_point(self, x: float, y: float) -> float:
        # returns the closest distance from the car to the planned route
        dx = self.x - float(x)
        dy = self.y - float(y)
        return float(np.min(np.hypot(dx, dy)))


@dataclass(frozen=True)
class PlannerConfig:
    xy_resolution: float = 0.25  # how fine the map grid is
    yaw_resolution: float = math.radians(15.0)  # how many heading buckets this uses
    motion_resolution: float = 0.08  # how small each simulated step is
    primitive_length: float = 0.45  # how far a single move rolls before a new choice
    steer_samples: int = 7  # how many steering choices get tried at each step
    max_iterations: int = 15000  # stop searching after this many tries
    goal_tolerance: float = 0.28  # close enough to the goal counts as arrived
    analytic_expansion_distance: float = 1.0  # when close, try to connect in one go
    obstacle_margin: float = 0.06  # extra space around obstacles
    steer_cost: float = 0.35  # for turns
    steer_change_cost: float = 0.6  # for wheel back and forth
    reverse_cost: float = 2.0
    direction_change_cost: float = 3.0
    heuristic_weight: float = 1.8  # how strongly this aims toward the goal
    smoothing_factor: float = 0.02  # how much this rounds off the final route
    smoothing_resolution: float = 0.05  # spacing for the rounded off route
    replan_distance: float = (
        0.15  # if the car drifts this far from the route, plan again
    )


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_gain: float = 0.55  # faster speed looks further ahead
    min_lookahead: float = 0.35  # do not look too close even at low speed
    max_lookahead: float = 1.0  # do not look too far even at high speed
    nominal_speed: float = 1.0  # normal cruising speed
    slow_speed: float = 0.4  # careful speed near tricky spots
    goal_slowdown_distance: float = 0.8  # start slowing as the car gets near the goal
    cusp_slowdown_distance: float = 0.45  # slow near a forward to reverse switch
    cusp_switch_distance: float = 0.12  # how close before the switch is accepted
    speed_gain: float = 0.75  # how strongly this corrects speed errors
    max_forward_wheel_speed: float = 15.0
    max_reverse_wheel_speed: float = 5.0


@dataclass
class _SearchNode:
    x_index: int
    y_index: int
    yaw_index: int
    direction: int
    x_values: list[float]
    y_values: list[float]
    yaw_values: list[float]
    directions: list[int]
    steer: float
    cost: float
    parent_key: tuple[int, int, int, int] | None

    @property
    def x(self) -> float:
        return self.x_values[-1]

    @property
    def y(self) -> float:
        return self.y_values[-1]

    @property
    def yaw(self) -> float:
        return self.yaw_values[-1]

    def key(self) -> tuple[int, int, int, int]:
        return (self.x_index, self.y_index, self.yaw_index, self.direction)


def _overlap_on_axis(
    center_delta: np.ndarray,
    axis: np.ndarray,
    a_axes: tuple[np.ndarray, np.ndarray],
    a_half: np.ndarray,
    b_axes: tuple[np.ndarray, np.ndarray],
    b_half: np.ndarray,
) -> bool:
    # checks if two rectangles overlap when viewed along one direction
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return True
    axis = axis / axis_norm
    proj_center = float(abs(np.dot(center_delta, axis)))
    proj_a = float(
        abs(np.dot(a_axes[0] * a_half[0], axis))
        + abs(np.dot(a_axes[1] * a_half[1], axis))
    )
    proj_b = float(
        abs(np.dot(b_axes[0] * b_half[0], axis))
        + abs(np.dot(b_axes[1] * b_half[1], axis))
    )
    return proj_center <= (proj_a + proj_b)


def vehicle_collides(
    x: float,
    y: float,
    yaw: float,
    obstacles: Iterable[ObstacleBox],
    bounds: EnvironmentBounds,
    geometry: VehicleGeometry,
    margin: float,
) -> bool:
    # returns true if the car would be outside the world or touching an obstacle
    if not (bounds.min_x <= x <= bounds.max_x and bounds.min_y <= y <= bounds.max_y):
        return True

    vehicle_center = np.array([x, y], dtype=np.float32)
    vehicle_half = np.array(
        [geometry.length * 0.5 + margin, geometry.width * 0.5 + margin],
        dtype=np.float32,
    )
    cy = float(math.cos(yaw))
    sy = float(math.sin(yaw))
    vehicle_axes = (
        np.array([cy, sy], dtype=np.float32),
        np.array([-sy, cy], dtype=np.float32),
    )
    world_axes = (
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )

    for obstacle in obstacles:
        obstacle_center = np.array(
            [obstacle.center_x, obstacle.center_y], dtype=np.float32
        )
        obstacle_half = np.array(
            [obstacle.size_x * 0.5, obstacle.size_y * 0.5], dtype=np.float32
        )
        delta = obstacle_center - vehicle_center
        if all(
            _overlap_on_axis(
                delta,
                axis,
                vehicle_axes,
                vehicle_half,
                world_axes,
                obstacle_half,
            )
            for axis in (*vehicle_axes, *world_axes)
        ):
            return True
    return False


def path_is_collision_free(
    path: PlannedPath,
    obstacles: Iterable[ObstacleBox],
    bounds: EnvironmentBounds,
    geometry: VehicleGeometry,
    margin: float,
) -> bool:
    return all(
        not vehicle_collides(x, y, yaw, obstacles, bounds, geometry, margin)
        for x, y, yaw in zip(path.x, path.y, path.yaw)
    )


class HybridAStarPlanner:
    """hybrid a* planner"""

    def __init__(
        self,
        geometry: VehicleGeometry,
        bounds: EnvironmentBounds,
        config: PlannerConfig | None = None,
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds
        self.config = PlannerConfig() if config is None else config
        self._steer_values = np.unique(
            np.append(
                np.linspace(
                    -self.geometry.max_steering_angle,
                    self.geometry.max_steering_angle,
                    self.config.steer_samples,
                ),
                0.0,
            )
        )

    def plan(
        self,
        start: VehicleState,
        goal_xy: np.ndarray,
        obstacles: list[ObstacleBox],
    ) -> PlannedPath | None:
        # keeps a todo list of places to try next, picking the most promising one first
        start_node = _SearchNode(
            x_index=self._x_index(start.x),
            y_index=self._y_index(start.y),
            yaw_index=self._yaw_index(start.yaw),
            direction=1,
            x_values=[float(start.x)],
            y_values=[float(start.y)],
            yaw_values=[wrap_to_pi(start.yaw)],
            directions=[1],
            steer=0.0,
            cost=0.0,
            parent_key=None,
        )

        open_nodes: dict[tuple[int, int, int, int], _SearchNode] = {
            start_node.key(): start_node
        }
        closed_nodes: dict[tuple[int, int, int, int], _SearchNode] = {}
        queue: list[tuple[float, int, tuple[int, int, int, int]]] = []
        counter = 0
        heapq.heappush(
            queue,
            (
                self._priority(start_node, goal_xy),
                counter,
                start_node.key(),
            ),
        )
        counter += 1

        for _ in range(self.config.max_iterations):
            if not queue:
                return None

            _, _, current_key = heapq.heappop(queue)
            current = open_nodes.pop(current_key, None)
            if current is None:
                continue
            closed_nodes[current_key] = current

            if self._goal_reached(current, goal_xy):
                return self._build_path(closed_nodes, current, obstacles)

            analytic_node = self._try_analytic_connection(current, goal_xy, obstacles)
            if analytic_node is not None:
                closed_nodes[analytic_node.key()] = analytic_node
                return self._build_path(closed_nodes, analytic_node, obstacles)

            # from the current pose try short moves with different steering
            for steer in self._steer_values:
                for direction in (1, -1):
                    neighbor = self._simulate_motion(
                        current, float(steer), direction, obstacles
                    )
                    if neighbor is None:
                        continue
                    existing = closed_nodes.get(neighbor.key())
                    if existing is not None and existing.cost <= neighbor.cost:
                        continue
                    existing = open_nodes.get(neighbor.key())
                    if existing is None or neighbor.cost < existing.cost:
                        open_nodes[neighbor.key()] = neighbor
                        heapq.heappush(
                            queue,
                            (
                                self._priority(neighbor, goal_xy),
                                counter,
                                neighbor.key(),
                            ),
                        )
                        counter += 1
        return None

    def _simulate_motion(
        self,
        current: _SearchNode,
        steer: float,
        direction: int,
        obstacles: list[ObstacleBox],
        travel_distance: float | None = None,
    ) -> _SearchNode | None:
        # simulates driving a short distance and records the points along the way
        x = current.x
        y = current.y
        yaw = current.yaw
        x_values: list[float] = []
        y_values: list[float] = []
        yaw_values: list[float] = []
        direction_values: list[int] = []

        path_length = (
            self.config.primitive_length
            if travel_distance is None
            else max(float(travel_distance), self.config.motion_resolution)
        )
        signed_step = self.config.motion_resolution * direction
        steps = max(2, int(math.ceil(path_length / self.config.motion_resolution)))

        for _ in range(steps):
            x += signed_step * math.cos(yaw)
            y += signed_step * math.sin(yaw)
            yaw = wrap_to_pi(
                yaw + signed_step / self.geometry.wheelbase * math.tan(steer)
            )
            if vehicle_collides(
                x,
                y,
                yaw,
                obstacles,
                self.bounds,
                self.geometry,
                self.config.obstacle_margin,
            ):
                return None
            x_values.append(x)
            y_values.append(y)
            yaw_values.append(yaw)
            direction_values.append(direction)

        # score this move so the search prefers short, smooth, mostly forward driving
        transition_cost = path_length
        transition_cost += self.config.steer_cost * abs(steer)
        transition_cost += self.config.steer_change_cost * abs(current.steer - steer)
        if direction < 0:
            transition_cost += self.config.reverse_cost * path_length
        if direction != current.direction:
            transition_cost += self.config.direction_change_cost

        return _SearchNode(
            x_index=self._x_index(x),
            y_index=self._y_index(y),
            yaw_index=self._yaw_index(yaw),
            direction=direction,
            x_values=x_values,
            y_values=y_values,
            yaw_values=yaw_values,
            directions=direction_values,
            steer=steer,
            cost=current.cost + transition_cost,
            parent_key=current.key(),
        )

    def _try_analytic_connection(
        self,
        current: _SearchNode,
        goal_xy: np.ndarray,
        obstacles: list[ObstacleBox],
    ) -> _SearchNode | None:
        # when already near the goal try to finish in one smooth move
        goal_dx = float(goal_xy[0] - current.x)
        goal_dy = float(goal_xy[1] - current.y)
        goal_distance = math.hypot(goal_dx, goal_dy)
        if goal_distance > self.config.analytic_expansion_distance:
            return None

        goal_heading = math.atan2(goal_dy, goal_dx)
        alpha = wrap_to_pi(goal_heading - current.yaw)
        direction = 1 if abs(alpha) <= math.pi * 0.5 else -1
        if direction < 0:
            alpha = wrap_to_pi(goal_heading + math.pi - current.yaw)
        steer = math.atan2(
            2.0 * self.geometry.wheelbase * math.sin(alpha), max(goal_distance, 0.1)
        )
        steer = float(
            np.clip(
                steer,
                -self.geometry.max_steering_angle,
                self.geometry.max_steering_angle,
            )
        )
        candidate = self._simulate_motion(
            current,
            steer,
            direction,
            obstacles,
            travel_distance=goal_distance,
        )
        if candidate is None:
            return None
        if not self._goal_reached(candidate, goal_xy):
            return None
        return candidate

    def _build_path(
        self,
        closed_nodes: dict[tuple[int, int, int, int], _SearchNode],
        goal_node: _SearchNode,
        obstacles: list[ObstacleBox],
    ) -> PlannedPath:
        # rebuild the full route by walking backward through the search tree
        reversed_x: list[float] = []
        reversed_y: list[float] = []
        reversed_yaw: list[float] = []
        reversed_direction: list[int] = []

        current: _SearchNode | None = goal_node
        while current is not None:
            reversed_x.extend(reversed(current.x_values))
            reversed_y.extend(reversed(current.y_values))
            reversed_yaw.extend(reversed(current.yaw_values))
            reversed_direction.extend(reversed(current.directions))
            current = (
                None if current.parent_key is None else closed_nodes[current.parent_key]
            )

        x = np.asarray(list(reversed(reversed_x)), dtype=np.float32)
        y = np.asarray(list(reversed(reversed_y)), dtype=np.float32)
        yaw = np.asarray(list(reversed(reversed_yaw)), dtype=np.float32)
        directions = np.asarray(list(reversed(reversed_direction)), dtype=np.int8)

        x, y, yaw, directions = self._dedupe_samples(x, y, yaw, directions)
        if len(directions) > 1:
            directions[0] = directions[1]
        raw_path = PlannedPath(
            x=x, y=y, yaw=yaw, directions=directions, cost=goal_node.cost
        )
        return smooth_path(
            raw_path,
            obstacles=obstacles,
            bounds=self.bounds,
            geometry=self.geometry,
            config=self.config,
        )

    def _dedupe_samples(
        self,
        x: np.ndarray,
        y: np.ndarray,
        yaw: np.ndarray,
        directions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # clean up tiny repeats that can happen when stitching segments together
        keep_mask = np.ones(len(x), dtype=bool)
        for idx in range(1, len(x)):
            if (
                abs(float(x[idx] - x[idx - 1])) < 1e-4
                and abs(float(y[idx] - y[idx - 1])) < 1e-4
                and abs(float(wrap_to_pi(float(yaw[idx] - yaw[idx - 1])))) < 1e-4
            ):
                keep_mask[idx] = False
        return x[keep_mask], y[keep_mask], yaw[keep_mask], directions[keep_mask]

    def _goal_reached(self, node: _SearchNode, goal_xy: np.ndarray) -> bool:
        return (
            math.hypot(float(goal_xy[0] - node.x), float(goal_xy[1] - node.y))
            <= self.config.goal_tolerance
        )

    def _priority(self, node: _SearchNode, goal_xy: np.ndarray) -> float:
        # prefers nodes that are cheaper so far and closer to the goal
        goal_dx = float(goal_xy[0] - node.x)
        goal_dy = float(goal_xy[1] - node.y)
        distance_cost = math.hypot(goal_dx, goal_dy)
        heading_cost = abs(wrap_to_pi(math.atan2(goal_dy, goal_dx) - node.yaw))
        return (
            node.cost
            + self.config.heuristic_weight * distance_cost
            + 0.15 * heading_cost
        )

    def _x_index(self, x: float) -> int:
        return int(round(x / self.config.xy_resolution))

    def _y_index(self, y: float) -> int:
        return int(round(y / self.config.xy_resolution))

    def _yaw_index(self, yaw: float) -> int:
        return int(round(wrap_to_pi(yaw) / self.config.yaw_resolution))


def smooth_path(
    path: PlannedPath,
    obstacles: list[ObstacleBox],
    bounds: EnvironmentBounds,
    geometry: VehicleGeometry,
    config: PlannerConfig,
) -> PlannedPath:
    # smoothing is optional and only used for forward routes
    # it makes the path less jagged, but it is skipped for reverse driving for now
    if len(path.x) < 4 or np.any(path.directions < 0):
        return path

    dx = np.diff(path.x)
    dy = np.diff(path.y)
    segment_lengths = np.hypot(dx, dy)
    keep_mask = np.concatenate(([True], segment_lengths > 1e-4))
    x = path.x[keep_mask]
    y = path.y[keep_mask]
    if len(x) < 4:
        return path

    arc_lengths = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    total_length = float(arc_lengths[-1])
    if total_length < 1e-5:
        return path

    # fits a smooth curve through the points and then resamples it evenly
    degree = min(3, len(x) - 1)
    smoothing = config.smoothing_factor * len(x)
    spline_x = UnivariateSpline(arc_lengths, x, k=degree, s=smoothing)
    spline_y = UnivariateSpline(arc_lengths, y, k=degree, s=smoothing)

    sample_count = max(
        len(path.x) * 3,
        int(math.ceil(total_length / config.smoothing_resolution)) + 1,
    )
    sampled = np.linspace(0.0, total_length, sample_count)
    smooth_x = spline_x(sampled).astype(np.float32)
    smooth_y = spline_y(sampled).astype(np.float32)
    smooth_x[0] = path.x[0]
    smooth_y[0] = path.y[0]
    smooth_x[-1] = path.x[-1]
    smooth_y[-1] = path.y[-1]

    dx_ds = spline_x.derivative(1)(sampled)
    dy_ds = spline_y.derivative(1)(sampled)
    smooth_yaw = np.arctan2(dy_ds, dx_ds).astype(np.float32)
    if len(smooth_yaw) >= 2:
        smooth_yaw[0] = math.atan2(
            float(smooth_y[1] - smooth_y[0]),
            float(smooth_x[1] - smooth_x[0]),
        )
        smooth_yaw[-1] = math.atan2(
            float(smooth_y[-1] - smooth_y[-2]),
            float(smooth_x[-1] - smooth_x[-2]),
        )

    smooth_path_candidate = PlannedPath(
        x=smooth_x,
        y=smooth_y,
        yaw=smooth_yaw,
        directions=np.ones(sample_count, dtype=np.int8),
        cost=path.cost,
    )
    if path_is_collision_free(
        smooth_path_candidate,
        obstacles=obstacles,
        bounds=bounds,
        geometry=geometry,
        margin=config.obstacle_margin,
    ):
        return smooth_path_candidate
    return path


class PurePursuitController:
    def __init__(
        self,
        geometry: VehicleGeometry,
        config: PurePursuitConfig | None = None,
    ) -> None:
        self.geometry = geometry
        self.config = PurePursuitConfig() if config is None else config
        self._target_index = 0

    def reset(self) -> None:
        self._target_index = 0

    def at_end(self, path: PlannedPath) -> bool:
        return self._target_index >= len(path.x) - 1

    def control(self, state: VehicleState, path: PlannedPath) -> tuple[float, float]:
        # pick a point ahead on the route and steer toward it
        # then picks a reasonable speed so the car slows near the end and near tight turns
        target_index, lookahead, segment_end = self._search_target_index(state, path)
        direction = int(path.directions[min(target_index, len(path.directions) - 1)])
        rear_x = state.x - direction * (self.geometry.wheelbase * 0.5) * math.cos(
            state.yaw
        )
        rear_y = state.y - direction * (self.geometry.wheelbase * 0.5) * math.sin(
            state.yaw
        )
        target_x = float(path.x[target_index])
        target_y = float(path.y[target_index])
        alpha = wrap_to_pi(math.atan2(target_y - rear_y, target_x - rear_x) - state.yaw)
        steering = direction * math.atan2(
            2.0 * self.geometry.wheelbase * math.sin(alpha),
            max(lookahead, 1e-3),
        )
        steering = float(
            np.clip(
                steering,
                -self.geometry.max_steering_angle,
                self.geometry.max_steering_angle,
            )
        )

        distance_to_goal = math.hypot(
            float(path.x[-1] - state.x), float(path.y[-1] - state.y)
        )
        target_speed = (
            self.config.slow_speed
            if distance_to_goal < self.config.goal_slowdown_distance
            else self.config.nominal_speed
        )
        if segment_end + 1 < len(path.x):
            distance_to_cusp = math.hypot(
                float(path.x[segment_end] - state.x),
                float(path.y[segment_end] - state.y),
            )
            if distance_to_cusp < self.config.cusp_slowdown_distance:
                target_speed = min(target_speed, self.config.slow_speed)
        target_speed *= direction
        target_speed *= max(
            0.45,
            1.0 - 0.35 * abs(steering) / max(self.geometry.max_steering_angle, 1e-6),
        )
        speed_error = target_speed - state.speed
        wheel_speed = (
            target_speed + self.config.speed_gain * speed_error
        ) / self.geometry.wheel_radius
        if direction >= 0:
            wheel_speed = float(
                np.clip(wheel_speed, 0.0, self.config.max_forward_wheel_speed)
            )
        else:
            wheel_speed = float(
                np.clip(wheel_speed, -self.config.max_reverse_wheel_speed, 0.0)
            )
        return wheel_speed, steering

    def _search_target_index(
        self,
        state: VehicleState,
        path: PlannedPath,
    ) -> tuple[int, float, int]:
        # keep moving the target forward as the car progresses along the route
        # if the route switches direction, it waits until the car is close to the switch point
        if self._target_index >= len(path.x):
            self._target_index = len(path.x) - 1

        while True:
            search_direction = int(
                path.directions[min(self._target_index, len(path.directions) - 1)]
            )
            segment_end = self._segment_end_index(path, self._target_index)
            rear_x = state.x - search_direction * (
                self.geometry.wheelbase * 0.5
            ) * math.cos(state.yaw)
            rear_y = state.y - search_direction * (
                self.geometry.wheelbase * 0.5
            ) * math.sin(state.yaw)

            nearest_index = self._target_index
            nearest_distance = math.hypot(
                float(path.x[nearest_index] - rear_x),
                float(path.y[nearest_index] - rear_y),
            )
            for idx in range(self._target_index + 1, segment_end + 1):
                candidate_distance = math.hypot(
                    float(path.x[idx] - rear_x),
                    float(path.y[idx] - rear_y),
                )
                if (
                    candidate_distance > nearest_distance
                    and idx > self._target_index + 2
                ):
                    break
                if candidate_distance < nearest_distance:
                    nearest_index = idx
                    nearest_distance = candidate_distance

            self._target_index = nearest_index
            distance_to_cusp = math.hypot(
                float(path.x[segment_end] - rear_x),
                float(path.y[segment_end] - rear_y),
            )
            if (
                self._target_index >= segment_end
                and segment_end + 1 < len(path.x)
                and distance_to_cusp <= self.config.cusp_switch_distance
            ):
                self._target_index = segment_end + 1
                continue

            lookahead = float(
                np.clip(
                    self.config.lookahead_gain * abs(state.speed)
                    + self.config.min_lookahead,
                    self.config.min_lookahead,
                    self.config.max_lookahead,
                )
            )

            while self._target_index + 1 <= segment_end:
                distance = math.hypot(
                    float(path.x[self._target_index] - rear_x),
                    float(path.y[self._target_index] - rear_y),
                )
                if distance >= lookahead:
                    break
                self._target_index += 1
            return self._target_index, lookahead, segment_end

    def _segment_end_index(self, path: PlannedPath, start_index: int) -> int:
        direction = int(path.directions[start_index])
        end_index = start_index
        while (
            end_index + 1 < len(path.directions)
            and int(path.directions[end_index + 1]) == direction
        ):
            end_index += 1
        return end_index


class TeacherPlanner:
    def __init__(
        self,
        geometry: VehicleGeometry,
        bounds: EnvironmentBounds,
        planner_config: PlannerConfig | None = None,
        controller_config: PurePursuitConfig | None = None,
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds
        self.planner_config = (
            PlannerConfig() if planner_config is None else planner_config
        )
        self.controller = PurePursuitController(geometry, controller_config)
        self.planner = HybridAStarPlanner(geometry, bounds, self.planner_config)
        self.path: PlannedPath | None = None
        self.goal_xy: np.ndarray | None = None

    def reset(self) -> None:
        self.path = None
        self.goal_xy = None
        self.controller.reset()

    def compute_action(
        self,
        state: VehicleState,
        goal_xy: np.ndarray,
        obstacles: list[ObstacleBox],
    ) -> tuple[float, float]:
        goal_xy = np.asarray(goal_xy, dtype=np.float32)
        if self.path is None or self._needs_replan(state, goal_xy):
            self.path = self.planner.plan(state, goal_xy, obstacles)
            self.goal_xy = goal_xy.copy()
            self.controller.reset()

        if self.path is None:
            raise RuntimeError(
                f"could not find a path from ({state.x:.2f}, {state.y:.2f}) "
                f"to ({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})."
            )

        # check if close enough to goal stop dirving
        if (
            math.hypot(float(goal_xy[0] - state.x), float(goal_xy[1] - state.y))
            <= self.planner_config.goal_tolerance
        ):
            return (0.0, 0.0)

        throttle, steering = self.controller.control(state, self.path)
        return (throttle, steering)

    def _needs_replan(self, state: VehicleState, goal_xy: np.ndarray) -> bool:
        if self.path is None or self.goal_xy is None:
            return True
        if float(np.linalg.norm(goal_xy - self.goal_xy)) > 1e-3:
            return True
        if (
            self.path.distance_to_point(state.x, state.y)
            > self.planner_config.replan_distance
        ):
            return True
        if self.controller.at_end(self.path):
            return True
        return False
