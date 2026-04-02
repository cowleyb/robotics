from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize


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
    collision_center_offset: float = 0.0
    rear_axle_offset: float = 0.0


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
        dx = self.x - x
        dy = self.y - y
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
    obstacle_margin: float = 0.10  # extra space around obstacles
    steer_cost: float = 0.35  # for turns
    steer_change_cost: float = 0.6  # for wheel back and forth
    reverse_cost: float = 2.0
    direction_change_cost: float = 3.0
    heuristic_weight: float = 1.8  # how strongly this aims toward the goal
    smoothing_factor: float = 0.02  # how much this rounds off the final route
    smoothing_resolution: float = 0.05  # spacing for the rounded off route
    replan_distance: float = 0.25
    escape_max_primitives: int = 3  # if the spawn starts inside the safety margin, allow a short physical-footprint escape first


@dataclass(frozen=True)
class KinematicMPCConfig:
    horizon_steps: int = 6
    control_interval_steps: int = 5  # solve once and reuse it for a few sim ticks
    sim_dt: float = 0.01
    reference_spacing: float = (
        0.08  # keep preview close enough to avoid cutting inside tight turns
    )
    nominal_speed: float = 1.0  # normal cruising speed
    slow_speed: float = 0.35  # careful speed near tricky spots
    goal_slowdown_distance: float = 0.8  # start slowing as the car gets near the goal
    cusp_slowdown_distance: float = 0.6  # slow earlier near a forward to reverse switch
    cusp_switch_distance: float = 0.16  # accept the switch a bit sooner when close
    actuator_response: float = 0.2  # matches the throttle smoothing in sim.world
    position_cost: float = 16.0
    yaw_cost: float = 4.0
    speed_cost: float = 0.8
    wheel_speed_cost: float = 0.02
    wheel_speed_rate_cost: float = 0.08
    steering_cost: float = 0.05
    steering_rate_cost: float = 0.9
    terminal_position_cost: float = 28.0
    terminal_yaw_cost: float = 6.0
    optimizer_maxiter: int = 14
    max_forward_wheel_speed: float = 20.0
    max_reverse_wheel_speed: float = 8.0


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
    cy = float(math.cos(yaw))
    sy = float(math.sin(yaw))
    vehicle_center = np.array(
        [
            x + geometry.collision_center_offset * cy,
            y + geometry.collision_center_offset * sy,
        ],
        dtype=np.float32,
    )
    vehicle_half = np.array(
        [geometry.length * 0.5 + margin, geometry.width * 0.5 + margin],
        dtype=np.float32,
    )
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
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds
        self.config = PlannerConfig()
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
        escape_prefix: PlannedPath | None = None
        if vehicle_collides(
            start.x,
            start.y,
            start.yaw,
            obstacles,
            self.bounds,
            self.geometry,
            self.config.obstacle_margin,
        ):
            if vehicle_collides(
                start.x,
                start.y,
                start.yaw,
                obstacles,
                self.bounds,
                self.geometry,
                0.0,
            ):
                return None
            escape_result = self._plan_buffer_escape(start_node, obstacles)
            if escape_result is None:
                return None
            escape_prefix, escape_end = escape_result
            start_node = _SearchNode(
                x_index=self._x_index(escape_end.x),
                y_index=self._y_index(escape_end.y),
                yaw_index=self._yaw_index(escape_end.yaw),
                direction=escape_end.direction,
                x_values=[float(escape_end.x)],
                y_values=[float(escape_end.y)],
                yaw_values=[wrap_to_pi(escape_end.yaw)],
                directions=[int(escape_end.direction)],
                steer=escape_end.steer,
                cost=escape_end.cost,
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
                return self._prepend_path(
                    escape_prefix,
                    self._build_path(closed_nodes, current, obstacles),
                )

            analytic_node = self._try_analytic_connection(current, goal_xy, obstacles)
            if analytic_node is not None:
                closed_nodes[analytic_node.key()] = analytic_node
                return self._prepend_path(
                    escape_prefix,
                    self._build_path(closed_nodes, analytic_node, obstacles),
                )

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
        collision_margin: float | None = None,
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
        margin = (
            self.config.obstacle_margin
            if collision_margin is None
            else float(collision_margin)
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
                margin,
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

    def _plan_buffer_escape(
        self,
        start_node: _SearchNode,
        obstacles: list[ObstacleBox],
    ) -> tuple[PlannedPath, _SearchNode] | None:
        frontier: list[tuple[list[_SearchNode], _SearchNode]] = [([], start_node)]
        for _ in range(self.config.escape_max_primitives):
            escape_candidates: list[tuple[float, list[_SearchNode]]] = []
            next_frontier: list[tuple[float, list[_SearchNode], _SearchNode]] = []
            for chain, current in frontier:
                for steer in self._steer_values:
                    for direction in (1, -1):
                        neighbor = self._simulate_motion(
                            current,
                            float(steer),
                            direction,
                            obstacles,
                            collision_margin=0.0,
                        )
                        if neighbor is None:
                            continue
                        neighbor_chain = [*chain, neighbor]
                        if not vehicle_collides(
                            neighbor.x,
                            neighbor.y,
                            neighbor.yaw,
                            obstacles,
                            self.bounds,
                            self.geometry,
                            self.config.obstacle_margin,
                        ):
                            escape_candidates.append((neighbor.cost, neighbor_chain))
                            continue
                        next_frontier.append((neighbor.cost, neighbor_chain, neighbor))
            if escape_candidates:
                _, best_chain = min(escape_candidates, key=lambda item: item[0])
                return (
                    self._build_chain_path(start_node, best_chain),
                    best_chain[-1],
                )
            next_frontier.sort(key=lambda item: item[0])
            frontier = [(chain, node) for _, chain, node in next_frontier]
        return None

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
        if abs(alpha) <= math.pi * 0.5:
            direction = 1
        else:
            direction = -1
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
        if candidate is not None and self._goal_reached(candidate, goal_xy):
            return candidate
        return None

    def _build_path(
        self,
        closed_nodes: dict[tuple[int, int, int, int], _SearchNode],
        goal_node: _SearchNode,
        obstacles: list[ObstacleBox],
    ) -> PlannedPath:
        # rebuild the full route by walking backward through the search tree
        raw_path = self._build_raw_path(closed_nodes, goal_node)
        return smooth_path(
            raw_path,
            obstacles=obstacles,
            bounds=self.bounds,
            geometry=self.geometry,
            config=self.config,
        )

    def _build_raw_path(
        self,
        closed_nodes: dict[tuple[int, int, int, int], _SearchNode],
        goal_node: _SearchNode,
    ) -> PlannedPath:
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
        return PlannedPath(
            x=x, y=y, yaw=yaw, directions=directions, cost=goal_node.cost
        )

    def _build_chain_path(
        self,
        start_node: _SearchNode,
        chain: list[_SearchNode],
    ) -> PlannedPath:
        x_values = [float(start_node.x)]
        y_values = [float(start_node.y)]
        yaw_values = [wrap_to_pi(start_node.yaw)]
        direction_values = [int(start_node.direction)]
        for node in chain:
            x_values.extend(node.x_values)
            y_values.extend(node.y_values)
            yaw_values.extend(node.yaw_values)
            direction_values.extend(node.directions)
        x = np.asarray(x_values, dtype=np.float32)
        y = np.asarray(y_values, dtype=np.float32)
        yaw = np.asarray(yaw_values, dtype=np.float32)
        directions = np.asarray(direction_values, dtype=np.int8)
        x, y, yaw, directions = self._dedupe_samples(x, y, yaw, directions)
        if len(directions) > 1:
            directions[0] = directions[1]
        return PlannedPath(
            x=x,
            y=y,
            yaw=yaw,
            directions=directions,
            cost=chain[-1].cost,
        )

    def _prepend_path(
        self,
        prefix: PlannedPath | None,
        path: PlannedPath,
    ) -> PlannedPath:
        if prefix is None:
            return path
        drop_first = (
            len(path.x) > 0
            and abs(float(path.x[0] - prefix.x[-1])) < 1e-4
            and abs(float(path.y[0] - prefix.y[-1])) < 1e-4
            and abs(wrap_to_pi(float(path.yaw[0] - prefix.yaw[-1]))) < 1e-4
        )
        start_index = 1 if drop_first else 0
        x = np.concatenate((prefix.x, path.x[start_index:])).astype(np.float32)
        y = np.concatenate((prefix.y, path.y[start_index:])).astype(np.float32)
        yaw = np.concatenate((prefix.yaw, path.yaw[start_index:])).astype(np.float32)
        directions = np.concatenate(
            (prefix.directions, path.directions[start_index:])
        ).astype(np.int8)
        x, y, yaw, directions = self._dedupe_samples(x, y, yaw, directions)
        if len(directions) > 1:
            directions[0] = directions[1]
        return PlannedPath(x=x, y=y, yaw=yaw, directions=directions, cost=path.cost)

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
                abs(x[idx] - x[idx - 1]) < 1e-4
                and abs(y[idx] - y[idx - 1]) < 1e-4
                and abs(wrap_to_pi(yaw[idx] - yaw[idx - 1])) < 1e-4
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
    max_feasible_curvature = abs(math.tan(geometry.max_steering_angle)) / max(
        geometry.wheelbase, 1e-6
    )
    if _path_max_curvature(smooth_path_candidate) > max_feasible_curvature + 1e-3:
        return path
    if path_is_collision_free(
        smooth_path_candidate,
        obstacles=obstacles,
        bounds=bounds,
        geometry=geometry,
        margin=config.obstacle_margin,
    ):
        return smooth_path_candidate
    return path


def _path_max_curvature(path: PlannedPath) -> float:
    if len(path.x) < 2:
        return 0.0

    segment_lengths = np.hypot(np.diff(path.x), np.diff(path.y))
    valid_segments = segment_lengths > 1e-6
    if not np.any(valid_segments):
        return 0.0

    yaw_deltas = np.asarray(
        [
            wrap_to_pi(float(path.yaw[idx + 1] - path.yaw[idx]))
            for idx in range(len(path.yaw) - 1)
        ],
        dtype=np.float32,
    )
    curvatures = np.abs(yaw_deltas[valid_segments]) / segment_lengths[valid_segments]
    if len(curvatures) == 0:
        return 0.0
    return float(np.max(curvatures))


class KinematicMPCController:
    def __init__(
        self,
        geometry: VehicleGeometry,
    ) -> None:
        self.geometry = geometry
        self.config = KinematicMPCConfig()
        self._target_index = 0
        self._steps_until_resolve = 0
        self._cached_command = (0.0, 0.0)
        self._cached_segment_end = 0
        self._cached_direction = 0
        self._last_solution: np.ndarray | None = None
        self._last_wheel_speed = 0.0
        self._last_steering = 0.0

    def reset(self) -> None:
        self._target_index = 0
        self._steps_until_resolve = 0
        self._cached_command = (0.0, 0.0)
        self._cached_segment_end = 0
        self._cached_direction = 0
        self._last_solution = None
        self._last_wheel_speed = 0.0
        self._last_steering = 0.0

    def at_end(self, path: PlannedPath) -> bool:
        return self._target_index >= len(path.x) - 1

    def control(self, state: VehicleState, path: PlannedPath) -> tuple[float, float]:
        target_index, segment_end = self._search_target_index(state, path)
        direction = int(path.directions[target_index])

        if (
            self._steps_until_resolve > 0
            and self._cached_segment_end == segment_end
            and self._cached_direction == direction
        ):
            self._steps_until_resolve -= 1
            return self._cached_command

        target_wheel_speed = self._target_wheel_speed(
            state, path, segment_end, direction
        )
        reference = self._build_reference(
            path=path,
            start_index=target_index,
            segment_end=segment_end,
            target_speed=target_wheel_speed * self.geometry.wheel_radius,
        )
        wheel_speed, steering, solution = self._solve_mpc(
            state=state,
            direction=direction,
            target_wheel_speed=target_wheel_speed,
            reference=reference,
        )

        self._last_solution = solution
        self._last_wheel_speed = wheel_speed
        self._last_steering = steering
        self._cached_command = (wheel_speed, steering)
        self._cached_segment_end = segment_end
        self._cached_direction = direction
        self._steps_until_resolve = self.config.control_interval_steps - 1
        return self._cached_command

    def _solve_mpc(
        self,
        state: VehicleState,
        direction: int,
        target_wheel_speed: float,
        reference: list[tuple[float, float, float, float]],
    ) -> tuple[float, float, np.ndarray]:
        horizon_steps = self.config.horizon_steps
        if direction >= 0:
            wheel_bounds = (0.0, self.config.max_forward_wheel_speed)
        else:
            wheel_bounds = (-self.config.max_reverse_wheel_speed, 0.0)
        bounds = [wheel_bounds] * horizon_steps + [
            (-self.geometry.max_steering_angle, self.geometry.max_steering_angle)
        ] * horizon_steps

        initial_guess = self._initial_guess(
            target_wheel_speed=target_wheel_speed,
            horizon_steps=horizon_steps,
        )
        objective = lambda controls: self._objective(
            controls=controls,
            state=state,
            reference=reference,
            target_wheel_speed=target_wheel_speed,
        )
        result = minimize(
            objective,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self.config.optimizer_maxiter},
        )
        solution = (
            np.asarray(result.x, dtype=np.float64)
            if np.all(np.isfinite(result.x))
            else initial_guess
        )
        wheel_speed = float(np.clip(solution[0], *wheel_bounds))
        steering = float(
            np.clip(
                solution[horizon_steps],
                -self.geometry.max_steering_angle,
                self.geometry.max_steering_angle,
            )
        )
        return (wheel_speed, steering, solution)

    def _objective(
        self,
        controls: np.ndarray,
        state: VehicleState,
        reference: list[tuple[float, float, float, float]],
        target_wheel_speed: float,
    ) -> float:
        horizon_steps = self.config.horizon_steps
        wheel_speeds = controls[:horizon_steps]
        steerings = controls[horizon_steps:]

        x = state.x
        y = state.y
        yaw = state.yaw
        applied_wheel_speed = state.speed / max(self.geometry.wheel_radius, 1e-6)
        previous_wheel_speed = self._last_wheel_speed
        previous_steering = self._last_steering
        cost = 0.0

        for step in range(horizon_steps):
            x, y, yaw, applied_wheel_speed = self._rollout_interval(
                x=x,
                y=y,
                yaw=yaw,
                applied_wheel_speed=applied_wheel_speed,
                wheel_speed_command=float(wheel_speeds[step]),
                steering=float(steerings[step]),
            )
            reference_x, reference_y, reference_yaw, reference_speed = reference[step]
            position_error = (x - reference_x) ** 2 + (y - reference_y) ** 2
            yaw_error = wrap_to_pi(yaw - reference_yaw)
            speed = applied_wheel_speed * self.geometry.wheel_radius
            speed_error = speed - reference_speed

            position_cost = (
                self.config.terminal_position_cost
                if step == horizon_steps - 1
                else self.config.position_cost
            )
            yaw_cost = (
                self.config.terminal_yaw_cost
                if step == horizon_steps - 1
                else self.config.yaw_cost
            )
            cost += position_cost * position_error
            cost += yaw_cost * yaw_error * yaw_error
            cost += self.config.speed_cost * speed_error * speed_error
            cost += (
                self.config.wheel_speed_cost
                * (wheel_speeds[step] - target_wheel_speed) ** 2
            )
            cost += (
                self.config.wheel_speed_rate_cost
                * (wheel_speeds[step] - previous_wheel_speed) ** 2
            )
            cost += self.config.steering_cost * steerings[step] * steerings[step]
            cost += (
                self.config.steering_rate_cost
                * (steerings[step] - previous_steering) ** 2
            )

            previous_wheel_speed = float(wheel_speeds[step])
            previous_steering = float(steerings[step])

        return float(cost)

    def _rollout_interval(
        self,
        x: float,
        y: float,
        yaw: float,
        applied_wheel_speed: float,
        wheel_speed_command: float,
        steering: float,
    ) -> tuple[float, float, float, float]:
        steering = float(
            np.clip(
                steering,
                -self.geometry.max_steering_angle,
                self.geometry.max_steering_angle,
            )
        )
        for _ in range(self.config.control_interval_steps):
            applied_wheel_speed += self.config.actuator_response * (
                wheel_speed_command - applied_wheel_speed
            )
            speed = applied_wheel_speed * self.geometry.wheel_radius
            x += speed * math.cos(yaw) * self.config.sim_dt
            y += speed * math.sin(yaw) * self.config.sim_dt
            yaw = wrap_to_pi(
                yaw
                + speed
                / max(self.geometry.wheelbase, 1e-6)
                * math.tan(steering)
                * self.config.sim_dt
            )
        return (x, y, yaw, applied_wheel_speed)

    def _initial_guess(
        self,
        target_wheel_speed: float,
        horizon_steps: int,
    ) -> np.ndarray:
        if (
            self._last_solution is not None
            and len(self._last_solution) == 2 * horizon_steps
        ):
            previous_wheels = self._last_solution[:horizon_steps]
            previous_steerings = self._last_solution[horizon_steps:]
            wheel_guess = np.concatenate((previous_wheels[1:], previous_wheels[-1:]))
            steering_guess = np.concatenate(
                (previous_steerings[1:], previous_steerings[-1:])
            )
        else:
            wheel_guess = np.full(horizon_steps, target_wheel_speed, dtype=np.float64)
            steering_guess = np.full(
                horizon_steps, self._last_steering, dtype=np.float64
            )
        return np.concatenate((wheel_guess, steering_guess))

    def _build_reference(
        self,
        path: PlannedPath,
        start_index: int,
        segment_end: int,
        target_speed: float,
    ) -> list[tuple[float, float, float, float]]:
        reference: list[tuple[float, float, float, float]] = []
        reference_index = start_index
        step_distance = max(
            self.config.reference_spacing,
            abs(target_speed) * self.config.sim_dt * self.config.control_interval_steps,
        )
        for _ in range(self.config.horizon_steps):
            reference_index = self._advance_reference_index(
                path=path,
                start_index=reference_index,
                end_index=segment_end,
                step_distance=step_distance,
            )
            reference.append(
                (
                    float(path.x[reference_index]),
                    float(path.y[reference_index]),
                    float(path.yaw[reference_index]),
                    float(target_speed),
                )
            )
        return reference

    def _advance_reference_index(
        self,
        path: PlannedPath,
        start_index: int,
        end_index: int,
        step_distance: float,
    ) -> int:
        index = start_index
        remaining = step_distance
        while index < end_index and remaining > 0.0:
            segment_distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            index += 1
            remaining -= max(segment_distance, 1e-4)
        return index

    def _target_wheel_speed(
        self,
        state: VehicleState,
        path: PlannedPath,
        segment_end: int,
        direction: int,
    ) -> float:
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
        wheel_speed = target_speed / max(self.geometry.wheel_radius, 1e-6)
        if direction >= 0:
            return float(np.clip(wheel_speed, 0.0, self.config.max_forward_wheel_speed))
        return float(np.clip(wheel_speed, -self.config.max_reverse_wheel_speed, 0.0))

    def _search_target_index(
        self,
        state: VehicleState,
        path: PlannedPath,
    ) -> tuple[int, int]:
        if self._target_index >= len(path.x):
            self._target_index = len(path.x) - 1

        while True:
            search_direction = int(
                path.directions[min(self._target_index, len(path.directions) - 1)]
            )
            segment_end = self._segment_end_index(path, self._target_index)
            rear_x, rear_y = self._tracking_point(state, search_direction)

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
                self._steps_until_resolve = 0
                self._last_solution = None
                continue

            return self._target_index, segment_end

    def _tracking_point(
        self, state: VehicleState, direction: int
    ) -> tuple[float, float]:
        return (state.x, state.y)

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
    ) -> None:
        self.geometry = geometry
        self.bounds = bounds
        self.controller = KinematicMPCController(geometry)
        self.planner = HybridAStarPlanner(geometry, bounds)
        self.planner_config = self.planner.config
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
        if (
            math.hypot(float(goal_xy[0] - state.x), float(goal_xy[1] - state.y))
            <= self.planner_config.goal_tolerance
        ):
            return (0.0, 0.0)
        planner_state = self._planner_state(state)

        if self.path is None or self._needs_replan(planner_state, goal_xy):
            self.path = self.planner.plan(planner_state, goal_xy, obstacles)
            self.goal_xy = goal_xy.copy()
            self.controller.reset()

        if self.path is None:
            raise RuntimeError(
                f"could not find a path from ({planner_state.x:.2f}, {planner_state.y:.2f}) "
                f"to ({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})."
            )

        throttle, steering = self.controller.control(planner_state, self.path)
        return (throttle, steering)

    def _needs_replan(self, state: VehicleState, goal_xy: np.ndarray) -> bool:
        return (
            self.path is None
            or self.goal_xy is None
            or float(np.linalg.norm(goal_xy - self.goal_xy)) > 1e-3
            or self.path.distance_to_point(state.x, state.y)
            > self.planner_config.replan_distance
        )

    def _planner_state(self, state: VehicleState) -> VehicleState:
        if abs(self.geometry.rear_axle_offset) < 1e-6:
            return state
        return VehicleState(
            x=state.x - self.geometry.rear_axle_offset * math.cos(state.yaw),
            y=state.y - self.geometry.rear_axle_offset * math.sin(state.yaw),
            yaw=state.yaw,
            speed=state.speed,
        )
