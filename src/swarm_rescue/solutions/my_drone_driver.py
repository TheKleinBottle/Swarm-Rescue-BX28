import math
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor
from swarm_rescue.simulation.utils.utils import normalize_angle

GRID_RESOLUTION = 20.0
FRONTIER_SEARCH_RADIUS = 35
FRONTIER_DISTANCE_WEIGHT = 1.0
FRONTIER_REPULSION_WEIGHT = 60.0
FRONTIER_VISIT_WEIGHT = 4.0
AGENT_REPULSION_RADIUS = 140.0
VICTIM_ASSIGNMENT_MARGIN = 20.0
VICTIM_GRAB_DISTANCE = 22.0
RESCUE_DROP_DISTANCE = 25.0
STALE_MESSAGE_TICKS = 150


class MyStatefulDrone(DroneAbstract):
    def __init__(self, identifier: Optional[int] = None, **kwargs):
        super().__init__(identifier=identifier, **kwargs)
        self.estimated_pos = np.array([0.0, 0.0])
        self.estimated_angle = 0.0
        self.last_gps = None

        self.occupancy: Dict[Tuple[int, int], int] = {}
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        self.known_agents: Dict[int, Tuple[np.ndarray, int]] = {}
        self.known_victims: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}

        self.state = "EXPLORE"
        self.current_target: Optional[np.ndarray] = None
        self.rescue_center_pos: Optional[np.ndarray] = None
        self.target_victim_key: Optional[Tuple[int, int]] = None

    def reset(self):
        self.estimated_pos = np.array([0.0, 0.0])
        self.estimated_angle = 0.0
        self.last_gps = None

        self.occupancy.clear()
        self.visit_counts.clear()
        self.known_agents.clear()
        self.known_victims.clear()

        self.state = "EXPLORE"
        self.current_target = None
        self.rescue_center_pos = None
        self.target_victim_key = None

    def world_to_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        return (int(math.floor(pos[0] / GRID_RESOLUTION)),
                int(math.floor(pos[1] / GRID_RESOLUTION)))

    def cell_to_world(self, cell: Tuple[int, int]) -> np.ndarray:
        return np.array([
            (cell[0] + 0.5) * GRID_RESOLUTION,
            (cell[1] + 0.5) * GRID_RESOLUTION,
        ])

    def update_estimate(self) -> None:
        gps_pos = self.measured_gps_position()
        compass_angle = self.measured_compass_angle()
        if gps_pos is not None and compass_angle is not None:
            self.estimated_pos = gps_pos
            self.estimated_angle = compass_angle
            self.last_gps = gps_pos
            return

        odom = self.odometer_values()
        if odom is not None:
            dist, alpha, theta = odom[0], odom[1], odom[2]
            move_angle = self.estimated_angle + alpha
            self.estimated_pos[0] += dist * math.cos(move_angle)
            self.estimated_pos[1] += dist * math.sin(move_angle)
            self.estimated_angle = normalize_angle(self.estimated_angle + theta)

    def mark_cell(self, cell: Tuple[int, int], value: int) -> None:
        current = self.occupancy.get(cell)
        if current is None or (current == 0 and value == 1):
            self.occupancy[cell] = value

    def update_grid_from_lidar(self) -> None:
        if self.lidar_is_disabled():
            return
        lidar_values = self.lidar_values()
        ray_angles = self.lidar_rays_angles()
        if lidar_values is None or ray_angles is None:
            return

        step = GRID_RESOLUTION * 0.5
        max_range = self.lidar().max_range

        for distance, angle in zip(lidar_values, ray_angles):
            if distance <= 0:
                continue
            ray_angle = self.estimated_angle + angle
            max_step = min(distance, max_range)
            steps = int(max_step / step)
            for idx in range(steps):
                sample_dist = idx * step
                sample_pos = self.estimated_pos + np.array([
                    sample_dist * math.cos(ray_angle),
                    sample_dist * math.sin(ray_angle),
                ])
                self.mark_cell(self.world_to_cell(sample_pos), 0)
            if distance < max_range * 0.98:
                hit_pos = self.estimated_pos + np.array([
                    distance * math.cos(ray_angle),
                    distance * math.sin(ray_angle),
                ])
                self.mark_cell(self.world_to_cell(hit_pos), 1)

    def update_visits(self) -> None:
        cell = self.world_to_cell(self.estimated_pos)
        self.mark_cell(cell, 0)
        self.visit_counts[cell] += 1

    def update_semantic(self) -> None:
        if self.semantic_is_disabled():
            return
        semantic_data = self.semantic_values()
        if not semantic_data:
            return

        for data in semantic_data:
            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                angle_global = self.estimated_angle + data.angle
                rx = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                ry = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                self.rescue_center_pos = np.array([rx, ry])
            if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                angle_global = self.estimated_angle + data.angle
                px = self.estimated_pos[0] + data.distance * math.cos(angle_global)
                py = self.estimated_pos[1] + data.distance * math.sin(angle_global)
                key = self.world_to_cell(np.array([px, py]))
                self.known_victims[key] = (np.array([px, py]), self.elapsed_timestep)

    def prune_messages(self) -> None:
        stale_limit = self.elapsed_timestep - STALE_MESSAGE_TICKS
        self.known_agents = {
            agent_id: (pos, ts)
            for agent_id, (pos, ts) in self.known_agents.items()
            if ts >= stale_limit
        }
        self.known_victims = {
            key: (pos, ts)
            for key, (pos, ts) in self.known_victims.items()
            if ts >= stale_limit
        }

    def select_frontier(self) -> Optional[np.ndarray]:
        free_cells = [cell for cell, value in self.occupancy.items() if value == 0]
        if not free_cells:
            return None

        frontier_cells = set()
        for cell in free_cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in self.occupancy:
                    frontier_cells.add(neighbor)

        if not frontier_cells:
            return None

        current_cell = self.world_to_cell(self.estimated_pos)
        candidates = []
        for cell in frontier_cells:
            if (abs(cell[0] - current_cell[0]) > FRONTIER_SEARCH_RADIUS or
                    abs(cell[1] - current_cell[1]) > FRONTIER_SEARCH_RADIUS):
                continue
            candidates.append(cell)

        if not candidates:
            candidates = list(frontier_cells)

        best_score = None
        best_cell = None
        for cell in candidates:
            cell_pos = self.cell_to_world(cell)
            dist = np.linalg.norm(cell_pos - self.estimated_pos)
            visit_penalty = self.visit_counts.get(cell, 0) * FRONTIER_VISIT_WEIGHT

            repulsion = 0.0
            for agent_pos, _ in self.known_agents.values():
                agent_dist = np.linalg.norm(cell_pos - agent_pos)
                if agent_dist < AGENT_REPULSION_RADIUS:
                    repulsion += (AGENT_REPULSION_RADIUS - agent_dist) / AGENT_REPULSION_RADIUS

            score = (dist * FRONTIER_DISTANCE_WEIGHT
                     + repulsion * FRONTIER_REPULSION_WEIGHT
                     + visit_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best_cell = cell

        if best_cell is None:
            return None
        return self.cell_to_world(best_cell)

    def choose_victim_target(self) -> Optional[Tuple[Tuple[int, int], np.ndarray]]:
        if not self.known_victims:
            return None

        best_key = None
        best_score = None
        for key, (pos, _) in self.known_victims.items():
            if not self.is_closest_agent(pos):
                continue
            score = np.linalg.norm(pos - self.estimated_pos)
            if best_score is None or score < best_score:
                best_score = score
                best_key = key

        if best_key is None:
            return None
        return best_key, self.known_victims[best_key][0]

    def is_closest_agent(self, target_pos: np.ndarray) -> bool:
        my_dist = np.linalg.norm(target_pos - self.estimated_pos)
        for agent_id, (pos, _) in self.known_agents.items():
            if agent_id == self.identifier:
                continue
            if np.linalg.norm(target_pos - pos) + VICTIM_ASSIGNMENT_MARGIN < my_dist:
                return False
        return True

    def move_to_target(self) -> CommandsDict:
        if self.current_target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        delta = self.current_target - self.estimated_pos
        dist = math.hypot(delta[0], delta[1])
        target_angle = math.atan2(delta[1], delta[0])
        angle_error = normalize_angle(target_angle - self.estimated_angle)

        rotation_cmd = max(-1.0, min(1.0, 2.5 * angle_error))
        forward_cmd = 0.6
        if dist < 80:
            forward_cmd = 0.4
        if dist < 30:
            forward_cmd = 0.2
        if abs(angle_error) > 0.4:
            forward_cmd = 0.0

        if not self.lidar_is_disabled():
            lidar_vals = self.lidar_values()
            if lidar_vals is not None and lidar_vals[len(lidar_vals) // 2] < 25:
                forward_cmd = 0.0
                rotation_cmd = 0.8

        grasper = 1 if self.grasped_wounded_persons() else 0

        return {
            "forward": forward_cmd,
            "lateral": 0.0,
            "rotation": rotation_cmd,
            "grasper": grasper,
        }

    def control(self) -> CommandsDict:
        self.update_estimate()
        self.update_grid_from_lidar()
        self.update_visits()
        self.update_semantic()
        self.merge_messages()
        self.prune_messages()

        if self.state == "RETURN" and not self.grasped_wounded_persons():
            self.state = "EXPLORE"
            self.target_victim_key = None
            self.current_target = None

        if self.grasped_wounded_persons() and self.rescue_center_pos is not None:
            self.state = "RETURN"
            self.current_target = self.rescue_center_pos
            if np.linalg.norm(self.estimated_pos - self.rescue_center_pos) < RESCUE_DROP_DISTANCE:
                return {
                    "forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0,
                }

        if self.state != "RETURN":
            victim_choice = self.choose_victim_target()
            if victim_choice is not None:
                victim_key, victim_pos = victim_choice
                self.state = "RESCUE"
                self.target_victim_key = victim_key
                self.current_target = victim_pos

        if self.state == "RESCUE":
            if self.target_victim_key in self.known_victims:
                self.current_target = self.known_victims[self.target_victim_key][0]
            if np.linalg.norm(self.estimated_pos - self.current_target) < VICTIM_GRAB_DISTANCE:
                return {
                    "forward": 0.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 1,
                }
            if self.grasped_wounded_persons():
                self.state = "RETURN"

        if self.state != "RESCUE":
            if self.current_target is None or np.linalg.norm(self.estimated_pos - self.current_target) < GRID_RESOLUTION:
                self.current_target = self.select_frontier()

        if self.current_target is None:
            self.current_target = self.estimated_pos.copy()

        return self.move_to_target()

    def define_message_for_all(self):
        victims_payload = [
            {"pos": pos.tolist(), "t": ts}
            for pos, ts in self.known_victims.values()
        ]
        return {
            "id": self.identifier,
            "pos": self.estimated_pos.tolist(),
            "t": self.elapsed_timestep,
            "victims": victims_payload,
            "rescue_center": None if self.rescue_center_pos is None else self.rescue_center_pos.tolist(),
        }

    def merge_messages(self) -> None:
        if not self.communicator:
            return
        for msg in self.communicator.received_messages:
            if not msg:
                continue
            sender = msg.get("id")
            pos = msg.get("pos")
            ts = msg.get("t")
            if sender is not None and pos is not None and ts is not None:
                self.known_agents[sender] = (np.array(pos), ts)
            rescue_center = msg.get("rescue_center")
            if rescue_center is not None and self.rescue_center_pos is None:
                self.rescue_center_pos = np.array(rescue_center)
            for victim in msg.get("victims", []):
                victim_pos = np.array(victim["pos"])
                victim_ts = victim["t"]
                key = self.world_to_cell(victim_pos)
                existing = self.known_victims.get(key)
                if existing is None or victim_ts > existing[1]:
                    self.known_victims[key] = (victim_pos, victim_ts)
