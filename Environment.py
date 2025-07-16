"""
Environment
"""
from __future__ import annotations
import os
import random
import sys
from collections import namedtuple
from enum import Enum
import pygame

pygame.init()
font = pygame.font.Font("arial.ttf", 25)

"""
Constants
"""
BLOCK = 20
# Speed constant for changing frame rate
SPEED = 10**14
MIN_SEP_BLOCKS = 6

# REWARD STRUCTURE
# REWARDS
REWARD_FIRE = +30
REWARD_RECHARGE = +20

# PENALTIES
PENALTY_NO_BATT_FIRE = -50
# Terminal state during training
PENALTY_COLLISION = -50 
PENALTY_NEAR_MISS = -2
PENALTY_NEAR_WALL = -3
# Penalty for no action
PENALTY_EMPTY_HOVER = -1 

# Reward shaping constant for progress toward fire
K_POS = 0.5 / BLOCK
PROGRESS_CLIP = 2

# Reward shaping constant for separation between drones
K_SEP = 0.3 / BLOCK
SEP_CLIP = 1

# Reward shaping constant for avoiding walls
K_WALL = 0.3 / BLOCK
WALL_CLIP = 1

"""
Direction Class
"""
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Grid coordinates
Point = namedtuple("Point", "x y")

"""
Helper Functions
"""
# Function to calculate the Manhattan distance between two points
def manhattan(a: Point, b: Point) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)

"""
Drone Environment AI Class
"""
# AGENT-ORIENTED SYSTEM REQUIREMENTS
# 3.) BOUNDARIES WHICH SEPARATE THE ENVIRONMENT AND THE AGENT ITSELF
# ENCAPSULATES THE WORLD STATE AND ONLY EXPOSES PERCEPTS AND AN ACTUATION INTERFACE
class DroneEnvironmentAI:
    # DroneEnvironmentAI initialiser
    # Initialise environment dimensions, counts and assets
    def __init__(self,
                 *,
                 w: int = 640,
                 h: int = 480,
                 num_fire: int = 1,
                 num_light: int = 0,
                 num_drones: int = 1,
                 is_simulation: bool = False,
                 min_sep_blocks: int = MIN_SEP_BLOCKS,
                 vision_scale: int = 10) -> None:

        self.w, self.h = w, h
        self.num_fire = num_fire
        self.num_light = num_light
        self.num_drones = num_drones
        self.is_simulation = is_simulation
        self.min_sep = max(1, min_sep_blocks) * BLOCK   # pixels

        # Pygame window
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Drone Fire-Extinguishing")
        self.bounds = pygame.Rect(0, 0, w - BLOCK, h - BLOCK)
        self.clock = pygame.time.Clock()

        # Sprite images for fires, drones, batteries, lightning and background
        fp = pygame.image.load(os.path.join("assets", "fire.png"))
        self.fire_img = pygame.transform.scale(fp, (BLOCK, BLOCK))

        dp = pygame.image.load(os.path.join("assets", "drone.png"))
        self.drone_img = pygame.transform.scale(dp, (BLOCK, BLOCK))

        bp = pygame.image.load(os.path.join("assets", "battery.png"))
        self.battery_img = pygame.transform.scale(bp, (BLOCK, BLOCK))

        lp = pygame.image.load(os.path.join("assets", "lightning.svg"))
        self.lightning_img = pygame.transform.scale(lp, (BLOCK, BLOCK))

        bg = pygame.image.load(os.path.join("assets", "forest_background.jpg"))
        self.forest_img = pygame.transform.scale(bg, (w, h))

        # Configure perception vision radius based on scale input
        self.vision_scale = max(1, min(10, vision_scale))      
        # Radius in 20 pixel blocks
        if self.vision_scale == 10:
            # Esentially unlimited perception vision radius
            self.vision_blocks = 9999

        else:
            radius_map = {9:20, 8:16, 7:12, 6:10, 5: 8, 4: 6, 3: 4, 2: 3, 1: 2}
            self.vision_blocks = radius_map[self.vision_scale]
        
        self.empty_batt_total: int = 0
        self.lightning: list[Point] = []
        self.batteries: list[Point] = []
        self.reset()

    """
    Helper Methods
    """
    # Restrict to remain within display bounds
    def _clamp_pt(self, x: int, y: int) -> tuple[int, int]:
        """Return **tuple** (x,y) that is guaranteed to stay visible."""
        x = max(self.bounds.left, min(self.bounds.right,  x))
        y = max(self.bounds.top, min(self.bounds.bottom, y))
        return x, y

    # Generate a random point inside the environment
    def _rand_grid_point(self) -> Point:
        return Point(random.randrange(self.bounds.left, self.bounds.right  + 1, BLOCK),
            random.randrange(self.bounds.top, self.bounds.bottom + 1, BLOCK))

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # SENSORS TEST WHETHER AN OBJECT IS WITHIN THE DRONE'S PERCEPTION VISION RADIUS
    def _within_vision(self, idx: int, pt: Point | None) -> bool:
        if pt is None:
            return False
        return manhattan(self.drones[idx], pt) <= self.vision_blocks * BLOCK

    # Randomly place drones, fires, lightning and batteries within environment
    def reset(self) -> None:
        # Drones
        self.drones, self.directions = [], []
        while len(self.drones) < self.num_drones:
            p = self._rand_grid_point()
            if all(manhattan(p, q) >= self.min_sep for q in self.drones):
                self.drones.append(p)
                self.directions.append(Direction.RIGHT)
        self.drone = [self.drones[0]]          

        # Lightning
        self._spawn_all_lightning()

        # Fires
        self._spawn_all_fires()

        # Batteries
        # Maximum battery capacity is 100 blocks
        self.battery_level = [100] * self.num_drones
        self._spawn_all_batteries()             

        # Reset counters and reward shaping caches
        self.score = 0
        self.frame_iteration = 0
        self.dist_total = 0
        self.collision_total = 0
        self.light_collision_total = 0
        self.empty_batt_total = 0
        self.prev_fire_dist = [self._dist_to_nearest_fire(i) for i in range(self.num_drones)]
        self.prev_sep = [self._dist_to_nearest_drone(i) for i in range(self.num_drones)]
        self.prev_wall_dist = [self._dist_to_wall(i) for i in range(self.num_drones)]

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 2.) INTERACTION WITH ENVIRONMENT THROUGH SPECIFIC ACTUATORS
    # ENVIRONMENT'S RESPONSE TO AGENTS' ACTUATOR COMMANDS
    def execute_step_multi(self, actions):
        self.frame_iteration += 1
        rewards = [0.0] * self.num_drones

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Move drones and deplete battery levels
        for i, act in enumerate(actions):
            self._move_drone(i, act)
            prev = self.battery_level[i]

            if self.battery_level[i] > 0:
                self.battery_level[i] -= 1

            if self.is_simulation and prev > 0 and self.battery_level[i] == 0:
                self.empty_batt_total += 1

        if self.is_simulation:
            self.dist_total += self.num_drones

        # Battery recharges
        for i in range(self.num_drones):
            if self.drones[i] in self.batteries:
                self.battery_level[i] = 100

                rewards[i] += REWARD_RECHARGE

                self.batteries.remove(self.drones[i])
                self._spawn_battery()

        # Apply reward shaping for fire progress, separation and wall avoidance
        for i in range(self.num_drones):
            # Progress towards fire
            d_now = self._dist_to_nearest_fire(i)
            r = K_POS * (self.prev_fire_dist[i] - d_now)
            rewards[i] += max(-PROGRESS_CLIP, min(PROGRESS_CLIP, r))
            self.prev_fire_dist[i] = d_now

            # Maintain separation
            s_now = self._dist_to_nearest_drone(i)
            r = K_SEP * (s_now - self.prev_sep[i])
            rewards[i] += max(-SEP_CLIP, min(SEP_CLIP, r))
            self.prev_sep[i] = s_now

            # Avoid walls
            w_now = self._dist_to_wall(i)
            r = K_WALL * (w_now - self.prev_wall_dist[i])
            rewards[i] += max(-WALL_CLIP, min(WALL_CLIP, r))
            self.prev_wall_dist[i] = w_now

            if any(manhattan(self.drones[i], self.drones[j]) < 2*BLOCK for j in range(self.num_drones) if j != i):
                rewards[i] += PENALTY_NEAR_MISS
            if w_now < BLOCK:
                rewards[i] += PENALTY_NEAR_WALL

        # Penalty for no action
        for i in range(self.num_drones):
            if self.battery_level[i] == 0:
                rewards[i] += PENALTY_EMPTY_HOVER
        
        # Handle invalid fire extinguishing attempts when depleted battery
        for i in range(self.num_drones):
            if (self.battery_level[i] == 0 and self.drones[i] in self.fires):
                rewards[i] += PENALTY_NO_BATT_FIRE

                if not self.is_simulation:               
                    return [PENALTY_NO_BATT_FIRE] * self.num_drones, True, self.score
                else:                                    
                    print(f"[SIM] Drone {i} tried to extinguish fire with 0 charge")

        # Valid fire extinguishing
        for i in range(self.num_drones):
            idx = next((j for j, f in enumerate(self.fires) if f == self.drones[i]), None)

            if idx is not None and self.battery_level[i] > 0:
                self.score += 1
                rewards[i] += REWARD_FIRE
                self.fires.pop(idx)
                self._spawn_fire()
                self.prev_fire_dist[i] = self._dist_to_nearest_fire(i)

        # Terminal states
        if not self.is_simulation:
            if self._training_collision() or self.frame_iteration > 100 * self.num_drones:
                return [PENALTY_COLLISION] * self.num_drones, True, self.score
            
        else:
            self.collision_total += self._count_drone_collisions()
            self.light_collision_total += sum(pt in self.lightning for pt in self.drones)

        self._draw()
        self.clock.tick(SPEED)
        return rewards, False, self.score

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # SENSORS RETURN THE NEAREST FIRE TO DRONE
    def get_closest_fire_for_drone(self, i):              
        visible = [f for f in self.fires if self._within_vision(i, f)]   
        p = self.drones[i]
        return min(self.fires, key=lambda f: manhattan(p, f)) if self.fires else p
    
    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # SENSORS RETURN THE NEAREST BATTERY TO DRONE
    def get_closest_battery_for_drone(self, i):
        visible = [f for f in self.fires if self._within_vision(i, f)]   
        p = self.drones[i]

        return min(self.batteries, key=lambda b: manhattan(p, b)) if self.batteries else p

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # SENSORS RETURN THE NEAREST LIGHTNING STORM TO DRONE
    def get_closest_light_for_drone(self, i):
        visible = [f for f in self.fires if self._within_vision(i, f)]   
        p = self.drones[i]
        return min(self.lightning, key=lambda l: manhattan(p, l)) if self.lightning else p

    # Determine if collision occurred
    def is_collision(self, pt: Point | None = None) -> bool:
        pt = pt or self.drones[0]
        solid = self.drones + self.lightning
        if self.is_simulation:                     
            return(pt in solid or pt.x < self.bounds.left  or pt.x > self.bounds.right or pt.y < self.bounds.top or pt.y > self.bounds.bottom)
        return(pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h or pt in solid)
    
    # Spawn one battery per drone
    def _spawn_all_batteries(self):
        self.batteries = []
        for _ in range(self.num_drones):        
            self._spawn_battery()

    def _spawn_battery(self):
        trials = 0
        while True:
            trials += 1
            p = self._rand_grid_point()
            if(p not in self.batteries and p not in self.fires and p not in self.lightning and p not in self.drones):
                self.batteries.append(p)
                break

            if trials > 1000:               
                self.batteries.append(p)
                break

    # Distance to nearest fire
    def _dist_to_nearest_fire(self, idx: int) -> int:
        p = self.drones[idx]
        visible = [f for f in self.fires if self._within_vision(idx, f)] 
        return min((manhattan(p, f) for f in self.fires), default=0)

    # Distance to nearest drone
    def _dist_to_nearest_drone(self, idx: int) -> int:
        p = self.drones[idx]
        visible = [f for f in self.fires if self._within_vision(idx, f)] 
        others = [q for j, q in enumerate(self.drones) if j != idx]
        return min((manhattan(p, q) for q in others), default=self.w + self.h)

    def _dist_to_wall(self, idx: int) -> int:
        p = self.drones[idx]
        return min(p.x, p.y, self.w - BLOCK - p.x, self.h - BLOCK - p.y)

    # Count collisions between drones
    def _count_drone_collisions(self) -> int:
        freq = {}
        for p in self.drones:
            freq[p] = freq.get(p, 0) + 1
        return sum((n * (n - 1)) // 2 for n in freq.values())

    # Spawn lightning
    def _spawn_all_lightning(self):
        self.lightning = []
        for _ in range(self.num_light):
            self._spawn_lightning()

    def _spawn_lightning(self):
        trials = 0
        while True:
            trials += 1
            p = self._rand_grid_point()
            if (p not in self.lightning and p not in getattr(self, "fires", []) and p not in self.drones):
                self.lightning.append(p)
                break
            if trials > 1000:
                self.lightning.append(p)
                break  

    # Spawn fires
    def _spawn_all_fires(self):
        self.fires = []
        for _ in range(self.num_fire):
            self._spawn_fire()

    def _spawn_fire(self):
        trials = 0
        while True:
            trials += 1
            p = self._rand_grid_point()

            if(p not in self.fires and p not in self.lightning and p not in self.drones):
                self.fires.append(p)
                break
            if trials > 1000:
                self.fires.append(p)
                break

    # Check for drone collisions to terminate training epoch
    def _training_collision(self) -> bool:
        for p in self.drones:
            if (p.x < 0 or p.x >= self.w or p.y < 0 or p.y >= self.h or p in self.lightning):
                return True
        return len(set(self.drones)) != len(self.drones)

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 2.) INTERACTION WITH ENVIRONMENT THROUGH SPECIFIC ACTUATORS
    # ACTUATION THROUGH EXPLICIT ACTION LIST CHANGING DRONE'S POSITION

    # 3.) BOUNDARIES WHICH SEPARATE THE ENVIRONMENT AND THE AGENT ITSELF
    # ENVIRONMENT CONSTRAINT BOUNDARY TO ENFORCE BOUNDS AROUND THE ENVIRONMENT TO PREVENT AGENT FROM LEAVING THE ENVIRONMENT
    def _move_drone(self, idx: int, action):
        dirs = [Direction.RIGHT, Direction.DOWN,
                Direction.LEFT,  Direction.UP]
        cur  = dirs.index(self.directions[idx])
        new_dir = (dirs[cur] if action == [1, 0, 0] else dirs[(cur + 1) % 4]  if action == [0, 1, 0] else dirs[(cur - 1) % 4])

        x, y = self.drones[idx]
        if   new_dir is Direction.RIGHT:
            x += BLOCK
        elif new_dir is Direction.LEFT:
            x -= BLOCK
        elif new_dir is Direction.DOWN:
            y += BLOCK
        else:
            y -= BLOCK

        # Prevent drone from leaving environment with invisible walls
        if self.is_simulation:
            x, y = self._clamp_pt(x, y)

        target = Point(x, y)

        if target in self.drones or target in self.lightning:
            alt = [dirs[(cur + 1) % 4], dirs[(cur - 1) % 4]]
            random.shuffle(alt)
            for d in alt:
                tx, ty = self.drones[idx]

                if d is Direction.RIGHT:
                    tx += BLOCK
                elif d is Direction.LEFT:
                    tx -= BLOCK
                elif d is Direction.DOWN:
                    ty += BLOCK
                else:
                    ty -= BLOCK

                tx = max(0, min(self.w - BLOCK, tx))
                ty = max(0, min(self.h - BLOCK, ty))

                if Point(tx, ty) not in self.drones + self.lightning:
                    new_dir, target = d, Point(tx, ty)
                    break

        self.directions[idx] = new_dir
        self.drones[idx] = target
        if idx == 0:
            self.drone[0] = target

    # Display background, sprites and perception vision radii
    def _draw(self):
        self.display.blit(self.forest_img, (0, 0))

        # Fires
        for f in self.fires:
            self.display.blit(self.fire_img, (f.x, f.y))

        # Perception vision radius
        if self.vision_scale < 10:
            radius_px = self.vision_blocks * BLOCK
            overlay = pygame.Surface((radius_px*2, radius_px*2), pygame.SRCALPHA)

            pygame.draw.circle(overlay, (128, 128, 128, 40), (radius_px, radius_px), radius_px)

            for p in self.drones:
                cx = p.x + BLOCK//2
                cy = p.y + BLOCK//2
                self.display.blit(overlay, (cx - radius_px, cy - radius_px))

        # Drones
        for p in self.drones:
            self.display.blit(self.drone_img, (p.x, p.y))

        # Lightning
        for l in self.lightning:
            self.display.blit(self.lightning_img, (l.x, l.y))

        # Batteries
        for b in self.batteries:
            self.display.blit(self.battery_img, (b.x, b.y))

        # Battery levels
        for idx, p in enumerate(self.drones):
            txt_surf = font.render(str(self.battery_level[idx]), True, (255, 255, 255))
            self.display.blit(txt_surf, (p.x + BLOCK // 2 - txt_surf.get_width() // 2, p.y - txt_surf.get_height() - 2))

        txt = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(txt, (5, 0))
        pygame.display.update()