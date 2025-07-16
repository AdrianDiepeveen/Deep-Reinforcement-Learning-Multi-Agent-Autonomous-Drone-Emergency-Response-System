"""
Multi Agent
Multiple drones are controlled by a shared policy where each drone:
1.) Observes its own local state via get_state_for_drone, producing a 22 dimensional percept.
2.) Selects an action using the shared policy.
3.) Executes that action in the environment, returning a reward and next state.
"""
from __future__ import annotations
import random
from collections import deque
import numpy as np

from Environment import Direction, Point, BLOCK
from Deep_Q_Learning_Model import NeuralNetwork, DQNTrainer
from Q_Learning_Model import QTable

# Shared hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 256
LR = 0.0005
GAMMA = 0.9
INPUT_DIM = 22

# Discretise state into key
def _discretise(state: np.ndarray, *, mode: str = "EXTINGUISH") -> tuple:
    # First 19 binary features
    bits  = tuple(state[:19].astype(int))
    dist_bin   = int(state[19] * 4)   
    idx_bin    = int(state[20] * 4)       
    charge_bin = int(state[21] * 5)      
    # Finite state machine mode bit
    mode_bit   = 1 if mode == "RECHARGE" else 0

    return bits + (dist_bin, idx_bin, charge_bin, mode_bit)

# Translate current and target directions into action
def _dir2action(cur_dir, target_pt, head):
    # Clockwise directions list
    DIRS = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    cur  = DIRS.index(cur_dir)
    dx   = target_pt.x - head.x
    dy   = target_pt.y - head.y

    # Movement toward target
    want = Direction.RIGHT if abs(dx) > abs(dy) and dx > 0 else \
           Direction.LEFT  if abs(dx) > abs(dy) and dx < 0 else \
           Direction.DOWN  if dy > 0 else Direction.UP
    
    # Relative turn delta
    delta = (DIRS.index(want) - cur) % 4

    return [int(delta == 0), int(delta == 1), int(delta == 3)]

"""
Agent Class
"""
class Agent:
    # Agent initialiser
    def __init__(self, num_drones: int = 1, *, algo: str = "Deep Q-learning"):
        self.num_drones = num_drones
        self.algo = algo
        self.n_games = 0
        self.epsilon = 0.0
        self.gamma = GAMMA

        # Finite State Machine thresholds for recharge mode
        self.mode = "EXTINGUISH" 
        self.low_thresh = 15       
        self.full_thresh = 90   

        # Initialise learning based on selected AI algorithm
        if algo == "Deep Q-learning":
            # Initialise replay buffer, online network and trainer
            self.memory = deque(maxlen = MAX_MEMORY)
            self.__dict__["model"] = NeuralNetwork(input_size = INPUT_DIM)
            self.trainer = DQNTrainer(self.__dict__["model"], lr = LR, gamma = self.gamma)

        elif algo == "Q-learning":
            # Initialise tabular Q-learning
            self.qtable = QTable(alpha = 0.10, gamma = self.gamma)


    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # BUILD 22 DIMENSIONAL PERCEPT VECTOR FROM ENVIRONMENT SENSOR READINGS

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 3.) BOUNDARIES WHICH SEPARATE THE ENVIRONMENT AND THE AGENT ITSELF
    # AGENT ONLY KNOWS HOW TO SENSE VIA get_state_for_drone METHOD AND ACTUATE BY RETURNING AN ACTION LIST
    def get_state_for_drone(self, game, idx: int) -> np.ndarray:
        
        # Get positions one block away in four directions
        head = game.drones[idx]
        pt_l = Point(head.x - BLOCK, head.y)
        pt_r = Point(head.x + BLOCK, head.y)
        pt_u = Point(head.x, head.y - BLOCK)
        pt_d = Point(head.x, head.y + BLOCK)

        # Get current heading direction flags
        dir_l = game.directions[idx] == Direction.LEFT
        dir_r = game.directions[idx] == Direction.RIGHT
        dir_u = game.directions[idx] == Direction.UP
        dir_d = game.directions[idx] == Direction.DOWN

        # Get nearest fire, battery or lightning
        fire  = game.get_closest_fire_for_drone(idx)
        batt  = game.get_closest_battery_for_drone(idx)
        storm = game.get_closest_light_for_drone(idx)

        # Get whether within perception vision radius 
        fire_vis = game._within_vision(idx, fire)              
        batt_vis  = game._within_vision(idx, batt)              
        storm_vis = game._within_vision(idx, storm)            

        # Get immediate dangers
        danger_straight = ((dir_r and game.is_collision(pt_r)) or
                           (dir_l and game.is_collision(pt_l)) or
                           (dir_u and game.is_collision(pt_u)) or
                           (dir_d and game.is_collision(pt_d)))
        
        danger_right = ((dir_u and game.is_collision(pt_r)) or
                        (dir_d and game.is_collision(pt_l)) or
                        (dir_l and game.is_collision(pt_u)) or
                        (dir_r and game.is_collision(pt_d)))
        
        danger_left = ((dir_d and game.is_collision(pt_r)) or
                       (dir_u and game.is_collision(pt_l)) or
                       (dir_r and game.is_collision(pt_u)) or
                       (dir_l and game.is_collision(pt_d)))

        # Normalise distance to nearest other drone
        if game.num_drones > 1:
            ds = [abs(head.x - q.x) + abs(head.y - q.y)
                  for j, q in enumerate(game.drones) if j != idx]
            dist_norm = min(ds) / (game.w + game.h)

        else:
            dist_norm = 1.0

        # Normalize drone index
        idx_norm = idx / (game.num_drones - 1) if game.num_drones > 1 else 0.0

        # Normalise drone battery charge level
        charge_norm = game.battery_level[idx] / 100.0

        # AGENT-ORIENTED SYSTEM REQUIREMENTS
        # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
        # NOW BUILD 22 DIMENSIONAL PERCEPT VECTOR FROM ENVIRONMENT SENSOR READINGS
        state = np.array([
            danger_straight,
            danger_right,
            danger_left,

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            fire.x < head.x if fire_vis else False,
            fire.x > head.x if fire_vis else False,
            fire.y < head.y if fire_vis else False,
            fire.y > head.y if fire_vis else False,

            storm.x < head.x if storm_vis else False,
            storm.x > head.x if storm_vis else False,
            storm.y < head.y if storm_vis else False,
            storm.y > head.y if storm_vis else False,

            batt.x < head.x if batt_vis else False,
            batt.x > head.x if batt_vis else False,
            batt.y < head.y if batt_vis else False,
            batt.y > head.y if batt_vis else False,

            dist_norm,
            idx_norm,
            charge_norm
        ], dtype=np.float32)

        return state

    # AGENT-ORIENTED SYSTEM REQUIREMENTS
    # 1.) PERCEPTION OF ENVIRONMENT THROUGH WELL DEFINED PERCEPT SEQUENCES
    # CONVERTING THE 22 DIMENSIONAL PERCEPT INTO PERCEPT SEQUENCES
    # Store transition in replay buffer for Deep Q-learning
    def remember(self, s, a, r, s_, done):
        if self.algo == "Deep Q-learning":
            self.memory.append((s, a, r, s_, done))

    # Train agent on past experiences
    def train_long_memory(self):
        if self.algo != "Deep Q-learning":
            return
        batch = (random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory)
        
        if batch:
            self.trainer.train_step(*zip(*batch))

    # Train agent on single transition
    def train_short_memory(self, s, a, r, s_, done):
        # Deep Q-learning
        if self.algo == "Deep Q-learning":
            return self.trainer.train_step(s, a, r, s_, done)

        # Tabular Q-learning update
        a_idx = int(np.argmax(a)) if len(a) > 1 else int(a)

        if self.algo == "Q-learning":
            self.qtable.update(_discretise(s,  mode=self.mode), a_idx, r, _discretise(s_, mode=self.mode), done)

        return 0.0

    # Compute Q-values for action selection
    def _q_values(self, state: np.ndarray) -> np.ndarray:
        if self.algo == "Deep Q-learning":
            return self.model.forward(state[np.newaxis, :])[0]
        
        if self.algo == "Q-learning":
            return self.qtable.predict(_discretise(state, mode=self.mode))

    # Choose action via Finite State Machine and Epsilon-greedy policy
    def get_action(self, state: np.ndarray) -> list[int]:
        # RECHARGE MODE
        # Update Finite State Machine based on current charge level
        # Please Note:
        # The Finite State Machine only decides when to switch between extinguish and recharge only when the battery crosses fixed thresholds
        # Every other decision is still driven by the Deep Q-learning's and Q-learning's Q-values and epsilon-greedy policy
        charge = state[-1] * 100  
        if self.mode == "EXTINGUISH" and charge <= self.low_thresh:
            self.mode = "RECHARGE"

        elif self.mode == "RECHARGE" and charge >= self.full_thresh:
            self.mode = "EXTINGUISH"

        # Select action
        if self.mode == "RECHARGE":
            game = self._last_game 
            idx = self._last_idx
            target = game.get_closest_battery_for_drone(idx)
            act = _dir2action(game.directions[idx], target, game.drones[idx])

        # EXTINGUISH MODE
        else:
            # Apply Epsilon-greedy exploration or exploitation
            self.epsilon = max(1, 60 - self.n_games)
            if random.randint(0, 200) < self.epsilon:
                # Explore
                move = random.randint(0, 2)                

            else:
                # Exploit
                move = int(self._q_values(state).argmax())

            act = [int(move == 0), int(move == 1), int(move == 2)]

        return act

    # Interface to AI model in order to save AI model
    @property
    def model(self):
        if self.algo == "Deep Q-learning":
            return self.__dict__["model"]
        return self

    # Save state of trained AI model
    def save(self, fname: str):
        if self.algo == "Deep Q-learning":
            self.__dict__["model"].save(fname)

        elif self.algo == "Q-learning":
            self.qtable.save(fname.replace(".npz", ".pkl"))

    # Load state of trained AI model
    def load(self, path: str):
        if self.algo == "Deep Q-learning":
            self.__dict__["model"].load(path)

        elif self.algo == "Q-learning":
            self.qtable.load(path)