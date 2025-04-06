# ctf_env_torch.py
import gymnasium
import numpy as np
import math
from gymnasium.spaces import Discrete, Box

# --- Constants (copied from marl_ctf_env.py) ---
BLUE_TEAM_BOUNDS = {
    "minLat": 37.75, "maxLat": 37.80, "minLng": -122.46, "maxLng": -122.41,
}
RED_TEAM_BOUNDS = {
    "minLat": 37.82, "maxLat": 37.87, "minLng": -122.28, "maxLng": -122.22,
}
ALL_BOUNDS = {
    "minLat": min(BLUE_TEAM_BOUNDS["minLat"], RED_TEAM_BOUNDS["minLat"]),
    "maxLat": max(BLUE_TEAM_BOUNDS["maxLat"], RED_TEAM_BOUNDS["maxLat"]),
    "minLng": min(BLUE_TEAM_BOUNDS["minLng"], RED_TEAM_BOUNDS["minLng"]),
    "maxLng": max(BLUE_TEAM_BOUNDS["maxLng"], RED_TEAM_BOUNDS["maxLng"]),
}

N_AGENTS_PER_TEAM = 8 # Keep consistent
TOTAL_AGENTS = N_AGENTS_PER_TEAM * 2
MAX_CYCLES = 1000 # Max steps per episode

MOVE_STEP_SIZE = 0.001
CAPTURE_RADIUS = 0.005
FLAG_SCORE_RADIUS = 0.005
TAG_RADIUS = 0.003 # Radius for agents tagging each other

# Reward constants
REWARD_STEP_PENALTY = -0.1
REWARD_MOVE_TOWARDS_ENEMY_FLAG_SCALE = 300.0
REWARD_MOVE_TOWARDS_ENEMY_AGENT_SCALE = 150.0 # Lower than flag reward
REWARD_PICKUP_FLAG = 10.0 # Increased
REWARD_TAG_ENEMY = 25.0   # New reward for tagging
REWARD_FLAG_SCORE_SCORER = 200.0 # Increased
REWARD_FLAG_SCORE_TEAMMATE = 50.0 # Increased

# --- NEW: Placeholder dimension for chat features ---
CHAT_FEATURE_DIM = 384 # Example size - adjust if using actual embeddings

ACTION_MAP = {
    0: (0.0, 0.0), 1: (MOVE_STEP_SIZE, 0.0), 2: (-MOVE_STEP_SIZE, 0.0),
    3: (0.0, MOVE_STEP_SIZE), 4: (0.0, -MOVE_STEP_SIZE),
}
N_ACTIONS = len(ACTION_MAP)

# --- Helper Functions (copied and adjusted) ---
def get_random_pos(bounds):
    lat = np.random.uniform(bounds["minLat"], bounds["maxLat"])
    lng = np.random.uniform(bounds["minLng"], bounds["maxLng"])
    return np.array([lat, lng], dtype=np.float32)

def distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def normalize_pos(pos):
    lat_range = ALL_BOUNDS["maxLat"] - ALL_BOUNDS["minLat"]
    lng_range = ALL_BOUNDS["maxLng"] - ALL_BOUNDS["minLng"]
    if pos is None or pos.shape != (2,): return np.array([0.5, 0.5], dtype=np.float32)
    lat = (pos[0] - ALL_BOUNDS["minLat"]) / lat_range if lat_range > 1e-6 else 0.5
    lng = (pos[1] - ALL_BOUNDS["minLng"]) / lng_range if lng_range > 1e-6 else 0.5
    return np.clip(np.array([lat, lng]), 0.0, 1.0).astype(np.float32)

class CaptureTheFlagEnvTorch:
    """
    Capture The Flag environment adapted for a standard synchronous PyTorch loop.
    - reset() returns initial observations for all agents.
    - step(actions) takes actions for all agents and returns next_obs, rewards, terms, truncs, infos.
    """
    metadata = { # Keep metadata for potential compatibility
        "render_modes": ["human"],
        "name": "capture_the_flag_torch_v1",
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        # Agent Setup
        self.blue_agents = [f"blue_{i}" for i in range(N_AGENTS_PER_TEAM)]
        self.red_agents = [f"red_{i}" for i in range(N_AGENTS_PER_TEAM)]
        self.possible_agents = self.blue_agents + self.red_agents
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.n_agents = len(self.possible_agents)

        # Calculate observation dimensions (ADDED BASE POSITIONS + CHAT)
        self_pos_dim = 2
        self_status_dim = 1
        teammate_pos_dim = 2 * (N_AGENTS_PER_TEAM - 1)
        opponent_pos_dim = 2 * N_AGENTS_PER_TEAM
        flag_pos_dim = 2 * 2 # Own and enemy flag CURRENT pos
        carrier_idx_dim = 1 * 2 # Own and enemy carrier index
        base_pos_dim = 2 * 2 # Own and enemy BASE pos
        chat_dim = CHAT_FEATURE_DIM # NEW chat features
        obs_dim = (
            self_pos_dim + self_status_dim + teammate_pos_dim + opponent_pos_dim +
            flag_pos_dim + carrier_idx_dim + base_pos_dim + chat_dim # Added chat_dim
        )

        # Define bounds (Adjusted for base positions and chat)
        # Bases are normalized coords [0, 1], chat features assumed normalized [-1, 1] for now
        # Adjust chat bounds if using different encoding
        non_chat_dims = obs_dim - carrier_idx_dim - chat_dim
        low_bounds = ([0.0] * non_chat_dims) + ([-1.0] * carrier_idx_dim) + ([-1.0] * chat_dim)
        high_bounds = ([1.0] * non_chat_dims) + ([float(TOTAL_AGENTS - 1)] * carrier_idx_dim) + ([1.0] * chat_dim)

        # Define spaces per agent (Updated shape)
        self.observation_space = Box(
            low=np.array(low_bounds, dtype=np.float32),
            high=np.array(high_bounds, dtype=np.float32),
            shape=(obs_dim,), # Updated shape
            dtype=np.float32
        )
        self.action_space = Discrete(N_ACTIONS)

        # Internal game state (initialized in reset)
        self._reset_game_state()
        self.cycles = 0


    def _reset_game_state(self):
        """Resets the internal state variables of the game."""
        self.agent_positions = {}
        self.agent_status = {} # 0: active, 1: has_flag
        self.blue_flag_pos = np.zeros(2, dtype=np.float32)
        self.red_flag_pos = np.zeros(2, dtype=np.float32)
        self.blue_flag_base = np.zeros(2, dtype=np.float32) # Base initialized in reset
        self.red_flag_base = np.zeros(2, dtype=np.float32) # Base initialized in reset
        self.blue_flag_carrier = None
        self.red_flag_carrier = None
        self.blue_flag_status = 0 # 0:base, 1:carried
        self.red_flag_status = 0  # 0:base, 1:carried
        self.scores = {"blue": 0, "red": 0}
        self.active_agents = set(self.possible_agents) # Keep track of active agents

    def reset(self, seed=None, options=None):
        """Resets the environment and returns initial observations for all agents."""
        if seed is not None:
            np.random.seed(seed)

        self._reset_game_state()
        self.cycles = 0

        # Initialize positions and status
        for agent_id in self.blue_agents:
            self.agent_positions[agent_id] = get_random_pos(BLUE_TEAM_BOUNDS)
            self.agent_status[agent_id] = 0
        for agent_id in self.red_agents:
            self.agent_positions[agent_id] = get_random_pos(RED_TEAM_BOUNDS)
            self.agent_status[agent_id] = 0

        # Initialize flags AND bases at random positions within team zones
        self.blue_flag_pos = get_random_pos(BLUE_TEAM_BOUNDS)
        self.red_flag_pos = get_random_pos(RED_TEAM_BOUNDS)
        self.blue_flag_base = self.blue_flag_pos.copy() # Base starts where flag starts
        self.red_flag_base = self.red_flag_pos.copy()   # Base starts where flag starts

        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        infos = {agent: {"score_blue": 0, "score_red": 0} for agent in self.possible_agents} # Initial info

        if self.render_mode == "human":
            self.render()

        # Return obs dict and initial info dict
        return observations, infos

    def observe(self, agent):
        """Generates the observation for a single agent."""
        if agent not in self.active_agents:
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs_list = []

        # Self Info
        obs_list.extend(normalize_pos(self.agent_positions.get(agent))) # Use .get for safety
        obs_list.append(np.float32(self.agent_status.get(agent, 0))) # Use .get

        # Teammate Info
        teammate_list = self.blue_agents if "blue" in agent else self.red_agents
        count = 0
        for tm_id in teammate_list:
            if tm_id != agent:
                if tm_id in self.active_agents:
                    obs_list.extend(normalize_pos(self.agent_positions.get(tm_id)))
                else:
                    obs_list.extend([np.float32(0.0), np.float32(0.0)])
                count += 1
        # assert count == N_AGENTS_PER_TEAM - 1 # Can add back if agent removal isn't planned

        # Opponent Info
        opponent_list = self.red_agents if "blue" in agent else self.blue_agents
        count = 0
        for op_id in opponent_list:
            if op_id in self.active_agents:
                obs_list.extend(normalize_pos(self.agent_positions.get(op_id)))
            else:
                obs_list.extend([np.float32(0.0), np.float32(0.0)])
            count += 1
        # assert count == N_AGENTS_PER_TEAM # Can add back

        # Flag CURRENT Position Info
        blue_flag_norm = normalize_pos(self.blue_flag_pos)
        red_flag_norm = normalize_pos(self.red_flag_pos)
        own_flag_pos, enemy_flag_pos = (blue_flag_norm, red_flag_norm) if "blue" in agent else (red_flag_norm, blue_flag_norm)
        obs_list.extend(own_flag_pos)
        obs_list.extend(enemy_flag_pos)

        # Flag Carrier Info
        blue_carrier_idx = self.agent_name_mapping.get(self.blue_flag_carrier, -1)
        red_carrier_idx = self.agent_name_mapping.get(self.red_flag_carrier, -1)
        own_carrier_idx, enemy_carrier_idx = (blue_carrier_idx, red_carrier_idx) if "blue" in agent else (red_carrier_idx, blue_carrier_idx)
        obs_list.append(np.float32(own_carrier_idx))
        obs_list.append(np.float32(enemy_carrier_idx))

        # --- NEW: Flag BASE Position Info ---
        blue_base_norm = normalize_pos(self.blue_flag_base)
        red_base_norm = normalize_pos(self.red_flag_base)
        own_base_pos, enemy_base_pos = (blue_base_norm, red_base_norm) if "blue" in agent else (red_base_norm, blue_base_norm)
        obs_list.extend(own_base_pos)
        obs_list.extend(enemy_base_pos)
        # --- END NEW ---

        # --- NEW: Add placeholder chat features (zeros for training simulation) ---
        # NOTE: For actual training with chat, you would need to:
        # 1. Modify the training loop (train_torch_iac.py) to simulate chat generation.
        # 2. Pass the simulated chat history to this function or the step method.
        # 3. Process the history here (e.g., using embeddings) instead of using zeros.
        chat_features = np.zeros(CHAT_FEATURE_DIM, dtype=np.float32)
        obs_list.extend(chat_features)
        # --- END NEW ---

        final_obs = np.array(obs_list, dtype=np.float32)

        # Final shape check - crucial for debugging
        expected_shape = self.observation_space.shape
        if final_obs.shape != expected_shape:
             # Ensure the obs_dim calculation above matches the number of elements added here
             raise ValueError(f"Observation shape mismatch for {agent}! Expected {expected_shape}, got {final_obs.shape}. List length: {len(obs_list)}")

        return final_obs

    def step(self, actions):
        """
        Performs a step for all agents simultaneously.
        Args:
            actions (dict): Dictionary mapping agent_id to action index.
        Returns:
            tuple: (observations, rewards, terminations, truncations, infos)
                   Dictionaries mapping agent_id to their respective values.
        """
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        # --- Calculate old distances (before move) ---
        old_distances_to_enemy_flag = {}
        old_distances_to_nearest_enemy = {}
        active_blue = [a for a in self.blue_agents if a in self.active_agents]
        active_red = [a for a in self.red_agents if a in self.active_agents]

        for agent_id in self.active_agents:
            current_pos = self.agent_positions[agent_id]
            is_blue = "blue" in agent_id

            # Distance to enemy flag
            enemy_flag_pos = self.red_flag_pos if is_blue else self.blue_flag_pos
            old_distances_to_enemy_flag[agent_id] = distance(current_pos, enemy_flag_pos)

            # Distance to nearest enemy
            min_dist_enemy = float('inf')
            opponents = active_red if is_blue else active_blue
            if opponents: # Only calculate if there are opponents
                for opp_id in opponents:
                    min_dist_enemy = min(min_dist_enemy, distance(current_pos, self.agent_positions[opp_id]))
            if min_dist_enemy != float('inf'):
                 old_distances_to_nearest_enemy[agent_id] = min_dist_enemy


        # --- Apply actions (Movement) ---
        new_positions = {} # Store new positions temporarily to avoid mid-step move side effects
        for agent_id, action in actions.items():
            if agent_id not in self.active_agents: continue

            move_lat, move_lng = ACTION_MAP[action]
            current_pos = self.agent_positions[agent_id]
            new_pos = current_pos + np.array([move_lat, move_lng])
            new_pos[0] = np.clip(new_pos[0], ALL_BOUNDS["minLat"], ALL_BOUNDS["maxLat"])
            new_pos[1] = np.clip(new_pos[1], ALL_BOUNDS["minLng"], ALL_BOUNDS["maxLng"])
            new_positions[agent_id] = new_pos # Store calculated new position

            # Basic step penalty
            rewards[agent_id] += REWARD_STEP_PENALTY

        # --- Update Positions and Move Flags ---
        for agent_id, new_pos in new_positions.items():
            self.agent_positions[agent_id] = new_pos
             # Move flag if carrying
            if self.blue_flag_carrier == agent_id: self.blue_flag_pos = new_pos.copy()
            if self.red_flag_carrier == agent_id: self.red_flag_pos = new_pos.copy()


        # --- Reward Shaping & Calculate New Distances (after move) ---
        active_blue = [a for a in self.blue_agents if a in self.active_agents] # Update active lists
        active_red = [a for a in self.red_agents if a in self.active_agents]
        for agent_id in list(self.active_agents): # Iterate over copy in case agent is removed
             if agent_id not in self.agent_positions: continue # Agent might have been removed by tag

             new_pos = self.agent_positions[agent_id]
             is_blue = "blue" in agent_id

             # Reward moving towards enemy flag
             enemy_flag_pos = self.red_flag_pos if is_blue else self.blue_flag_pos
             new_distance_to_flag = distance(new_pos, enemy_flag_pos)
             old_distance_flag = old_distances_to_enemy_flag.get(agent_id, new_distance_to_flag + 1.0)
             if new_distance_to_flag < old_distance_flag:
                 distance_improved = old_distance_flag - new_distance_to_flag
                 rewards[agent_id] += distance_improved * REWARD_MOVE_TOWARDS_ENEMY_FLAG_SCALE

             # Reward moving towards nearest enemy
             min_dist_enemy = float('inf')
             opponents = active_red if is_blue else active_blue
             if opponents:
                 for opp_id in opponents:
                     # Ensure opponent is still active and has a position before calculating distance
                     if opp_id in self.active_agents and opp_id in self.agent_positions:
                         min_dist_enemy = min(min_dist_enemy, distance(new_pos, self.agent_positions[opp_id]))

             if min_dist_enemy != float('inf'):
                 old_distance_enemy = old_distances_to_nearest_enemy.get(agent_id, min_dist_enemy + 1.0)
                 if min_dist_enemy < old_distance_enemy:
                     distance_improved = old_distance_enemy - min_dist_enemy
                     rewards[agent_id] += distance_improved * REWARD_MOVE_TOWARDS_ENEMY_AGENT_SCALE


        # --- Implement Tagging Logic ---
        tagged_agents = set()
        # Iterate through pairs of active opponents
        current_active_blue = [a for a in self.blue_agents if a in self.active_agents]
        current_active_red = [a for a in self.red_agents if a in self.active_agents]

        for blue_agent_id in current_active_blue:
            if blue_agent_id in tagged_agents: continue # Already tagged this cycle
            for red_agent_id in current_active_red:
                if red_agent_id in tagged_agents: continue # Already tagged this cycle

                pos_blue = self.agent_positions[blue_agent_id]
                pos_red = self.agent_positions[red_agent_id]

                if distance(pos_blue, pos_red) <= TAG_RADIUS:
                    print(f"[Game Event Cycle {self.cycles}] Tag! {blue_agent_id} and {red_agent_id}")
                    # Mark both agents for removal
                    tagged_agents.add(blue_agent_id)
                    tagged_agents.add(red_agent_id)

                    # Assign tag rewards
                    rewards[blue_agent_id] += REWARD_TAG_ENEMY
                    rewards[red_agent_id] += REWARD_TAG_ENEMY

                    # Handle flag dropping
                    if self.blue_flag_carrier == red_agent_id:
                        print(f"   Red agent {red_agent_id} dropped Blue flag.")
                        self.blue_flag_pos = self.blue_flag_base.copy()
                        self.blue_flag_status = 0
                        self.agent_status[red_agent_id] = 0
                        self.blue_flag_carrier = None
                    if self.red_flag_carrier == blue_agent_id:
                        print(f"   Blue agent {blue_agent_id} dropped Red flag.")
                        self.red_flag_pos = self.red_flag_base.copy()
                        self.red_flag_status = 0
                        self.agent_status[blue_agent_id] = 0
                        self.red_flag_carrier = None

                    # Since a tag occurred involving these agents, break inner loop
                    # and continue with the next blue agent. Avoids double tagging issues in one step.
                    break # Move to the next blue agent

        # Remove tagged agents from the game
        for agent_id in tagged_agents:
             if agent_id in self.active_agents:
                 self.active_agents.remove(agent_id)
                 # Optional: Zero out observation? The observe func already handles inactive agents.
                 # Optional: Set termination flag for removed agents?
                 terminations[agent_id] = True # Agent's episode ends if tagged
                 # print(f"   Agent {agent_id} removed due to tag.")
             # Clean up dangling references (though observe should handle missing keys)
             if agent_id in self.agent_positions: del self.agent_positions[agent_id]
             if agent_id in self.agent_status: del self.agent_status[agent_id]


        # --- Update Game State (Flags, Scoring) ---
        # Flag Pickup (Check only active agents)
        pickup_occurred = False
        # Check if red flag can be picked up by an active blue agent
        if self.red_flag_status == 0:
            for blue_agent in self.blue_agents:
                 # Ensure agent is active, exists in positions, and is close enough
                 if blue_agent in self.active_agents and blue_agent in self.agent_positions and \
                    distance(self.agent_positions[blue_agent], self.red_flag_pos) <= CAPTURE_RADIUS:
                      self.red_flag_carrier = blue_agent
                      self.red_flag_status = 1
                      self.agent_status[blue_agent] = 1 # has_flag
                      rewards[blue_agent] += REWARD_PICKUP_FLAG
                      print(f"[Game Event Cycle {self.cycles}] {blue_agent} picked up Red flag!")
                      pickup_occurred = True
                      break # Only one agent can pick up per cycle

        # Check if blue flag can be picked up by an active red agent
        if not pickup_occurred and self.blue_flag_status == 0:
             for red_agent in self.red_agents:
                 if red_agent in self.active_agents and red_agent in self.agent_positions and \
                    distance(self.agent_positions[red_agent], self.blue_flag_pos) <= CAPTURE_RADIUS:
                     self.blue_flag_carrier = red_agent
                     self.blue_flag_status = 1
                     self.agent_status[red_agent] = 1 # has_flag
                     rewards[red_agent] += REWARD_PICKUP_FLAG
                     print(f"[Game Event Cycle {self.cycles}] {red_agent} picked up Blue flag!")
                     break # Only one agent can pick up per cycle


        # Scoring (Check only active agents)
        scored_this_cycle = False
        if self.red_flag_carrier is not None: # Blue has Red flag
            carrier = self.red_flag_carrier
            # Ensure carrier is still active before checking score
            if carrier in self.active_agents and carrier in self.agent_positions and \
               distance(self.agent_positions[carrier], self.blue_flag_base) <= FLAG_SCORE_RADIUS:
                 self.scores["blue"] += 1
                 rewards[carrier] += REWARD_FLAG_SCORE_SCORER # Scorer reward
                 for tm in self.blue_agents: # Team reward
                      if tm in self.active_agents and tm != carrier: rewards[tm] += REWARD_FLAG_SCORE_TEAMMATE
                 print(f"[Game Event Cycle {self.cycles}] Blue scored! Agent: {carrier}")
                 scored_this_cycle = True
                 # Reset flag state after score
                 self.red_flag_pos = self.red_flag_base.copy() # Reset to its base
                 self.red_flag_status = 0
                 if carrier in self.agent_status: self.agent_status[carrier] = 0 # No longer has flag
                 self.red_flag_carrier = None

        if not scored_this_cycle and self.blue_flag_carrier is not None: # Red has Blue flag
            carrier = self.blue_flag_carrier
            if carrier in self.active_agents and carrier in self.agent_positions and \
               distance(self.agent_positions[carrier], self.red_flag_base) <= FLAG_SCORE_RADIUS:
                 self.scores["red"] += 1
                 rewards[carrier] += REWARD_FLAG_SCORE_SCORER # Scorer reward
                 for tm in self.red_agents: # Team reward
                      if tm in self.active_agents and tm != carrier: rewards[tm] += REWARD_FLAG_SCORE_TEAMMATE
                 print(f"[Game Event Cycle {self.cycles}] Red scored! Agent: {carrier}")
                 scored_this_cycle = True
                 # Reset flag state after score
                 self.blue_flag_pos = self.blue_flag_base.copy() # Reset to its base
                 self.blue_flag_status = 0
                 if carrier in self.agent_status: self.agent_status[carrier] = 0 # No longer has flag
                 self.blue_flag_carrier = None

        # --- REMOVED old tagging/return logic as it's handled by collision now ---

        # --- Update Cycle Count and Check Termination/Truncation ---
        self.cycles += 1
        # Check if all agents on a team are tagged out
        no_active_blue = all(agent not in self.active_agents for agent in self.blue_agents)
        no_active_red = all(agent not in self.active_agents for agent in self.red_agents)
        game_over = no_active_blue or no_active_red or scored_this_cycle

        if game_over:
            for ag in self.possible_agents: # Terminate all agents if game is over
                 # If not already terminated by tag, set termination flag
                 if not terminations[ag]:
                    terminations[ag] = True
            print(f"[Game Event Cycle {self.cycles}] Episode ended. Score: B {self.scores['blue']}-R {self.scores['red']}. Cause: {'Score' if scored_this_cycle else 'Team Wiped'}")

        # Check truncation based on max cycles AFTER game over check
        if self.cycles >= MAX_CYCLES and not game_over:
            for ag in self.possible_agents:
                if not terminations[ag]: # Only truncate if not already terminated
                    truncations[ag] = True
            game_over = True # Set game_over flag for logic below
            print(f"[Game Event Cycle {self.cycles}] Episode truncated due to max cycles.")


        # --- Prepare return values ---
        # Generate observations only for currently active agents? No, need full dict for trainer.
        # Observe handles inactive agents by returning zeros.
        observations = {agent: self.observe(agent) for agent in self.possible_agents}

        # Update infos for all agents (even inactive/terminated ones)
        current_info = {"score_blue": self.scores["blue"], "score_red": self.scores["red"]}
        for agent_id in self.possible_agents:
             infos[agent_id] = current_info.copy()


        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos


    def render(self):
        """Renders the environment state (basic text)."""
        if self.render_mode != "human": return

        print(f"--- Cycle {self.cycles} ---")
        print(f"Scores: Blue {self.scores['blue']} - Red {self.scores['red']}")
        bf_c = self.blue_flag_carrier if self.blue_flag_carrier else 'Base'
        rf_c = self.red_flag_carrier if self.red_flag_carrier else 'Base'
        # Also print base locations
        print(f"Blue Flag @ [{self.blue_flag_pos[0]:.4f},{self.blue_flag_pos[1]:.4f}] (Base: [{self.blue_flag_base[0]:.4f},{self.blue_flag_base[1]:.4f}], Carrier: {bf_c})")
        print(f"Red Flag @ [{self.red_flag_pos[0]:.4f},{self.red_flag_pos[1]:.4f}] (Base: [{self.red_flag_base[0]:.4f},{self.red_flag_base[1]:.4f}], Carrier: {rf_c})")
        print("Agents:")
        for agent_id in sorted(self.possible_agents): # Sort for consistent order
            if agent_id in self.active_agents:
                team = "B" if "blue" in agent_id else "R"
                status = self.agent_status.get(agent_id, 0)
                stat_str = "Active" if status == 0 else "HasFlag"
                pos = self.agent_positions[agent_id]
                print(f"  {agent_id} ({team}): [{pos[0]:.4f},{pos[1]:.4f}] ({stat_str})")
            # else: print(f"  {agent_id}: (Inactive)") # Optionally show inactive
        print("-" * 40)

    def close(self):
        """Clean up resources."""
        pass

# --- Example Usage / Testing (Updated Obs Dim Print) ---
if __name__ == "__main__":
    print("\nRunning manual Torch loop example...")
    env = CaptureTheFlagEnvTorch(render_mode="human")
    print(f"Observation space shape: {env.observation_space.shape}") # Verify new shape
    obs, info = env.reset()
    print("Sample observation (blue_0):", obs['blue_0']) # Print one obs to see structure

    step_count = 0
    max_total_steps = 50 # Limit steps for testing

    episode_rewards = {agent: 0.0 for agent in env.possible_agents}
    done = False

    while not done and step_count < max_total_steps:
        actions = {agent: env.action_space.sample() for agent in env.possible_agents if agent in env.active_agents}
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        for agent_id, reward in rewards.items():
            if agent_id in episode_rewards:
                episode_rewards[agent_id] += reward
        obs = next_obs
        term_values = terminations.values()
        trunc_values = truncations.values()
        if any(term_values) or any(trunc_values):
            print(f"Episode finished at step {step_count+1}")
            done = True
        step_count += 1

    print("\n--- Episode Summary ---")
    print(f"Final Scores: Blue {env.scores['blue']} - Red {env.scores['red']}")
    print("Total Rewards per Agent:")
    for agent_id, total_reward in sorted(episode_rewards.items()):
        print(f"  {agent_id}: {total_reward:.2f}")

    env.close()
    print(f"Manual loop finished after {step_count} steps.")
