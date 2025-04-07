# here's the main file for the MARL API
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer

# --- Import necessary components from training and env scripts ---
# Assuming train_torch_iac.py and ctf_env_torch.py are in the same directory or accessible
try:
    from train_torch_iac import Agent, HIDDEN_DIM, device # Need Agent class, hidden_dim, device
    from ctf_env_torch import (
        CaptureTheFlagEnvTorch, N_AGENTS_PER_TEAM, N_ACTIONS, TOTAL_AGENTS,
        ACTION_MAP, ALL_BOUNDS, normalize_pos, distance, CAPTURE_RADIUS, FLAG_SCORE_RADIUS,
        BLUE_TEAM_BOUNDS, RED_TEAM_BOUNDS, CHAT_FEATURE_DIM # Import bounds if needed for base pos logic
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Ensure train_torch_iac.py and ctf_env_torch.py are accessible.")
    exit(1)

# --- Configuration ---
MODEL_EPISODE = 200 # Choose which saved models to load
VERSION = "v7"
MODEL_DIR = os.path.expanduser(f"~/torch_results/ctf_iac_{VERSION}")
if not os.path.exists(MODEL_DIR):
    print(f"ERROR: Model directory not found at {MODEL_DIR}")
    exit(1)

# --- Global Variables ---
# Will be populated during startup
env_temp = None # Temporary env instance for dims, not for simulation state
agents: Dict[str, Agent] = {}
obs_dim = 0
action_dim = 0
st_model = None
blue_base_pos_global: np.ndarray = None
red_base_pos_global: np.ndarray = None
possible_agents_global: List[str] = []
agent_name_mapping_global: Dict[str, int] = {}

# --- Pydantic Models for API ---
class AgentState(BaseModel):
    lat: float
    lng: float
    status: int # 0 = active, 1 = has_flag (consistent with env)

class FlagState(BaseModel):
    lat: float
    lng: float
    status: int # 0 = base/dropped, 1 = carried
    carrier: Optional[str] = None # Agent ID or None

class ChatMessageIn(BaseModel):
    sender: str
    message: str
    timestamp: float # Assuming timestamp is float epoch seconds
    isComplete: bool

# --- UPDATED: Input GameState model ---
class GameState(BaseModel):
    agents: Dict[str, AgentState] # agent_id -> state
    blue_flag: FlagState
    red_flag: FlagState
    # Added optional base positions from frontend (though global ones are used for logic)
    blue_base: Optional[Dict[str, float]] = None
    red_base: Optional[Dict[str, float]] = None
    # Added chat messages
    blue_chat: Optional[List[ChatMessageIn]] = Field(default_factory=list)
    red_chat: Optional[List[ChatMessageIn]] = Field(default_factory=list)
    # Added scores from frontend (primarily for context, backend calculates next scores)
    score_blue: int = 0
    score_red: int = 0

class GameStateOut(BaseModel):
    agents: Dict[str, AgentState] # agent_id -> state
    blue_flag: FlagState
    red_flag: FlagState
    score_blue: int
    score_red: int

# --- FastAPI App Setup ---
app = FastAPI(title="MARL CTF Inference API")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Function to Reconstruct Observation ---
# This avoids needing a full env instance per request
# Takes current state dictionaries, returns observation numpy array for one agent
def get_observation_for_agent(
    agent_id: str,
    agent_positions: Dict[str, np.ndarray],
    agent_statuses: Dict[str, int],
    blue_flag_pos: np.ndarray,
    red_flag_pos: np.ndarray,
    blue_flag_carrier: Optional[str],
    red_flag_carrier: Optional[str],
    blue_base_pos: np.ndarray,
    red_base_pos: np.ndarray,
    possible_agents: List[str],
    agent_name_mapping: Dict[str, int],
    blue_chat: List[ChatMessageIn],
    red_chat: List[ChatMessageIn],
    st_model_instance: SentenceTransformer # Pass the loaded model
) -> np.ndarray:

    obs_list = []
    current_pos = agent_positions[agent_id]
    current_status = agent_statuses[agent_id]

    # Self Info
    obs_list.extend(normalize_pos(current_pos))
    obs_list.append(np.float32(current_status))

    # Teammate Info
    team_list = [ag for ag in possible_agents if ag.startswith(agent_id.split('_')[0])]
    for tm_id in team_list:
        if tm_id != agent_id:
            if tm_id in agent_positions:
                obs_list.extend(normalize_pos(agent_positions[tm_id]))
            else: # Should not happen if state is complete
                obs_list.extend([np.float32(0.0), np.float32(0.0)])

    # Opponent Info
    opponent_list = [ag for ag in possible_agents if not ag.startswith(agent_id.split('_')[0])]
    for op_id in opponent_list:
        if op_id in agent_positions:
            obs_list.extend(normalize_pos(agent_positions[op_id]))
        else:
             obs_list.extend([np.float32(0.0), np.float32(0.0)])

    # Flag Info
    blue_flag_norm = normalize_pos(blue_flag_pos)
    red_flag_norm = normalize_pos(red_flag_pos)
    own_flag_pos, enemy_flag_pos = (blue_flag_norm, red_flag_norm) if agent_id.startswith("blue") else (red_flag_norm, blue_flag_norm)
    obs_list.extend(own_flag_pos)
    obs_list.extend(enemy_flag_pos)

    # Flag Carrier Info
    blue_carrier_idx = agent_name_mapping.get(blue_flag_carrier, -1)
    red_carrier_idx = agent_name_mapping.get(red_flag_carrier, -1)
    own_carrier_idx, enemy_carrier_idx = (blue_carrier_idx, red_carrier_idx) if agent_id.startswith("blue") else (red_carrier_idx, blue_carrier_idx)
    obs_list.append(np.float32(own_carrier_idx))
    obs_list.append(np.float32(enemy_carrier_idx))

    # Flag BASE Position Info
    blue_base_norm = normalize_pos(blue_base_pos)
    red_base_norm = normalize_pos(red_base_pos)
    own_base_pos, enemy_base_pos = (blue_base_norm, red_base_norm) if agent_id.startswith("blue") else (red_base_norm, blue_base_norm)
    obs_list.extend(own_base_pos)
    obs_list.extend(enemy_base_pos)

    is_blue_team = agent_id.startswith("blue")
    relevant_chat = blue_chat if is_blue_team else red_chat
    
    # Select recent messages (e.g., last 5)
    recent_messages = relevant_chat[-5:]
    
    # Prepare texts for embedding
    texts_to_embed = []
    if recent_messages:
        for msg in recent_messages:
            # Simple concatenation of sender and message
            texts_to_embed.append(f"{msg.sender}: {msg.message}")

    chat_features = np.zeros(CHAT_FEATURE_DIM, dtype=np.float32)
    if texts_to_embed and st_model_instance:
        try:
            # Get embeddings (returns a list of numpy arrays)
            embeddings = st_model_instance.encode(texts_to_embed)
            # Average pooling
            if len(embeddings) > 0:
                chat_features = np.mean(embeddings, axis=0).astype(np.float32)
        except Exception as e:
             print(f"[Warn] Error embedding chat for {agent_id}: {e}") # Log error but continue

    obs_list.extend(chat_features)

    final_obs = np.array(obs_list, dtype=np.float32)

    # Validate shape (important!)
    global obs_dim
    if final_obs.shape[0] != obs_dim:
         raise ValueError(f"Observation shape mismatch for {agent_id}! Expected {obs_dim}, got {final_obs.shape[0]}. List length: {len(obs_list)}")

    return final_obs


# --- Load Models on Startup ---
@app.on_event("startup")
async def load_models_and_bases():
    global env_temp, agents, obs_dim, action_dim
    global blue_base_pos_global, red_base_pos_global
    global possible_agents_global, agent_name_mapping_global
    global st_model
    print("Loading environment, models, sentence transformer, and base positions...")
    try:
        # --- Load Sentence Transformer Model ---
        print("Loading Sentence Transformer model...")
        st_model = SentenceTransformer('all-MiniLM-L6-v2') # Load the model
        print("Sentence Transformer model loaded.")
        # -------------------------------------

        # Create a temporary env just to get space info and agent lists
        env_temp = CaptureTheFlagEnvTorch()
        _, _ = env_temp.reset(seed=42) # Use a fixed seed for deterministic base init if desired
        obs_dim = env_temp.observation_space.shape[0]
        action_dim = env_temp.action_space.n
        possible_agents_list = env_temp.possible_agents
        agent_map = env_temp.agent_name_mapping

        blue_base_pos_global = env_temp.blue_flag_base.copy()
        red_base_pos_global = env_temp.red_flag_base.copy()
        possible_agents_global = env_temp.possible_agents
        agent_name_mapping_global = env_temp.agent_name_mapping
        print(f"Blue Base: {blue_base_pos_global}, Red Base: {red_base_pos_global}")

        print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, Agents: {len(possible_agents_list)}")

        for agent_id in possible_agents_list:
            agent_instance = Agent(
                agent_id, obs_dim, action_dim, HIDDEN_DIM,
                lr_actor=0, lr_critic=0 # LRs don't matter for inference
            )
            model_path_prefix = os.path.join(MODEL_DIR, f"agent_{agent_id}_ep{MODEL_EPISODE}")
            actor_path = f"{model_path_prefix}_actor.pth"
            critic_path = f"{model_path_prefix}_critic.pth"

            if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                 raise FileNotFoundError(f"Model files not found for {agent_id} at episode {MODEL_EPISODE} in {MODEL_DIR}")

            agent_instance.load_models(model_path_prefix)
            agent_instance.actor.eval() # Set actor to evaluation mode
            agent_instance.critic.eval() # Set critic to evaluation mode
            agents[agent_id] = agent_instance
            print(f"Loaded models for {agent_id}")

        print(f"Models loaded successfully for {len(agents)} agents.")

    except Exception as e:
        print(f"FATAL ERROR during model loading: {e}")
        # Depending on deployment, might want to exit or raise further
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


# --- Inference Endpoint ---
@app.post("/step", response_model=GameStateOut)
async def run_step(current_state: GameState):
    """
    Accepts the current game state (including chat and scores), runs one step of inference for all agents,
    updates the state based on actions, checks scoring, and returns the new state including scores.
    NOTE: Chat messages are received but NOT currently used by the agents' models.
    """
    if not agents:
        raise HTTPException(status_code=500, detail="Models not loaded yet.")
    if not env_temp:
        raise HTTPException(status_code=500, detail="Environment info not loaded.")

    # --- 1. Extract current state into formats needed ---
    agent_positions_np = {id: np.array([s.lat, s.lng], dtype=np.float32) for id, s in current_state.agents.items()}
    agent_statuses_dict = {id: s.status for id, s in current_state.agents.items()}
    blue_flag_pos_np = np.array([current_state.blue_flag.lat, current_state.blue_flag.lng], dtype=np.float32)
    red_flag_pos_np = np.array([current_state.red_flag.lat, current_state.red_flag.lng], dtype=np.float32)
    blue_flag_carrier = current_state.blue_flag.carrier
    red_flag_carrier = current_state.red_flag.carrier
    blue_flag_status = current_state.blue_flag.status
    red_flag_status = current_state.red_flag.status

    blue_chat_history = current_state.blue_chat
    red_chat_history = current_state.red_chat
    # print(f"Received Blue Chat: {len(blue_chat_history)} messages") # Optional Debug
    # print(f"Received Red Chat: {len(red_chat_history)} messages")   # Optional Debug
    # ---

    # Store original base positions needed for resets (Use globally loaded ones)
    # No need to extract from current_state, use blue_base_pos_global and red_base_pos_global

    # --- 2. Get actions from models ---
    actions_dict = {}
    with torch.no_grad():
        for agent_id, agent_instance in agents.items():
            if agent_id not in agent_positions_np: continue # Skip if agent somehow missing

            # Generate observation for this agent based on current FULL state
            observation = get_observation_for_agent(
                agent_id,
                agent_positions_np,
                agent_statuses_dict,
                blue_flag_pos_np,
                red_flag_pos_np,
                blue_flag_carrier,
                red_flag_carrier,
                blue_base_pos_global, # Use global base
                red_base_pos_global,  # Use global base
                possible_agents_global, # Use global list
                agent_name_mapping_global, # Use global mapping
                blue_chat_history, # Pass blue chat
                red_chat_history,  # Pass red chat
                st_model           # Pass the loaded ST model
            )

            # Get action from agent's actor network
            action, _ = agent_instance.select_action(observation)
            actions_dict[agent_id] = action

    # --- 3. Simulate one step based on actions (apply movement, check flags/scoring) ---
    next_agent_positions_np = agent_positions_np.copy()
    next_agent_statuses_dict = agent_statuses_dict.copy()
    next_blue_flag_pos_np = blue_flag_pos_np.copy()
    next_red_flag_pos_np = red_flag_pos_np.copy()
    next_blue_flag_carrier = blue_flag_carrier
    next_red_flag_carrier = red_flag_carrier
    next_blue_flag_status = blue_flag_status
    next_red_flag_status = red_flag_status
    scores = {"blue": current_state.score_blue, "red": current_state.score_red} # Initialize scores from input

    # Apply movement
    for agent_id, action in actions_dict.items():
        move_lat, move_lng = ACTION_MAP[action]
        current_pos = next_agent_positions_np[agent_id]
        new_pos = current_pos + np.array([move_lat, move_lng])
        # Clip to global bounds
        new_pos[0] = np.clip(new_pos[0], ALL_BOUNDS["minLat"], ALL_BOUNDS["maxLat"])
        new_pos[1] = np.clip(new_pos[1], ALL_BOUNDS["minLng"], ALL_BOUNDS["maxLng"])
        next_agent_positions_np[agent_id] = new_pos

        # Move flag if carrying
        if next_blue_flag_carrier == agent_id: next_blue_flag_pos_np = new_pos.copy()
        if next_red_flag_carrier == agent_id: next_red_flag_pos_np = new_pos.copy()

    # Check flag pickups (simplified logic from env)
    pickup_occurred = False
    if next_red_flag_status == 0: # If red flag is at base/dropped
        for blue_agent in env_temp.blue_agents:
             if blue_agent in next_agent_positions_np and distance(next_agent_positions_np[blue_agent], next_red_flag_pos_np) <= CAPTURE_RADIUS:
                  next_red_flag_carrier = blue_agent
                  next_red_flag_status = 1
                  next_agent_statuses_dict[blue_agent] = 1 # has_flag
                  pickup_occurred = True
                  print(f"[API Step] {blue_agent} picked up Red flag")
                  break # One pickup per step
    if not pickup_occurred and next_blue_flag_status == 0: # If blue flag is at base/dropped
         for red_agent in env_temp.red_agents:
             if red_agent in next_agent_positions_np and distance(next_agent_positions_np[red_agent], next_blue_flag_pos_np) <= CAPTURE_RADIUS:
                 next_blue_flag_carrier = red_agent
                 next_blue_flag_status = 1
                 next_agent_statuses_dict[red_agent] = 1 # has_flag
                 print(f"[API Step] {red_agent} picked up Blue flag")
                 break

    # Check scoring (simplified logic from env)
    scored_this_cycle = False
    if next_red_flag_carrier is not None: # Blue has Red flag
        carrier = next_red_flag_carrier
        # Use the *global* blue base pos for scoring check
        if carrier in next_agent_positions_np and \
           distance(next_agent_positions_np[carrier], blue_base_pos_global) <= FLAG_SCORE_RADIUS:
             scores["blue"] += 1
             print(f"[API Step] Blue scored! Agent: {carrier}")
             scored_this_cycle = True
             # Reset red flag state
             next_red_flag_pos_np = red_base_pos_global # Reset to its base
             next_red_flag_status = 0
             if carrier in next_agent_statuses_dict: # Check if carrier still exists
                 next_agent_statuses_dict[carrier] = 0 # No longer has flag
             next_red_flag_carrier = None

    if not scored_this_cycle and next_blue_flag_carrier is not None: # Red has Blue flag
        carrier = next_blue_flag_carrier
        # Use the *global* red base pos for scoring check
        if carrier in next_agent_positions_np and \
           distance(next_agent_positions_np[carrier], red_base_pos_global) <= FLAG_SCORE_RADIUS:
             scores["red"] += 1
             print(f"[API Step] Red scored! Agent: {carrier}")
             scored_this_cycle = True
             # Reset blue flag state
             next_blue_flag_pos_np = blue_base_pos_global # Reset to its base
             next_blue_flag_status = 0
             if carrier in next_agent_statuses_dict: # Check if carrier still exists
                 next_agent_statuses_dict[carrier] = 0 # No longer has flag
             next_blue_flag_carrier = None

    # TODO: Add Tagging logic if needed
    # TODO: Add Flag Return logic if needed

    # --- 4. Format the next state into the response model ---
    response_agents = {
        id: AgentState(lat=pos[0], lng=pos[1], status=next_agent_statuses_dict[id])
        for id, pos in next_agent_positions_np.items()
    }
    response_blue_flag = FlagState(
        lat=next_blue_flag_pos_np[0], lng=next_blue_flag_pos_np[1],
        status=next_blue_flag_status, carrier=next_blue_flag_carrier
    )
    response_red_flag = FlagState(
        lat=next_red_flag_pos_np[0], lng=next_red_flag_pos_np[1],
        status=next_red_flag_status, carrier=next_red_flag_carrier
    )

    return GameStateOut(
        agents=response_agents,
        blue_flag=response_blue_flag,
        red_flag=response_red_flag,
        score_blue=scores["blue"],
        score_red=scores["red"]
    )

# --- Root endpoint for basic check ---
@app.get("/")
async def root():
    return {"message": "MARL CTF Inference API is running."}

# --- Optional: Add command to run easily ---
if __name__ == "__main__":
    import uvicorn
    new_port = 8888 # Define your new port
    print(f"Starting API server on http://localhost:{new_port}") # Updated print
    # Use reload=True only for development
    uvicorn.run("main:app", host="0.0.0.0", port=new_port, reload=False) # Use the new port variable
    # For production, use a proper ASGI server like gunicorn+uvicorn workers
    # gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000