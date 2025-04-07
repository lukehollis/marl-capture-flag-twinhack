# train_torch_iac.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from ctf_env_torch import CaptureTheFlagEnvTorch, N_AGENTS_PER_TEAM, N_ACTIONS # Import the new env

# --- Configuration ---
VERSION = "v7"
ENV_NAME = f"capture_the_flag_torch_{VERSION}"
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99 # Discount factor
EPSILON = 0.2 # PPO clip parameter (can be used in actor loss, optional for basic IAC)
ENTROPY_COEFF = 0.01 # Entropy bonus coefficient
VALUE_LOSS_COEFF = 0.5 # Critic loss coefficient
MAX_GRAD_NORM = 0.5 # Gradient clipping
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 1000 # Match env MAX_CYCLES or adjust as needed
UPDATE_INTERVAL = 20 # Number of steps to collect before updating networks
BATCH_SIZE = 64 # Minibatch size for updates
HIDDEN_DIM = 128
SAVE_INTERVAL = 100 # Save models every N episodes
RESULTS_DIR = os.path.expanduser(f"~/torch_results/ctf_iac_{VERSION}")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Define Networks ---

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_actor = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc_actor(x)
        return action_logits # Return logits for Categorical distribution

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1) # Output a single value estimate

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_critic(x)
        return value

# --- Agent Class ---

class Agent:
    def __init__(self, agent_id, obs_dim, action_dim, hidden_dim, lr_actor, lr_critic):
        self.id = agent_id
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(obs_dim, hidden_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.memory = [] # Simple list to store transitions (s, a, r, s', done)

    def select_action(self, state):
        """Selects action based on policy, returns action, log_prob."""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits = self.actor(state)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.cpu().item() # Return integer action and scalar log_prob

    def store_transition(self, state, action, reward, next_state, done):
        # Store raw numpy/python types, convert to tensors during update
        self.memory.append((state, action, reward, next_state, done))

    def update(self, gamma):
        """Performs a gradient update using collected experience."""
        if not self.memory:
            return 0.0, 0.0 # No loss if no memory

        # Convert memory to batches of tensors
        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device) # Action indices need LongTensor
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device) # 0.0 or 1.0

        # --- Critic Update ---
        with torch.no_grad():
            next_values = self.critic(next_states)
            # Target for value function: R_t + gamma * V(S_{t+1}) * (1 - done)
            target_values = rewards + gamma * next_values * (1 - dones)

        current_values = self.critic(states)
        critic_loss = F.mse_loss(current_values, target_values)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
        self.optimizer_critic.step()

        # --- Actor Update ---
        with torch.no_grad():
            # Advantage: A(s,a) = Q(s,a) - V(s) = (r + gamma*V(s')) - V(s)
            advantages = target_values - current_values

        action_logits = self.actor(states)
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions.squeeze(1)).unsqueeze(1) # Ensure shape matches advantages
        entropy = action_dist.entropy().mean()

        # Policy Gradient Loss: -log_prob * Advantage
        actor_loss = -(log_probs * advantages).mean()

        # Total Loss: Actor Loss - Entropy Bonus
        total_actor_loss = actor_loss - ENTROPY_COEFF * entropy

        self.optimizer_actor.zero_grad()
        total_actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
        self.optimizer_actor.step()

        # Clear memory after update
        self.memory = []

        return total_actor_loss.item(), critic_loss.item()

    def save_models(self, path):
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth", map_location=device))


# --- Main Training Loop ---
if __name__ == "__main__":
    env = CaptureTheFlagEnvTorch() # render_mode="human" for visualization
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Create agents
    agents = {
        agent_id: Agent(
            agent_id, obs_dim, action_dim, HIDDEN_DIM,
            LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC
        )
        for agent_id in env.possible_agents
    }

    # Setup TensorBoard
    LOG_DIR = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Saving results to: {RESULTS_DIR}")
    print(f"Logging TensorBoard data to: {LOG_DIR}")

    total_steps = 0
    episode_rewards_history = []

    for i_episode in range(1, NUM_EPISODES + 1):
        observations, _ = env.reset()
        episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
        steps_in_episode = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            total_steps += 1
            steps_in_episode += 1

            # Collect actions from all agents
            actions_dict = {}
            log_probs_dict = {} # Store log_probs if needed for more advanced updates (like PPO)
            for agent_id, agent in agents.items():
                if agent_id in observations: # Check if agent is still active (obs exists)
                    action, log_prob = agent.select_action(observations[agent_id])
                    actions_dict[agent_id] = action
                    log_probs_dict[agent_id] = log_prob
                # else: handle potentially inactive agents if needed

            # Step the environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions_dict)

            # Check if episode ended for any agent
            done_dict = {agent_id: terminations.get(agent_id, False) or truncations.get(agent_id, False)
                         for agent_id in env.possible_agents}
            episode_ended = any(done_dict.values())

            # Store transitions for each agent
            for agent_id, agent in agents.items():
                 if agent_id in observations: # Only store if agent took an action this step
                     state = observations[agent_id]
                     action = actions_dict[agent_id]
                     reward = rewards.get(agent_id, 0.0)
                     next_state = next_observations.get(agent_id, np.zeros(obs_dim, dtype=np.float32)) # Use zero obs if agent terminated
                     done = done_dict.get(agent_id, False)

                     agent.store_transition(state, action, reward, next_state, done)
                     episode_rewards[agent_id] += reward


            observations = next_observations

            # --- Perform Updates ---
            # Update periodically based on total steps or episode end
            # Update based on total_steps collected across all agents
            if total_steps % (UPDATE_INTERVAL * env.n_agents) == 0 or episode_ended:
                 actor_losses = []
                 critic_losses = []
                 print(f"Updating networks at episode {i_episode}, step {step+1} (total steps: {total_steps})")
                 for agent_id, agent in agents.items():
                     if agent.memory: # Only update if agent has experience
                         a_loss, c_loss = agent.update(GAMMA)
                         actor_losses.append(a_loss)
                         critic_losses.append(c_loss)
                 if actor_losses:
                     avg_actor_loss = np.mean(actor_losses)
                     avg_critic_loss = np.mean(critic_losses)
                     print(f"  Avg Actor Loss: {avg_actor_loss:.4f}, Avg Critic Loss: {avg_critic_loss:.4f}")
                     writer.add_scalar('Loss/AvgActor', avg_actor_loss, total_steps)
                     writer.add_scalar('Loss/AvgCritic', avg_critic_loss, total_steps)


            if episode_ended:
                break # Exit step loop

        # --- Logging ---
        avg_episode_reward = np.mean(list(episode_rewards.values()))
        blue_score = infos.get(env.blue_agents[0], {}).get('score_blue', 0) # More robust info access
        red_score = infos.get(env.blue_agents[0], {}).get('score_red', 0)   # More robust info access
        episode_rewards_history.append(avg_episode_reward)
        print(f"Episode {i_episode}: Steps={steps_in_episode}, Avg Reward={avg_episode_reward:.2f}, "
              f"Blue Score: {blue_score}, Red Score: {red_score}")

        # Add episode metrics to TensorBoard
        writer.add_scalar('Reward/AverageEpisodeReward', avg_episode_reward, i_episode)
        writer.add_scalar('Score/Blue', blue_score, i_episode)
        writer.add_scalar('Score/Red', red_score, i_episode)
        writer.add_scalar('Progress/StepsPerEpisode', steps_in_episode, i_episode)

        # --- Save Models ---
        if i_episode % SAVE_INTERVAL == 0:
            print(f"Saving models at episode {i_episode}...")
            for agent_id, agent in agents.items():
                agent.save_models(os.path.join(RESULTS_DIR, f"agent_{agent_id}_ep{i_episode}"))

    # --- Cleanup ---
    writer.close() # Close the writer
    env.close()
    print("Training finished.")

    # TODO: Add plotting of rewards history