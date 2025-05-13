# Phase 3: Model Implementation and Hybrid Training
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import copy

# Assuming data_preparation and environment are accessible
import data_preparation
# import environment # Environment not needed for offline-only training

# --- Constants and Hyperparameters ---
STATE_DIM = 4 # week, disc_inv, price_idx, tier_idx (as encoded)
ACTION_DIM = 4 # keep, drop 54, drop 48, drop 36

# Training Params
GAMMA = 0.99           # Discount factor
LR = 5e-4              # Learning rate for Adam optimizer
BATCH_SIZE = 128       # Minibatch size for training
# BUFFER_SIZE = int(1e6) # Replay buffer size (only needed for online)
# TARGET_UPDATE_FREQ = 1000 # How often to update the target network (online steps)

# Epsilon-Greedy (only needed for online)
# EPS_START = 0.1
# EPS_END = 0.01
# EPS_DECAY = 0.995

# Offline Training (CQL) Params
CQL_ALPHA = 5.0        # Conservative Regularization weight (Needs tuning)
OFFLINE_TARGET_UPDATE_FREQ = 100 # How often to update target net in offline phase (steps)

# Files
# PRETRAINED_MODEL_FILE = 'pretrained_model.pth' # No longer needed
FINAL_MODEL_FILE = 'offline_trained_model.pth' # Default final model name

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Replay Buffer Tuple (used for offline data structure, but buffer class not needed)
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# --- Neural Network Architecture ---
class QNetwork(nn.Module):
    """Simple Feed-Forward Neural Network for Q-value approximation."""
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Replay Buffer (for Online phase - COMMENTED OUT) ---
# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""
#     def __init__(self, action_size, buffer_size, batch_size, seed):
#         self.action_size = action_size
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.seed = random.seed(seed)
#
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to memory."""
#         e = Transition(state, action, reward, next_state, done)
#         self.memory.append(e)
#
#     def sample(self):
#         """Randomly sample a batch of experiences from memory."""
#         experiences = random.sample(self.memory, k=self.batch_size)
#
#         # Convert to tensors
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(DEVICE)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)
#
#         return (states, actions, rewards, next_states, dones)
#
#     def __len__(self):
#         return len(self.memory)

# --- RL Agent (Handles Networks, Offline Buffer, Learning) ---
class RLAgent():
    """Interacts with and learns from the offline data."""
    def __init__(self, state_size, action_size, offline_buffer, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.copy_weights(self.qnetwork_local, self.qnetwork_target) # Initialize target net

        # Offline Replay Buffer (provided)
        self.offline_buffer = offline_buffer # List of (s, a, r, s', d) tuples
        if not isinstance(self.offline_buffer, list):
             try:
                 self.offline_buffer = list(self.offline_buffer)
             except TypeError:
                 raise ValueError("offline_buffer must be a list or convertible to a list of tuples.")

        print(f"Agent initialized with {len(self.offline_buffer)} offline experiences.")

        # Online Replay Buffer (Not used in this setup)
        # self.online_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Training step counter (for target network updates)
        self.t_step_offline = 0
        # self.t_step_online = 0 # Not used

    def copy_weights(self, source_network, target_network):
        """Copies weights from source to target network."""
        target_network.load_state_dict(source_network.state_dict())

    # --- Offline Training (CQL) ---
    def train_offline_cql(self, epochs, batch_size, model_save_path, cql_alpha=CQL_ALPHA, target_update_freq=OFFLINE_TARGET_UPDATE_FREQ):
        """Trains the agent using the offline buffer with CQL and saves the model."""
        if not self.offline_buffer:
            print("Offline buffer is empty. Skipping offline training.")
            return

        if len(self.offline_buffer) < batch_size:
            print(f"Warning: Offline buffer size ({len(self.offline_buffer)}) is smaller than batch size ({batch_size}). Training cannot proceed.")
            # Or adjust batch size: batch_size = len(self.offline_buffer)
            return

        num_batches = len(self.offline_buffer) // batch_size
        print(f"Starting CQL Offline Training for {epochs} epochs ({num_batches} batches per epoch)...")
        print(f"CQL Alpha: {cql_alpha}, Target Update Freq: {target_update_freq}, Batch Size: {batch_size}")

        losses = []
        q1_losses = []
        cql_losses = []

        self.qnetwork_local.train() # Set model to training mode

        for epoch in range(1, epochs + 1):
            # Shuffle offline buffer indices for batch sampling
            indices = np.arange(len(self.offline_buffer))
            np.random.shuffle(indices)

            epoch_loss = 0.0
            epoch_q1_loss = 0.0
            epoch_cql_loss = 0.0

            for i in range(num_batches):
                # Sample a minibatch from the offline buffer
                batch_indices = indices[i * batch_size:(i + 1) * batch_size]
                try:
                    minibatch = [self.offline_buffer[idx] for idx in batch_indices]
                    if not all(isinstance(item, tuple) and len(item) == 5 for item in minibatch):
                        print(f"Error: Invalid data format in minibatch at epoch {epoch}, batch {i}.")
                        continue # Skip this batch

                    # Convert batch to tensors
                    states = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(DEVICE)
                    actions = torch.from_numpy(np.array([e[1] for e in minibatch])).long().unsqueeze(1).to(DEVICE)
                    rewards = torch.from_numpy(np.array([e[2] for e in minibatch])).float().unsqueeze(1).to(DEVICE)
                    next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(DEVICE)
                    dones = torch.from_numpy(np.array([e[4] for e in minibatch]).astype(np.uint8)).float().unsqueeze(1).to(DEVICE)
                except Exception as e:
                     print(f"Error processing batch at epoch {epoch}, batch {i}: {e}")
                     # Potentially log the problematic minibatch or indices
                     continue # Skip this batch

                # --- CQL Loss Calculation ---
                # 1. Standard DQN Loss (Bellman Error)
                with torch.no_grad():
                    Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
                    Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

                Q_expected = self.qnetwork_local(states).gather(1, actions)
                q1_loss = F.mse_loss(Q_expected, Q_targets)

                # 2. CQL Regularization Term
                q_values_all_actions = self.qnetwork_local(states)
                logsumexp_q = torch.logsumexp(q_values_all_actions, dim=1, keepdim=True)
                q_value_taken_action = self.qnetwork_local(states).gather(1, actions)
                cql_loss = (logsumexp_q - q_value_taken_action).mean()

                # --- Total Loss --- #
                loss = q1_loss + cql_alpha * cql_loss

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_q1_loss += q1_loss.item()
                epoch_cql_loss += cql_loss.item()

                # Update target network
                self.t_step_offline = (self.t_step_offline + 1)
                if self.t_step_offline % target_update_freq == 0:
                    self.copy_weights(self.qnetwork_local, self.qnetwork_target)
                    # print(f"Target network updated at step {self.t_step_offline}")

            # Ensure division by zero doesn't occur if num_batches is 0
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_q1_loss = epoch_q1_loss / num_batches
                avg_cql_loss = epoch_cql_loss / num_batches
                losses.append(avg_loss)
                q1_losses.append(avg_q1_loss)
                cql_losses.append(avg_cql_loss)

                if epoch % 100 == 0 or epoch == epochs:
                    print(f"Epoch {epoch}/{epochs}\tAvg Loss: {avg_loss:.4f}\tQ1 Loss: {avg_q1_loss:.4f}\tCQL Loss: {avg_cql_loss:.4f}")
            else:
                 print(f"Epoch {epoch}/{epochs} skipped (num_batches=0).")

        print("Offline CQL training finished.")
        self.save_model(model_save_path) # Save the final model
        self.qnetwork_local.eval() # Set model to eval mode after training

    # --- Online Fine-tuning (COMMENTED OUT) --- #
    # def online_step(self, state, action, reward, next_state, done):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     # Save experience / reward
    #     self.online_memory.add(state, action, reward, next_state, done)
    #
    #     # Learn, if enough samples are available in memory
    #     if len(self.online_memory) > BATCH_SIZE:
    #         experiences = self.online_memory.sample()
    #         self._learn_dqn(experiences, GAMMA)
    #
    # def _learn_dqn(self, experiences, gamma):
    #     """Update value parameters using given batch of experience tuples (Standard DQN update)."""
    #     states, actions, rewards, next_states, dones = experiences
    #
    #     # Get max predicted Q values (for next states) from target model
    #     Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    #     # Compute Q targets for current states
    #     Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #
    #     # Get expected Q values from local model
    #     Q_expected = self.qnetwork_local(states).gather(1, actions)
    #
    #     # Compute loss
    #     loss = F.mse_loss(Q_expected, Q_targets)
    #     # Minimize the loss
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     # Update target network
    #     self.t_step_online = (self.t_step_online + 1) % TARGET_UPDATE_FREQ
    #     if self.t_step_online == 0:
    #         self.copy_weights(self.qnetwork_local, self.qnetwork_target)


    def act(self, state, eps=0., valid_actions_mask=None):
        """Returns actions for given state as per current policy.
           Only uses exploitation (eps=0) by default after offline training.
           Applies action masking if provided.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval() # Set model to evaluation mode
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # No need to switch back to train mode if only doing inference/evaluation
        # self.qnetwork_local.train()

        # Action selection (primarily exploitation)
        if random.random() > eps: # If eps > 0, allows for some exploration if needed
            # Exploitation: choose the best valid action
            if valid_actions_mask is not None:
                # Ensure mask is numpy array
                if isinstance(valid_actions_mask, torch.Tensor):
                    valid_actions_mask = valid_actions_mask.cpu().numpy()
                elif not isinstance(valid_actions_mask, np.ndarray):
                     valid_actions_mask = np.array(valid_actions_mask)

                masked_action_values = action_values.cpu().data.numpy()[0].copy()

                # Check dimensions if necessary
                # print("Mask shape:", valid_actions_mask.shape)
                # print("Action values shape:", masked_action_values.shape)
                if masked_action_values.shape[0] != valid_actions_mask.shape[0]:
                    print("Error: Action mask dimension mismatch!")
                    # Handle error: maybe return default action 0
                    return 0

                masked_action_values[valid_actions_mask == 0] = -np.inf
                return np.argmax(masked_action_values).astype(int)
            else:
                # No mask provided, choose best overall action
                return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            # Exploration (if eps > 0)
            if valid_actions_mask is not None:
                if isinstance(valid_actions_mask, torch.Tensor):
                    valid_actions_mask = valid_actions_mask.cpu().numpy()
                elif not isinstance(valid_actions_mask, np.ndarray):
                     valid_actions_mask = np.array(valid_actions_mask)

                valid_indices = np.where(valid_actions_mask == 1)[0]
                if len(valid_indices) > 0:
                    return random.choice(valid_indices).astype(int)
                else:
                    return 0 # Fallback
            else:
                return random.choice(np.arange(self.action_size)).astype(int)

    # --- Model Persistence ---
    def save_model(self, filepath):
        """Saves the local Q-network weights."""
        try:
            # Get the directory part of the filepath
            dir_name = os.path.dirname(filepath)
            # Only create directories if dir_name is not empty (i.e., filepath includes a path)
            if dir_name:
                 os.makedirs(dir_name, exist_ok=True)
            # Save the model state dictionary
            torch.save(self.qnetwork_local.state_dict(), filepath)
            print(f"Model saved successfully to {filepath}") # Changed message to reflect success here
        except Exception as e:
             print(f"Error saving model to {filepath}: {e}")


    def load_model(self, filepath):
        """Loads weights into the local Q-network."""
        if os.path.exists(filepath):
            try:
                self.qnetwork_local.load_state_dict(torch.load(filepath, map_location=DEVICE))
                # Also copy to target network after loading, useful if target net needed later
                self.copy_weights(self.qnetwork_local, self.qnetwork_target)
                self.qnetwork_local.eval() # Set to evaluation mode after loading
                self.qnetwork_target.eval()
                print(f"Model loaded from {filepath}")
            except Exception as e:
                print(f"Error loading model from {filepath}: {e}")
        else:
            print(f"Warning: Model file {filepath} not found. Cannot load model.")

# --- Online Fine-tuning Loop (COMMENTED OUT) ---
# def train_online_dqn(agent, env, n_episodes=2000, max_t=data_preparation.N_WEEKS,
#                        eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
#     """Fine-tunes the agent using online interaction with the environment.
#     (REMOVED/COMMENTED OUT for offline-only workflow)
#     """
#     # ... implementation removed ...
#     pass


# Example of how to potentially split offline buffer (if needed for validation during offline training)
# from sklearn.model_selection import train_test_split
# train_buffer, val_buffer = train_test_split(agent.offline_buffer, test_size=0.1, random_state=seed)
# Monitor performance on val_buffer using standard Q-loss (without CQL term)


if __name__ == '__main__':
    # Example usage / testing (Focus on Offline Training)
    print("Running model_training.py standalone example (Offline focus)...")

    # 1. Prepare dummy offline data
    print("Creating dummy offline buffer...")
    dummy_buffer = []
    for _ in range(1000): # Small dummy buffer
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a = random.randint(0, ACTION_DIM - 1)
        r = random.random() * 100
        s_prime = np.random.rand(STATE_DIM).astype(np.float32)
        d = random.random() < 0.1
        dummy_buffer.append((s, a, r, s_prime, d))
    print(f"Dummy buffer size: {len(dummy_buffer)}")

    # 2. Initialize Agent
    agent = RLAgent(state_size=STATE_DIM, action_size=ACTION_DIM, offline_buffer=dummy_buffer, seed=0)

    # 3. Run Offline Training
    model_save_path = os.path.join('Reinforcemente_Learning', 'standalone_test_' + FINAL_MODEL_FILE)
    agent.train_offline_cql(epochs=5, batch_size=32, model_save_path=model_save_path)

    # 4. Test loading saved model
    print("\nTesting model loading...")
    agent_loaded = RLAgent(state_size=STATE_DIM, action_size=ACTION_DIM, offline_buffer=[], seed=1) # New agent
    agent_loaded.load_model(model_save_path) # Load the model saved by offline training

    # 5. Test inference with loaded model (optional)
    if agent_loaded.qnetwork_local:
        dummy_state = np.random.rand(STATE_DIM).astype(np.float32)
        dummy_mask = np.array([1, 1, 0, 0]) # Example mask
        action = agent_loaded.act(dummy_state, eps=0.0, valid_actions_mask=dummy_mask) # Act deterministically
        print(f"Loaded agent - Sample deterministic action for state {dummy_state}: {action}")

    print("\nStandalone example finished.") 