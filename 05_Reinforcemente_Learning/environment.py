# Phase 2: Environment Design (for Online Fine-tuning & Evaluation)
import numpy as np
import gym
from gym import spaces

# Assuming data_preparation.py is in the same directory or accessible in PYTHONPATH
import data_preparation # For constants and encoding functions

class RetailMarkdownEnv(gym.Env):
    """Custom Environment for Retail Markdown Simulation.
    Follows the OpenAI Gym interface.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, lift_tier, initial_inventory=data_preparation.INITIAL_INVENTORY,
                 initial_price=data_preparation.INITIAL_PRICE, n_weeks=data_preparation.N_WEEKS,
                 penalty_per_unit=data_preparation.TERMINAL_PENALTY_PER_UNIT):
        super(RetailMarkdownEnv, self).__init__()

        self.initial_inventory = initial_inventory
        self.initial_price = initial_price
        self.n_weeks = n_weeks
        self.penalty_per_unit = penalty_per_unit
        self.lift_tier = lift_tier # Tier for this instance (1, 2, or 3)

        # Defined price levels from problem description/data_preparation
        self.price_levels = data_preparation.PRICE_LEVELS # [60.0, 54.0, 48.0, 36.0]
        self.num_actions = 4 # 0: keep, 1: drop to 54, 2: drop to 48, 3: drop to 36
        self.action_space = spaces.Discrete(self.num_actions)

        # Define observation space
        # State: [week, discretized_inventory, price_index, lift_tier_index]
        # week: 1 to N_WEEKS (or N_WEEKS + 1 for terminal)
        # discretized_inventory: 0 to INVENTORY_BINS - 1
        # price_index: 0 to len(price_levels) - 1
        # lift_tier_index: 0 to num_tiers - 1 (assuming 3 tiers -> 0, 1, 2)
        self.max_week_val = self.n_weeks + 1 # To accommodate terminal state week
        self.max_inv_val = data_preparation.INVENTORY_BINS -1
        self.max_price_idx_val = len(self.price_levels) -1
        self.max_lift_tier_idx_val = 3 - 1 # Assuming 3 tiers

        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 0], dtype=np.float32),
            high=np.array([self.max_week_val, self.max_inv_val, self.max_price_idx_val, self.max_lift_tier_idx_val], dtype=np.float32),
            dtype=np.float32
        )

        # Demand lift factors (Tier 1, Tier 2, Tier 3)
        # These correspond to price levels: $60, $54, $48, $36
        # Lift factor is 1.0 for $60 base price.
        # Tiers given: 1.30, 1.79, 2.81. These are TOTAL demand factors at optimal prices.
        # The problem states: "Known demand lift tiers (1.30, 1.79, 2.81) and their mapping to item 'combos'"
        # It also implies lift factors are PER PRICE LEVEL.
        # We need a clear lift table: (tier, price) -> lift_factor
        # Example: If 1.30, 1.79, 2.81 are for the *best* markdown for that tier, how does it map to specific prices?
        # For now, let's make a simplified assumption for the lift table structure.
        # Assume the provided tiers [1.30, 1.79, 2.81] are the *maximum lift factors achievable for that tier*,
        # likely at the lowest price ($36). Lift at $60 is always 1.0.
        # We need to interpolate or define lifts for intermediate prices ($54, $48).

        # Simplified lift table: {tier: {price: lift_factor}}
        # Tier 1 max lift = 1.30
        # Tier 2 max lift = 1.79
        # Tier 3 max lift = 2.81
        # Assuming linear interpolation of lift between $60 (lift=1) and $36 (max_lift_for_tier)
        self.lift_factors_table = {
            1: self._interpolate_lifts(1.0, 1.30), # Tier 1
            2: self._interpolate_lifts(1.0, 1.79), # Tier 2
            3: self._interpolate_lifts(1.0, 2.81)  # Tier 3
        }

        # Environment state variables
        self.current_week = 0
        self.current_inventory = 0
        self.current_price = 0.0
        self.current_lift_tier_idx = self.lift_tier -1 # 0-indexed

        self.rng = np.random.default_rng() # Random number generator for demand

    def _interpolate_lifts(self, lift_at_60, lift_at_36):
        """Helper to linearly interpolate lift factors for intermediate prices."""
        # Prices: $60, $54, $48, $36
        # Lift at $60 is lift_at_60
        # Lift at $36 is lift_at_36
        # Lifts for $54 and $48 are interpolated.
        # Relative distances from $60: $54 (6), $48 (12), $36 (24)
        # Total range for interpolation: $60 - $36 = $24
        price_map = {}
        price_map[60.0] = lift_at_60
        price_map[36.0] = lift_at_36
        price_map[54.0] = lift_at_60 + (lift_at_36 - lift_at_60) * ( (60.0 - 54.0) / (60.0 - 36.0) )
        price_map[48.0] = lift_at_60 + (lift_at_36 - lift_at_60) * ( (60.0 - 48.0) / (60.0 - 36.0) )
        # Round for stability
        for p in price_map: price_map[p] = round(price_map[p], 3)
        return price_map

    def _get_lift_factor(self, price, tier):
        """Gets the lift factor for a given price and tier."""
        tier_lifts = self.lift_factors_table.get(tier)
        if not tier_lifts:
            # print(f"Warning: Lift tier {tier} not found in lift table. Defaulting to tier 1.")
            tier_lifts = self.lift_factors_table.get(1, {60.0: 1.0, 54.0: 1.0, 48.0: 1.0, 36.0: 1.0})
        return tier_lifts.get(round(price,2), 1.0) # Default to 1.0 if price not found

    def _simulate_demand(self):
        """Simulates demand for the current week based on price and lift tier.
        Base demand at $60 is sampled from a distribution (e.g., Normal, Poisson).
        Problem statement: "stochastic demand"
        Let's assume base demand at $60 is N(mu, sigma), e.g., N(100, 20), truncated at 0.
        """
        base_demand_mu = 100
        base_demand_sigma = 20 # Std dev
        base_demand_at_60 = self.rng.normal(base_demand_mu, base_demand_sigma)
        base_demand_at_60 = max(0, int(round(base_demand_at_60))) # Ensure non-negative integer

        lift_factor = self._get_lift_factor(self.current_price, self.lift_tier)
        actual_demand = base_demand_at_60 * lift_factor
        return int(round(actual_demand))

    def _get_target_price(self, action):
        """Determines the target price based on the current price and action."""
        # Action: 0=keep, 1=drop to 54, 2=drop to 48, 3=drop to 36
        if action == 0: # Keep price
            return self.current_price
        elif action == 1: # Drop to $54
            return 54.0
        elif action == 2: # Drop to $48
            return 48.0
        elif action == 3: # Drop to $36
            return 36.0
        else:
            # print(f"Warning: Invalid action index {action}. Keeping current price.")
            return self.current_price # Should not happen with discrete action space

    def _is_action_valid(self, target_price):
        """Checks if the target price is a valid markdown (not higher than current).
           And if it's one of the allowed price levels.
        """
        if round(target_price,2) not in self.price_levels:
            return False # Target price is not one of the allowed levels
        if round(target_price,2) > round(self.current_price,2):
            return False # Cannot increase price
        return True

    def get_valid_actions_mask(self):
        """Returns a mask of valid actions. 1 if valid, 0 if invalid."""
        mask = np.zeros(self.num_actions, dtype=np.int8)
        for action_idx in range(self.num_actions):
            target_price = self._get_target_price(action_idx)
            # Condition 1: Target price must be a valid price level (implicitly handled by _get_target_price)
            # Condition 2: Target price must not be higher than current price.
            # Condition 3: If action is a specific drop, current price must be higher than target.
            #              If action is 'keep', it's always valid unless it implies a drop to same price.

            if action_idx == 0: # Keep price
                mask[action_idx] = 1
                continue

            # For drop actions (1, 2, 3)
            if target_price < self.current_price:
                mask[action_idx] = 1
            elif target_price == self.current_price and action_idx != 0: # Trying to drop to current price
                mask[action_idx] = 0 # This specific drop action isn't meaningful, use action 0
            else: # Target price >= current_price for a drop action
                mask[action_idx] = 0
        return mask

    def _encode_current_state(self):
        """Encodes the current environment state using the shared encoding function."""
        discretized_inv = data_preparation.discretize_inventory(self.current_inventory)
        # State: [week, discretized_inventory, price_index, lift_tier_index]
        return data_preparation.encode_state(
            self.current_week,
            discretized_inv,
            self.current_price,
            self.lift_tier # lift_tier is 1,2,3; encode_state handles indexing
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.current_week = 1
        self.current_inventory = self.initial_inventory
        self.current_price = self.initial_price # Start at $60
        # self.current_lift_tier_idx is fixed by constructor

        observation = self._encode_current_state()
        info = {"valid_actions_mask": self.get_valid_actions_mask()}
        return observation, info

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not (0 <= action < self.num_actions):
            raise ValueError(f"Invalid action: {action}. Action must be an int in [0, {self.num_actions-1}] ")

        target_price = self._get_target_price(action)

        # Action Validation (as per Phase 2.4.1 in prompt)
        # If agent chooses an action that implies dropping to a price P,
        # but current price is already <= P, this is effectively "keep price".
        # The mask should ideally prevent strictly invalid actions (price increase).

        # If the action chosen implies a markdown but the price is already at or below that target markdown,
        # treat it as 'keep current price' effectively.
        # Or, rely on action masking to prevent agent selecting such non-progressing drop actions.
        # The core idea is that the *resulting price* is what matters for demand simulation.

        original_price_for_step = self.current_price

        # Update current_price based on valid action
        # If action is e.g. "drop to 54" (action=1) but current price is 48, this is invalid.
        # The agent should use action masking from `get_valid_actions_mask()`.
        # We assume the action passed to step() is one that *can* be taken from current state.
        if self._is_action_valid(target_price):
            self.current_price = target_price
        else:
            # This case suggests the agent picked an action that is not valid from the current state/price.
            # This should ideally be handled by action masking *before* calling step.
            # If it happens, penalize or simply maintain state.
            # For now, if an invalid action is somehow passed, we keep the current price.
            pass # Price remains self.current_price (which was original_price_for_step)

        # Simulate Demand
        demand = self._simulate_demand()

        # Calculate Revenue & Inventory
        sales_this_week = min(demand, self.current_inventory)
        reward = self.current_price * sales_this_week
        self.current_inventory -= sales_this_week
        self.current_inventory = max(0, self.current_inventory) # Ensure non-negative

        # Determine Next State components
        # next_price is the price maintained into the next week (price after current action)
        next_s_price = self.current_price
        next_s_inventory_raw = self.current_inventory

        # Increment week
        self.current_week += 1

        # Check done condition
        done = (self.current_week > self.n_weeks) or (self.current_inventory <= 0)

        # Terminal Reward/Penalty (if done)
        if done:
            if self.current_week > self.n_weeks: # End of horizon
                reward -= self.current_inventory * self.penalty_per_unit # Penalty for leftover
            # If done due to inventory_out_of_stock before n_weeks, no additional penalty/reward here
            # (unless specified differently)

        # Encode next_state
        next_s_week = self.current_week
        next_s_disc_inventory = data_preparation.discretize_inventory(next_s_inventory_raw)

        next_observation = data_preparation.encode_state(
            next_s_week,
            next_s_disc_inventory,
            next_s_price,
            self.lift_tier
        )

        info = {"valid_actions_mask": self.get_valid_actions_mask() if not done else np.ones(self.num_actions) }
        # Metadata for debugging or analysis
        info['sales'] = sales_this_week
        info['demand'] = demand
        info['week'] = self.current_week -1 # week during which action was taken
        info['price_action_taken_at'] = original_price_for_step
        info['price_after_action'] = self.current_price

        return next_observation, reward, done, False, info # Gym v26 returns 5 values

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Week: {self.current_week-1}, Inv: {self.current_inventory:.0f}, Price: ${self.current_price:.2f}, Tier: {self.lift_tier}")
        else:
            super(RetailMarkdownEnv, self).render(mode=mode) # Or raise error for unsupported modes

    def close(self):
        pass

# Example usage (for testing)
if __name__ == '__main__':
    print("Testing RetailMarkdownEnv...")
    target_tier = 1 # Test with Tier 1
    env = RetailMarkdownEnv(lift_tier=target_tier)
    obs, info = env.reset()
    print(f"Initial Obs: {obs}, Info: {info}")

    total_reward = 0
    done = False
    week = 0
    max_weeks = data_preparation.N_WEEKS

    while not done and week < max_weeks:
        week += 1
        env.render()
        # Sample a random valid action
        valid_actions_mask = info.get("valid_actions_mask", np.ones(env.action_space.n)) # Default to all if no mask
        valid_action_indices = np.where(valid_actions_mask == 1)[0]
        if len(valid_action_indices) == 0:
            print("No valid actions! Defaulting to action 0 (keep price).")
            action = 0 # Fallback, though should not happen if logic is correct
        else:
            action = env.rng.choice(valid_action_indices)

        print(f"Week {week} - Taking action: {action} (Valid: {valid_action_indices})")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        print(f"  -> Obs: {obs}, Reward: {reward:.2f}, Done: {done}")
        print(f"  -> Info: Sales: {info.get('sales')}, Demand: {info.get('demand')}, Price Set: {info.get('price_after_action')}")
        print(f"  -> Cumulative Reward: {total_reward:.2f}\n")

    env.render() # Final state
    print(f"Episode finished after {week} weeks.")
    print(f"Final Inventory: {env.current_inventory}")
    print(f"Total Reward: {total_reward:.2f}")
    env.close()

    # Test observation space and action space compliance
    print(f"Observation Space: {env.observation_space}")
    print(f"Sample Obs: {env.observation_space.sample()}")
    print(f"Action Space: {env.action_space}")
    print(f"Sample Action: {env.action_space.sample()}") 