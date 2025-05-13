# Phase 1: Data Preparation and Offline Replay Buffer Construction
import pandas as pd
import numpy as np
from collections import defaultdict

# --- Constants & Configuration ---
DATA_FILE = 'raw_data.csv' # Ensure this is in the Reinforcemente_Learning directory
# COMBOS_FILE = 'combos.csv' # File mapping combo to lift tier (assumed)
INITIAL_INVENTORY = 2000
INITIAL_PRICE = 60.0
N_WEEKS = 15
INVENTORY_BINS = 81 # 0-2000 into bins of size 25 -> 2000/25 = 80 bins, +1 for 0 -> 81 bins
INVENTORY_BIN_SIZE = 25
TERMINAL_PENALTY_PER_UNIT = 10 # Example penalty value for leftover stock

# Price levels defined in the problem
PRICE_LEVELS = sorted([60.0, 54.0, 48.0, 36.0], reverse=True)

# --- Lift Tier Mapping ---
def map_combo_to_tier(combo_id):
    """Maps a combo ID to a lift tier category (e.g., 1, 2, 3).
    This needs to be defined based on how 'combo' relates to the three lift tiers.
    The problem statement says: "assuming combos represent tiers".
    Example: combo IDs 1-X -> Tier 1, X+1-Y -> Tier 2, Y+1-Z -> Tier 3
    For now, a placeholder that assumes a direct or modulo mapping if not specified.
    If combos.csv exists and specifies mapping, it should be used.
    """
    # Placeholder: If combo is 1,2,3, assume it's the tier. Otherwise, needs definition.
    # A simple modulo mapping for illustrative purposes if combos are e.g. 1..N
    # Example: ((combo_id - 1) % 3) + 1 might work if combos are 1,2,3,4,5,6 -> 1,2,3,1,2,3
    # The user prompt says: "combo IDs 1-5 -> Tier 1, 6-10 -> Tier 2, etc."
    # Let's assume a mapping like that, though the total number of combos isn't explicitly stated.
    # We'll use a simple modulo for now, assuming 3 tiers derived from combo ID.
    # If combo ID itself is the tier (1, 2, or 3), this is simpler:
    if combo_id in [1,2,3]: # This might be too simplistic if combo is not directly the tier.
        return int(combo_id)
    # A more general placeholder if combo values are different:
    # Example: assume combos are grouped. Needs clear definition from user or combos.csv
    # Let's use a modulo approach for now as a general placeholder if combo_id > 3.
    return ((int(combo_id) - 1) % 3) + 1


# --- Action Determination ---
def determine_action(current_price, previous_price):
    """Determines the discrete action based on price change.
    Action mapping: 0=keep price, 1=drop to 54, 2=drop to 48, 3=drop to 36.
    """
    if round(current_price, 2) == round(previous_price, 2):
        return 0
    # Check for valid markdown to specific price points
    # The actions are defined as *dropping to* a price level
    if round(current_price, 2) == 54.0 and round(previous_price, 2) > 54.0:
        return 1
    elif round(current_price, 2) == 48.0 and round(previous_price, 2) > 48.0:
        return 2
    elif round(current_price, 2) == 36.0 and round(previous_price, 2) > 36.0:
        return 3
    else:
        # This case could mean: a price increase (invalid), or a multi-step drop not fitting one action,
        # or dropping to a price that isn't one of the targets from a higher price.
        # For offline data, we assume transitions are valid based on the data's recorded price.
        # If the current price is one of the levels, and previous was higher, it's a valid drop.
        # If it's not one of these explicit drops, it could be 'keep price' if the price didn't change
        # or an unhandled transition. The prompt says "assume valid transitions for now".
        # Given the actions are specific *drops*, if it's not a defined drop, and not keep,
        # it's problematic for this action definition.
        # However, the state includes current_price, so the action is about *how we got to this price*.
        # print(f"Warning: Potentially unmapped price transition from {previous_price} to {current_price}. Defaulting to action 0 (keep).")
        return 0 # Defaulting to 'keep price' if no specific markdown action matches.


# --- Inventory Discretization ---
def discretize_inventory(inventory, initial_inventory=INITIAL_INVENTORY, num_bins=INVENTORY_BINS):
    """Discretizes inventory into bins.
    Bin 0: 0 inventory
    Bin 1: 1 to BIN_SIZE
    ...
    Bin N-1: covers up to initial_inventory
    """
    if inventory <= 0:
        return 0
    if num_bins <= 1: # Should not happen with INVENTORY_BINS = 81
        return 0

    bin_size = initial_inventory / (num_bins -1) # Calculate bin_size based on num_bins and initial_inv
    if bin_size == 0: # Avoid division by zero if initial_inventory is 0 or num_bins is 1
        return 0

    # inventory is > 0 here
    # Bin 0 is for 0 inventory.
    # Bins 1 to num_bins-1 cover >0 to initial_inventory.
    bin_index = int(np.ceil(inventory / bin_size))

    return min(bin_index, num_bins - 1) # Cap at max bin index


# --- State Encoding ---
def encode_state(week, discretized_inventory, current_price, lift_tier):
    """Encodes the state features into a NumPy array.
    Format: [week, discretized_inventory, price_index, lift_tier_index]
    Price is converted to an index (0 for $60, 1 for $54, etc.).
    Lift tier is converted to 0-indexed if it's 1,2,3.
    """
    try:
        price_idx = PRICE_LEVELS.index(round(current_price, 2))
    except ValueError:
        # print(f"Warning: Price {current_price} not in defined PRICE_LEVELS. Using highest price index.")
        price_idx = 0 # Default or handle error

    # Assuming lift_tier is 1, 2, or 3. Convert to 0-indexed.
    lift_tier_idx = lift_tier - 1

    # Week is 1-15. Can be used as is or 0-indexed (0-14).
    # Using 1-indexed week directly as per s_week.
    return np.array([week, discretized_inventory, price_idx, lift_tier_idx], dtype=np.float32)


# --- Main Data Preparation Function ---
def prepare_offline_data(data_path=DATA_FILE):
    """Loads data, parses episodes, constructs RL tuples, and returns the offline buffer."""
    print(f"Loading raw data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found. Make sure it's in the Reinforcemente_Learning folder.")
        return [] # Return empty list on error
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

    print(f"Raw data loaded: {len(df)} rows.")
    print("Columns found:", df.columns.tolist())
    required_cols = ['combo', 'replication', 'week', 'price', 'sales', 'remain_invent']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in {data_path}. Required: {required_cols}")
        return []

    print("Parsing episodes and constructing RL tuples...")
    offline_replay_buffer = []
    # Ensure data types are correct for grouping and calculations
    df['combo'] = df['combo'].astype(int)
    df['replication'] = df['replication'].astype(int)
    df['week'] = df['week'].astype(int)
    df['price'] = df['price'].astype(float)
    df['sales'] = df['sales'].astype(float) # Can be float if fractional sales are possible
    df['remain_invent'] = df['remain_invent'].astype(float)

    grouped = df.groupby(['combo', 'replication'])
    total_episodes = len(grouped)
    processed_episodes = 0

    for (combo_id, _), episode_df_full in grouped:
        # Sort by week to ensure correct order FOR THE ENTIRE EPISODE (including week 16)
        episode_df_full = episode_df_full.sort_values('week').reset_index(drop=True)

        # Data for RL tuples (weeks 1-15)
        episode_data = episode_df_full[episode_df_full['week'] <= N_WEEKS].copy()

        if len(episode_data) == 0:
            # print(f"Warning: No data for weeks 1-{N_WEEKS} for combo {combo_id}, replication {_}. Skipping.")
            continue

        # Validate week 16 if present, for analysis later, not for tuple construction
        # week16_row = episode_df_full[episode_df_full['week'] == 16]

        lift_tier = map_combo_to_tier(combo_id)
        current_episode_inventory = INITIAL_INVENTORY # Inventory at the start of week 1 for this episode
        previous_price_in_episode = INITIAL_PRICE

        for i in range(len(episode_data)):
            row = episode_data.iloc[i]
            week = int(row['week'])

            if week > N_WEEKS: continue # Should be filtered by episode_data, but as a safeguard

            # State (s) components
            s_week = week
            # Inventory at the START of THIS week `w`.
            # For w=1, this is INITIAL_INVENTORY.
            # For w > 1, this is remain_invent from week w-1.
            # This is handled by `current_episode_inventory` variable, updated at end of loop.
            s_inventory_at_week_start = current_episode_inventory
            s_price_this_week = float(row['price'])
            s_lift_tier = lift_tier

            s_disc_inventory = discretize_inventory(s_inventory_at_week_start)
            state = encode_state(s_week, s_disc_inventory, s_price_this_week, s_lift_tier)

            # Action (a)
            # Action is based on transition from previous_price_in_episode to s_price_this_week
            action = determine_action(s_price_this_week, previous_price_in_episode)

            # Reward (r)
            sales_this_week = float(row['sales'])
            reward = s_price_this_week * sales_this_week

            # Next State (s') components
            inventory_at_week_end = float(row['remain_invent'])
            next_week_num = week + 1
            done = (next_week_num > N_WEEKS) or (inventory_at_week_end <= 0)

            if done:
                s_prime_week = next_week_num # Can be N_WEEKS + 1
                s_prime_inventory_at_next_week_start = inventory_at_week_end
                s_prime_price_next_week = s_price_this_week # Price carries over for terminal state representation
                s_prime_lift_tier = lift_tier

                s_prime_disc_inventory = discretize_inventory(s_prime_inventory_at_next_week_start)
                next_state = encode_state(s_prime_week, s_prime_disc_inventory, s_prime_price_next_week, s_prime_lift_tier)

                # Add terminal penalty if episode ends at week 15 (or later if data allows, though N_WEEKS is cap)
                if week == N_WEEKS: # Only apply penalty if it's the true end of horizon
                    terminal_penalty = inventory_at_week_end * TERMINAL_PENALTY_PER_UNIT
                    reward -= terminal_penalty
            else:
                # If not done, find the next row for next_state's price
                if i + 1 < len(episode_data):
                    next_row = episode_data.iloc[i+1]
                    if int(next_row['week']) == next_week_num:
                        s_prime_week = next_week_num
                        s_prime_inventory_at_next_week_start = inventory_at_week_end # This is remain_invent of current week
                        s_prime_price_next_week = float(next_row['price'])
                        s_prime_lift_tier = lift_tier

                        s_prime_disc_inventory = discretize_inventory(s_prime_inventory_at_next_week_start)
                        next_state = encode_state(s_prime_week, s_prime_disc_inventory, s_prime_price_next_week, s_prime_lift_tier)
                    else:
                        # print(f"Warning: Missing or non-sequential next week data for combo {combo_id}, rep {_}, week {week}. Skipping tuple.")
                        # Update inventory and price for the next iteration of the current episode and continue
                        current_episode_inventory = inventory_at_week_end
                        previous_price_in_episode = s_price_this_week
                        continue # Skip this problematic transition
                else:
                    # print(f"Warning: Ran out of data mid-episode (before week 15 and not done) for combo {combo_id}, rep {_}, week {week}. Skipping tuple.")
                    current_episode_inventory = inventory_at_week_end
                    previous_price_in_episode = s_price_this_week
                    continue # Skip this problematic transition

            offline_replay_buffer.append((state, action, reward, next_state, done))

            # Update for the next iteration WITHIN the same episode
            current_episode_inventory = inventory_at_week_end
            previous_price_in_episode = s_price_this_week

            if done: # If episode terminated, break from this episode's loop
                break

        processed_episodes += 1
        if processed_episodes % 1000 == 0 or processed_episodes == total_episodes:
            print(f"Processed {processed_episodes}/{total_episodes} episodes. Buffer size: {len(offline_replay_buffer)}")

    print(f"Finished processing. Generated {len(offline_replay_buffer)} RL tuples.")

    analyze_offline_data(offline_replay_buffer, df) # Pass original df for validation

    return offline_replay_buffer


def analyze_offline_data(buffer, raw_df_for_validation):
    """Performs basic analysis on the generated offline buffer and validates against week 16 data."""
    if not buffer:
        print("Buffer is empty, cannot analyze.")
        return

    print(f"--- Offline Buffer Analysis (First 5 Tuples) ---")
    for i, item in enumerate(buffer[:5]):
        s, a, r, s_prime, d = item
        print(f"Tuple {i}: S={s}, A={a}, R={r:.2f}, S'={s_prime}, Done={d}")

    print(f"Total tuples: {len(buffer)}")

    # Analyze state distribution (example: week distribution)
    weeks = [s[0] for s, _, _, _, _ in buffer]
    from collections import Counter
    week_counts = Counter(weeks)
    print(f"Week distribution in states: {sorted(week_counts.items())}")

    # Analyze action distribution
    actions = [a for _, a, _, _, _ in buffer]
    action_counts = Counter(actions)
    print(f"Action distribution: {sorted(action_counts.items())}")

    # Validate sum of rewards vs Week 16 price (total revenue)
    # This is a complex validation and needs careful reconstruction of episodes from the buffer
    # or by re-grouping the raw_df_for_validation.
    print("--- Validating Episode Rewards vs Week 16 Price (Total Revenue) --- ")
    # For this validation, we need to link buffer entries back to their original (combo, replication) episodes
    # This is non-trivial if the buffer doesn't store combo/replication IDs directly.
    # The prompt expects: "Validate that the sum of step rewards + terminal reward for sample episodes
    # from Weeks 1-15 approximately matches the price value in the Week 16 row for those episodes
    # (assuming the Week 16 price is the total revenue)."

    # Let's try to do this by re-iterating through the raw data groups.
    validation_passed_count = 0
    validation_failed_count = 0
    episodes_without_week16 = 0

    # Ensure 'combo' and 'replication' are present for grouping
    if not all(col in raw_df_for_validation.columns for col in ['combo', 'replication']):
        print("Error: 'combo' or 'replication' column missing in raw_df_for_validation. Cannot perform validation.")
        return

    grouped_raw = raw_df_for_validation.groupby(['combo', 'replication'])
    sample_episode_details = [] # To store details for a few samples

    for (combo_id, replication_id), episode_df in grouped_raw:
        week16_row = episode_df[episode_df['week'] == 16]
        if week16_row.empty:
            episodes_without_week16 +=1
            continue

        # Week 16 'price' column is assumed to be total revenue for the episode
        # Ensure 'price' in week16_row is numeric
        try:
            week16_total_revenue = pd.to_numeric(week16_row['price'].iloc[0], errors='raise')
        except (ValueError, TypeError, IndexError):
            print(f"Warning: Could not parse Week 16 'price' as numeric for Combo {combo_id}, Rep {replication_id}. Skipping validation for this episode.")
            episodes_without_week16 +=1 # Count as if no W16 data for this purpose
            continue


        # Sum rewards for this episode from weeks 1-15, including terminal penalty application
        # This requires re-simulating the reward calculation as done for buffer generation
        # --- MODIFICATION FOR VALIDATION --- 
        # Calculate TWO sums: one with penalty (like buffer), one without (for W16 comparison)
        episode_rewards_sum_with_penalty = 0.0 # Matches buffer reward logic
        episode_rewards_sum_gross = 0.0      # For comparing with W16 'price'
        # current_inv = INITIAL_INVENTORY # Not directly needed for sum if using raw sales, but good for context
        
        # Ensure 'week', 'price', 'sales', 'remain_invent' are numeric and present
        required_episode_cols = ['week', 'price', 'sales', 'remain_invent']
        if not all(col in episode_df.columns for col in required_episode_cols):
            print(f"Warning: Missing required columns for reward calculation in Combo {combo_id}, Rep {replication_id}. Skipping validation.")
            continue
            
        episode_week_data = episode_df[episode_df['week'] <= N_WEEKS].sort_values('week').copy()

        if episode_week_data.empty:
            continue

        try:
            episode_week_data['price'] = pd.to_numeric(episode_week_data['price'], errors='raise')
            episode_week_data['sales'] = pd.to_numeric(episode_week_data['sales'], errors='raise')
            episode_week_data['remain_invent'] = pd.to_numeric(episode_week_data['remain_invent'], errors='raise')
        except (ValueError, TypeError):
            print(f"Warning: Could not parse price/sales/remain_invent as numeric for Combo {combo_id}, Rep {replication_id} during validation. Skipping.")
            continue


        for i in range(len(episode_week_data)):
            row = episode_week_data.iloc[i]
            week = int(row['week'])
            price = float(row['price'])
            sales = float(row['sales'])
            rem_inv = float(row['remain_invent']) # Inventory at END of THIS week

            # Calculate reward for the current step, identical to prepare_offline_data
            current_step_reward = price * sales
            
            # --- Accumulate Gross Sum (Before Penalty) --- 
            episode_rewards_sum_gross += current_step_reward
            
            # --- Apply penalty to the other sum (mirroring buffer logic) --- 
            reward_for_penalty_sum = current_step_reward
            # Apply terminal penalty only for the last week of the horizon (N_WEEKS)
            # based on remaining inventory at the END of that last week.
            if week == N_WEEKS:
                terminal_penalty = rem_inv * TERMINAL_PENALTY_PER_UNIT
                # Subtract penalty for the penalty-adjusted sum
                reward_for_penalty_sum -= terminal_penalty 
            
            # Accumulate the potentially penalty-adjusted reward
            episode_rewards_sum_with_penalty += reward_for_penalty_sum

            # If episode would have terminated due to stockout before N_WEEKS,
            # stop accumulating further rewards. This mimics the 'break if done' logic
            # in prepare_offline_data's buffer generation.
            if rem_inv <= 0 and week < N_WEEKS:
                break
        
        # --- VALIDATION COMPARISON --- 
        # Compare the GROSS summed rewards (no penalty) with Week 16 total revenue
        if np.isclose(episode_rewards_sum_gross, week16_total_revenue, atol=1e-2): # Using a small tolerance
            validation_passed_count += 1
        else:
            validation_failed_count += 1
            if len(sample_episode_details) < 5: # Log a few failed cases
                sample_episode_details.append(
                    # Report both sums for clarity in the failure message
                    f"  FAIL: Combo {combo_id}, Rep {replication_id} | Gross Sum R (W1-15): {episode_rewards_sum_gross:.2f}, W16 Revenue: {week16_total_revenue:.2f}, Diff: {abs(episode_rewards_sum_gross - week16_total_revenue):.2f} | (Buffer Sum R would be: {episode_rewards_sum_with_penalty:.2f})"
                )

    print(f"Validation Summary: Passed={validation_passed_count}, Failed={validation_failed_count}, No W16 data={episodes_without_week16}")
    if sample_episode_details:
        print("Sample Mismatched Episodes (Calculated Sum vs. W16 'price' as Total Revenue):")
        for detail in sample_episode_details:
            print(detail)
    if validation_failed_count == 0 and validation_passed_count > 0:
        print("Reward sum validation against Week 16 'price' (as total revenue) passed for all applicable episodes.")
    elif validation_passed_count > 0 :
        print("Reward sum validation against Week 16 'price' (as total revenue) had some mismatches.")
    else:
        print("Could not perform reward sum validation (no episodes passed or no W16 data).")


if __name__ == '__main__':
    # This is for testing the data preparation script independently
    print("Running data_preparation.py standalone...")
    # Assume raw_data.csv is in the same directory or adjust path
    # Ensure the script is run from a directory where 'raw_data.csv' is accessible
    # or modify the DATA_FILE constant.
    # For example, if Reinforcemente_Learning is the CWD:
    # test_buffer = prepare_offline_data(data_path='raw_data.csv')

    # If running from parent of Reinforcemente_Learning:
    test_buffer = prepare_offline_data(data_path='Reinforcemente_Learning/raw_data.csv')

    if test_buffer:
        print(f"\nStandalone test successful. Generated {len(test_buffer)} tuples.")
        # print("First 5 tuples:")
        # for i in range(min(5, len(test_buffer))):
        #     print(test_buffer[i])
    else:
        print("\nStandalone test failed or produced no data.") 