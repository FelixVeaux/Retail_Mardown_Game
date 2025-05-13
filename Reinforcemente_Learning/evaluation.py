# Phase 4: Policy Evaluation and Analysis
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming other modules are accessible
import environment
import model_training
import data_preparation

# --- RL Agent Evaluation ---
def load_trained_model(model_path=model_training.FINAL_MODEL_FILE):
    """Loads a pre-trained Q-network model for evaluation."""
    model = model_training.QNetwork(model_training.STATE_DIM, model_training.ACTION_DIM, seed=0).to(model_training.DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=model_training.DEVICE))
        model.eval() # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def evaluate_policy(env, model, n_episodes=1000):
    """Evaluates the learned RL policy deterministically (epsilon=0) with action masking.

    Args:
        env: The simulation environment instance.
        model: The loaded, trained Q-network model.
        n_episodes: Number of episodes to run for evaluation.

    Returns:
        A dictionary containing evaluation results (rewards, inventories, etc.).
    """
    print(f"--- Evaluating RL Policy ({n_episodes} episodes) ---")
    if model is None:
        print("Model not provided or failed to load. Skipping RL evaluation.")
        return None

    all_rewards = []
    final_inventories = []
    episode_lengths = []
    markdown_actions_taken = defaultdict(int) # Count how many times each markdown action was used

    for i in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        valid_actions_mask = info.get("valid_actions_mask", np.ones(env.action_space.n))

        while not done and steps < data_preparation.N_WEEKS:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(model_training.DEVICE)
            with torch.no_grad():
                action_values = model(state_tensor)

            # Apply mask and choose best valid action
            masked_action_values = action_values.cpu().data.numpy()[0].copy()
            masked_action_values[valid_actions_mask == 0] = -np.inf
            action = np.argmax(masked_action_values).astype(int)

            if action != 0: # Count markdown actions (excluding 'keep price')
                markdown_actions_taken[action] += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            valid_actions_mask = info.get("valid_actions_mask", np.ones(env.action_space.n))

            state = next_state
            episode_reward += reward
            steps += 1

        all_rewards.append(episode_reward)
        final_inventories.append(env.current_inventory) # Inventory at the end
        episode_lengths.append(steps)

        if (i + 1) % (n_episodes // 10) == 0:
            print(f"\rEvaluated episode {i+1}/{n_episodes}", end="")
    print("\nRL Policy Evaluation finished.")

    results = {
        "policy_name": "RL Agent (Fine-tuned)",
        "rewards": np.array(all_rewards),
        "final_inventories": np.array(final_inventories),
        "episode_lengths": np.array(episode_lengths),
        "markdown_action_counts": dict(markdown_actions_taken),
        "n_episodes": n_episodes
    }
    return results


# --- Benchmark Policies ---
def linear_markdown_policy(week, current_price, current_inventory, valid_actions_mask):
    """Simple linear markdown: drop price every X weeks.
       Example: Drop every 5 weeks (week 5, 10, 15)
    """
    # Determine target price based on week
    if week == 5 and current_price == 60.0:
        target_action = 1 # Drop to 54
    elif week == 10 and current_price == 54.0:
        target_action = 2 # Drop to 48
    elif week == 15 and current_price == 48.0:
        target_action = 3 # Drop to 36
    else:
        target_action = 0 # Keep price

    # Check if the target action is valid, otherwise keep price
    if target_action != 0 and valid_actions_mask[target_action] == 1:
        return target_action
    else:
        return 0 # Default to keep price if target drop isn't valid or needed

def rule_based_policy(week, current_price, current_inventory, valid_actions_mask):
    """Example rule-based policy based on inventory level.
       Drops price faster if inventory is high.
    """
    # Use discretized inventory for rules
    disc_inv = data_preparation.discretize_inventory(current_inventory)
    inventory_ratio = current_inventory / data_preparation.INITIAL_INVENTORY
    target_action = 0 # Default keep price

    # Example Rules (can be much more complex)
    if current_price == 60.0:
        if inventory_ratio > 0.8 and week >= 5:
            target_action = 1 # Drop to 54
        elif inventory_ratio > 0.6 and week >= 8:
            target_action = 1
    elif current_price == 54.0:
        if inventory_ratio > 0.5 and week >= 10:
            target_action = 2 # Drop to 48
        elif inventory_ratio > 0.3 and week >= 12:
             target_action = 2
    elif current_price == 48.0:
        if inventory_ratio > 0.2 and week >= 13:
            target_action = 3 # Drop to 36
        elif inventory_ratio > 0.1 and week >= 14:
            target_action = 3

    # Check validity
    if target_action != 0 and valid_actions_mask[target_action] == 1:
        return target_action
    else:
        # If the intended action is not valid, check if keeping price is valid
        if valid_actions_mask[0] == 1:
            return 0
        else:
            # If even keeping price is invalid (shouldn't happen unless end state?), find any valid action
            valid_indices = np.where(valid_actions_mask == 1)[0]
            return valid_indices[0] if len(valid_indices) > 0 else 0 # Fallback

def random_policy(week, current_price, current_inventory, valid_actions_mask):
    """Chooses a random valid action."""
    valid_indices = np.where(valid_actions_mask == 1)[0]
    if len(valid_indices) > 0:
        return np.random.choice(valid_indices)
    else:
        return 0 # Fallback if no valid actions (should only happen at end state)


def run_benchmark(env, policy_func, policy_name, n_episodes=1000):
    """Runs evaluation for a given benchmark policy function."""
    print(f"--- Evaluating Benchmark: {policy_name} ({n_episodes} episodes) ---")
    all_rewards = []
    final_inventories = []
    episode_lengths = []
    markdown_actions_taken = defaultdict(int)

    for i in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        valid_actions_mask = info.get("valid_actions_mask", np.ones(env.action_space.n))

        while not done and steps < data_preparation.N_WEEKS:
            week = int(state[0])
            current_price = data_preparation.PRICE_LEVELS[int(state[2])] # Decode price from index
            # Inventory needs to be 'undiscretized' for policy logic if needed, or use discretized form
            # Benchmark policies here use raw inventory ratio, so we need env internal state
            current_inventory_raw = env.current_inventory # Get from env directly

            action = policy_func(week, current_price, current_inventory_raw, valid_actions_mask)

            if action != 0:
                markdown_actions_taken[action] += 1

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            valid_actions_mask = info.get("valid_actions_mask", np.ones(env.action_space.n))

            state = next_state
            episode_reward += reward
            steps += 1

        all_rewards.append(episode_reward)
        final_inventories.append(env.current_inventory)
        episode_lengths.append(steps)

        if (i + 1) % (n_episodes // 10) == 0:
            print(f"\rEvaluated episode {i+1}/{n_episodes}", end="")
    print(f"\n{policy_name} Evaluation finished.")

    results = {
        "policy_name": policy_name,
        "rewards": np.array(all_rewards),
        "final_inventories": np.array(final_inventories),
        "episode_lengths": np.array(episode_lengths),
        "markdown_action_counts": dict(markdown_actions_taken),
        "n_episodes": n_episodes
    }
    return results

# --- Analysis and Visualization ---
def analyze_results(rl_results, benchmark_results_list):
    """Analyzes and compares results from RL agent and benchmarks."""
    print("\n--- Evaluation Analysis --- ")

    all_results = []
    if rl_results:
        all_results.append(rl_results)
    all_results.extend(benchmark_results_list)

    if not all_results:
        print("No evaluation results to analyze.")
        return

    # --- Summary Statistics --- #
    summary_data = []
    for res in all_results:
        mean_reward = np.mean(res["rewards"])
        std_reward = np.std(res["rewards"])
        median_reward = np.median(res["rewards"])
        mean_final_inv = np.mean(res["final_inventories"])
        median_final_inv = np.median(res["final_inventories"])
        print(f"Policy: {res['policy_name']}")
        print(f"  Avg Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Median Reward: {median_reward:.2f}")
        print(f"  Avg Final Inv: {mean_final_inv:.2f}")
        print(f"  Median Final Inv: {median_final_inv:.2f}")
        # print(f"  Markdown Counts: {res.get('markdown_action_counts', {})}")
        summary_data.append({
            "Policy": res["policy_name"],
            "Mean Reward": mean_reward,
            "Median Reward": median_reward,
            "Std Reward": std_reward,
            "Mean Final Inv": mean_final_inv,
            "Median Final Inv": median_final_inv
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(summary_df.to_string(index=False, float_format="{:.2f}".format))

    # --- Plotting --- #
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except IOError:
        print("Seaborn style 'seaborn-v0_8-darkgrid' not found. Using default style.")
        # plt.style.use('ggplot') # Fallback style

    # 1. Reward Distribution Plot
    plt.figure(figsize=(12, 7))
    plot_data = []
    policy_names = []
    for res in all_results:
        plot_data.append(res["rewards"])
        policy_names.append(res["policy_name"])

    sns.boxplot(data=plot_data, palette="viridis")
    plt.xticks(ticks=range(len(policy_names)), labels=policy_names, rotation=15, ha="right")
    plt.title(f"Distribution of Total Rewards per Episode (N={all_results[0]['n_episodes'] if all_results else 'N/A'})")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig("evaluation_reward_distribution.png")
    print("\nSaved reward distribution plot to evaluation_reward_distribution.png")
    # plt.show()
    plt.close()

    # 2. Final Inventory Distribution Plot
    plt.figure(figsize=(12, 7))
    plot_data_inv = []
    for res in all_results:
        plot_data_inv.append(res["final_inventories"])

    sns.boxplot(data=plot_data_inv, palette="viridis")
    plt.xticks(ticks=range(len(policy_names)), labels=policy_names, rotation=15, ha="right")
    plt.title(f"Distribution of Final Inventories per Episode (N={all_results[0]['n_episodes'] if all_results else 'N/A'})")
    plt.ylabel("Final Inventory")
    plt.tight_layout()
    plt.savefig("evaluation_final_inventory_distribution.png")
    print("Saved final inventory distribution plot to evaluation_final_inventory_distribution.png")
    # plt.show()
    plt.close()

# --- Sensitivity Analysis (Placeholder) ---
def run_sensitivity_analysis(env_class, model, base_lift_tier, n_episodes=500):
    """Placeholder for sensitivity analysis (e.g., varying demand, initial inventory)."""
    print("\n--- Sensitivity Analysis (Placeholder) --- ")
    # Example: Test performance on different lift tiers
    for tier in [1, 2, 3]:
        print(f"\nTesting on Lift Tier: {tier}")
        try:
            env_sens = env_class(lift_tier=tier)
            results = evaluate_policy(env_sens, model, n_episodes=n_episodes)
            if results:
                mean_reward = np.mean(results["rewards"])
                mean_final_inv = np.mean(results["final_inventories"])
                print(f"  Avg Reward: {mean_reward:.2f}")
                print(f"  Avg Final Inv: {mean_final_inv:.2f}")
            env_sens.close()
        except Exception as e:
            print(f"  Error running sensitivity test for tier {tier}: {e}")

    # Could also vary initial inventory, demand parameters in env, etc.

if __name__ == '__main__':
    # Example usage / testing of evaluation functions
    print("Running evaluation.py standalone example...")

    # 1. Setup Environment
    try:
        eval_env = environment.RetailMarkdownEnv(lift_tier=1) # Example tier 1
    except Exception as e:
        print(f"Error initializing environment: {e}")
        eval_env = None

    if eval_env:
        # 2. Load the fine-tuned RL Model (assuming it exists from model_training step)
        rl_model = load_trained_model(model_training.FINAL_MODEL_FILE)

        # 3. Evaluate RL Policy
        if rl_model:
            rl_eval_results = evaluate_policy(eval_env, rl_model, n_episodes=100) # Fewer episodes for testing
        else:
            print("RL model not found or failed to load. Skipping RL evaluation.")
            rl_eval_results = None

        # 4. Evaluate Benchmark Policies
        benchmark_policies_map = {
            "Linear Markdown": linear_markdown_policy,
            "Rule-Based": rule_based_policy,
            "Random Valid Actions": random_policy
        }
        benchmark_eval_results_list = []
        for name, policy_func in benchmark_policies_map.items():
            results = run_benchmark(eval_env, policy_func, name, n_episodes=100)
            benchmark_eval_results_list.append(results)

        # 5. Analyze and Compare Results
        analyze_results(rl_eval_results, benchmark_eval_results_list)

        # 6. Run Sensitivity Analysis (Optional)
        # run_sensitivity_analysis(environment.RetailMarkdownEnv, rl_model, base_lift_tier=1, n_episodes=50)

        eval_env.close()
    else:
        print("Skipping evaluation standalone test due to environment initialization error.")

    print("\nStandalone evaluation example finished.")
 