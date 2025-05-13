import sys
import os
import numpy as np
import matplotlib.pyplot as plt # Added for plots in evaluation
import csv
import re
import time
import argparse  # For command-line arguments

# --- Selenium Imports (Add these) ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options # To run headless if desired
# Consider using webdriver_manager if chromedriver path is not fixed
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service

# === Local Module Imports ===
# Ensure the Reinforcemente_Learning directory is in the Python path
# Adjust this path manipulation if your script structure is different
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) # Go up one level if needed
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Now import modules from the project
from Reinforcemente_Learning import environment, model_training, evaluation, inference # <-- Changed env_interaction to environment

# --- Configuration Flags ---
# Control which phases of the script run
RUN_OFFLINE_COLLECTION = False # Set to True to collect data from random policy
RUN_OFFLINE_TRAINING = False   # Set to True to train the model from offline data
RUN_EVALUATION = False       # Set to True to evaluate trained model vs baseline
RUN_INTERACTIVE_PLAY = True # Renamed from RUN_INFERENCE_DEMO

# Parameters (can be loaded from a config file)
OFFLINE_BUFFER_FILE = 'offline_replay_buffer.npz' # Relative path, used by other scripts potentially
# PRETRAINED_MODEL_FILE = 'pretrained_model.pth' # No longer needed as separate step
# Construct path relative to the script's location
FINAL_MODEL_FILE = os.path.join(script_dir, 'offline_trained_model.pth')

OFFLINE_EPOCHS = 5000  # Adjust as needed
# ONLINE_EPISODES = 2000 # Not needed for online fine-tuning
EVALUATION_EPISODES = 1000
TARGET_LIFT_TIER = 1 # Example: Assume tier 1 for evaluation/play, adjust if needed
SEED = 42 # Seed for reproducibility

# --- Interactive Play Parameters (Add these) ---
GAME_URL = "https://www.randhawa.us/games/retailer/nyu.html"
DEFAULT_NUM_GAMES_TO_PLAY = 1 # How many games the RL agent should play
RESULTS_CSV_FILE = os.path.join(script_dir, "rl_game_outcomes.csv")

# --- Helper Functions for Interactive Play (Add these) ---

def setup_browser():
    """Sets up and returns a configured Chrome browser instance."""
    print("Setting up browser...")
    chrome_options = Options()
    # Uncomment to run headless (no GUI)
    # chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Optional: Disable images for speed
    # chrome_options.add_argument("--disable-images")
    # chrome_options.add_argument("--blink-settings=imagesEnabled=false")
    chrome_options.page_load_strategy = 'normal' # or 'eager'

    try:
        # Use webdriver_manager if installed
        # from webdriver_manager.chrome import ChromeDriverManager
        # from selenium.webdriver.chrome.service import Service
        # browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        # --- OR specify path to chromedriver if not using manager ---
        # Example: Requires chromedriver executable in the same directory or in PATH
        browser = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error initializing WebDriver: {e}")
        print("Ensure chromedriver is installed and accessible (either in PATH or via webdriver_manager).")
        raise

    # Set timeouts
    browser.set_page_load_timeout(20)
    browser.set_script_timeout(10)

    try:
        print(f"Navigating to {GAME_URL}...")
        browser.get(GAME_URL)
        browser.maximize_window()
        # Wait for the start button to be present as a check
        WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#practiceButton"))
        )
        print("Browser setup complete.")
    except TimeoutException:
        print(f"Timeout waiting for game page elements at {GAME_URL}.")
        browser.quit()
        raise
    except Exception as e:
        print(f"Error navigating to game page: {e}")
        browser.quit()
        raise

    return browser

def get_game_state(driver):
    """Extracts the current week, price, and inventory from the results table."""
    # Placeholder - Implement using Selenium table reading
    print("    Getting game state...")
    try:
        table = driver.find_element(By.CSS_SELECTOR, "#result-table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        if len(rows) > 1: # Check if there's at least one data row
            last_row = rows[-1]
            cells = last_row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 4:
                week = int(cells[0].text)
                price = float(cells[1].text) # Price is likely float
                inventory = int(cells[3].text)
                print(f"    State: Week={week}, Price={price}, Inv={inventory}")
                return {'week': week, 'price': price, 'inventory': inventory, 'game_over': False}
            else:
                 print("    Warning: Last table row doesn't have enough cells.")
                 return {'game_over': True} # Indicate issue or end
        else:
             # If only header row exists, game likely hasn't started providing data rows
             # Or it's the very beginning, state needs default values
             print("    State: Table empty or only header. Assuming start.")
             # You might need to handle the very first state differently if needed
             # For now, let's assume we only call this after week 1 data appears.
             # If called before start, might need default start state (week 1, price 60, inv 2000)
             return None # Indicate state not ready yet or initial state

    except NoSuchElementException:
        print("    Error: Results table not found.")
        return {'game_over': True}
    except Exception as e:
        print(f"    Error getting game state: {e}")
        return {'game_over': True}

def click_action_button(driver, wait, action_index):
    """Clicks the button corresponding to the RL agent's action index."""
    action_map = {
        0: 'maintainButton',
        1: 'tenButton',
        2: 'twentyButton',
        3: 'fortyButton'
    }
    button_id = action_map.get(action_index)
    if button_id:
        print(f"    Clicking Action: {action_index} (Button ID: {button_id})")
        try:
            button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"#{button_id}")))
            button.click()
            return True
        except TimeoutException:
            print(f"    Warning: Action button {button_id} not clickable (likely disabled).")
            # If the intended action button is disabled, maybe try 'maintain' (action 0)?
            # Or just report failure.
            if action_index != 0:
                 print("    Attempting to click 'maintain' as fallback...")
                 return click_action_button(driver, wait, 0) # Try maintain if original fails
            else:
                 print("    Error: 'maintain' button also not clickable.")
                 return False # Failed to click even maintain
        except Exception as e:
            print(f"    Error clicking button {button_id}: {e}")
            return False
    else:
        print(f"    Error: Invalid action index {action_index}")
        return False

def get_final_results(driver, wait):
    """Extracts final revenue and perfect foresight revenue."""
    print("    Getting final results...")
    try:
        # Wait for revenue element to be stable/present
        revenue_elem = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#rev")))
        perfect_elem = driver.find_element(By.CSS_SELECTOR, "#perfect")

        # Clean text to get numbers
        revenue_text = revenue_elem.text
        perfect_text = perfect_elem.text
        revenue = int(re.sub(r'[^0-9]', '', revenue_text)) if revenue_text else 0
        perfect = int(re.sub(r'[^0-9]', '', perfect_text)) if perfect_text else 0
        print(f"    Results: Agent Revenue={revenue}, Optimal Revenue={perfect}")
        return {'agent_revenue': revenue, 'optimal_revenue': perfect}
    except TimeoutException:
        print("    Error: Timeout waiting for final results elements (#rev). Game might not have finished correctly.")
        return {'agent_revenue': 0, 'optimal_revenue': 0} # Indicate failure
    except NoSuchElementException:
        print("    Error: Could not find final result elements (#rev, #perfect).")
        return {'agent_revenue': 0, 'optimal_revenue': 0}
    except Exception as e:
        print(f"    Error extracting final results: {e}")
        return {'agent_revenue': 0, 'optimal_revenue': 0}

def start_game(driver, wait):
    """Clicks the start game button."""
    print("Starting new game...")
    try:
        start_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#practiceButton")))
        start_button.click()
        # Wait for the first row of the table to appear after starting
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#result-table tr:nth-child(2)")))
        print("Game started.")
        return True
    except Exception as e:
        print(f"Error starting game: {e}")
        # Try refreshing and starting again once?
        try:
             print("Refreshing page and trying to start again...")
             driver.refresh()
             start_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#practiceButton")))
             start_button.click()
             WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#result-table tr:nth-child(2)")))
             print("Game started after refresh.")
             return True
        except Exception as e2:
             print(f"Error starting game even after refresh: {e2}")
             return False

def play_one_game_with_rl(driver, inference_module, lift_tier):
    """Plays one full game using the RL agent to decide actions."""
    wait = WebDriverWait(driver, 5) # Shorter wait for actions within game

    if not start_game(driver, wait):
        return None # Failed to start

    game_results = []
    current_week = 0

    while current_week < 15: # Game has 15 weeks where decisions are made
        time.sleep(0.2) # Small delay for page update
        state = get_game_state(driver)

        if state is None or state['game_over']:
            print("Game ended prematurely or state error.")
            break # Exit loop if game over signal or error

        current_week = state['week']
        current_price = state['price']
        current_inventory = state['inventory']

        # Week 1 state is shown, but first action is for week 2
        if current_week >= 15:
             print("Reached end of decision weeks.")
             break

        # Predict action for the *next* week (week + 1)
        # The table shows the state *after* the decision for that week was implicitly made (week 1 shows price 60)
        # Our action influences the state starting from week 2
        print(f"  Predicting action for week {current_week + 1}")
        action_idx = inference_module.predict_action(
            week=current_week + 1, # Predict for the upcoming week
            inventory=current_inventory,
            current_price=current_price,
            lift_tier=lift_tier
        )

        if action_idx == -1:
            print("  RL Agent prediction failed. Stopping game.")
            return None # Agent failed

        # Click the button for the chosen action
        if not click_action_button(driver, wait, action_idx):
            print("  Failed to click action button. Stopping game.")
            return None # Selenium failed

    # After the loop (game finished or broke)
    final_results = get_final_results(driver, wait)
    return final_results

# --- Main Interactive Session Function ---
def run_interactive_session(num_games_to_play, target_lift_tier):
    """Runs the interactive game playing session with the RL agent."""
    print(f"\n--- Running Interactive Online Play with RL Agent for {num_games_to_play} game(s) ---")
    browser = None
    all_game_outcomes = []

    try:
        print(f"Loading model for inference from: {FINAL_MODEL_FILE}")
        inference_module = inference.InferenceModule(model_path=FINAL_MODEL_FILE)
        if not inference_module.model:
            raise Exception(f"Could not load inference model from {FINAL_MODEL_FILE}.")
        print("Inference module loaded successfully.")

        browser = setup_browser()

        for game_num in range(1, num_games_to_play + 1):
            print(f"\n--- Playing Game {game_num}/{num_games_to_play} (Lift Tier: {target_lift_tier}) ---")
            game_outcome = play_one_game_with_rl(browser, inference_module, target_lift_tier)

            if game_outcome:
                all_game_outcomes.append({
                    'game_number': game_num,
                    'agent_revenue': game_outcome['agent_revenue'],
                    'optimal_revenue': game_outcome['optimal_revenue'],
                    'lift_tier': target_lift_tier
                })
                print(f"Game {game_num} finished. Agent Revenue: {game_outcome['agent_revenue']}, Optimal Revenue: {game_outcome['optimal_revenue']}")
            else:
                print(f"Game {game_num} failed or was interrupted.")
            time.sleep(1) # Small pause before next game

        print("\nAll interactive play sessions finished.")

    except Exception as e:
        print(f"Error during Interactive Play setup/loop: {e}")
    finally:
        if browser:
            print("Closing browser...")
            browser.quit()

        if all_game_outcomes:
            print(f"\nSaving game outcomes to {RESULTS_CSV_FILE}...")
            try:
                # Ensure results directory exists (if RESULTS_CSV_FILE includes a path)
                os.makedirs(os.path.dirname(RESULTS_CSV_FILE), exist_ok=True)
                with open(RESULTS_CSV_FILE, 'w', newline='') as csvfile:
                    fieldnames = ['game_number', 'agent_revenue', 'optimal_revenue', 'lift_tier']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_game_outcomes)
                print(f"Results saved successfully to {RESULTS_CSV_FILE}")
            except Exception as e:
                print(f"Error saving results to CSV: {e}")
        else:
            print("No game outcomes to save.")

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL agent for interactive web game play.")
    parser.add_argument(
        "--num_games",
        type=int,
        default=DEFAULT_NUM_GAMES_TO_PLAY,
        help=f"Number of games the RL agent should play (default: {DEFAULT_NUM_GAMES_TO_PLAY})"
    )
    parser.add_argument(
        "--lift_tier",
        type=int,
        default=TARGET_LIFT_TIER,
        help=f"Target lift tier to simulate (default: {TARGET_LIFT_TIER})"
    )
    args = parser.parse_args()

    # Check if only interactive play is intended
    # This script is primarily for interactive play, so we call it directly.
    # The RUN_INTERACTIVE_PLAY flag is implicitly True by running this script.
    # If other phases were to be run from here, more complex logic would be needed.

    print("Starting script: Interactive RL Agent Game Play")
    # Set seed for reproducibility if any stochastic parts exist in agent/env not covered by model
    np.random.seed(SEED)
    # Potentially torch.manual_seed(SEED) if torch is used directly for random numbers here

    run_interactive_session(num_games_to_play=args.num_games, target_lift_tier=args.lift_tier)

    print("\n--- Script Finished ---")



    