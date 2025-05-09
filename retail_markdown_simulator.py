#!/usr/bin/env python3
"""
Retail Markdown Game Simulator
==============================
This script automates playing the Retailer Game using predefined price markdown strategies.
It records results in two CSV files:
- week_detail.csv: Contains weekly data for each simulation
- outcome.csv: Contains final performance metrics for each simulation
"""

import csv
import time
import re
import pandas as pd
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import traceback

# Game URL
GAME_URL = "https://www.randhawa.us/games/retailer/nyu.html"

def generate_price_strategies():
    """
    Generate all valid price markdown strategies and returns as a DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame with all valid price combinations
    """
    # All possible prices
    prices = [60, 54, 48, 36]
    
    # Generate all valid combinations (non-increasing prices)
    valid_combos = []
    combo_id = 0
    
    for combo in itertools.product(prices, repeat=14):
        # Check if prices are non-increasing
        if all(combo[i] >= combo[i+1] for i in range(13)):
            # First week is always 60
            full_combo = [combo_id, 60] + list(combo)
            valid_combos.append(full_combo)
            combo_id += 1
    
    # Create DataFrame
    columns = ["combo_number"] + [str(i) for i in range(1, 16)]
    df = pd.DataFrame(valid_combos, columns=columns)
    
    return df

def convert_to_actions(price_df):
    """
    Convert price values to action letters.
    
    Args:
        price_df: DataFrame with price values
        
    Returns:
        pandas.DataFrame: DataFrame with action letters
    """
    action_df = pd.DataFrame()
    action_df['combo_number'] = price_df['combo_number']
    
    # First week is always maintain (A)
    action_df['1'] = 'A'
    
    # For subsequent weeks, determine the action based on price change
    for week in range(2, 16):
        curr_week = str(week)
        prev_week = str(week - 1)
        
        actions = []
        for _, row in price_df.iterrows():
            prev_price = row[prev_week]
            curr_price = row[curr_week]
            
            if curr_price == prev_price:
                action = 'A'  # Maintain
            elif curr_price == 54:
                action = 'B'  # 10% markdown
            elif curr_price == 48:
                action = 'C'  # 20% markdown
            elif curr_price == 36:
                action = 'D'  # 40% markdown
            else:
                action = '?'  # Unknown (shouldn't happen)
                
            actions.append(action)
            
        action_df[curr_week] = actions
    
    return action_df

def setup_browser():
    """
    Set up and return a configured browser instance.
    
    Returns:
        webdriver.Chrome: Configured Chrome WebDriver
    """
    chrome_options = Options()
    # Uncomment to run headless (no GUI)
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    # Initialize the browser
    try:
        # If using webdriver_manager
        from webdriver_manager.chrome import ChromeDriverManager
        browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    except:
        # Fallback to local chromedriver
        browser = webdriver.Chrome(options=chrome_options)
        
    # Navigate to the game and maximize window
    browser.get(GAME_URL)
    browser.maximize_window()
    
    return browser

def run_simulation(browser, combo_row, simulation_number):
    """
    Run a single simulation using the specified pricing strategy.
    
    Args:
        browser: WebDriver instance
        combo_row: DataFrame row with actions for each week
        simulation_number: Identifier for this simulation
    
    Returns:
        tuple: (week_data, outcome_data) containing results
    """
    wait = WebDriverWait(browser, 2)
    
    # Map letters to button IDs
    action_map = {
        'A': 'maintainButton',  # Maintain price
        'B': 'tenButton',       # 10% markdown
        'C': 'twentyButton',    # 20% markdown
        'D': 'fortyButton'      # 40% markdown
    }
    
    # Click the "Start Game" button
    try:
        start_button = wait.until(EC.element_to_be_clickable((By.ID, "practiceButton")))
        start_button.click()
        time.sleep(0.00001)  # Wait for game to initialize
    except Exception as e:
        # If error, refresh and try again
        print(f"Error starting game, refreshing page: {e}")
        browser.refresh()
        #time.sleep(0.00002)
        wait.until(EC.presence_of_element_located((By.ID, "practiceButton")))
        start_button = wait.until(EC.element_to_be_clickable((By.ID, "practiceButton")))
        start_button.click()
        time.sleep(0.0002)
    
    combo_id = int(combo_row['combo_number'])
    week_data = []
    
    # Execute the strategy for weeks 2-15 (week 1 is automatic)
    for week in range(2, 16):
        try:
            week_str = str(week)
            action_letter = combo_row[week_str]
            button_id = action_map[action_letter]
            
            # Wait for and click the appropriate button
            button = wait.until(EC.element_to_be_clickable((By.ID, button_id)))
            button.click()
            #time.sleep(0.000001)  # Allow time for game to update
            
        except TimeoutException:
            #print(f"Button not clickable in simulation {simulation_number}, week {week}")
            # Some actions might be disabled due to previous choices
            continue
        except Exception as e:
            #print(f"Error in week {week}: {e}")
            continue
    
    # Wait for final results
    #time.sleep(0.0001)
    wait.until(EC.presence_of_element_located((By.ID, "rev")))
    
    # Extract results from all weeks
    try:
        # Find the results table
        table = wait.until(EC.presence_of_element_located((By.ID, "result-table")))
        rows = table.find_elements(By.TAG_NAME, "tr")
        
        # Skip header row
        for row in rows[1:]:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) >= 4:
                week_num = int(cells[0].text)
                price = int(cells[1].text)
                sales = int(cells[2].text)
                inventory = int(cells[3].text)
                
                week_data.append([
                    simulation_number,
                    combo_id,
                    week_num,
                    price,
                    sales,
                    inventory
                ])
    except Exception as e:
        print(f"Error extracting week data: {e}")
    
    # Extract final outcome
    try:
        # Find revenue and perfect foresight elements
        revenue_elem = browser.find_element(By.ID, "rev")
        perfect_elem = browser.find_element(By.ID, "perfect")
        
        # Clean the text to get just the numbers
        revenue = int(re.sub(r'[^0-9]', '', revenue_elem.text))
        perfect = int(re.sub(r'[^0-9]', '', perfect_elem.text))
        
        # Extract difference percentage (from percentage element if available)
        try:
            pct_elem = browser.find_element(By.ID, "percentage")
            difference_pct = float(re.sub(r'[^0-9.]', '', pct_elem.text))
        except:
            # Calculate manually if element not found
            difference_pct = round(((perfect - revenue) / perfect) * 100, 2) if perfect > 0 else 0
        
        outcome_data = [
            simulation_number,
            combo_id,
            revenue,
            perfect,
            difference_pct
        ]
    except Exception as e:
        print(f"Error extracting outcome: {e}")
        outcome_data = [simulation_number, combo_id, 0, 0, 0]
    
    return week_data, outcome_data

def main():
    """Main function to run the simulation"""
    # Either generate strategies or use a pre-saved CSV
    try:
        action_df = pd.read_csv('action_df.csv')
        print(f"Loaded {len(action_df)} strategy combinations from action_df.csv")
    except:
        print("Creating price strategies...")
        price_df = generate_price_strategies()
        action_df = convert_to_actions(price_df)
        action_df.to_csv('action_df.csv', index=False)
        print(f"Generated {len(action_df)} strategy combinations")
    
    # Initialize data collectors
    all_week_data = []
    all_outcome_data = []
    
    # Initialize browser
    browser = setup_browser()
    
    # How many simulations to run (set to a small number for testing)
    num_simulations = 680  # Change this to run more or fewer simulations
    
    try:
        # Loop through each combo in the action_df (limited by num_simulations)
        for index, combo_row in action_df.head(num_simulations).iterrows():
            simulation_number = index + 1
            print(f"Running simulation {simulation_number}/{num_simulations}: Combo {combo_row['combo_number']}")
            
            try:
                # Run the simulation and collect results
                week_data, outcome_data = run_simulation(browser, combo_row, simulation_number)
                
                # Store results
                all_week_data.extend(week_data)
                all_outcome_data.append(outcome_data)
                
                # Save results every 10 simulations
                if simulation_number % 200 == 0:
                    print(f"Saving interim results after {simulation_number} simulations...")
                    
                    # Save week detail data
                    week_df = pd.DataFrame(
                        all_week_data, 
                        columns=["Simulation Number", "comboID", "Week", "Price", "Sales", "Remaining Inventory"]
                    )
                    week_df.to_csv(f"week_detail_interim_{simulation_number}.csv", index=False)
                    
                    # Save outcome data
                    outcome_df = pd.DataFrame(
                        all_outcome_data, 
                        columns=["Simulation Number", "comboID", "Your revenue", "Perfect foresight strategy", "Difference (%)"]
                    )
                    outcome_df.to_csv(f"outcome_interim_{simulation_number}.csv", index=False)
                
            except Exception as e:
                print(f"Error in simulation {simulation_number}: {e}")
                traceback.print_exc()
                continue
    
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    
    finally:
        # Always close the browser when done
        browser.quit()
        
        # Save final results
        if all_week_data:
            week_df = pd.DataFrame(
                all_week_data, 
                columns=["Simulation Number", "comboID", "Week", "Price", "Sales", "Remaining Inventory"]
            )
            #week_df.to_csv("week_detail.csv", index=False)
            write_header = not os.path.exists("week_detail.csv")
            week_df.to_csv("week_detail.csv", mode='a', index=False, header=write_header)
        
        if all_outcome_data:
            outcome_df = pd.DataFrame(
                all_outcome_data, 
                columns=["Simulation Number", "comboID", "Your revenue", "Perfect foresight strategy", "Difference (%)"]
            )
            #outcome_df.to_csv("outcome.csv", index=False)
            write_header = not os.path.exists("outcome.csv")
            outcome_df.to_csv("outcome.csv", mode='a', index=False, header=write_header)
        
        print("Simulation completed. Results saved to week_detail.csv and outcome.csv")

if __name__ == "__main__":
    main() 