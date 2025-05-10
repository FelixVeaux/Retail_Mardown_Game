# Dynamic Programming Notebook

## Overview
This Jupyter Notebook implements a complete workflow to optimize and simulate the **Retailer Game** using dynamic programming–style decision structuring, with a core Gurobi mixed-integer model. It includes:

1. **Data imports** (pandas, NumPy, statsmodels) for analysis and summary statistics.  
2. **`optimize_monotonic_schedule(d60)`**  
   - A Gurobi-based optimizer computing an optimal monotonic price schedule over 15 weeks given first-week demand.  
3. **`create_price_dict(schedule)`**  
   - Converts Gurobi’s binary week–price assignment into a `{price: weeks}` dictionary.  
4. **`simulate_game_with_auto_demand(headless=False)`**  
   - A Selenium-driven function that:  
     - Launches the Retailer Game in Chrome  
     - Reads actual Week-1 sales (used as `d60`)  
     - Runs the optimizer to get the pricing schedule  
     - Automates button clicks to play the remaining 14 weeks per that schedule  
     - Scrapes final revenue and perfect-inventory values to compute a performance score  
5. **`batch_simulate_optimal(n_runs, headless=False)`**  
   - Repeats the single-run simulation, skipping infeasible optimizer cases, and aggregates:  
     - First-week demands (`d60`)  
     - Price schedules (`{price: weeks}`)  
     - Final revenues, perfect values, and scores  
     - Summary statistics (mean score, standard deviation, 95% CI)  
6. **Main block**  
   - Demonstrates calling `batch_simulate_optimal` for multiple runs in headless mode and prints a results summary.  

---

## Dependencies
- Python 3.7+  
- Gurobi Python API (`gurobipy`)  
- Selenium & webdriver-manager  
- numpy, pandas, statsmodels  

```bash
pip install gurobipy selenium webdriver-manager numpy pandas statsmodels
