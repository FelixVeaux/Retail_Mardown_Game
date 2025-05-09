# Retail Markdown Game Simulator

This script automates playing the Retailer Game (https://www.randhawa.us/games/retailer/nyu.html) using different markdown pricing strategies to find the optimal approach.

## Requirements

- Python 3.6+
- Chrome browser installed
- The following Python packages:
  - selenium
  - pandas
  - webdriver-manager

## Installation

1. Install the required packages:
```
pip install selenium pandas webdriver-manager
```

2. Make sure you have Chrome browser installed on your system.

## Usage

1. Run the script with:
```
python retail_markdown_simulator.py
```

2. The script will:
   - Generate or load all valid price markdown strategies
   - Run the simulation for each strategy (limited to 10 by default)
   - Save results to CSV files

3. The results will be saved in two CSV files:
   - `week_detail.csv`: Contains details of each week for each simulation
   - `outcome.csv`: Contains the performance metrics for each simulation

## Configuration

You can modify the following parameters in the script:

- `num_simulations`: Number of simulations to run (default: 10)
- Chrome options: Uncomment the headless option to run without a visible browser

## Troubleshooting

If you encounter errors:

1. Make sure Chrome is installed and up to date
2. Check that your chromedriver is compatible with your Chrome version
3. Increase the sleep times if the game isn't loading properly
4. Check the console output for specific error messages 