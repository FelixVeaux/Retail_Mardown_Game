# Retail Markdown Game Simulator

A simulation tool for analyzing retail markdown strategies in a competitive market environment. This project allows users to test different retail pricing strategies and observe their impact on profitability over a 15-week period.

## Project Overview

This repository contains a retail markdown simulation game that recreates the decision-making process retailers face when determining markdown strategies. The simulation allows users to:

- Test different pricing strategies across a 15-week selling season
- Compete against multiple retail competitors with their own pricing strategies
- Analyze profitability outcomes based on different markdown approaches
- Visualize the impact of markdown decisions on sales and revenue

## Repository Structure

- **01_Selenuim-Try_2.ipynb**: Main Jupyter notebook containing the simulation code and execution logic
- **action_df.csv**: Contains the various markdown strategies/actions that can be taken
- **combos_df.csv**: Contains combinations of pricing levels across the 15-week period
- **instructions.txt**: Detailed explanation of the game rules and operation
- **README.md**: This file - overview and documentation of the repository
- **01_FINAL_DATA/**: Directory containing final data and results
- **02_Backup_Interim/**: Directory containing interim backup versions
- **10_Old_Code/**: Directory containing previous iterations of the code

## Installation Requirements

This project requires:

- Python 3.x
- Jupyter Notebook
- Selenium WebDriver (for browser automation)
- Common data science libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Usage Instructions

1. **Setup**: Clone this repository and ensure all dependencies are installed
2. **Open the Notebook**: Open `01_Selenuim-Try_2.ipynb` in Jupyter Notebook
3. **Run Simulation**: Execute the notebook cells to run the retail markdown simulation
4. **Strategy Analysis**:
   - The simulation allows testing different price markdown strategies over a 15-week period
   - Price options include keeping the original price (60) or discounting to lower levels (54, 48, 36)
   - The simulation compares your strategy against competitors and evaluates profitability

## Game Rules

The retail markdown game simulates a market with multiple retailers selling identical products with:

- Initial inventory of 100 units
- 15-week selling season
- Starting price of $60
- Optional markdown prices of $54, $48, and $36
- Market demand influenced by pricing decisions
- Competitor pricing strategies that adapt to market conditions

The goal is to maximize profit by choosing an optimal markdown strategy across the 15-week period.

## Output Files

The simulation generates visualizations and performance metrics directly in the notebook, including:

- Sales volume by week
- Revenue performance
- Profit comparison against competitors
- Optimal markdown strategy analysis

## Contributing

Contributions to improve the simulation model or add new features are welcome. Please feel free to submit pull requests with your enhancements.

## License

This project is provided for educational and research purposes. 