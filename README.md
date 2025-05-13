# Retail Markdown Game Simulator


A comprehensive simulation and optimization tool for retail markdown strategies in competitive markets, helping retailers maximize revenue through data-driven pricing decisions.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Game Rules](#game-rules)
- [Data Description](#data-description)
- [Code Components](#code-components)
- [Installation Requirements](#installation-requirements)
- [Usage Instructions](#usage-instructions)
- [Results](#results)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview

The Retail Markdown Game Simulator recreates the real-world decision-making process retailers face when optimizing pricing strategies for seasonal products. This repository contains a comprehensive toolkit for:

- Simulating and analyzing different markdown strategies across a 15-week retail selling season
- Optimizing pricing decisions using mathematical programming and reinforcement learning
- Collecting and analyzing simulation data using web automation
- Visualizing the impact of pricing decisions on inventory, sales, and revenue

The project simulates a fashion apparel retail environment where you start with 2,000 units of inventory and must make optimal pricing decisions to maximize revenue over a 15-week period, balancing immediate sales against future opportunities.

## Project Structure

```
Retail_Markdown_Game/
├── 01_Final_Simulation_Data/       # Analysis results and processed data
│   ├── 01-Week_Data/               # Detailed weekly simulation data
│   ├── 02-Outcome_Data/            # Simulation outcome summaries
│   ├── 02_File_Merging.ipynb       # Data integration notebook
│   ├── 03_Descriptive_Analysis_Differance.ipynb  # Performance analysis
│   └── *.csv, *.png                # Result data and visualizations
├── 02_Backup_Interim/              # Interim backup versions
├── 03_Simulation/                  # Core simulation components
│   ├── 01_Experiment Setup.ipynb   # Simulation configuration
│   ├── 02_Selenuim_Data_Collection.ipynb  # Web automation for data collection
│   ├── action_df.csv               # Available pricing actions
│   ├── action_df_eliminated.csv    # Filtered pricing actions
│   ├── action_df_prioritized.csv   # Prioritized pricing actions
│   └── combos_df.csv               # Combinations of pricing strategies
├── 04_Linear Optimization/         # Mathematical optimization approaches
│   ├── 01_DynamicProgramming.ipynb # DP solution implementation
│   ├── 02_DynamicProgramming.ipynb # Alternative DP implementation
│   ├── Demand_Sample1.xlsx         # Sample demand data
│   └── README.md                   # Optimization module documentation
├── 05_Reinforcemente_Learning/     # RL-based optimization approach
│   ├── environment.py              # RL environment implementation
│   ├── model_training.py           # RL model training code
│   ├── evaluation.py               # RL model evaluation
│   ├── inference.py                # Model inference pipeline
│   ├── main.ipynb                  # Main RL workflow notebook
│   └── *.csv, *.png, *.pth         # Results and model checkpoints
├── instructions.txt                # Detailed game rules and specifications
├── README.md                       # This file
└── .gitattributes                  # Git configuration
```

## Game Rules

The retail markdown simulation operates with these key parameters:

- **Initial inventory**: 2,000 units
- **Season length**: 15-week selling season
- **Starting price**: $60 (mandatory for Week 1)
- **Markdown options**:
  - Maintain current price (0% markdown)
  - 10% markdown ($54)
  - 20% markdown ($48)
  - 40% markdown ($36) - final markdown, ends decision-making

**Key constraints**:
- Once a markdown is applied, prices cannot be increased
- Markdowns are sequential (e.g., after 20% markdown, you cannot choose 10%)
- A 40% markdown ends further pricing decisions for all future weeks
- Any unsold inventory at week 15 has zero salvage value

The objective is to maximize total revenue by making optimal pricing decisions throughout the season, balancing immediate sales with long-term revenue potential.

## Data Description

The project uses and generates several key datasets:

### Input Data
- **action_df.csv**: Contains all possible markdown strategies and actions
- **combos_df.csv**: Contains combinations of pricing levels across the 15-week period
- **action_df_prioritized.csv**: Filtered set of high-potential pricing strategies

### Output Data
- **Week Detail CSV**: Weekly performance for each simulation
  - Simulation Number, comboID, Week, Price, Sales, Remaining Inventory
  
- **Outcome CSV**: Summary of each simulation run
  - Simulation Number, comboID, Your Revenue, Perfect Foresight Strategy, Difference (%)

### Analysis Data
- **combo_performance_difference.png**: Visualization of strategy performance gaps
- **combo_difference_distribution.png**: Distribution of performance differences
- **combo_statistical_analysis.csv**: Statistical comparison of strategies

## Code Components

The project is organized into four main components:

### 1. Simulation Framework
- **03_Simulation/01_Experiment Setup.ipynb**: Configures simulation parameters and strategy combinations
- **03_Simulation/02_Selenuim_Data_Collection.ipynb**: Uses Selenium WebDriver to automate the data collection process from the web-based game

### 2. Data Analysis
- **01_Final_Simulation_Data/02_File_Merging.ipynb**: Integrates data from multiple simulation runs
- **01_Final_Simulation_Data/03_Descriptive_Analysis_Differance.ipynb**: Analyzes performance differences between strategies

### 3. Optimization Methods
- **04_Linear Optimization/01_DynamicProgramming.ipynb**: Implements dynamic programming to find optimal markdown strategies

### 4. Reinforcement Learning
- **05_Reinforcemente_Learning/environment.py**: Custom RL environment for the markdown problem
- **05_Reinforcemente_Learning/model_training.py**: Training pipeline for RL agents
- **05_Reinforcemente_Learning/evaluation.py**: Evaluation framework for trained models
- **05_Reinforcemente_Learning/main.ipynb**: End-to-end RL workflow

## Installation Requirements

### Prerequisites
- Python 3.8+
- Jupyter Notebook/Lab

### Dependencies
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
selenium>=4.0.0
torch>=1.9.0  # For reinforcement learning
scipy>=1.7.0
openpyxl>=3.0.9  # For Excel file handling
```

### Setup Instructions
1. Clone this repository
```bash
git clone https://github.com/yourusername/Retail_Markdown_Game.git
cd Retail_Markdown_Game
```

2. Install required dependencies
```bash
pip install -r requirements.txt  # Note: Create this file based on dependencies above
```

3. Install appropriate WebDriver for Selenium (if using simulation data collection)
   - For Chrome: [ChromeDriver](https://sites.google.com/chromium.org/driver/)
   - For Firefox: [GeckoDriver](https://github.com/mozilla/geckodriver/releases)

## Usage Instructions

### Running Simulations
1. Navigate to the Simulation directory:
```bash
cd 03_Simulation
```

2. Open and run the experiment setup notebook:
```bash
jupyter notebook "01_Experiment Setup.ipynb"
```

3. Run the data collection notebook to gather simulation results:
```bash
jupyter notebook "02_Selenuim_Data_Collection.ipynb"
```

### Analyzing Results
1. Navigate to the analysis directory:
```bash
cd ../01_Final_Simulation_Data
```

2. Run the analysis notebook:
```bash
jupyter notebook "03_Descriptive_Analysis_Differance.ipynb"
```

### Example: Dynamic Programming Optimization
1. Navigate to the optimization directory:
```bash
cd ../04_Linear\ Optimization
```

2. Open and run the dynamic programming notebook:
```bash
jupyter notebook "01_DynamicProgramming.ipynb"
```

### Example: Reinforcement Learning
1. Navigate to the RL directory:
```bash
cd ../05_Reinforcemente_Learning
```

2. Open and run the main RL notebook:
```bash
jupyter notebook "main.ipynb"
```

## Results

The project provides several key insights:

1. **Strategy Performance Analysis**: Visualizations in the `01_Final_Simulation_Data` directory demonstrate the performance gap between various markdown strategies and the perfect foresight benchmark.

2. **Optimal Timing**: The analysis reveals that markdown timing is critical, with early aggressive markdowns often underperforming compared to strategic delayed markdowns.

3. **Method Comparison**: Performance comparison between:
   - Rule-based markdown strategies
   - Dynamic programming optimization
   - Reinforcement learning approaches
   
4. **Inventory Management**: Analysis of ending inventory levels and their impact on overall revenue.

Key visualizations include:
- Performance difference distribution across strategy combinations
- Statistical analysis of strategy effectiveness
- Reward and inventory distribution under RL policies

## Contributors

- Felix Veaux
- Juan David Ovalle
- Jean Pool Nieto
- Mirza Abubacker

## Acknowledgments

- Retail markdown game interface provided by [randhawa.us](https://www.randhawa.us/games/retailer/nyu.html)
- Special thanks to contributors and researchers in the field of retail analytics and revenue management 
