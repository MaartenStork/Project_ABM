# Fishery Agent-Based Model (ABM)

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> An advanced Agent-Based Model simulating small-scale artisanal fishery dynamics with cooperation traits and Marine Protected Areas (MPAs)

## Overview

This repository contains a sophisticated Agent-Based Model (ABM) that simulates the complex dynamics of artisanal fisheries. The model investigates how **fishing behavior** (cooperation traits) and **differing fish species** affect fish populations and fishery yields.

## Quick Start

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Project_ABM.git
cd Project_ABM
```

2. **Install dependencies**

```bash
pip install numpy matplotlib pandas tqdm SALib imageio seaborn adjustText tkinter
```

### Run Your First Simulation

**Option 1: Interactive GUI (Recommended)**

```bash
python DynamicCoopUI.py
```

- Configure parameters through the tabbed interface
- Choose visualization options
- Click "Run Simulation"

**Option 2: Command Line**

```bash
python DynamicCoop.py
```

## Table of Contents

- [Model Architecture](#model-architecture)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Analysis Tools](#analysis-tools)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Model Architecture

### Agent Types

| Agent Type               | Description                                | Key Behaviors                                         |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------- |
| **Fishing Agents** | Pirogues with different cooperation traits | Movement, fishing, trust dynamics, imitation          |
| **Fish Agents**    | Schooling fish populations                 | Boid-based movement (separation, alignment, cohesion) |

### Cooperation Spectrum

```
Fully Non-Cooperative ←→ Non-Cooperative ←→ Conditional ←→ Cooperative ←→ Fully Cooperative
```

- **Trust-based interactions**: Agents build/lose trust based on observed behaviors
- **Dynamic cooperation**: Cooperation levels can change based on local conditions
- **Threshold behaviors**: Agents respond to fish density and peer cooperation

## Usage Guide

### GUI Interface

The graphical interface provides five main tabs:

| Tab                         | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| **Basic Parameters**  | Simulation steps, carrying capacity      |
| **Fisher Parameters** | Number and cooperation distribution      |
| **MPA Settings**      | Marine Protected Area configuration      |
| **Trust Parameters**  | Trust dynamics and behavioral thresholds |
| **Visualization**     | Real-time plotting and animation options |

### Visualization Options

- **No Live Visualization**: Maximum performance
- **Live Plots**: Real-time population and catch graphs
- **Movement Visualization**: Interactive spatial view (press spacebar to pause)

### Analysis Tools

#### Sensitivity Analysis

```bash
# Comprehensive analysis (OFAT, Sobol, Morris)
python sens_anl.py

# Quick test with reduced samples
python sens_anl.py quick

# Specific analysis types
python sens_anl.py first_total
```

## Configuration

### Cooperation Scenarios

| Scenario                   | Fully Non-Coop | Non-Coop | Conditional | Coop | Fully Coop |
| -------------------------- | -------------- | -------- | ----------- | ---- | ---------- |
| **Low Cooperation**  | 6              | 5        | 4           | 3    | 2          |
| **Intermediate**     | 4              | 4        | 4           | 4    | 4          |
| **High Cooperation** | 2              | 3        | 4           | 5    | 6          |

## Analysis Tools

### Built-in Sensitivity Analysis

- **OFAT (One-Factor-At-A-Time)**: Individual parameter impact assessment
- **Sobol Analysis**: Global sensitivity with interaction effects
- **Morris Screening**: Efficient parameter screening for large parameter spaces

### Specialized Analysis Scripts

| Script                                | Purpose                              |
| ------------------------------------- | ------------------------------------ |
| `morris_analysis_no_mpa.py`         | Morris analysis for no-MPA scenarios |
| `parameter_equilibrium_analyzer.py` | Equilibrium state analysis           |
| `balanced_stability_analysis.py`    | System stability assessment          |
| `five_param_sens.py`                | Focused analysis on key parameters   |

## Project Structure

```
Project_ABM/
├── DynamicCoopUI.py              # GUI interface
├── DynamicCoop.py                # Main simulation
├── parameters.py                 # Configuration
├── fish.py                       # Fish initialization
├── SimulationVisualizer.py       # Real-time visualization
├── Analysis Scripts/
│   ├── sens_anl.py                  # Sensitivity analysis
│   ├── morris_analysis_no_mpa.py    # Morris (no MPA)
│   ├── plot_sobol_results.py       # Sobol visualization
│   └── ...
├── Original/                     # Original implementations
├── TestingMaarten/              # Development versions
├── plots/                       # Generated visualizations
└── simulation_output/           # Results data
```

## Examples

### Parameter Sensitivity Results

The model provides comprehensive sensitivity analysis showing which parameters most significantly impact:

- Fish population dynamics
- Fishing yields
- Cooperation emergence
- MPA effectiveness

### Trust Dynamics

Trust evolves dynamically based on:

- Observed fishing behavior
- Spatial proximity
- Historical interactions
- Local fish density

## Requirements

```txt
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.1.0
tqdm>=4.50.0
SALib>=1.4.0
imageio>=2.9.0
seaborn>=0.11.0
adjustText>=0.7.0
```

## Contact

**Project Maintainer**: Maarten Stork

- Email: maartenastork@gmail.com
- GitHub: [@MaartenStork](https://github.com/yourusername)
