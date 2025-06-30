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
- [Analysis Tools](#analysis-tools)
- [Project Structure](#project-structure)
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
