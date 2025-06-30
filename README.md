# Adaptive Cooperative Harvesting: Dynamic Cooperation and Sustainability in Agent-Based Fisheries
By Mikko Brandon, Lucas Keijzer, Yoad van Praag, Maarten Stork, for Agent Based Modeling at UvA, 2025.

## General aspects

The Agent-Based Model (ABM) is an abstracted version of a real life fishery.
It was extended from an existing ABM created by Owusu et al., where the focus
lied with assessing the effect a Marine Protected Area (MPA) has on the sustainability
of fisheries. In the model, Owusu et al. defined cooperation as the complement
of fishing effort, and fixed it at various levels to assess the effect of
cooperative fishing in different environments (MPA/non-MPA). Our goal was to
make the cooperation of fishers dynamic and dependent on the fisher's
environment, to see if it is possible to create conditions under which
sustainable fishing arises from such dynamic cooperation.

The model comprises two types of agents, the fishing agents and fish
agents. Agents are initially randomly distributed on a two-dimensional space
that wraps around representing the fishing ground. Fishers are represented by
dots of different hues, with lighter hues representing higher cooperation.
Fish differ in various parameters, such as speed and reproduction rate. Different
combinations of values for these parameters are represented through different
colors.

A report detailing the design of the model further is included in the repository
(`report.pdf`).

## Running Code

The file `dynamic_coop.py` contains the model code (in python) and is the script to
run for a single simulation. It generates a `simulation.GIF` file in the simulation_output
directory. The GIF shows the movement and interactions between agents over the span
of one simulation.

Default parameters can be modified in `parameters/parameters.py` in the BaseParameters class.
These parameters detail the starting values and constants of each simulation. They may also be
modified through code, as is done on occasion for sensitivity analyses.

The agents and their update actions are defined in `agents/fish.py` and `agents/fisher.py`.
The classes were created post-hoc in code cleanup, because of which object-oriented
programming is mostly not practiced in the main code. As such, most class functions
had to be made static.

Various sensitivity analysis files have been created and can be run. One run may take
many hours depending on the number of timesteps used for each simulation, the number of
parameters to evaluate, the number of parameter values to evaluate for, and access to
proper computational hardware.

# Relevant references

K. A. Owusu, M. M. Kulesz, and A. Merico. [Extraction behaviour and income inequalities resulting from a common pool resource exploitation](https://www.mdpi.com/2071-1050/11/2/536). *Sustainability*, **11**(536), 2019.
