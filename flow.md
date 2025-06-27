
flowchart TD
  A[Start script\n(experiment_label)] --> B[initialize(experiment_label)]
  B --> B1(delete_prev_sim)
  B --> B2[create fisher agents & fish agents\n– set traits, positions, trust scores]
  B --> B3[init global data structures\n– time1=0, total_fish_count, trust_history, etc.]

  subgraph Main Simulation Loop
    direction TB
    C1([time1 < n?]) -->|Yes| C2[update_one_unit_time]
    C2 --> C3[observe() → save frame]
    C3 --> C4[time1 incremented by update_one_unit_time]
    C4 --> C1
    C1 -->|No| D
  end

  B3 --> C1

  D[End of loop] --> E[plot_summary() → save summary figures]
  E --> F[save_cooperation_data(), save_trust_data()]
  F --> G[create_gif(), delete_prev_sim()]
  G --> H[End]

flowchart TD
  A[Start script\n(experiment_label)] --> B[initialize(experiment_label)]
  B --> B1(delete_prev_sim)
  B --> B2[create fisher agents & fish agents\n– set traits, positions, trust scores]
  B --> B3[init global data structures\n– time1=0, total_fish_count, trust_history, etc.]

  subgraph Main Simulation Loop
    direction TB
    C1([time1 < n?]) -->|Yes| C2[update_one_unit_time]
    C2 --> C3[observe() → save frame]
    C3 --> C4[time1 incremented by update_one_unit_time]
    C4 --> C1
    C1 -->|No| D
  end

  B3 --> C1

  D[End of loop] --> E[plot_summary() → save summary figures]
  E --> F[save_cooperation_data(), save_trust_data()]
  F --> G[create_gif(), delete_prev_sim()]
  G --> H[End]

  %% Detail inside update_one_unit_time
  subgraph update_one_unit_time
    direction LR
    U1[update_fish()\n– move & reproduce fish] --> U2[update_trust()\n– trust_score adjustments]
    U2 --> U3[adjust_effort_based_on_trust()\n– conditional cooperators adjust effort]
    U3 --> U4[track_trust_metrics()\n– append avg trust]
    U4 --> U5[check_threshold_behavior()\n– threshold-based effort changes]
    U5 --> U6[fishermen harvest & move]\n
    U6 --> U7[imitation if time1 % imitation_period == 0]
    U7 --> U8[track_cooperation_levels()\n– append counts & avg cooperation]
    U8 --> U9[write current step to CSV]
  end

  C2 --> update_one_unit_time

  %% Detail inside update_one_unit_time
  subgraph update_one_unit_time
    direction LR
    U1[update_fish()\n– move & reproduce fish] --> U2[update_trust()\n– trust_score adjustments]
    U2 --> U3[adjust_effort_based_on_trust()\n– conditional cooperators adjust effort]
    U3 --> U4[track_trust_metrics()\n– append avg trust]
    U4 --> U5[check_threshold_behavior()\n– threshold-based effort changes]
    U5 --> U6[fishermen harvest & move]\n
    U6 --> U7[imitation if time1 % imitation_period == 0]
    U7 --> U8[track_cooperation_levels()\n– append counts & avg cooperation]
    U8 --> U9[write current step to CSV]
  end

  C2 --> update_one_unit_time
