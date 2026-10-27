# Predictive Lifted-State Rocket Flight Control with Online EDMDc, MPC, and Residual Neural Networks

This repository implements a data-driven flight control algorithm to maintain the trajectory of rockets during turbulent atmospheric ascent. 

Traditional flight controllers often struggle when facing rapid and unpredictable changes like: non-linear aerodynamics, fast fuel burn, and sudden wind gusts. This framework simplifies these dynamics by lifting the complex, non-linear rocket behavior into a higher-dimensional linear space. 

By making the dyanmics linear, the flight computer can acuratly predict and influence the future state of the system using Quadratic Programs at 50 Hz on low-power avionics.

## Core Architecture

The architecture runs a high speed control loop that lifts the current physical state vector of the rocket, predicts the future state vector, determines the necessary control inputs, and decodes back into a physical state space:

*   **Linear Space Embedding (EDMDc):** The Extended Dynamic Mode Decomposition with control algorithm takes the current 11 dimensional current physical state vector of the rocket: x = [h, v<sub>z</sub>, x<sub>p</sub>, v<sub>x</sub>, y<sub>p</sub>, v<sub>y</sub>, θ<sub>p</sub>, ω<sub>p</sub>, θ<sub>y</sub>, ω<sub>y</sub>, m], and breaks it into two, 4 dimensional state vectors for pitch and yaw respectively: x<sub>k</sub> = [x<sub>p</sub>, v<sub>x</sub>, θ<sub>p</sub>, ω<sub>p</sub>] & y<sub>k</sub> = [y<sub>p</sub>, v<sub>y</sub>, θ<sub>y</sub>, ω<sub>y</sub>] ([h, v<sub>z</sub>,  m] are omitted in this step). The 4 dimensional vectors are then lifted via a quadratic polynomial lifting map onto a 15 dimensional observable space for each: z<sub>k</sub> = ψ(x<sub>k</sub>). In this high dimensional state space, the nonlinearities of the physical system behave approximately linear. The transposed lifted state vector is then premultiplied by the linear prediction matrix 'A': A · z<sub>k</sub>, which returns a vector prediction of how the future states would evolve naturally; this future state vector is denoted as z<sub>k+1</sub>. 
*
*   standard physical observations—such as 3D position, velocity, attitude (pitch/yaw), and angular rates—and maps them into a 15-state quadratic polynomial observable space. In this lifted space, the highly coupled non-linear equations behave linearly, allowing standard linear Model Predictive Control (MPC) to directly calculate optimal two-axis Thrust Vector Control (TVC) nozzle gimbal angles.
*   **Online Koopman Adaptation:** Rocket dynamics change rapidly as air density drops and the vehicle lightens. The controller maintains a rolling sliding-window memory buffer of recent state transitions. Every few time steps, it runs a recursive regression to update the linear prediction matrices (A, B, and C) on the fly, ensuring the predictive model updates alongside the shifting flight envelope.
*   **Physics-Residual Neural Network** To catch unmodeled structural dynamics, sensor noise, or wind anomalies that escape the data-driven linear model, a lightweight Multi-Layer Perceptron (MLP) is run in parallel. Written in raw NumPy for fast inference, this network trains in real time via stochastic gradient descent over an experience replay buffer. It functions as a dynamic trim layer, injecting a residual correction to eliminate steady-state tracking errors.

## Repository Layout

```text
├── artifacts/plots/
│   ├── data/
│   │   ├── adaptive_study_results.json
│   │   └── study_results.json
│   └── plots/
│       ├── mpc_waypoint_comparison.png       <- Comparative flight performance curves
│       ├── pid_vs_mpc_comparison.png         <- 30-run Monte Carlo behavior profiles
│       └── rocket_3d_interactive.html        <- Interactive HTML visualization
├── src/
│   ├── edmd.py                               <- EDMDc regression architecture
│   ├── lifting.py                            <- Polynomial lifting maps & decoding
│   ├── mpc.py                                <- Projected gradient descent solver
│   ├── residual_nn.py                        <- NumPy-native real-time error-correction MLP
│   ├── rocket_model.py                       <- 11-DOF rigid body model & RK4 method
│   ├── simulator.py                          <- Real-time flight loop coordination & wind models
│   └── visualization.py                      <- HTML visualization code
├── FINDINGS_REPORT.pdf                       <- Technical report
└── README.md
```

## Results

The framework was thoroughly validated using a 30-run statistical Monte Carlo flight campaign under identical turbulent wind conditions to see how it stands up against a classic cascaded PID/PD autopilot:

*   **62.3% Drift Reduction:** Under continuous stochastic crosswinds, the adaptive Koopman MPC safely restricted cross-track errors, cutting 3D lateral RMS drift by more than half compared to PID.
*   **12.9% Smoother Actuation:** Instead of wildly over-correcting, the predictive engine executes subtle, proactive attitude tilts into the wind. This creates a much smoother mechanical profile and lowers high-frequency gimbal strain.
*   **75.7% Recovery in Stress Tests:** When suddenly hit mid-flight with an unexpected 35% spike in drag, 25% shift in inertia, and a drop in actuator effectiveness, the online adaptive loop successfully recovered the vehicle, slashing incremental errors by 75.7% over static configurations.
*   **100% Waypoint Capture:** During precision 3D target navigation tests under heavy wind gusts, the controller consistently guided the rocket through target zones with sub-meter vertical accuracy.

## AI Usage Statement

Artificial intelligence was used in this project to identify bugs, tune PID and MPC weights, and create the full stack visualization for the model's results. AI was in no way used to formulate, derive or create the core mathematics and frameworks.
