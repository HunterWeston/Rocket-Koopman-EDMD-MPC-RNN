# Predictive Lifted-State Rocket Flight Control with Online EDMDc, MPC, and Residual Neural Networks

This repository implements a data-driven flight control system designed to maintain the trajectory of rockets during turbulent atmospheric ascent. 

Traditional flight controllers often struggle when facing non-linear aerodynamics, fast fuel burn, and sudden wind gusts. This framework solves that by translating complex, non-linear rocket behavior into a higher-dimensional linear space. 

By making the math linear, the flight computer can solve complex trajectory optimization problems via simple Quadratic Programs at 50 Hz on low-power avionics.

## Core Architecture

The architecture runs a high-speed control loop that balances global linear models with rapid real-time adaptations:

*   **The Linear Space Embedding (EDMDc):** Extended Dynamic Mode Decomposition with control takes standard physical observations—such as 3D position, velocity, attitude (pitch/yaw), and angular rates—and maps them into a 15-state quadratic polynomial observable space. In this lifted space, the highly coupled non-linear equations behave linearly, allowing standard linear Model Predictive Control (MPC) to directly calculate optimal two-axis Thrust Vector Control (TVC) nozzle gimbal angles.
*   **Online Koopman Adaptation ( Shifting Regimes):** Rocket dynamics change rapidly as air density drops and the vehicle lightens. The controller maintains a rolling sliding-window memory buffer of recent state transitions. Every few time steps, it runs a recursive regression to update the linear prediction matrices (A, B, and C) on the fly, ensuring the predictive model updates alongside the shifting flight envelope.
*   **Physics-Residual Neural Network (The Corrector):** To catch unmodeled structural dynamics, sensor noise, or wind anomalies that escape the data-driven linear model, a lightweight Multi-Layer Perceptron (MLP) is run in parallel. Written in raw NumPy for fast inference, this network trains in real time via stochastic gradient descent over an experience replay buffer. It functions as a dynamic trim layer, injecting a residual correction to eliminate steady-state tracking errors.

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
