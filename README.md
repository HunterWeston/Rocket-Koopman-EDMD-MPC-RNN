# Predictive Lifted-State Rocket Flight Control with Online EDMDc, MPC, and Residual Neural Networks

This repository implements a data-driven flight control algorithm to maintain rocket trajectories throughout turbulent atmospheric ascent.

Traditional flight controllers regularly struggle with correcting rapid, unpredictable changes such as non-linear aerodynamics, fast fuel burn, and sudden wind gusts. This framework simplifies these dynamics by lifting the complex, non-linear rocket behavior into a higher-dimensional linear space.

By linearizing the dynamics, the flight computer can accurately predict and influence the future state of the system using Quadratic Programs at 50 Hz on low power avionics.

## Core Architecture

   **EDMDc Prediction Loop:**
  1. **Lifting:** The Extended Dynamic Mode Decomposition with control algorithm takes the current 11 dimensional physical state vector of the rocket: x = [h, v<sub>z</sub>, x<sub>p</sub>, v<sub>x</sub>, y<sub>p</sub>, v<sub>y</sub>, θ<sub>p</sub>, ω<sub>p</sub>, θ<sub>y</sub>, ω<sub>y</sub>, m], and breaks it into two, 4 dimensional state vectors for pitch and yaw, respectively: x<sub>k</sub> = [x<sub>p</sub>, v<sub>x</sub>, θ<sub>p</sub>, ω<sub>p</sub>] & y<sub>k</sub> = [y<sub>p</sub>, v<sub>y</sub>, θ<sub>y</sub>, ω<sub>y</sub>] ([h, v<sub>z</sub>, m] are omitted in this step). The 4 dimensional vectors are then lifted via a quadratic polynomial lifting map onto a 15 dimensional observable space for each: z<sub>k</sub> = ψ(x<sub>k</sub>). In this high dimensional state space, the nonlinearities of the physical system behave approximately linearly.
  2. **Passive Evolution:** The lifted state vector is then premultiplied by the linear prediction matrix _A_ (15 × 15): A · z<sub>k</sub>, which returns a vector prediction of how the future states would evolve naturally.
  3. **Control Input:** To incorporate active control into the loop, the scalar value u<sub>k</sub> is added and is defined as the current TVC gimbal angle (u<sub>k</sub> = δ<sub>p</sub> and u<sub>k</sub> = δ<sub>y</sub> for pitch and yaw, respectively). The future state vector from step 2 is enhanced by adding the product of the control input and the linear control effectiveness matrix _B_ (15 × 1), yielding a final lifted predicted state vector: z<sub>k+1</sub> ≈ (A · z<sub>k</sub>) + (B · u<sub>k</sub>).
  4. **Decoding:** z<sub>k+1</sub> has the dimensions 15 × 1 and cannot be interpreted by the model, so it must be projected back down into a low dimensional physical state space (4 × 1). This is done by applying a linear decoding matrix _C_ (4 × 15) which recovers the original state vector through the mapping: x<sub>k+1</sub> ≈ C · z<sub>k+1</sub>.

   **Online Adaptation:** Rocket dynamics change rapidly as air density drops and the vehicle loses fuel weight. To account for this, the flight computer updates the model in real time by logging snapshot matrices of physical and lifted states in a rolling memory window. Every 8 timesteps (timestep = 0.02 sec), a linear regression recalculates the A, B, and C matrices. This helps ensure that unmodeled variations such as air pressure, weight, servo imprecision, or unexpected friction along the airframe don’t have a sustained effect on model performance during flight.
   
   **Physics ResNet:** Even with the online adaptation, polynomial lifting maps cannot accurately account for rapid changes like wind shear or sudden sensor noise. To address this, the model runs a Multilayer Perceptron neural network in parallel with the oEDMDc algorithm. The network takes the rocket's current physical state vector and control inputs and maps them to estimate model error. This learned correction is then added to the final decoded output of the prediction loop: x<sub>k+1</sub> ≈ C · z<sub>k+1</sub> + r<sub>θ</sub>(x<sub>k</sub>, u<sub>k</sub>).

## Results

The framework was validated using a 30 run Monte Carlo flight simulation under identical wind conditions and thrust curves to compare with a classic cascaded PID controller

*   **62.3% Drift Reduction:** Under continuous stochastic crosswinds, the adaptive Koopman MPC cut the 3D lateral RMS drift by more than half compared to PID.
*   **12.9% Smoother Actuation:** Instead of overcorrecting, the Koopman MPC performed stable attitude tilts into the wind. This created a  smoother mechanical profile and lower high frequency gimbal strain compared to PID.
*   **75.7% Recovery in Stress Tests:** When the rocket was hit with an unexpected 35% spike in drag, a 25% shift in inertia, and a sudden decrease in actuator force, the online adaptive loop recovered the vehicle, lowering errors by 75.7% compared to the static EDMDc algorithm.
*   **100% Waypoint Accuracy:** During 3D target navigation tests under heavy wind gusts, the controller guided the rocket through a target zone over 200 lateral meters away within 8 meters 100% of the time, and within 3 meters 20.0% of the time.

## AI Usage Statement

Artificial intelligence was used in this project to identify bugs, tune PID and MPC weights, and create the HTML visualization for the model’s results. AI was not used to formulate, derive, or create the core mathematics or frameworks.
