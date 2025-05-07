# NMPC Code Implementation

This directory contains the Python code for the NMPC project comparing Direct Multiple Shooting (DMS) and Direct Collocation (DC) methods for the Crazyflie 2.1 quadcopter.

## Descriptions

* **`NMPC_solver.py`**: This is the core file defining the NMPC logic.
    * `dynamics` class: Implements the 13-state Crazyflie dynamics model using CasADi symbolic variables. Includes physical parameters of the drone.
    * `cost` class: Defines the quadratic stage cost (tracking error for state and control deviation from hover) and terminal cost (weighted tracking error). Includes options for using slack variables on terminal constraints.
    * `solver` class: Encapsulates the NMPC setup.
        * Initializes model and cost parameters.
        * Sets up the Optimal Control Problem (OCP) structure.
        * `create_dms_solver()`: Formulates the Non-Linear Program (NLP) using the Direct Multiple Shooting method with an RK4 integrator.
        * `create_dc_solver()`: Formulates the NLP using the Direct Collocation method (Gauss-Legendre scheme).
        * `solve()`: Calls the IPOPT solver via CasADi to solve the formulated NLP.
        * `extract_next_state()`: Parses the solution from the solver for closed-loop execution.
        * `run_mpc()`: Implements the closed-loop MPC simulation logic (for simulation purposes).
        * `control_to_drone()`: Converts the calculated optimal motor speeds (KRPM) into the [roll, pitch, yaw rate, thrust] setpoint format required by the Crazyflie firmware/API.

* **`drone_control_open_loop.py`**:
    * A script designed to send a *pre-calculated sequence* of control setpoints `[roll, pitch, yaw_rate, thrust]` to the Crazyflie.
    * It reads this sequence (potentially from a file like `control_array.txt`, although the exact implementation details might vary).
    * Uses the `cflib` library to connect to the drone and send commands via `cf.commander.send_setpoint`.
    * Intended for testing playback of trajectories or specific open-loop maneuvers.

* **`drone_control_closed_loop.py`**:
    * The main script for demonstrating *closed-loop NMPC* control.
    * Imports the `solver` from `NMPC_solver.py`.
    * Uses `cflib` to:
        * Connect to the specified Crazyflie (`uri`).
        * Log real-time state estimates (position, attitude, velocities, rates) from the drone (`state_callback`).
        * Send computed control setpoints (`cf.commander.send_setpoint`).
    * Initializes the NMPC solver (defaults to DMS in the provided version).
    * Runs a loop that: gets the latest state, solves the NMPC OCP for the optimal control, converts the control to the required setpoint format, and sends it to the drone.
    * Contains parameters for NMPC tuning (Q/R matrices, timing, bounds) and the target `state_reference`.

## Dependencies

The code requires Python 3 and the following libraries:

* `casadi`
* `numpy`
* `matplotlib` (Primarily for analysis/plotting, may not be strictly required to run the control loops)
* `cflib`

Install using pip:
```bash
pip install casadi numpy matplotlib cflib
