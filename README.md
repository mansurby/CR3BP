# Circular Restricted Three-Body Problem (CR3BP) with Zero-Velocity Curves

Author: Mansur Mohammed Bala  
File: `CR3BP with zero velocity curves.py`

This project simulates the **Circular Restricted Three-Body Problem (CR3BP)** using a 4th-order Runge-Kutta integrator. It computes **orbits, zero-velocity curves, forbidden regions, Lagrange points, and Poincaré sections**.

## Features

- Integrates CR3BP equations of motion in a rotating synodic frame.
- Computes **Jacobi constant** and checks its conservation.
- Calculates **effective potential** and zero-velocity curves (forbidden regions).
- Approximates **Lagrange points** (L1–L5).
- Transforms trajectories from synodic to inertial frame.
- Generates **Poincaré sections** for dynamical analysis.
- Produces plots:
  - Synodic frame orbit with zero-velocity curves
  - Orbit in inertial frame
  - Zero-velocity curves for multiple Jacobi constants
  - Jacobi constant evolution
  - Poincaré section
  - Effective potential contours

## Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies using pip:
```bash
pip install numpy matplotlib

## Usage

Run the script directly:

python CR3BP with zero velocity curves.py

## Adjustable Parameters

Mass parameter: mu (default 1e-3)

Time step: h

Simulation duration: tf (in units of orbital periods)

Initial conditions: [x0, y0, z0, vx0, vy0, vz0]

Desired Jacobi constant: C_desired

The script automatically computes the initial velocity vy0 to match the Jacobi constant.

Output

Trajectory file: cr3bp_trajectory.txt (contains positions, velocities, and Jacobi constant over time)

Plots: cr3bp_results.png (contains all visualizations)

Console output includes:

Mass parameters

Initial Jacobi constant

Lagrange points

Jacobi constant statistics

Functions Overview

cr3bp_eom(t, state, mu): Equations of motion.

jacobi_constant(state, mu): Computes Jacobi constant.

effective_potential(x, y, mu): Calculates 2U.

compute_zero_velocity_curves(mu, CJ): Generates zero-velocity curves.

find_lagrange_points(mu): Returns approximate L1–L5 coordinates.

rk4_step(t, state, h, mu): Performs 4th-order Runge-Kutta integration.

integrate_cr3bp(t0, tf, h, initial_state, mu): Integrates the full trajectory.

synodic_to_inertial(x, y, z, t): Converts synodic frame to inertial coordinates.

poincare_section(trajectory, plane='y', direction='positive'): Computes Poincaré crossings.

# References

Murray, C.D., & Dermott, S.F. Solar System Dynamics, Cambridge University Press.

Classical CR3BP theory in rotating frames.
