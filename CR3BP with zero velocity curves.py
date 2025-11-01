import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Implementation of 4th-Order Runge-Kutta Integration Method for the Circular Restricted Three-Body Problem (CR3BP) with Zero-Velocity Curves and Forbidden Regions
'Author: Mansur Mohammed Bala'


def cr3bp_eom(t, state, mu):

    x, y, z, vx, vy, vz = state

    # Distance to primary bodies
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    # Velocity derivatives
    dxdt = vx
    dydt = vy
    dzdt = vz

    # Acceleration equations in rotating frame
    dvxdt = 2*vy + x - (1 - mu)*(x + mu)/r1**3 - mu*(x - (1 - mu))/r2**3
    dvydt = -2*vx + y - (1 - mu)*y/r1**3 - mu*y/r2**3
    dvzdt = -(1 - mu)*z/r1**3 - mu*z/r2**3

    return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])


# Jacobi Constant
def jacobi_constant(state, mu):
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2)

    # Jacobi constant formula (Eq. 3.27 from the book)
    C = x**2 + y**2 + 2*(1 - mu)/r1 + 2*mu/r2 - (vx**2 + vy**2 + vz**2)

    return C


# Effective Potential
def effective_potential(x, y, mu):
    """
    Calculate the effective potential U (Eq. 3.39 from book)
    2U = n^2(x^2 + y^2) + 2*mu1/r1 + 2*mu2/r2
    with n = 1 (mean motion)
    """
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)

    # Avoid division by zero
    r1 = np.maximum(r1, 1e-10)
    r2 = np.maximum(r2, 1e-10)

    U = 0.5 * (x**2 + y**2) + (1 - mu)/r1 + mu/r2

    return 2 * U  # Return 2U as in Eq. 3.38

# Zero Velocity Curves


def compute_zero_velocity_curves(mu, CJ, xlim=(-2, 2), ylim=(-2, 2), resolution=500):
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Calculating 2U for each points
    two_U = effective_potential(X, Y, mu)
    forbidden = two_U > CJ

    return X, Y, two_U, forbidden


def find_lagrange_points(mu):
    # Collinear points (approximate)
    L1 = np.array([1 - mu - (mu/3)**(1/3), 0])
    L2 = np.array([1 - mu + (mu/3)**(1/3), 0])
    L3 = np.array([-1 - mu - 5*mu/12, 0])

    # Triangular points (exact)
    L4 = np.array([0.5 - mu, np.sqrt(3)/2])
    L5 = np.array([0.5 - mu, -np.sqrt(3)/2])

    return {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5}


# Runge-Kutta 4th Order Step
def rk4_step(t, state, h, mu):

    k1 = cr3bp_eom(t, state, mu)
    k2 = cr3bp_eom(t + h/2, state + h*k1/2, mu)
    k3 = cr3bp_eom(t + h/2, state + h*k2/2, mu)
    k4 = cr3bp_eom(t + h, state + h*k3, mu)

    new_state = state + h*(k1 + 2*k2 + 2*k3 + k4)/6

    return new_state


def integrate_cr3bp(t0, tf, h, initial_state, mu, save_file=None):

    # Initialize storage arrays
    n_steps = int((tf - t0) / h) + 1
    t_array = np.zeros(n_steps)
    states = np.zeros((n_steps, 6))
    C_array = np.zeros(n_steps)

    # Initial conditions
    t = t0
    state = np.array(initial_state)

    if save_file:
        f = open(save_file, 'w')
        f.write("t\tx\ty\tz\tvx\tvy\tvz\tC_jacobi\n")

    # Integration loop
    for i in range(n_steps):
        # Store current state
        t_array[i] = t
        states[i] = state
        C_array[i] = jacobi_constant(state, mu)

        # Write to file
        if save_file:
            f.write(f"{t:.6f}\t{state[0]:.6f}\t{state[1]:.6f}\t{state[2]:.6f}\t"
                    f"{state[3]:.6f}\t{state[4]:.6f}\t{state[5]:.6f}\t{C_array[i]:.6f}\n")

        if i % 1000 == 0:
            progress = 100 * t / tf
            print(f"Progress: {progress:.2f}%")

        # Check if final time is reached
        if t >= tf:
            break

        state = rk4_step(t, state, h, mu)
        t += h

    if save_file:
        f.close()

    # Trajectory
    trajectory = {
        't': t_array[:i+1],
        'x': states[:i+1, 0],
        'y': states[:i+1, 1],
        'z': states[:i+1, 2],
        'vx': states[:i+1, 3],
        'vy': states[:i+1, 4],
        'vz': states[:i+1, 5],
        'C': C_array[:i+1]
    }

    return trajectory


def synodic_to_inertial(x, y, z, t):

    xi = x * np.cos(t) - y * np.sin(t)
    eta = x * np.sin(t) + y * np.cos(t)
    zeta = z

    return xi, eta, zeta


def poincare_section(trajectory, plane='y', direction='positive'):

    if plane == 'y':
        coord = trajectory['y']
        vel = trajectory['vy']
        cross_coords = {'x': [], 'vx': []}

        for i in range(1, len(coord)):
            if direction == 'positive':
                if coord[i-1] < 0 and coord[i] >= 0 and vel[i] > 0:
                    cross_coords['x'].append(trajectory['x'][i])
                    cross_coords['vx'].append(trajectory['vx'][i])
            elif direction == 'negative':
                if coord[i-1] > 0 and coord[i] <= 0 and vel[i] < 0:
                    cross_coords['x'].append(trajectory['x'][i])
                    cross_coords['vx'].append(trajectory['vx'][i])

    return cross_coords


# SIMULATION
if __name__ == "__main__":

    # System parameters
    mu = 1e-3  # Mass parameter

    # Time parameters
    t0 = 0.0
    tf = 100 * (2*np.pi)  # 100 orbital periods
    h = 0.0001

    plot_downsample = 100  # Plot every 100th point to reduce memory

    # Initial conditions
    x0 = 0.55
    y0 = 0.0
    z0 = 0.0
    vx0 = 0.0
    vz0 = 0.0

    # Calculate vy0 for a specific Jacobi constant
    C_desired = 3.07
    r1 = np.sqrt((x0 + mu)**2 + y0**2 + z0**2)
    r2 = np.sqrt((x0 - (1 - mu))**2 + y0**2 + z0**2)

    # Validate before computing vy0
    velocity_squared = x0**2 + y0**2 + 2 * \
        (1 - mu)/r1 + 2*mu/r2 - vx0**2 - C_desired
    if velocity_squared < 0:
        print(
            f"Error: Invalid initial conditions (velocity_squared = {velocity_squared:.6f})")
        print("Adjusting C_desired...")
        C_desired = x0**2 + y0**2 + 2*(1 - mu)/r1 + 2*mu/r2 - vx0**2 - 0.01
        velocity_squared = 0.01

    vy0 = np.sqrt(velocity_squared)

    initial_state = [x0, y0, z0, vx0, vy0, vz0]

    print("="*70)
    print("CIRCULAR RESTRICTED THREE-BODY PROBLEM")
    print("="*70)
    print(f"Mass parameter μ = {mu}")
    print(f"Primary masses: m1 = {1-mu:.3f}, m2 = {mu:.3f}")
    print(
        f"Initial Jacobi constant CJ = {jacobi_constant(initial_state, mu):.6f}")
    print(f"Time span: {t0} to {tf/(2*np.pi):.1f} periods")
    print(f"Time step: {h}")
    print("="*70)

    # Find Lagrange points
    lagrange = find_lagrange_points(mu)
    print("\nApproximate Lagrange point locations:")
    for name, pos in lagrange.items():
        print(f"  {name}: ({pos[0]:.4f}, {pos[1]:.4f})")

    # Integrate the equations of motion
    print("\nStarting integration...")
    trajectory = integrate_cr3bp(t0, tf, h, initial_state, mu,
                                 save_file="cr3bp_trajectory.txt")

    print("\nIntegration complete")

    # Transform to inertial frame
    xi, eta, zeta = synodic_to_inertial(trajectory['x'], trajectory['y'],
                                        trajectory['z'], trajectory['t'])

    # Compute Poincaré section
    poincare = poincare_section(trajectory, plane='y', direction='positive')

    # Check Jacobi constant conservation
    C_error = np.std(trajectory['C'])
    C_mean = np.mean(trajectory['C'])
    print(f"\nJacobi constant statistics:")
    print(f"  Mean: {C_mean:.10f}")
    print(f"  Std deviation: {C_error:.2e}")
    print(f"  Relative error: {C_error/C_mean:.2e}")

    # PLOT SECTION
    fig = plt.figure(figsize=(16, 12))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Synodic frame orbit with zero-velocity curves
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot zero-velocity curves for this trajectory's CJ
    X, Y, two_U, forbidden = compute_zero_velocity_curves(mu, C_mean,
                                                          xlim=(-1.5, 1.5),
                                                          ylim=(-1.5, 1.5))

    # Shade forbidden regions (where 2U > CJ)
    ax1.contourf(X, Y, forbidden.astype(int), levels=[0.5, 1.5],
                 colors=['gray'], alpha=0.3)

    # Plot zero-velocity curve boundary (where 2U = CJ)
    contour = ax1.contour(X, Y, two_U, levels=[C_mean], colors='red',
                          linewidths=2, linestyles='--')
    ax1.clabel(contour, inline=True, fontsize=8, fmt='CJ=%.2f')

    # Plot trajectory (downsampled for memory efficiency)
    traj_indices = np.arange(0, len(trajectory['x']), plot_downsample)
    ax1.plot(trajectory['x'][traj_indices], trajectory['y'][traj_indices],
             'g-', linewidth=0.8, label='Orbit')
    ax1.plot(-mu, 0, 'yo', markersize=12, label=f'm₁ (μ={1-mu:.3f})')
    ax1.plot(1-mu, 0, 'bo', markersize=8, label=f'm₂ (μ={mu:.3f})')
    ax1.plot(x0, y0, 'r*', markersize=10, label='Initial position')

    # Plot Lagrange points
    for name, pos in lagrange.items():
        if -1.5 <= pos[0] <= 1.5 and -1.5 <= pos[1] <= 1.5:
            ax1.plot(pos[0], pos[1], 'kx', markersize=8, markeredgewidth=2)
            ax1.text(pos[0]+0.05, pos[1]+0.05, name, fontsize=8)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Orbit with Zero-Velocity Curves',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    # 2. Inertial frame orbit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(xi[traj_indices], eta[traj_indices], 'r-', linewidth=0.5)
    ax2.set_xlabel('ξ', fontsize=12)
    ax2.set_ylabel('η', fontsize=12)
    ax2.set_title('Orbit in Inertial Frame', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Zero-velocity curves for multiple CJ values
    ax3 = fig.add_subplot(gs[0, 2])
    CJ_values = [C_mean + 0.3, C_mean, C_mean - 0.3]
    colors = ['blue', 'red', 'green']

    for CJ_val, color in zip(CJ_values, colors):
        X, Y, two_U, forbidden = compute_zero_velocity_curves(mu, CJ_val,
                                                              xlim=(-1.5, 1.5),
                                                              ylim=(-1.5, 1.5))
        ax3.contour(X, Y, two_U, levels=[CJ_val], colors=color,
                    linewidths=2, linestyles='-')
        ax3.contourf(X, Y, forbidden.astype(int), levels=[0.5, 1.5],
                     colors=[color], alpha=0.1)

    ax3.plot(-mu, 0, 'yo', markersize=12)
    ax3.plot(1-mu, 0, 'bo', markersize=8)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title('Zero-Velocity Curves\n(Different CJ values)',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)

    # 4. Jacobi constant evolution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(trajectory['t'][traj_indices]/(2*np.pi),
             trajectory['C'][traj_indices], 'purple', linewidth=1)
    ax4.axhline(y=C_mean, color='red', linestyle='--', linewidth=1,
                label=f'Mean = {C_mean:.6f}')
    ax4.set_xlabel('Time (orbital periods)', fontsize=12)
    ax4.set_ylabel('Jacobi Constant CJ', fontsize=12)
    ax4.set_title(
        f'Jacobi Constant (σ = {C_error:.2e})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Poincaré section
    ax5 = fig.add_subplot(gs[1, 1])
    if len(poincare['x']) > 0:
        ax5.plot(poincare['x'], poincare['vx'], 'b.', markersize=2)
        ax5.set_xlabel('x', fontsize=12)
        ax5.set_ylabel('vₓ', fontsize=12)
        ax5.set_title(f'Poincaré Section (y=0, vy>0)\nn={len(poincare["x"])}',
                      fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No crossings detected',
                 ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Poincaré Section', fontsize=12, fontweight='bold')

    # 6. Effective potential contour plot
    ax6 = fig.add_subplot(gs[1, 2])
    X, Y, two_U, _ = compute_zero_velocity_curves(mu, C_mean,
                                                  xlim=(-1.5, 1.5),
                                                  ylim=(-1.5, 1.5))
    contour = ax6.contourf(X, Y, two_U, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax6, label='2U')
    ax6.plot(-mu, 0, 'yo', markersize=12)
    ax6.plot(1-mu, 0, 'bo', markersize=8)
    ax6.contour(X, Y, two_U, levels=[C_mean], colors='red', linewidths=2)
    ax6.set_xlabel('x', fontsize=12)
    ax6.set_ylabel('y', fontsize=12)
    ax6.set_title('Effective Potential 2U', fontsize=12, fontweight='bold')
    ax6.axis('equal')
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)

    fig.suptitle(f'Circular Restricted Three-Body Problem (μ = {mu})',
                 fontsize=14, fontweight='bold')

    plt.savefig('cr3bp_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved to 'cr3bp_results.png' and 'cr3bp_trajectory.txt'")
    print("="*70)
