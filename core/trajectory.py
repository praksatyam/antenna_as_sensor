from .array import *

def screen_bounds(Sx, Sy, de):
    """Return (x_min, x_max, y_min, y_max) of the physical screen area."""
    x_max =  (Sx - 1) / 2 * de
    y_max =  (Sy - 1) / 2 * de
    return -x_max, x_max, -y_max, y_max


def trajectory_linear(N, dt, Sx, Sy, de, z_f=None):
    """
    Type 1: Linear sweep from one point to another.
    Finger moves in a straight line — used for model validation.
    """
    if z_f is None:
        z_f = z_f0
    x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)

    # Start bottom-left, end top-right
    x_start, y_start = x_min * 0.8, y_min * 0.8
    x_end,   y_end   = x_max * 0.8, y_max * 0.8

    t   = np.linspace(0, 1, N)
    x_f = x_start + (x_end - x_start) * t
    y_f = y_start + (y_end - y_start) * t
    z_f_arr = np.full(N, z_f)

    return np.stack([x_f, y_f, z_f_arr], axis=1)   # (N, 3)


def trajectory_sinusoidal(N, dt, Sx, Sy, de, z_f=None):
    """
    Type 2: Sinusoidal curved path — realistic swipe gesture.
    """
    if z_f is None:
        z_f = z_f0
    x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)

    t = np.linspace(0, 2 * np.pi, N)

    # X sweeps full width, Y follows a sine (figure-8 style)
    x_f = (x_max * 0.8) * np.sin(t)
    y_f = (y_max * 0.8) * np.sin(2 * t)   # double frequency → Lissajous curve

    # Slight height variation
    z_f_arr = z_f + 0.003 * np.sin(3 * t)

    return np.stack([x_f, y_f, z_f_arr], axis=1)


def trajectory_random_walk(N, dt, Sx, Sy, de, z_f=None, seed=42):
    """
    Type 3: Smoothed random walk — arbitrary realistic gesture.
    Uses small random steps smoothed with a moving average to ensure
    continuity. Position is clipped to screen bounds.
    """
    if z_f is None:
        z_f = z_f0
    rng = np.random.default_rng(seed)
    x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)

    # Step size proportional to screen size
    step = de * 0.3

    # Raw random steps
    dx = rng.normal(0, step, N)
    dy = rng.normal(0, step, N)

    # Smooth with moving average (window = 10 steps)
    dx = uniform_filter1d(dx, size=10)
    dy = uniform_filter1d(dy, size=10)

    # Integrate to get positions
    x_f = np.cumsum(dx)
    y_f = np.cumsum(dy)

    # Center and scale to fit within 80% of screen
    x_f = x_f - x_f.mean()
    y_f = y_f - y_f.mean()
    scale = min((x_max * 0.8) / (np.abs(x_f).max() + 1e-9),
                (y_max * 0.8) / (np.abs(y_f).max() + 1e-9))
    x_f *= scale
    y_f *= scale

    # Clip to bounds (safety)
    x_f = np.clip(x_f, x_min, x_max)
    y_f = np.clip(y_f, y_min, y_max)

    z_f_arr = np.full(N, z_f)
    return np.stack([x_f, y_f, z_f_arr], axis=1)

# ─────────────────────────────────────────────
# 6. FULL SIMULATION
# ─────────────────────────────────────────────

def simulate(trajectory, array_pos, sigma_f, k, sigma_n2, C=None):
    """
    Run the full simulation over a trajectory.

    If C is provided, applies mutual coupling:
        y_coupled(t) = C @ y_ideal(t) + n(t)

    Note: noise is added AFTER coupling because the coupling happens
    in the antenna layer before the receiver chain introduces noise.
    When C=None the model reduces to the ideal uncoupled case.

    Args:
        trajectory : (N, 3) finger positions over time
        array_pos  : (S, 3) antenna element positions
        sigma_f    : float
        k          : float
        sigma_n2   : float
        C          : (S, S) complex coupling matrix, or None for ideal

    Returns:
        Y : (N, S) complex array — array response at each time step
    """
    N = len(trajectory)
    S = len(array_pos)
    Y = np.zeros((N, S), dtype=complex)

    for n in range(N):
        # Step 1: compute ideal noiseless backscatter at each element
        r      = compute_distances(array_pos, trajectory[n])
        signal = sigma_f * np.exp(-1j * 2 * k * r) / (r ** 2)

        # Step 2: apply mutual coupling (mixes signals across elements)
        if C is not None:
            signal = C @ signal     # (S,S) @ (S,) → (S,)

        # Step 3: add receiver noise after coupling
        noise  = (np.random.randn(S) + 1j * np.random.randn(S)) \
                 * np.sqrt(sigma_n2 / 2)

        Y[n, :] = signal + noise

    return Y



# ─────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────

def plot_array_geometry(array_pos, Sx, Sy, de):
    """Plot the antenna array layout."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(array_pos[:, 0] * 1e3, array_pos[:, 1] * 1e3,
               c='steelblue', s=80, zorder=3, label='Antenna elements')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'ScreenAnt Array Geometry ({Sx}×{Sy}, de={de*1e3:.1f} mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_trajectories(array_pos, traj_lin, traj_sin, traj_rnd, Sx, Sy, de):
    """Plot all three trajectories over the array footprint."""
    x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles  = ['Type 1: Linear', 'Type 2: Sinusoidal', 'Type 3: Random Walk']
    trajs   = [traj_lin, traj_sin, traj_rnd]
    colors  = ['#e63946', '#2a9d8f', '#e9c46a']

    for ax, traj, title, color in zip(axes, trajs, titles, colors):
        # Screen boundary
        rect = plt.Rectangle((x_min*1e3, y_min*1e3),
                              (x_max-x_min)*1e3, (y_max-y_min)*1e3,
                              linewidth=1.5, edgecolor='gray',
                              facecolor='#f0f0f0', zorder=0)
        ax.add_patch(rect)

        # Array elements
        ax.scatter(array_pos[:, 0]*1e3, array_pos[:, 1]*1e3,
                   c='steelblue', s=30, zorder=2, alpha=0.6)

        # Trajectory with time colormap
        sc = ax.scatter(traj[:, 0]*1e3, traj[:, 1]*1e3,
                        c=np.arange(len(traj)), cmap='plasma',
                        s=8, zorder=3)
        ax.plot(traj[:, 0]*1e3, traj[:, 1]*1e3,
                color=color, alpha=0.4, linewidth=1, zorder=2)

        # Start / end markers
        ax.scatter(*traj[0, :2]*1e3,  marker='o', s=120,
                   color='green', zorder=5, label='Start')
        ax.scatter(*traj[-1, :2]*1e3, marker='X', s=120,
                   color='red',   zorder=5, label='End')

        ax.set_xlim(x_min*1e3*1.1, x_max*1e3*1.1)
        ax.set_ylim(y_min*1e3*1.1, y_max*1e3*1.1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label='Time step')

    plt.suptitle('Finger Trajectories Above ScreenAnt Array', fontsize=13)
    plt.tight_layout()
    return fig


def plot_array_response(Y, trajectory, array_pos, traj_name='Linear'):
    """
    Plot the magnitude of the array response over time.
    Shows: (1) received power per element over time, (2) spatial heatmap
    at selected time steps.
    """
    N, S = Y.shape
    Sx_loc = int(np.sqrt(S))
    power  = np.abs(Y) ** 2   # (N, S)

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 4, figure=fig)

    # ── Top: power per element over time ──
    ax_top = fig.add_subplot(gs[0, :])
    im = ax_top.imshow(power.T, aspect='auto', origin='lower',
                       cmap='inferno',
                       extent=[0, N, 0, S])
    ax_top.set_xlabel('Time step')
    ax_top.set_ylabel('Antenna element index')
    ax_top.set_title(f'Received Power per Element Over Time — {traj_name}')
    plt.colorbar(im, ax=ax_top, label='Power (|y|²)')

    # ── Bottom: spatial heatmap at 4 time snapshots ──
    snapshots = [0, N//4, N//2, 3*N//4]
    snap_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]

    for ax, t_idx in zip(snap_axes, snapshots):
        grid = power[t_idx].reshape(Sx_loc, Sx_loc)
        im2  = ax.imshow(grid, cmap='hot', origin='upper')
        # Mark true finger position (nearest element)
        fpos = trajectory[t_idx, :2] * 1e3
        ax.set_title(f't={t_idx}  finger=({fpos[0]:.1f},{fpos[1]:.1f})mm',
                     fontsize=8)
        ax.set_xlabel('Col'); ax.set_ylabel('Row')
        plt.colorbar(im2, ax=ax)

    plt.suptitle(f'Array Response — {traj_name} Trajectory', fontsize=13)
    plt.tight_layout()
    return fig


def plot_signal_vs_time(Y, trajectory, element_idx=24):
    """
    Plot the real/imaginary parts and phase of the received signal
    at a single element (default: center element) over time.
    """
    N = len(trajectory)
    t_axis = np.arange(N) * dt * 1e3   # in ms

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t_axis, np.real(Y[:, element_idx]), color='steelblue')
    axes[0].set_ylabel('Real part')
    axes[0].set_title(f'Signal at element {element_idx} (center) over time')

    axes[1].plot(t_axis, np.imag(Y[:, element_idx]), color='coral')
    axes[1].set_ylabel('Imaginary part')

    axes[2].plot(t_axis, np.angle(Y[:, element_idx]), color='seagreen')
    axes[2].set_ylabel('Phase [rad]')
    axes[2].set_xlabel('Time [ms]')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_coupling_comparison(Y_ideal, Y_coupled, trajectory, epsilon,
                             element_idx=24):
    """
    Side-by-side comparison of ideal vs coupled array response.
    Shows:
      - Power heatmap (ideal vs coupled) at a mid-trajectory snapshot
      - Signal magnitude at center element over time (both cases)
      - Difference (corruption introduced by coupling)
    """
    N, S  = Y_ideal.shape
    Sx_l  = int(np.sqrt(S))
    t_mid = N // 2
    t_ax  = np.arange(N) * dt * 1e3   # ms

    fig   = plt.figure(figsize=(14, 9))
    gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Row 0: spatial power heatmaps at t_mid ──
    for col, (Y, label) in enumerate([(Y_ideal,  'Ideal  (ε=0)'),
                                       (Y_coupled, f'Coupled (ε={epsilon})')]):
        ax  = fig.add_subplot(gs[0, col])
        pwr = (np.abs(Y[t_mid]) ** 2).reshape(Sx_l, Sx_l)
        im  = ax.imshow(pwr, cmap='hot', origin='upper')
        ax.set_title(f'{label}\nPower at t={t_mid}', fontsize=10)
        ax.set_xlabel('Col'); ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax)

    # Difference heatmap
    ax_diff = fig.add_subplot(gs[0, 2])
    diff_pwr = np.abs(
        (np.abs(Y_coupled[t_mid])**2) - (np.abs(Y_ideal[t_mid])**2)
    ).reshape(Sx_l, Sx_l)
    im3 = ax_diff.imshow(diff_pwr, cmap='bwr', origin='upper')
    ax_diff.set_title(f'|Power difference|\nat t={t_mid}', fontsize=10)
    ax_diff.set_xlabel('Col'); ax_diff.set_ylabel('Row')
    plt.colorbar(im3, ax=ax_diff)

    # ── Row 1: signal magnitude over time at center element ──
    ax_sig = fig.add_subplot(gs[1, :2])
    ax_sig.plot(t_ax, np.abs(Y_ideal[:, element_idx]),
                label='Ideal (ε=0)', color='steelblue', linewidth=1.5)
    ax_sig.plot(t_ax, np.abs(Y_coupled[:, element_idx]),
                label=f'Coupled (ε={epsilon})', color='coral',
                linewidth=1.5, linestyle='--')
    ax_sig.set_xlabel('Time [ms]')
    ax_sig.set_ylabel('|y| — Signal Magnitude')
    ax_sig.set_title(f'Signal magnitude at center element (idx={element_idx})')
    ax_sig.legend()
    ax_sig.grid(True, alpha=0.3)

    # ── Row 1 right: coupling effect per element (mean over time) ──
    ax_bar = fig.add_subplot(gs[1, 2])
    mean_diff = np.mean(np.abs(np.abs(Y_coupled)**2 - np.abs(Y_ideal)**2),
                        axis=0)
    ax_bar.bar(np.arange(S), mean_diff, color='mediumpurple', alpha=0.8)
    ax_bar.set_xlabel('Antenna element index')
    ax_bar.set_ylabel('Mean |ΔPower|')
    ax_bar.set_title('Mean coupling distortion\nper element (over time)')
    ax_bar.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Mutual Coupling Effect — ε={epsilon}  '
                 f'(Linear trajectory)', fontsize=13, fontweight='bold')
    return fig


def plot_coupling_sweep(trajectory, array_pos, sigma_f, k, sigma_n2,
                        epsilons=None):
    """
    Sweep epsilon from 0 to 1 and show how the array response changes.
    Metric: mean squared deviation from ideal response across all
    elements and time steps.
    """
    if epsilons is None:
        epsilons = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    Y_ideal = simulate(trajectory, array_pos, sigma_f, k, sigma_n2, C=None)
    deviations = []

    for eps in epsilons:
        C_eps   = build_coupling_matrix(array_pos, k, eps)
        Y_c     = simulate(trajectory, array_pos, sigma_f, k, sigma_n2, C=C_eps)
        msd     = np.mean(np.abs(Y_c - Y_ideal) ** 2)
        deviations.append(msd)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epsilons, deviations, 'o-', color='#e63946', linewidth=2,
            markersize=7)
    ax.set_xlabel('Coupling strength ε', fontsize=12)
    ax.set_ylabel('Mean squared deviation from ideal', fontsize=12)
    ax.set_title('Array Response Distortion vs. Coupling Strength', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    return fig, deviations