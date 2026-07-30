import numpy as np
import matplotlib.pyplot as plt
import logging


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_torque_log(filename="torque_log.npz"):
    try:
        # Load data
        loaded = np.load(filename)
        data = loaded['data']
        logger.info(f"Loaded data shape: {data.shape}")
    except FileNotFoundError:
        logger.error(f"File {filename} not found. Run the robot control first.")
        return

    # --- Parse data columns ---
    # log_buffer append order: [t_now, target(6), q_act(6), torque_p(6), torque_d(6), torque_target(6)]
    # Total columns: 1 + 6 + 6 + 6 + 6 + 6 = 31

    time_stamps = data[:, 0]
    target_q = data[:, 1:7]
    actual_q = data[:, 7:13]
    torque_p = data[:, 13:19]
    torque_d = data[:, 19:25]
    torque_cmd = data[:, 25:31]

    joint_names = ["Base", "Shoulder", "Elbow", "Wrist 1", "Wrist 2", "Wrist 3"]

    # ==========================================
    # Chart 1: Position Tracking (Target vs Actual)
    # ==========================================
    fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    fig1.suptitle('Joint Position Tracking', fontsize=16)

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axs1[row, col]
        ax.plot(time_stamps, target_q[:, i], 'r--', label='Target', linewidth=1.5)
        ax.plot(time_stamps, actual_q[:, i], 'b-', label='Actual', alpha=0.7)
        ax.set_title(f"Joint {i}: {joint_names[i]}")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.tight_layout()

    # ==========================================
    # Chart 2: Tracking Error (Error = Target - Actual)
    # Used to observe sagging or oscillation caused by gravity
    # ==========================================
    error_q = target_q - actual_q
    # Convert to degrees for intuitiveness
    error_deg = np.degrees(error_q)

    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    fig2.suptitle('Tracking Error (Degrees)', fontsize=16)

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axs2[row, col]
        ax.plot(time_stamps, error_deg[:, i], 'k-', linewidth=1)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)  # 0 degree baseline
        ax.set_title(f"Joint {i} Error ({joint_names[i]})")
        ax.set_ylabel("Error (deg)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # ==========================================
    # Chart 3: Torque Command Breakdown (P-term vs D-term)
    # Used to debug PID parameters and observe noise
    # ==========================================
    fig3, axs3 = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    fig3.suptitle('Torque Command Composition (Nm)', fontsize=16)

    for i in range(6):
        row = i // 3
        col = i % 3
        ax = axs3[row, col]

        # Plot total torque
        ax.plot(time_stamps, torque_cmd[:, i], 'k-', label='Total Cmd', linewidth=1.0, alpha=0.8)
        # Plot P-term
        ax.plot(time_stamps, torque_p[:, i], 'g--', label='P-term', linewidth=0.8, alpha=0.6)
        # Plot D-term (usually D-term has the most noise)
        ax.plot(time_stamps, torque_d[:, i], 'b--', label='D-term', linewidth=0.8, alpha=0.6)

        ax.set_title(f"Joint {i} Torque")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run this file directly to plot
    plot_torque_log("torque_log.npz")
