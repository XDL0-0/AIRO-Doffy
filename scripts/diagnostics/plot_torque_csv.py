import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_joints(csv_path: str):
    """Read torque_log.csv and plot target/q_act/torque curves for 6 joints."""

    # Read CSV
    df = pd.read_csv(csv_path)
    t = df["time"]

    # Prepare canvas: 3 rows (Position, PD components, Output torque) x 6 columns (Each joint)
    fig, axes = plt.subplots(3, 6, figsize=(20, 9), sharex=True)
    fig.suptitle("UR Torque Control Log", fontsize=16)

    for j in range(6):
        # Data columns
        tgt = df[f"target_{j}"]
        q = df[f"q_act_{j}"]
        tau_p = df[f"torque_p_{j}"]
        tau_d = df[f"torque_d_{j}"]
        tau_cmd = df[f"torque_cmd_{j}"]

        # Row 1: Position
        ax = axes[0, j]
        ax.plot(t, tgt, label="target", linewidth=1.0)
        ax.plot(t, q, label="actual", linewidth=1.0)
        ax.set_title(f"Joint {j}")
        ax.grid(True)
        if j == 0:
            ax.set_ylabel("Position [rad]")
        if j == 5:
            ax.legend(fontsize=8)

        # Row 2: PD components
        ax = axes[1, j]
        ax.plot(t, tau_p, label="P term", linewidth=1.0)
        ax.plot(t, tau_d, label="D term", linewidth=1.0)
        ax.grid(True)
        if j == 0:
            ax.set_ylabel("Torque [Nm]")
        if j == 5:
            ax.legend(fontsize=8)

        # Row 3: Final output
        ax = axes[2, j]
        ax.plot(t, tau_cmd, color="black", linewidth=1.0)
        ax.grid(True)
        ax.set_xlabel("time [s]")
        if j == 0:
            ax.set_ylabel("Torque cmd [Nm]")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ---------- Run directly in IDE ----------
if __name__ == "__main__":
    # Change this to your own log file path
    csv_path = "torque_log.csv"
    plot_all_joints(csv_path)
