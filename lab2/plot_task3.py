import csv
import numpy as np
import matplotlib.pyplot as plt


# Set this to match your controller's friction coefficient
MU = 0.6  # <-- change if your code uses a different mu


def load_contact_forces(csv_path: str):
    """Load t, Fn (normal), Ft (tangential magnitude) from CSV."""
    t, Fn, Ft = [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t.append(float(row["t"]))
            Fn.append(float(row["F_normal"]))
            Ft.append(float(row["F_tangent"]))
    return np.array(t), np.array(Fn), np.array(Ft)


def main():
    t, Fn, Ft = load_contact_forces("contact_forces.csv")

    # Basic sanity checks
    if len(t) == 0:
        raise RuntimeError("No data found in contact_forces.csv")

    # Sometimes during impacts you may briefly get negative/near-zero Fn
    # (depending on sign convention / contact model). For friction-cone visualization,
    # keep only samples with meaningful normal force.
    mask = np.isfinite(Fn) & np.isfinite(Ft) & (Fn > 1e-6)
    t2, Fn2, Ft2 = t[mask], Fn[mask], Ft[mask]

    # -------- Plot 1: Friction cone scatter (Ft vs Fn) --------
    plt.figure()
    plt.plot(Fn2, Ft2, ".", markersize=3, label="Solver samples")

    # Plot the friction limit line Ft = mu * Fn
    fn_line = np.linspace(Fn2.min(), Fn2.max(), 200)
    plt.plot(fn_line, MU * fn_line, label=r"$F_t = \mu F_n$")

    plt.xlabel(r"Normal force $F_n = F_z$ [N]")
    plt.ylabel(r"Tangential force $F_t = \sqrt{F_x^2+F_y^2}$ [N]")
    plt.title("Friction cone check (linearized cone)")
    plt.grid(True)
    plt.legend()

    # -------- Plot 2: Time series (optional but useful) --------
    plt.figure()
    plt.plot(t2, Fn2, label=r"$F_n$")
    plt.plot(t2, Ft2, label=r"$F_t$")
    plt.plot(t2, MU * Fn2, label=r"$\mu F_n$")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Normal vs tangential force over time")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
