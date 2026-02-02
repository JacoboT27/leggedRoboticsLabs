import csv
import numpy as np
import matplotlib.pyplot as plt

MU = 0.005  # must match QP friction

Ft, Fz, Fx1, Fy1 = [], [], [], []

with open("contact_wrench.csv", "r") as f:
    r = csv.DictReader(f)
    for row in r:
        Fx = float(row["Fx"])
        Fy = float(row["Fy"])
        Fz_i = float(row["Fz"])

        Ft_i = np.sqrt(Fx**2 + Fy**2)

        Ft.append(Ft_i)
        Fz.append(Fz_i)
        Fx1.append(Fx)
        Fy1.append(Fy)


Ft = np.array(Ft)
Fz = np.array(Fz)
Fx1 = np.array(Fx1)
Fy1 = np.array(Fy1)

# Keep valid contact samples
mask = np.isfinite(Fz) & np.isfinite(Ft) & (Fz > 1e-6)
Ft = Ft[mask]
Fz = Fz[mask]

# Build the cone boundary line: Fz = (1/mu) * Ft  (equivalent to Ft = mu*Fz)
ft_max = MU * Fz.max()          # maximum Ft that would be on the boundary at max Fz
ft_line = np.linspace(0.0, ft_max, 200)
fz_line = (1.0 / MU) * ft_line


max_ft = max(Ft)
max_fz = max(Fz)

mean_ft = np.mean(Ft)
mean_fz = np.mean(Fz)

std_ft = np.std(Ft)
std_fz = np.std(Fz)

print("-------------- Stats --------------------")
print(f"Max Tangential Force: {max_ft:.3f}")
print(f"Max Normal Force: {max_fz:.3f}")
print(f"Mean Tangential Force: {mean_ft:.3f}")
print(f"Mean Normal Force: {mean_fz:.3f}")
print(f"Std Dev Tangential Force: {std_ft:.3f}")
print(f"Std Dev Normal Force: {std_fz:.3f}")
print("-----------------------------------------")

print("-------- Test --------")
# Safety mask: valid contact only
mask = np.isfinite(Fx1) & np.isfinite(Fy1) & np.isfinite(Fz) & (Fz > 1e-6)

Fx_v = Fx1[mask]
Fy_v = Fy1[mask]
Fz_v = Fz[mask]

# L-infinity tangential force (what the QP actually constrains)
F_inf = np.maximum(np.abs(Fx_v), np.abs(Fy_v))

# Friction utilization (should be <= 1)
u = F_inf / (MU * Fz_v)

print("Max pyramid utilization u_max =", u.max())
print("Mean pyramid utilization =", u.mean())
print("Number of violations =", np.sum(u > 1.0 + 1e-6))





# Plot
plt.figure(figsize=(7, 5))
plt.scatter(Fz, Ft, s=10, label="Solver samples")         # s controls marker size
plt.plot(fz_line, ft_line, linewidth=2, label=rf"$F_t=\mu F_z$ (μ={MU})")

plt.ylabel(r"Tangential force magnitude $F_t=\sqrt{F_x^2+F_y^2}$ [N]")
plt.xlabel(r"Normal force $F_z$ [N]")
plt.title("Friction cone (right-half cross-section)")

plt.ylim(0.0, ft_max)            # IMPORTANT: shows the wedge clearly
plt.xlim(0.0, Fz.max() * 1.05)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(u)
plt.axhline(1.0, linestyle="--", color="r", label="friction limit")
plt.xlabel("Sample index")
plt.ylabel(r"$u = \max(|F_x|,|F_y|)/(\mu F_z)$")
plt.title("Friction pyramid utilization")
plt.grid(True)
plt.legend()
plt.show()


