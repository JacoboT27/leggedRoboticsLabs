import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MU = 0.005  # must match QP friction

path = "lab2/mu-0.005.parquet"  # <-- Low friction

pf = pq.ParquetFile(path)
print("\nArrow schema:\n", pf.schema_arrow)
cols = [
    "current_forces_contact_pos_3",     #fx
    "current_forces_contact_pos_4",     #fy
    "current_forces_contact_pos_5",     #fz
]

df = pd.read_parquet(path, columns=cols)  # reads only these columns

fx_low = df["current_forces_contact_pos_3"].to_numpy()
fy_low = df["current_forces_contact_pos_4"].to_numpy()
fz_low = df["current_forces_contact_pos_5"].to_numpy()

ft_low = np.sqrt(fx_low**2+fy_low**2)


# Build the cone boundary line: Fz = (1/mu) * Ft  (equivalent to Ft = mu*Fz)
ft_max = MU * fz_low.max() +0.8          # maximum Ft that would be on the boundary at max Fz
ft_line = np.linspace(0.0, ft_max, 200)
fz_line = (1.0 / MU) * ft_line



path = "lab2/mu-0.5.parquet"  # <-- Default friction

pf = pq.ParquetFile(path)

cols = [
    "current_forces_contact_pos_3",     #fx
    "current_forces_contact_pos_4",     #fy
    "current_forces_contact_pos_5",     #fz
]

df = pd.read_parquet(path, columns=cols)  # reads only these columns


fx_def = df["current_forces_contact_pos_3"].to_numpy()
fy_def = df["current_forces_contact_pos_4"].to_numpy()
fz_def = df["current_forces_contact_pos_5"].to_numpy()

ft_def = np.sqrt(fx_def**2+fy_def**2)

print("-------- Test --------")
# L-infinity tangential force (what the QP actually constrains)
F_inf = np.maximum(np.abs(fx_low), np.abs(fy_low))
# Friction utilization (should be <= 1)
u = F_inf / (MU * fz_low)
print("Max pyramid utilization u_max =", u.max())
print("Mean pyramid utilization =", u.mean())
print("Number of violations =", np.sum(u > 1.0 + 1e-6))

# ---- Build boundary lines using BOTH datasets (prevents clipping) ----
max_fz = max(np.max(fz_low), np.max(fz_def))
fz_line = np.linspace(0.0, max_fz * 1.05, 400)

# Cone in Ft-Fz plane (L2 cone): Ft = mu * Fz
ft_cone = MU * fz_line

# Pyramid (L∞) projected into Ft-Fz plane:
# If |Fx|<=muFz and |Fy|<=muFz, then Ft = sqrt(Fx^2+Fy^2) <= sqrt(2)*muFz
ft_pyramid_envelope = np.sqrt(2.0) * MU * fz_line

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.scatter(fz_low, ft_low, s=10, color='red',  label="Low friction (0.005)")
plt.scatter(fz_def, ft_def, s=10, color='blue', label="Default friction (0.5)")
plt.plot(fz_line, ft_cone, linewidth=2, color="orange", linestyle="--",label=rf"Cone: $F_t=\mu F_z$ (μ={MU})")
plt.plot(fz_line, ft_pyramid_envelope, linewidth=2, linestyle="--", color="green",label=rf"Pyramid envelope: $F_t=\sqrt{{2}}\mu F_z$ (μ={MU})")
plt.ylabel(r"Tangential force $F_t=\sqrt{F_x^2+F_y^2}$ [N]", fontsize=16)
plt.xlabel(r"Normal force $F_z$ [N]", fontsize=16)
plt.title("Controller Friction Inequality", fontsize=19)
# ---- FIXED LIMITS (use maxima across BOTH datasets and the envelope line) ----
max_ft = max(np.max(ft_low), np.max(ft_def), np.max(ft_pyramid_envelope))
plt.xlim(0.0, max_fz * 1.05)
plt.ylim(0.0, max_ft * 1.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.legend(fontsize=13)
plt.tight_layout()
plt.show()


eps = 1e-9  # avoid divide-by-zero
# --- Normalize Fx, Fy by Fz ---
fz_safe_low = np.maximum(fz_low, eps)  # if fz can be 0; if sign can be negative, see note below
fx_hat_low = fx_low / fz_safe_low
fy_hat_low = fy_low / fz_safe_low
# --- Normalize Fx, Fy by Fz ---
fz_safe_def = np.maximum(fz_def, eps)  # if fz can be 0; if sign can be negative, see note below
fx_hat_def = fx_def / fz_safe_def
fy_hat_def = fy_def / fz_safe_def


# =========================
# Panel A: Fx/Fz vs Fy/Fz
# =========================
plt.figure(figsize=(8, 6))
plt.scatter(fx_hat_low, fy_hat_low, s=8, alpha=0.7, label="$ \mu = 0.005$ (Fx/Fz, Fy/Fz)", color='blue')
plt.scatter(fx_hat_def, fy_hat_def, s=8, alpha=0.7, label="$ \mu = 0.5$ (Fx/Fz, Fy/Fz)", color='red')
# Square boundary (pyramid): |fx_hat|<=mu and |fy_hat|<=mu
square_x = [-MU,  MU,  MU, -MU, -MU]
square_y = [-MU, -MU,  MU,  MU, -MU]
plt.plot(square_x, square_y, linewidth=2, label=rf"Pyramid: $|F_x|,|F_y|\leq \mu F_z$")

# Circle boundary (cone): sqrt(fx_hat^2 + fy_hat^2) = mu
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(MU*np.cos(theta), MU*np.sin(theta), linewidth=2, linestyle="--",label=rf"Cone: $\sqrt{{F_x^2+F_y^2}}\leq \mu F_z$")
plt.xlabel(r"$F_x/F_z$", fontsize=16)
plt.ylabel(r"$F_y/F_z$", fontsize=16)
plt.title("Tangential components normalized by $F_z$", fontsize=19)

# Make limits symmetric and slightly padded
lim = MU * 1.5
# If data exceeds, expand automatically
lim = max(lim, np.nanmax(np.abs(fx_hat_low))*1.05, np.nanmax(np.abs(fy_hat_low))*1.05)
plt.ylim(-MU-0.0005, lim+0.0015)
plt.xlim(-MU-0.0005, lim)
plt.grid(True)
plt.legend(fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
