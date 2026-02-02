import csv
import matplotlib.pyplot as plt

t = []
com_x = []
com_y = []
com_z = []

t_high = []
com_x_high = []
com_y_high = []
com_z_high = []

t_low = []
com_x_low = []
com_y_low = []
com_z_low = []

with open("com_log_default.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["t"]))
        com_x.append(float(row["com_x"]))
        com_y.append(float(row["com_y"]))
        com_z.append(float(row["com_z"]))

with open("com_log_high_weight.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_high.append(float(row["t"]))
        com_x_high.append(float(row["com_x"]))
        com_y_high.append(float(row["com_y"]))
        com_z_high.append(float(row["com_z"]))

with open("com_log_low_weight.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_low.append(float(row["t"]))
        com_x_low.append(float(row["com_x"]))
        com_y_low.append(float(row["com_y"]))
        com_z_low.append(float(row["com_z"]))

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(t, com_x, label="Default")
plt.plot(t_high, com_x_high, label="High Weight")
plt.plot(t_low, com_x_low, label="Low Weight")
plt.xlabel("Time [s]")
plt.ylabel("CoM X position [m]")
plt.title("Center of Mass trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, com_y, label="Default")
plt.plot(t_high, com_y_high, label="High Weight")
plt.plot(t_low, com_y_low, label="Low Weight")
plt.xlabel("Time [s]")
plt.ylabel("CoM Y position [m]")
plt.title("Center of Mass trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, com_z, label="Default")
plt.plot(t_high, com_z_high, label="High Weight")
plt.plot(t_low, com_z_low, label="Low Weight")
plt.xlabel("Time [s]")
plt.ylabel("CoM Z position [m]")
plt.title("Center of Mass trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

