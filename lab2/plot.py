import csv
import matplotlib.pyplot as plt

t = []
com_x = []
com_y = []
com_z = []

with open("com_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t.append(float(row["t"]))
        com_x.append(float(row["com_x"]))
        com_y.append(float(row["com_y"]))
        com_z.append(float(row["com_z"]))

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(t, com_x, label="CoM x")
plt.plot(t, com_y, label="CoM y")
plt.plot(t, com_z, label="CoM z")

plt.xlabel("Time [s]")
plt.ylabel("CoM position [m]")
plt.title("Center of Mass trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
