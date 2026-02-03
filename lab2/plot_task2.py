import csv
import matplotlib.pyplot as plt
import numpy as np

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
plt.title("Center of Mass X value")
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
plt.title("Center of Mass Y value")
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
plt.title("Center of Mass Z value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

com_x = np.array(com_x)
com_y = np.array(com_y)
com_z = np.array(com_z)
com_x_high = np.array(com_x_high)
com_y_high = np.array(com_y_high)
com_z_high = np.array(com_z_high)
com_x_low = np.array(com_x_low)
com_y_low = np.array(com_y_low)
com_z_low = np.array(com_z_low)

com = np.sqrt(com_x**2 + com_y**2 + com_z**2)
com_high = np.sqrt(com_x_high**2 + com_y_high**2 + com_z_high**2)
com_low = np.sqrt(com_x_low**2 + com_y_low**2 + com_z_low**2)

plt.figure(figsize=(8, 5))
plt.plot(t, com, label="Default")
plt.plot(t_high, com_high, label="High Weight")
plt.plot(t_low, com_low, label="Low Weight")
plt.xlabel("Time [s]")
plt.ylabel("CoM position [m]")
plt.title("Center of Mass Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


dt = 0.01
start = int(10.0 / dt)
z = com_z_low[start:] - np.mean(com_z_low[start:])  # remove DC

N = len(z)
freqs_low = np.fft.rfftfreq(N, d=dt)
Z_low = np.abs(np.fft.rfft(z))

# Plot
plt.plot(freqs_low, Z_low)
plt.xlim(0, 3)  # up to 3 Hz
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of CoM Z motion Low Priority")
plt.grid(True)
plt.show()

dt = 0.01
start = int(10.0 / dt)
z = com_z_high[start:] - np.mean(com_z_high[start:])  # remove DC

N = len(z)
freqs_high = np.fft.rfftfreq(N, d=dt)
Z_high = np.abs(np.fft.rfft(z))

# Plot
plt.plot(freqs_high, Z_high)
plt.xlim(0, 3)  # up to 3 Hz
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of CoM Z motion High Priority")
plt.grid(True)
plt.show()

dt = 0.01
start = int(10.0 / dt)
z = com_z[start:] - np.mean(com_z[start:])  # remove DC

N = len(z)
freqs = np.fft.rfftfreq(N, d=dt)
Z = np.abs(np.fft.rfft(z))

# Plot
plt.plot(freqs, Z)
plt.xlim(0, 3)  # up to 3 Hz
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of CoM Z motion Default")
plt.grid(True)
plt.show()


# Plot
plt.plot(freqs, Z, label = "Default")
plt.plot(freqs_low, Z_low, label="Low Priority")
plt.plot(freqs_high, Z_high, label="High Priority")
plt.xlim(0, 0.4)  # up to 0.4 Hz
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of CoM Z motion")
plt.grid(True)
plt.legend()
plt.show()
