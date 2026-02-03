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

# Position Plots

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
plt.plot(t, com, label="No Swing Task")
plt.plot(t_high, com_high, label="High Weight")
plt.plot(t_low, com_low, label="Low Weight")
plt.xlabel("Time [s]", fontsize = 15)
plt.ylabel("CoM position [m]", fontsize = 15)
plt.title("Center of Mass Trajectory", fontsize = 22)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Error Plots

x_des = 0.00297544
y_des = 0.01088117
z_des = 0.75
com_x_error = com_x - x_des
com_y_error = com_y - y_des
com_z_error = com_z - z_des
com_x_error_high = com_x_high - x_des
com_y_error_high = com_y_high - y_des
com_z_error_high = com_z_high - z_des
com_x_error_low = com_x_low - x_des
com_y_error_low = com_y_low - y_des
com_z_error_low = com_z_low - z_des
com_error = np.sqrt(com_x_error**2 + com_y_error**2 + com_z_error**2)
com_high_error = np.sqrt(com_x_error_high**2 + com_y_error_high**2 + com_z_error_high**2)
com_low_error = np.sqrt(com_x_error_low**2 + com_y_error_low**2 + com_z_error_low**2)

plt.figure(figsize=(8, 5))
plt.plot(t, com_error, label="No Swing Task Error")
plt.plot(t_high, com_high_error, label="High Weight Error")
plt.plot(t_low, com_low_error, label="Low Weight Error")
plt.xlabel("Time [s]", fontsize = 15)
plt.ylabel("CoM position Error [m]", fontsize = 15)
plt.title("Center of Mass Error Trajectory", fontsize = 22)
plt.legend(fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()


def metrics(e):
    e = np.asarray(e)
    mse  = np.mean(e**2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(e))
    mx   = np.max(np.abs(e))
    return mse, rmse, mae, mx

for name, e in [
    ("Default", com_error),
    ("High priority", com_high_error),
    ("Low priority", com_low_error),
]:
    mse, rmse, mae, mx = metrics(e)
    print(f"{name}:  MSE={mse:.3e}  RMSE={rmse:.3e}  MAE={mae:.3e}  Max={mx:.3e}")









# FFT Plots
dt = 0.01
start = int(1.0 / dt)

z = com_z_low[start:] - np.mean(com_z_low[start:])  # remove DC
N = len(z)
freqs_low = np.fft.rfftfreq(N, d=dt)
Z_low = np.abs(np.fft.rfft(z))


z = com_z_high[start:] - np.mean(com_z_high[start:])  # remove DC
N = len(z)
freqs_high = np.fft.rfftfreq(N, d=dt)
Z_high = np.abs(np.fft.rfft(z))


z = com_z[start:] - np.mean(com_z[start:])  # remove DC
N = len(z)
freqs = np.fft.rfftfreq(N, d=dt)
Z = np.abs(np.fft.rfft(z))


# Plot
plt.plot(freqs, Z, label = "Default")
plt.plot(freqs_low, Z_low, label="Low Priority")
plt.plot(freqs_high, Z_high, label="High Priority")
plt.xlim(0, 0.5)  # up to 0.4 Hz
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("FFT of CoM Z motion")
plt.grid(True)
plt.legend()
plt.show()
