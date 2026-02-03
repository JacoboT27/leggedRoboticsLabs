import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Settings
FILE_1 = 'mu-0.5.parquet'
FILE_2 = 'mu-0.005.parquet'
MU = 0.005

# 2. Load
df1 = pd.read_parquet(FILE_1)
df2 = pd.read_parquet(FILE_2)

# Check if columns exist (debugging)
if 'current_forces_contact_pos_0' not in df1.columns:
    print("Error: Columns not found. Available columns are:", df1.columns.tolist())
    exit()

# 3. Extract Data (Force X, Y, Z for Left Foot)
# File 1
f1_x = df1['current_forces_contact_pos_0'].to_numpy()
f1_y = df1['current_forces_contact_pos_1'].to_numpy()
f1_z = df1['current_forces_contact_pos_2'].to_numpy()
f1_t = np.sqrt(f1_x**2 + f1_y**2)

# File 2
f2_x = df2['current_forces_contact_pos_0'].to_numpy()
f2_y = df2['current_forces_contact_pos_1'].to_numpy()
f2_z = df2['current_forces_contact_pos_2'].to_numpy()
f2_t = np.sqrt(f2_x**2 + f2_y**2)

# 4. Plot
plt.figure()

# Data points
plt.scatter(f1_t, f1_z, color='blue', label=FILE_1, s=5)
plt.scatter(f2_t, f2_z, color='red', label=FILE_2, s=5)

# Cone lines
z_max = max(f1_z.max(), f2_z.max())
plt.plot([0, MU * z_max], [0, z_max], 'k', label='Cone Limit')
plt.plot([0, -MU * z_max], [0, z_max], 'k')

plt.xlabel('Tangential Force (Ft)')
plt.ylabel('Normal Force (Fz)')
plt.legend()
plt.show()