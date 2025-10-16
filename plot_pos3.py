import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# --- 1. Load and Clean Data ---
try:
    df = pd.read_csv('SDB_10-Mar-2022_V1.0.csv', skiprows=53)
    df.rename(columns={'#time': 'time'}, inplace=True)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Error: 'SDB_10-Mar-2022_V1.0.csv' not found.")
    exit()

# --- 2. Process Data for Plotting ---

# Convert position units from km to Earth Radii (RE)
earth_radius_km = 6371.0
df['pos_x_re'] = df['pos_x'] / earth_radius_km
df['pos_y_re'] = df['pos_y'] / earth_radius_km

# Remove rows with missing position data
df.dropna(subset=['pos_x_re', 'pos_y_re'], inplace=True)

df['year_numeric'] = pd.to_datetime(df['time'], unit='s').dt.year

# Calculate the B-field phi angle from Bx and By components
phi_rad = np.arctan2(df['By_us'], df['Bx_us'])
df['B_phi_calculated'] = np.degrees(phi_rad)
df.loc[df['B_phi_calculated'] < 0, 'B_phi_calculated'] += 360

# Apply the y-flip condition using our calculated phi angle
is_ortho_parker = (
    ((df['B_phi_calculated'] >= 45) & (df['B_phi_calculated'] <= 135)) |
    ((df['B_phi_calculated'] >= 225) & (df['B_phi_calculated'] <= 315))
)
df['pos_y_plot'] = np.where(is_ortho_parker, -df['pos_y_re'], df['pos_y_re'])


# --- 3. Setup and Create Plots ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Location of Bow Shock Crossings', fontsize=16)

# Setup axes properties
for ax in axes.flat:
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    circle = plt.Circle((0, 0), 1, color='black', fill=True, zorder=2)
    ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X ($R_E$) GSE')
    ax.set_ylabel('Y ($R_E$) GSE')
    ax.grid(True, linestyle=':', alpha=0.6)

# --- Plot with Final, Corrected Color Ranges ---
# (a) Time: Range 2015 to 2021
sc1 = axes[0, 0].scatter(
    df['pos_x_re'], df['pos_y_plot'], c=df['year_numeric'], cmap='viridis',
    s=10, alpha=0.7, vmin=2015, vmax=2020  # Set color range
)
fig.colorbar(sc1, ax=axes[0, 0], label='Time (Year)')
axes[0, 0].set_title('(a) Color Coded by Time')

# (b) Dynamic Pressure: Log scale from 0.1 to 10
sc2 = axes[0, 1].scatter(
    df['pos_x_re'], df['pos_y_plot'], c=df['Pdyn_us'], cmap='viridis',
    s=10, alpha=0.7, norm=colors.LogNorm(vmin=0.1, vmax=10) # Set log color range
)
fig.colorbar(sc2, ax=axes[0, 1], label='dyn. P (nPa)')
axes[0, 1].set_title('(b) Color Coded by Dynamic Pressure')

# (c) Alfven Mach Number: Range 0 to 25
sc3 = axes[1, 0].scatter(
    df['pos_x_re'], df['pos_y_plot'], c=df['MA'], cmap='viridis',
    s=10, alpha=0.7, vmin=0, vmax=20 # Set color range
)
fig.colorbar(sc3, ax=axes[1, 0], label='$M_A$')
axes[1, 0].set_title('(c) Color Coded by Alfven Mach Number')

# (d) Shock Angle: Range 0 to 90
sc4 = axes[1, 1].scatter(
    df['pos_x_re'], df['pos_y_plot'], c=df['thBn'], cmap='viridis',
    s=10, alpha=0.7, vmin=0, vmax=80 # Set color range
)
fig.colorbar(sc4, ax=axes[1, 1], label=r'$\theta_{Bn}$ ($^\circ$)')
axes[1, 1].set_title(r'(d) Color Coded by Shock Angle ($\theta_{Bn}$)')

# --- 4. Final Touches ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()