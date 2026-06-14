import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import config
from matplotlib.widgets import Slider

# Enable interactive mode to keep both windows open
plt.ion()

# --- 1. Load and Clean Data ---
try:
    df = pd.read_csv(config.PLOT_POS_CSV, skiprows=53)
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

# --- Plots ---
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

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.show()  # Non-blocking show for first window

# --- 4. Create Second Window with Shock Angle Analysis ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Bow Shock Crossings: Shock Angle Analysis', fontsize=16)

# Setup common properties for all subplots
def setup_axis(ax, title):
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    circle = plt.Circle((0, 0), 1, color='black', fill=True, zorder=2)
    ax.add_patch(circle)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X ($R_E$) GSE')
    ax.set_ylabel('Y ($R_E$) GSE')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(title)

# Filter data for different angle ranges
df_clean = df.dropna(subset=['thBn'])  # Remove NaN values
low_angle = df_clean[df_clean['thBn'] < 45]
high_angle = df_clean[df_clean['thBn'] >= 45]

# Plot 1: Angles < 45 degrees
setup_axis(axes2[0], r'Quasi-Parallel Shocks ($\theta_{Bn}$ < 45°)')
if len(low_angle) > 0:
    sc1 = axes2[0].scatter(
        low_angle['pos_x_re'], low_angle['pos_y_plot'], 
        c=low_angle['thBn'], cmap='plasma', s=15, alpha=0.7, 
        vmin=0, vmax=45
    )
    cbar1 = fig2.colorbar(sc1, ax=axes2[0])
    cbar1.set_label(r'$\theta_{Bn}$ ($^\circ$)')

# Plot 2: Angles >= 45 degrees
setup_axis(axes2[1], r'Quasi-Perpendicular Shocks ($\theta_{Bn}$ ≥ 45°)')
if len(high_angle) > 0:
    sc2 = axes2[1].scatter(
        high_angle['pos_x_re'], high_angle['pos_y_plot'], 
        c=high_angle['thBn'], cmap='plasma', s=15, alpha=0.7, 
        vmin=45, vmax=90
    )
    cbar2 = fig2.colorbar(sc2, ax=axes2[1])
    cbar2.set_label(r'$\theta_{Bn}$ ($^\circ$)')

# Plot 3: Interactive slider plot
setup_axis(axes2[2], r'Interactive: Adjust $\theta_{Bn}$ Range')

# Create initial scatter plot (empty)
scatter_interactive = axes2[2].scatter([], [], c=[], cmap='plasma', s=15, alpha=0.7)

# Add space for slider
plt.subplots_adjust(bottom=0.25)

# Create slider
ax_slider = plt.axes([0.67, 0.1, 0.25, 0.03])
angle_slider = Slider(
    ax_slider, r'$\theta_{Bn}$ Range', 0, 80, 
    valinit=45, valfmt='%.0f°'
)

# Slider update function
def update_plot(val):
    angle_threshold = angle_slider.val
    
    # Filter data based on slider value (±5 degrees range)
    mask = (df_clean['thBn'] >= angle_threshold - 5) & (df_clean['thBn'] <= angle_threshold + 5)
    filtered_data = df_clean[mask]
    
    # Clear and replot
    axes2[2].clear()
    setup_axis(axes2[2], f'$\\theta_{{Bn}}$ = {angle_threshold:.0f}° ± 5°')
    
    if len(filtered_data) > 0:
        sc = axes2[2].scatter(
            filtered_data['pos_x_re'], filtered_data['pos_y_plot'], 
            c=filtered_data['thBn'], cmap='plasma', s=15, alpha=0.7, 
            vmin=0, vmax=90
        )
        # Update the plot
        axes2[2].text(0.02, 0.98, f'N = {len(filtered_data)} points', 
                     transform=axes2[2].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig2.canvas.draw()

# Connect slider to update function
angle_slider.on_changed(update_plot)

# Initialize the interactive plot
update_plot(45)

fig2.show()  # Non-blocking show for second window

# Keep both windows open
plt.ioff()  # Turn off interactive mode
plt.show()  # Final blocking show to keep script running