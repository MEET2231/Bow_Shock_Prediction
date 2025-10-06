import numpy as np
import matplotlib.pyplot as plt
import cdflib

# --- Configuration ---
# The CDF file created by your classifier script
input_cdf_file = 'output_labels.cdf' 

# --- Main Script ---

try:
    # 1. Read probabilities and epoch data from the input CDF file
    print(f"Reading data from '{input_cdf_file}'...")
    with cdflib.CDF(input_cdf_file) as cdf_file:
        probs = cdf_file.varget('probability')
        epoch = cdf_file.varget('epoch')
    print("Data loaded successfully.")

    # 2. Convert TT2000 epoch to Python datetime objects
    # This is the key step for changing the x-axis
    print("Converting TT2000 timestamps to readable dates...")
    datetimes = cdflib.cdfepoch.to_datetime(epoch)

    # 3. Calculate the probability difference (same logic as before)
    print("Calculating probability difference and finding transitions...")
    # Calculate P(0) + P(1) - P(2)
    raw_diff = probs[:, 0] + probs[:, 1] - probs[:, 2]
    
    # Set to 0 when absolute value is less than 0.9 and round others
    prob_diff_thresholded = np.where(np.abs(raw_diff) < 0.90, 0, np.round(raw_diff, 0))

    # Apply moving median filter with a window size of 12
    window_size = 12
    prob_diff_median = np.zeros_like(prob_diff_thresholded)
    for i in range(len(prob_diff_thresholded)):
        start = max(0, i - window_size // 2)
        end = min(len(prob_diff_thresholded), i + window_size // 2 + 1)
        prob_diff_median[i] = np.median(prob_diff_thresholded[start:end])

    # Calculate the derivative to find the exact moment of transition
    prob_diff_derivative = np.zeros_like(prob_diff_median)
    prob_diff_derivative[1:] = prob_diff_median[1:] - prob_diff_median[:-1]
    
    # The final data for the y-axis
    shock_indicator = -prob_diff_derivative
    print("Calculations complete.")

    # 4. Plot the results with a real time axis
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(datetimes, shock_indicator, marker='.', linestyle='-', color='royalblue', markersize=8, zorder=10)
    
    # Add vertical lines for detected shocks to make them stand out
    shock_indices = np.where(np.abs(shock_indicator) > 0)[0]
    for index in shock_indices:
        ax.axvline(datetimes[index], color='crimson', linestyle='--', linewidth=0.8, alpha=0.7)

    # Formatting the plot for clarity
    ax.set_title('Shock Crossing Indicator vs. Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (UTC)', fontsize=12)
    ax.set_ylabel('Shock Indicator (Derivative of $\Delta p$)', fontsize=12)
    ax.set_ylim(-2.5, 2.5) # Set Y-limits to focus on the +/- 2 spikes
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Improve date formatting on the x-axis
    from matplotlib.dates import DateFormatter
    date_form = DateFormatter("%Y-%m-%d\n%H:%M:%S")
    ax.xaxis.set_major_formatter(date_form)

    plt.tight_layout()
    plt.show()
    print("Plot displayed.")

except FileNotFoundError:
    print(f"Error: The file '{input_cdf_file}' was not found.")
    print("Please make sure the CDF file is in the same directory as this script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")