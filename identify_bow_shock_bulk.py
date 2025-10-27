import os
import glob
import numpy as np
import cdflib
import csv



import config

def find_shocks_in_file(cdf_path):
    """
    Identifies shock crossings in a single labeled CDF file.

    Args:
        cdf_path (str): The full path to the input 'labels_*.cdf' file.

    Returns:
        list: A list of lists, where each inner list contains the 
              [epoch, crossing_direction, source_filename] for a detected shock.
              Returns an empty list if no shocks are found or an error occurs.
    """
    try:
        with cdflib.CDF(cdf_path) as cdf_file:
            probs = cdf_file.varget('probability')
            epoch = cdf_file.varget('epoch')
        
        if probs is None or epoch is None or len(epoch) < 2:
            # Not enough data to process
            return []

        # Calculate P(0) + P(1) - P(2)
        raw_diff = probs[:, 0] + probs[:, 1] - probs[:, 2]
        
        # Apply threshold and rounding
        prob_diff_thresholded = np.where(np.abs(raw_diff) < 0.90, 0, np.round(raw_diff, 0))

        # Apply moving median filter
        window_size = 12
        prob_diff_median = np.zeros_like(prob_diff_thresholded)
        for i in range(len(prob_diff_thresholded)):
            start = max(0, i - window_size // 2)
            end = min(len(prob_diff_thresholded), i + window_size // 2 + 1)
            prob_diff_median[i] = np.median(prob_diff_thresholded[start:end])

        # Calculate derivative to find transitions
        prob_diff_derivative = np.zeros_like(prob_diff_median)
        prob_diff_derivative[1:] = prob_diff_median[1:] - prob_diff_median[:-1]
        
        shock_indicator = -prob_diff_derivative

        # Find indices where the absolute value of the indicator is greater than 1
        # This corresponds to a full jump from -1 to 1 or 1 to -1.
        shock_indices = np.where(np.abs(shock_indicator) >= 1)[0]
        
        found_shocks = []
        if len(shock_indices) > 0:
            source_filename = os.path.basename(cdf_path)
            for index in shock_indices:
                shock_epoch = epoch[index]
                indicator_value = shock_indicator[index]
                
                # A transition from SW/IF (+1) to MSH (-1) results in a derivative of -2.
                # Our indicator is -derivative, so a positive value (+2) is INBOUND.
                if indicator_value > 0:
                    direction = 'Inbound (SW -> MSH)'
                else:
                    direction = 'Outbound (MSH -> SW)'
                
                found_shocks.append([shock_epoch, direction, source_filename])
        
        return found_shocks

    except Exception as e:
        print(f"  [ERROR] Could not process file {os.path.basename(cdf_path)}. Reason: {e}")
        return []


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. DEFINE PATHS
    # Directory containing your 'labels_*.cdf' files
    input_directory = config.PROCESSED_DATA_2017_11
    
    # The final output CSV file
    output_csv_file = config.BOW_SHOCK_CROSSINGS_CSV

    # 2. FIND ALL LABELED CDF FILES
    # The pattern 'labels_*.cdf' ensures we only get the files from the previous step
    labeled_files = glob.glob(os.path.join(input_directory, 'labels_*.cdf'))
    
    if not labeled_files:
        print(f"No 'labels_*.cdf' files found in '{input_directory}'. Please check the path and filenames.")
    else:
        print(f"Found {len(labeled_files)} labeled CDF files to process.\n")

    # 3. PROCESS EACH FILE AND COLLECT ALL SHOCK EVENTS
    all_shock_events = []
    for cdf_path in sorted(labeled_files): # Sorting ensures chronological processing
        print(f"Scanning: {os.path.basename(cdf_path)}")
        shocks_from_file = find_shocks_in_file(cdf_path)
        if shocks_from_file:
            print(f"  --> Found {len(shocks_from_file)} shock crossing(s).")
            all_shock_events.extend(shocks_from_file)
        else:
            print("  --> No shock crossings found.")
    
    # 4. WRITE ALL COLLECTED EVENTS TO A SINGLE CSV FILE
    print(f"\nFound a total of {len(all_shock_events)} shock crossings for the month.")
    
    if all_shock_events:
        print(f"Writing all events to '{output_csv_file}'...")
        try:
            with open(output_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write the header
                writer.writerow(['Epoch_TT2000', 'Crossing_Direction', 'Source_File'])
                # Write all the event data
                writer.writerows(all_shock_events)
            print("CSV file created successfully.")
        except Exception as e:
            print(f"  [ERROR] Failed to write CSV file. Reason: {e}")

    print("\n--- Shock identification complete. ---")