"""
MEET Magnetosphere Region Classifier - Batch Processor

This script automates the classification of magnetosphere regions for an entire
directory of MMS FPI CDF files.

Working Logic:
-------------
1.  Setup: Defines input and output directories and loads the pre-trained CNN model once.
2.  File Discovery: Scans the input directory for all '.cdf' files.
3.  Batch Loop: Iterates through each discovered CDF file.
    a.  For each input file, it determines a unique, corresponding output filename.
    b.  It calls a processing function to perform the classification.
4.  Per-File Processing (process_cdf_file function):
    a.  Loads a single CDF file and processes its data in chunks for memory efficiency.
    b.  Preprocesses the data using normalization.
    c.  Classifies each time-step using the CNN model.
    d.  Saves the results (labels, probabilities, epoch) to a new CDF file in the
        output directory.

This automates the data generation pipeline, preparing a large dataset for further
analysis, such as shock detection.

Author: MEET
Date: September 2023 (Modified for batch processing in October 2025)
"""
import os
import glob
from tensorflow.keras.models import load_model
import cdflib
import numpy as np
import config

# --- Global Configuration & Helper Functions ---

regions = {-1: 'Unknown', 0: 'Solar Wind', 1: 'Foreshock', 2: 'Magnetosheath', 3: 'Magnetosphere'}
lbl_spec = {'Copyright': 'Meet Amitbhai Modi (modimeet05@gmail.com)'}

def normalize_data(X, verbose=False):
    """ Compute logarithm and normalize the data for learning. """
    # Optimized version - same logic but more efficient
    if verbose: print('Normalizing data array', X.shape)
    
    nonzero_mask = ~np.isclose(X, 0, rtol=0, atol=1e-30)
    if not np.any(nonzero_mask):
        print('Warning! All elements of X are zero, returning a zero-array.')
        return X
    
    min_value = np.min(X[nonzero_mask])
    X = np.where(nonzero_mask, X, min_value)
    X = np.log10(X)
    
    X_min = X.min()
    X -= X_min
    
    X_max = X.max()
    if X_max > 0:
        X /= X_max
    
    X = np.roll(X, 16, axis=X.ndim-2)
    return X

def process_cdf_file(input_file_path, output_file_path, model):
    """
    Processes a single MMS FPI CDF file, classifies regions, and saves the output.
    """
    print(f"--- Processing: {os.path.basename(input_file_path)} ---")
    
    try:
        with cdflib.CDF(input_file_path) as fpi_cdf_file:
            var_name = 'mms1_dis_dist_fast'
            if var_name not in fpi_cdf_file.cdf_info().zVariables:
                print(f"  [ERROR] Variable '{var_name}' not found in file. Skipping.")
                return

            var_info = fpi_cdf_file.varinq(var_name)
            var_info_epoch = fpi_cdf_file.varinq('Epoch')

            chunk_size = 200  # Increased chunk size for potentially better performance
            num_records = getattr(var_info, "Last_Rec", -1) + 1
            epoch_records = getattr(var_info_epoch, "Last_Rec", -1) + 1
            
            if num_records == 0:
                print("  [INFO] File contains no data records. Skipping.")
                return

            all_predictions, all_epoch, all_labels = [], [], []

            for start_idx in range(0, num_records, chunk_size):
                end_idx = min(start_idx + chunk_size, num_records)
                
                dist_chunk = fpi_cdf_file.varget(var_name, startrec=start_idx, endrec=end_idx-1)
                epoch_chunk = fpi_cdf_file.varget('Epoch', startrec=start_idx, endrec=min(end_idx-1, epoch_records-1))
                
                dist_chunk_norm = normalize_data(dist_chunk)
                dist_chunk_reshaped = dist_chunk_norm.reshape(dist_chunk_norm.shape + (1,))

                chunk_predictions = model.predict(dist_chunk_reshaped, verbose=0)
                chunk_label = chunk_predictions.argmax(axis=1)

                all_predictions.append(chunk_predictions)
                all_labels.append(chunk_label)
                all_epoch.extend(epoch_chunk)

        if not all_predictions:
            print("  [INFO] No data was processed from this file. Skipping.")
            return

        epoch = np.array(all_epoch)
        predictions = np.vstack(all_predictions)
        label = np.hstack(all_labels)

        # --- Exporting results as a new CDF file ---
        with cdflib.cdfwrite.CDF(output_file_path, cdf_spec=lbl_spec, delete=True) as lbl_cdf_file:
            # Define variable specifications (condensed for clarity)
            vs_pred = {'Variable': 'label', 'Data_Type': 1, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable', 'Dim_Sizes': [], 'Compress': 6, 'Pad': np.array([-1], dtype=np.int8)}
            attrs_pred = {'VAR_NOTES': 'Predicted label' + str(regions), 'Original_Source_File': os.path.basename(input_file_path)}
            lbl_cdf_file.write_var(vs_pred, var_attrs=attrs_pred, var_data=label)

            vs_prob = {'Variable': 'probability', 'Data_Type': 21, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable', 'Dim_Sizes': [len(regions)-1], 'Compress': 6, 'Pad': np.array([0.], dtype=np.float32)}
            attrs_prob = {'VAR_NOTES': 'Probability of the predicted label.', 'Original_Source_File': os.path.basename(input_file_path)}
            lbl_cdf_file.write_var(vs_prob, var_attrs=attrs_prob, var_data=predictions)

            epoch_info = {'Variable': 'epoch', 'Data_Type': 33, 'Num_Elements': 1, 'Rec_Vary': True, 'Var_Type': 'zVariable', 'Dim_Sizes': [], 'Compress': 6, 'Pad': np.array([-9223372036854775807], dtype=np.int64)}
            epoch_attrs = {'CATDESC': 'Nanoseconds since J2000', 'UNITS': 'ns', 'TIME_BASE': 'J2000'}
            lbl_cdf_file.write_var(epoch_info, var_attrs=epoch_attrs, var_data=epoch)

        print(f"  [SUCCESS] Saved results to {os.path.basename(output_file_path)}")

    except Exception as e:
        print(f"  [ERROR] Failed to process file. Reason: {str(e)}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. DEFINE PATHS
    input_directory = config.FPI_FAST_L2_DIS_DIST_2023_1
    output_directory = config.PROCESSED_DATA_2023_1
    model_path = config.CNN_MODEL_201711_VERIFY

    # 2. CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST
    os.makedirs(output_directory, exist_ok=True)
    
    # 3. LOAD THE MODEL ONCE
    print("Loading CNN model...")
    try:
        model = load_model(model_path)
        model.summary()
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"FATAL: Could not load model from {model_path}. Error: {e}")
        exit() # Exit if model can't be loaded

    # 4. FIND ALL CDF FILES IN THE INPUT DIRECTORY
    cdf_files_to_process = glob.glob(os.path.join(input_directory, '*.cdf'))
    
    if not cdf_files_to_process:
        print(f"No '.cdf' files found in '{input_directory}'. Please check the path.")
    else:
        print(f"Found {len(cdf_files_to_process)} CDF files to process.\n")
    
    # 5. LOOP THROUGH FILES AND PROCESS EACH ONE
    for input_path in cdf_files_to_process:
        # Construct a proper output filename
        base_filename = os.path.basename(input_path)
        output_filename = f"labels_{base_filename}"
        output_path = os.path.join(output_directory, output_filename)
        
        # Process the current file
        process_cdf_file(input_path, output_path, model)

    print("\n--- Batch processing complete. ---")   