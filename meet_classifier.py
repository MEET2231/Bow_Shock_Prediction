"""
MEET Magnetosphere Region Classifier

This script classifies magnetosphere regions using a pre-trained CNN model based on MMS (Magnetospheric
Multiscale Mission) Fast Plasma Investigation (FPI) particle distribution data.

Working Logic:
-------------
1. Data Loading:
   - Loads a CDF (Common Data Format) file containing MMS FPI distribution data
   - Processes data in chunks to handle large files efficiently

2. Data Preprocessing:
   - Normalizes the distribution data using log transformation and scaling
   - Prepares the data in the format expected by the CNN model (adds channel dimension)

3. Classification:
   - Uses a pre-trained CNN model to classify each distribution into one of four regions:
     * 0/Blue: Solar Wind (SW)
     * 1/Black: Ion Foreshock (1/F)
     * 2/Yellow: Magnetosheath (MSH)
     * 3/Red: Magnetosphere (MSP)

4. Visualization:
   - Displays color-coded classification results over time (using plot_utils module)
   - Uses a custom color scheme to distinguish between different regions
   - Includes a legend for region identification

The classifier helps identify different regions in Earth's magnetosphere and surrounding space,
which is critical for studying space physics phenomena like magnetic reconnection, shocks, and
boundary layers.

Author: MEET
Date: September 2023
"""

from tensorflow.keras.models import load_model
import cdflib
import numpy as np
import matplotlib.pyplot as plt
# Import custom plotting utilities
import plot_utils

regions = {-1: 'Unknown', 0: 'Solar Wind', 1: 'Foreshock', 2: 'Magnetosheath', 3: 'Magnetosphere'}
lbl_spec = {'Copyright': 'Meet Amitbhai Modi (modimeet05@gmail.com)'}

def normalize_data(X, verbose=True):
    """ Compute logarithm and normalize the data for learning.

    Parameters:
        X - [epoch, Phi, Theta, Energy]
    
    Returns:
        Normalized array with same shape as input
    """
    # Optimized version - same logic but more efficient
    if verbose:
        print('Normalizing data array', X.shape)
    
    # Find minimum non-zero value (with error handling)
    nonzero_mask = ~np.isclose(X, 0, rtol=0, atol=1e-30)
    if not np.any(nonzero_mask):
        print('Warning! All elements of X are zero, returning a zero-array.')
        return X
    
    min_value = np.min(X[nonzero_mask])
    
    # Replace zeros with minimum value
    if verbose:
        print('Replacing zeros with min...')
    X = np.where(nonzero_mask, X, min_value)
    
    # Log transform
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    
    # Normalize to [0, 1] range
    if verbose:
        print('Subtracting min...')
    X_min = X.min()
    X -= X_min
        
    if verbose:
        print('Normalizing to 1...')
    X_max = X.max()
    if X_max > 0:  # Avoid division by zero
        X /= X_max
    
    if verbose:
        print('Rolling along Phi...')
    X = np.roll(X, 16, axis=X.ndim-2)
    return X


model = load_model(r"D:\mms\Data\models\cnn_dis_201711_verify.h5")
# model.summary()

fpi_cdf_file = cdflib.CDF(r"D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2018\11\mms1_fpi_fast_l2_dis-dist_20181114160000_v3.4.0.cdf")
var_name = 'mms1' + '_' + 'dis' + '_dist_fast'
var_info = fpi_cdf_file.varinq(var_name)
var_info_epoch = fpi_cdf_file.varinq('Epoch')

print(var_info)
print(var_info_epoch)

chunk_size = 50 
num_records = getattr(var_info, "Last_Rec", -1) + 1
epoch_records = getattr(var_info_epoch, "Last_Rec", -1) + 1

print(f"Number of records in {var_name}: {num_records}")
print(f"Number of records in Epoch: {epoch_records}")

all_predictions = []
all_epoch = []
all_labels = []

for start_idx in range(0, num_records, chunk_size):
    end_idx = min(start_idx + chunk_size, num_records)
    print(f"Processing records {start_idx} to {end_idx-1} of {num_records}")

    try:
        dist_chunk = fpi_cdf_file.varget(var_name, startrec=start_idx, endrec=end_idx-1)
        epoch_chunk = fpi_cdf_file.varget('Epoch', startrec=start_idx, endrec=min(end_idx-1, epoch_records-1))
        # dist_chunk = dist_chunk.swapaxes(1, 3)
        # Prepare data for classificator: normalize and add extra dimension - 'channel'
        dist_chunk = normalize_data(dist_chunk, verbose=False)
        dist_chunk = dist_chunk.reshape(dist_chunk.shape + (1,))

        chunk_predictions = model.predict(dist_chunk, verbose=0)
        chunk_label = chunk_predictions.argmax(axis=1)
        all_predictions.append(chunk_predictions)
        all_labels.append(chunk_label)
        all_epoch.extend(epoch_chunk)
    except Exception as e:
        print(f"Error processing chunk {start_idx}-{end_idx-1}: {str(e)}")
        continue
# Concatenate all results
if len(all_predictions) > 0:
        import numpy as np
        epoch = np.array(all_epoch)
        predictions = np.vstack(all_predictions) if len(all_predictions) > 1 else all_predictions[0]
        label = np.hstack(all_labels) if len(all_labels) > 1 else all_labels[0]
else:
    print(f"Warning: No valid data processed from {fpi_cdf_file}.")



# Exporting results as a CDF file  
lbl_cdf_file = cdflib.cdfwrite.CDF('output_labels.cdf', cdf_spec=lbl_spec, delete=True)
vs_pred = {
    'Variable': 'label',   # Name of the variable inside the CDF
    'Data_Type': 1,                     # CDF_INT1 (8-bit integer, signed)
    'Num_Elements': 1,                  # Each record holds 1 element (scalar, not a vector)
    'Rec_Vary': True,                   # Records vary (each time step can have different values)
    'Var_Type': 'zVariable',            # It's a zVariable (modern, flexible CDF variable type)
    'Dim_Sizes': [],                    # No extra dimensions (1D variable over time)
    'Num_Dims': 0,                      # Explicitly says: zero dimensions besides record index
    'Sparse': 'No_sparse',              # Don’t use sparse storage (all records stored fully)
    'Compress': 6,                      # Apply compression (gzip level 6)
    'Pad': np.array([-1], dtype=np.int8) # Fill value for missing records (-1 means no label)
}
attrs_pred = {
    'VAR_NOTES': 'Predicted label' + str(regions),
    'filename': 'mms1_fpi_fast_l2_dis-dist_20181114160000_v3.4.0.cdf'
}
lbl_cdf_file.write_var(vs_pred, var_attrs=attrs_pred, var_data=label)

# writing probabilities
vs_prob = {
    'Variable': 'probability',  # name, e.g. probability_mms1_dis_fast
    'Data_Type': 21,                         # CDF_REAL4 (32-bit float)
    'Num_Elements': 1,                       # one element per dimension
    'Rec_Vary': True,                        # varies with each record (time step)
    'Var_Type': 'zVariable',                 # modern flexible variable type
    'Dim_Sizes': [len(regions)-1],           # vector length = number of classes - 1
    'Num_Dims': 0,                           # dimension info (0 means defined in Dim_Sizes only)
    'Sparse': 'No_sparse',                   # don’t use sparse storage
    'Compress': 6,                           # gzip compression level 6
    'Pad': np.array([0.], dtype=np.float32)  # fill value = 0.0 for missing data
}
attrs_prob = {
    'VAR_NOTES': 'Probability of the predicted label.',
    'filename': 'mms1_fpi_fast_l2_dis-dist_20181114160000_v3.4.0.cdf'
}
lbl_cdf_file.write_var(vs_prob, var_attrs=attrs_prob, var_data=predictions)

# writiing epoch
epoch_info = {
    'Variable': 'epoch',           # name, e.g. epoch_mms1_dis_fast
    'Data_Type': 33,                            # CDF_TIME_TT2000 (nanoseconds since J2000)
    'Num_Elements': 1,                          # one timestamp per record
    'Rec_Vary': True,                           # changes for every record
    'Var_Type': 'zVariable',                    # modern flexible variable type
    'Dim_Sizes': [],                            # scalar (no extra dims)
    'Num_Dims': 0,                              # no extra dimensions
    'Sparse': 'No_sparse',                      # don’t use sparse storage
    'Compress': 6,                              # gzip compression level 6
    'Pad': np.array([-9223372036854775807], dtype=np.int64), # fill value for missing epochs
    'Data_Type_Description': 'CDF_TIME_TT2000'  # human-readable description
}
epoch_attrs = {
    'CATDESC': 'Nanoseconds since J2000',       # description
    'DELTA_PLUS_VAR': 'Epoch_plus_var',         # references to optional timing uncertainty variables
    'DELTA_MINUS_VAR': 'Epoch_minus_var',
    'FIELDNAM': 'Time tags',                    # field name
    'FILLVAL': np.array([-9223372036854775808], dtype=np.int64), # missing-value code
    'LABLAXIS': 'Epoch',                        # label for plotting
    'MONOTON': 'INCREASE',                      # expected to increase monotonically
    'REFERENCE_POSITION': 'Rotating Earth Geoid', # time reference frame
    'SCALETYP': 'linear',                       # scaling type
    'SI_CONVERSION': '1.0e-9>s',                # nanoseconds to seconds
    'TIME_BASE': 'J2000',                       # epoch zero point
    'TIME_SCALE': 'Terrestrial Time',           # time scale
    'UNITS': 'ns',                              # units
    'VALIDMIN': np.array([-315575942816000000], dtype=np.int64), # min valid epoch
    'VALIDMAX': np.array([3155716868184000000], dtype=np.int64), # max valid epoch
    'VAR_NOTES': 'MMS1 FPI/DIS Fast Survey data begin-time; derived from packet time.',
    'VAR_TYPE': 'support_data'                  # indicates it’s an auxiliary variable
}
lbl_cdf_file.write_var(epoch_info, var_attrs=epoch_attrs, var_data=epoch)
print(getattr(epoch_info, 'Variable', 'Variable name not available'))








# Ensure data is available
if len(all_predictions) > 0:
    print(f"Total epochs processed: {len(epoch)}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Check for class distribution
    unique_labels, label_counts = np.unique(label, return_counts=True)
    print("Class distribution:")
    for lbl, cnt in zip(unique_labels, label_counts):
        print(f"  Class {int(lbl)}: {cnt} samples ({cnt/len(label):.2%})")
    
    # Plot the region classifications using the plot_utils module
    plot_utils.plot_region_classifications(epoch, label)
    
    # Optionally, you can also generate a combined plot with energy spectrogram
    # To use this, you need to generate energy spectrogram data first
    # Example:
    # energy_var = 'mms1_dis_energy_fast'
    # if energy_var in fpi_cdf_file.cdf_info().zVariables:
    #     energy_data = fpi_cdf_file.varget(energy_var)
    #     energy_spectrogram = generate_energy_spectrogram(fpi_cdf_file, var_name, num_records)
    #     plot_utils.plot_combined_spectrograms(epoch, label, energy_spectrogram, energy_data)
else:
    print("No data available to plot.")


