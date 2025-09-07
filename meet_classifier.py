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
   - Displays color-coded classification results over time
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

    
# Ensure data is available
if len(all_predictions) > 0:
    print(f"Total epochs processed: {len(epoch)}")
    print(f"Predictions shape: {predictions.shape}")
    # Plot 1: CNN Class Output as Color-coded Bar
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    # Define color mapping: 0=Blue, 1=Black, 2=Yellow, 3=Red
    class_colors = ['blue', 'black', 'yellow', 'red']
    cmap = mcolors.ListedColormap(class_colors)
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(12, 2))
    # Show as an image: shape (1, N)
    plt.imshow(label[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm)
    plt.yticks([])
    plt.xlabel('Epoch')
    plt.title('CNN Class Output (Blue: SW, Black: 1/F, Yellow: MSH, Red: MSP)')

    # Custom legend
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='0/SW'),
        Patch(facecolor='black', edgecolor='k', label='1/F'),
        Patch(facecolor='yellow', edgecolor='k', label='2/MSH'),
        Patch(facecolor='red', edgecolor='k', label='3/MSP')
    ]
    plt.legend(handles=legend_elements, loc='upper right', ncol=4, bbox_to_anchor=(1, 1.5))
    plt.tight_layout()
    plt.show()
else:
    print("No data available to plot.")
