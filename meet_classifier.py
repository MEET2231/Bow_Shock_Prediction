from tensorflow.keras.models import load_model
import cdflib
import numpy as np
import matplotlib.pyplot as plt


def normalize_data(X, verbose=True):
    """ Compute logarithm and normalize the data for learning.

    Parameters:
        X - [epoch, Phi, Theta, Energy]

    """

    # Old way
    if verbose:
        print('Normalizing data array', X.shape)
    try:
        min_value = np.min(X[np.nonzero(X)])
    except ValueError:
        print('Warning! All elements of X are zero, returning a zero-array.')
        return X
    if verbose:
        print('Replacing zeros with min...')
    X = np.where(np.isclose(X, 0, rtol=0, atol=1e-30), min_value, X)
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X -= X.min()

    '''
    # New way
    min_value = 1e-30
    if verbose:
        print('Replacing negatives with zeros...')
    X = np.where(X > 0, X, min_value)
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X += 30
    '''
    if verbose:
        print('Normalizing to 1...')
    X /= X.max()
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
    # Plot 1: Epoch vs Probability
    plt.figure(figsize=(10, 5))
    plt.plot(epoch, predictions[:, 1], label='Probability of Class 1')
    plt.xlabel('Epoch')
    plt.ylabel('Probability')
    plt.title('Epoch vs Probability')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot 2: CNN Class Output as Color-coded Bar
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
