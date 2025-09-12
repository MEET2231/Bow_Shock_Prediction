import numpy as np
import matplotlib.pyplot as plt
import cdflib

# Input and output file paths
input_cdf = 'output_labels.cdf'
output_cdf = 'prob_diff.cdf'

# Read probabilities from the input CDF file
cdf_file = cdflib.CDF(input_cdf)
# Assuming the probabilities are stored in a variable named 'probabilities' with shape (N, 3)
probs = cdf_file.varget('probability')

# Calculate P(0) + P(1) - P(2) for each reading
raw_diff = probs[:, 0] + probs[:, 1] - probs[:, 2]
# Set to 0 when absolute value is less than 0.9
prob_diff = np.where(np.abs(raw_diff) < 0.90, 0, np.round(raw_diff, 0))

# Apply moving median filter with a window size of 12 (adjust as needed)
window_size = 12
prob_diff_median = np.zeros_like(prob_diff)
for i in range(len(prob_diff)):
    start = max(0, i - window_size // 2)
    end = min(len(prob_diff), i + window_size // 2 + 1)
    prob_diff_median[i] = np.median(prob_diff[start:end])

# Calculate the derivative (difference between consecutive values)
prob_diff_derivative = np.zeros_like(prob_diff_median)
prob_diff_derivative[1:] = prob_diff_median[1:] - prob_diff_median[:-1]
# First value has no previous value, so set it to 0
prob_diff_derivative[0] = 0

# Update prob_diff to be the derivative for plotting
prob_diff = -prob_diff_derivative


# Plot the probability difference
plt.figure(figsize=(10, 5))
plt.plot(prob_diff, marker='o', linestyle='-', color='b')
plt.title('Probability Difference: P(0) + P(1) - P(2)')
plt.xlabel('Reading Index')
plt.ylabel('Probability Difference')
plt.grid(True)
plt.tight_layout()
plt.show()
