import cdflib


cdf_file_path = r"D:\mms\workspace\Bow_Shock_Prediction\output_labels.cdf"

cdf_file = cdflib.cdfread.CDF(cdf_file_path)
print(cdf_file.cdf_info())

count = 0

for label in cdf_file.varget('label'):
    print(label,count)
    count += 1

print(f"Total labels: {count}")
