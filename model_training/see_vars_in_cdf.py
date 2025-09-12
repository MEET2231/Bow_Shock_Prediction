import cdflib


cdf_file_path = r"D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2017\11\labeled\mms1_fpi_fast_l2_dis-dist_20171102020000_v3.4.0.cdf"

cdf_file = cdflib.cdfread.CDF(cdf_file_path)
info = cdf_file.cdf_info()
print(info)
# Get all variable names in the CDF file
variables = info.zVariables
print(f"Total number of variables: {len(variables)}")

# Print all variable names
print("List of all variables:")
for i, var in enumerate(variables):
    print(f"{i+1}. {var}")

print("mms1_dis_phi_fast")
print(cdf_file.varattsget('mms1_dis_phi_fast'))
print("mms1_dis_theta_fast")
print(cdf_file.varattsget('mms1_dis_theta_fast'))
print("mms1_dis_energy_fast")
print(cdf_file.varattsget('mms1_dis_energy_fast'))


# print(cdf_file.varget('label_mms1_fpi_fast_dis_dist_20171108040000'))
# print(cdf_file.varget('probability_mms1_fpi_fast_dis_dist_20171108040000'))
# print(cdf_file.varget('epoch_mms1_fpi_fast_dis_dist_20171108040000'))

# var_info_epoch = cdf_file.varinq('epoch_mms1_fpi_fast_dis_dist_20171108040000')
# epoch_records = getattr(var_info_epoch, "Last_Rec", -1) + 1
# print(f"Total epoch records: {epoch_records}")

# count = 0

# for label in cdf_file.varget('label'):
#     print(label,count)
#     count += 1

# print(f"Total labels: {count}")
#extra commit comment

print(cdf_file.varget('label'))
print(cdf_file.varget('label_epoch'))