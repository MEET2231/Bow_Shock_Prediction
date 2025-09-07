from spacepy.pycdf import CDF


cdf_file_path = r"D:\mms\Data\mms\mms1\fpi\fast\l2\dis-dist\2018\11\mms1_fpi_fast_l2_dis-dist_20181114160000_v3.4.0.cdf"

with CDF(cdf_file_path) as cdf:

    variables = list(cdf.keys())
    print("Variables in the CDF file:")
    print(variables)


    variable_name = variables[0]  
    data = cdf[variable_name][:]
    print(f"\nData for variable '{variable_name}':")
    print(data)