"""
Configuration file for the Bow Shock Prediction project.

This file contains all the paths to data, models, and output files.
Using a centralized configuration file makes it easier to manage file paths
and share the code with others.
"""

import os

# --- Base Directories ---
# It's good practice to define a base data directory if possible.
# This makes it easier to move the whole data folder.
BASE_DATA_DIR = r"D:\mms\Data"
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Model Files ---
MODEL_DIR = os.path.join(BASE_DATA_DIR, "models")
CNN_MODEL_201711_VERIFY = os.path.join(MODEL_DIR, "cnn_dis_201711_verify.h5")
CNN_FINAL_MODEL = os.path.join(BASE_PROJECT_DIR, "model_training", "mms_cnn_final_model.h5")


# --- Input Data ---
FPI_FAST_L2_DIS_DIST_2023_1 = os.path.join(BASE_DATA_DIR, r"mms\mms1\fpi\fast\l2\dis-dist\2023\1")
FPI_FAST_L2_DIS_DIST_2019_2 = os.path.join(BASE_DATA_DIR, r"mms\mms1\fpi\fast\l2\dis-dist\2019\2")
LABELED_DATA_2017_11 = os.path.join(BASE_DATA_DIR, r"mms\mms1\fpi\fast\l2\dis-dist\2017\11\labeled")
LABELS_HUMAN_201711_CDF = r"D:\mms\OLSHEVSKY\mmslearning\labels_human\labels_fpi_fast_dis_dist_201711.cdf"
SHOCK_DATABASE_CSV = r"D:\mms\BOW_SHOCK\database_and_overview_plots\SDB_10-Mar-2022_V1.0.csv"
PLOT_POS_CSV = "SDB_10-Mar-2022_V1.0.csv"


# --- Processed/Output Data ---
PROCESSED_DATA_2023_1 = os.path.join(BASE_DATA_DIR, "processed", "2023", "1")
PROCESSED_DATA_2017_11 = os.path.join(BASE_DATA_DIR, "processed", "2017", "11")
BOW_SHOCK_CROSSINGS_CSV = os.path.join(PROCESSED_DATA_2017_11, "bow_shock_crossings_2017_11.csv")
OUTPUT_LABELS_CDF = os.path.join(BASE_PROJECT_DIR, "output_labels.cdf")
