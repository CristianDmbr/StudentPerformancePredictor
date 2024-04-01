import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

dataBase = pd.read_csv('Database/Academic Database.csv')
pd.set_option('display.max_columns', None)

####################################################################################
# Adding new columns:

# Calculate the average of SABER PRO'S 'QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO' metrics
dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)
dataBase["Performance_SABER_PRO"] = np.select(
    [
        dataBase['Average_Score_SABER_PRO'] >= 90,
        dataBase['Average_Score_SABER_PRO'] >= 80,
        dataBase['Average_Score_SABER_PRO'] >= 70,
        dataBase['Average_Score_SABER_PRO'] >= 60
    ],
    [
        'Best',
        'Very Good',
        'Good',
        'Pass'
    ],
    default='Fail'
)

# Calculate the average of Saber 11 competences
dataBase["Average_Score_Saber11"] = dataBase[['MAT_S11', 'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11']].mean(axis=1)
dataBase["Performance_Saber11"] = np.select(
    [
        dataBase['Average_Score_Saber11'] >= 90,
        dataBase['Average_Score_Saber11'] >= 80,
        dataBase['Average_Score_Saber11'] >= 70,
        dataBase['Average_Score_Saber11'] >= 60
    ],
    [
        'Best',
        'Very Good',
        'Good',
        'Pass'
    ],
    default='Fail'
)

####################################################################################
# Replace EDU_FATHER and EDU_MOTHER records of “Not sure” and “0” and “Ninguno” (meaning None) with “None”
dataBase["EDU_FATHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["EDU_MOTHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)

# Replace OOC_FATHER and OCC_MOTHER records of “0”, “Home” and “Retires” with "unemployed"
dataBase["OCC_FATHER"].replace({"0": "Unemployed", "Home": "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["OCC_MOTHER"].replace({"0": "Unemployed", "Home": "Unemployed", "Retired": "Unemployed"}, inplace=True)

# Change “Esta clasificada en otro Level del SISBEN” into "It is classified in another SISBEN Level" from SISBEN
dataBase["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN": "It is classified in another SISBEN Level"},
                           inplace=True)

# In PEOPLE_HOUSE replace “Nueve” into “Nine” and “Once” into “Eleven”
dataBase["PEOPLE_HOUSE"].replace({"Nueve": "Nine", "Once": "Eleven"}, inplace=True)

# Replace “No” and “O” to “0 hours per week”
dataBase["JOB"].replace({"No": "0 hours per week", "0": "0 hours per week"}, inplace=True)

####################################################################################
# Remove unnecessary features
dataBase.drop(columns=['COD_S11', 'COD_SPRO', 'SCHOOL_NAME', 'UNIVERSITY', 'Unnamed: 9'], inplace=True)

####################################################################################

# Data Splitting:
# Given data
X_raw_columns = set(['GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'OCC_FATHER', 'OCC_MOTHER',
       'STRATUM', 'SISBEN', 'PEOPLE_HOUSE', 'INTERNET', 'TV',
       'COMPUTER', 'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD', 'FRESH', 'PHONE',
       'MOBILE', 'REVENUE', 'JOB', 'SCHOOL_NAT', 'SCHOOL_TYPE', 'MAT_S11',
       'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11', 'ACADEMIC_PROGRAM', 'QR_PRO',
       'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO', 'FEP_PRO', 'G_SC',
       'PERCENTILE', '2ND_DECILE', 'QUARTILE', 'SEL', 'SEL_IHE',
       'Average_Score_Saber11', 'Performance_Saber11'])

specified_columns = {
    'GENDER': 'gender',
    'EDU_FATHER': 'edu_father',
    'EDU_MOTHER': 'edu_mother',
    'OCC_FATHER': 'occ_father',
    'OCC_MOTHER': 'occ_mother',
    'STRATUM': 'stratum',
    'SISBEN': 'sisben',
    'PEOPLE_HOUSE': 'people_house',
    'INTERNET': 'internet',
    'TV': 'tv',
    'COMPUTER': 'computer',
    'WASHING_MCH': 'washing_machine',
    'MIC_OVEN': 'microwave_oven',
    'CAR': 'car',
    'DVD': 'dvd',
    'FRESH': 'fresh_food',
    'PHONE': 'phone',
    'MOBILE': 'mobile',
    'REVENUE': 'family_revenue',
    'JOB': 'job',
    'SCHOOL_NAT': 'school_nat',
    'SCHOOL_TYPE': 'school_type',
    'MAT_S11': 'mat_s11',
    'CR_S11': 'cr_s11',
    'CC_S11': 'cc_s11',
    'BIO_S11': 'bio_s11',
    'ENG_S11': 'eng_s11',
    'ACADEMIC_PROGRAM': 'academic_program',
    'QR_PRO': 'qr_pro',
    'CR_PRO': 'cr_pro',
    'CC_PRO': 'cc_pro',
    'ENG_PRO': 'eng_pro',
    'WC_PRO': 'wc_pro',
    'FEP_PRO': 'fep_pro',
    'G_SC': 'g_sc',
    'PERCENTILE': 'percentile',
    '2ND_DECILE': 'second_decile', 
    'QUARTILE': 'quartile',
    'SEL': 'sel',
    'SEL_IHE': 'sel_ihe',
    'Average_Score_Saber11': 'average_score_saber11',
    'Performance_Saber11': 'performance_saber11'
}

# Get the set of variables in X_raw that are not in the specified_columns
missing_columns = X_raw_columns - set(specified_columns.keys())

print("Variables in X_raw but not in the specified dictionary:")
print(missing_columns)
