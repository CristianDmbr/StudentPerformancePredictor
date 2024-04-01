import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV

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
dataBase["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN": "It is classified in another SISBEN Level"}, inplace=True)

# In PEOPLE_HOUSE replace “Nueve” into “Nine” and “Once” into “Eleven”
dataBase["PEOPLE_HOUSE"].replace({"Nueve": "Nine", "Once": "Eleven"}, inplace=True)

# Replace “No” and “O” to “0 hours per week”
dataBase["JOB"].replace({"No": "0 hours per week", "0": "0 hours per week"}, inplace=True)

####################################################################################
# Remove unnecessary features
dataBase.drop(columns=['COD_S11', 'COD_SPRO', 'SCHOOL_NAME', 'UNIVERSITY', 'Unnamed: 9'], inplace=True)

####################################################################################

# Data Splitting:
X_raw = dataBase[['GENDER', 'EDU_FATHER', 'EDU_MOTHER', 'OCC_FATHER', 'OCC_MOTHER',
                  'STRATUM', 'SISBEN', 'PEOPLE_HOUSE', 'INTERNET', 'TV',
                  'COMPUTER', 'WASHING_MCH', 'MIC_OVEN', 'CAR', 'DVD', 'FRESH', 'PHONE',
                  'MOBILE', 'REVENUE', 'JOB', 'SCHOOL_NAT', 'SCHOOL_TYPE', 'MAT_S11',
                  'CR_S11', 'CC_S11', 'BIO_S11', 'ENG_S11', 'ACADEMIC_PROGRAM', 'QR_PRO',
                  'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO', 'FEP_PRO', 'G_SC',
                  'PERCENTILE', '2ND_DECILE', 'QUARTILE', 'SEL', 'SEL_IHE',
                  'Average_Score_SABER_PRO', 'Average_Score_Saber11', 'Performance_Saber11']]

y_raw = dataBase["Performance_SABER_PRO"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

# Identify columns with missing values
missing_columns = X_train_raw.columns[X_train_raw.isnull().any()]

# Separate numerical and categorical features
numerical_features = X_raw.select_dtypes(include=np.number).columns
categorical_features = X_raw.select_dtypes(exclude=np.number).columns

# Handle missing values for numerical features
if not X_train_raw[numerical_features].empty:
    numeric_imputer = SimpleImputer(strategy="mean")
    X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features])
    X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features])
else:
    print("No numerical features to impute.")

# Handle missing values for categorical features
if not X_train_raw[categorical_features].empty:    
    categoric_imputer = SimpleImputer(strategy="most_frequent")
    X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features])
    X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features])
else:
    print("No categorical features to impute.")

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')  # Ignores unknown categories
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed)
# For test data, use the categories learned from the training data
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed)

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded.toarray()), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded.toarray()), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

####################################################################################
# Hyperparameter Tuning for Nearest Centroid Classifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Nearest Centroid
param_grid_nc = {
    'metric': ['euclidean', 'manhattan'],
    'shrink_threshold': [None, 0.1, 0.5, 1.0]
}

# Instantiate the grid search for Nearest Centroid
grid_search_nc = GridSearchCV(NearestCentroid(), param_grid_nc, cv=5, scoring='accuracy')

# Perform the grid search for Nearest Centroid
grid_search_nc.fit(X_train_scaled, y_train)

# Print the best hyperparameters for Nearest Centroid
print("Best hyperparameters found for Nearest Centroid:")
print(grid_search_nc.best_params_)
print()

# Print out the results of each hyperparameter for Nearest Centroid
print("Grid search results for Nearest Centroid:")
results_nc = grid_search_nc.cv_results_
for mean_score, params in zip(results_nc['mean_test_score'], results_nc['params']):
    print(f"Mean accuracy: {mean_score:.3f} with parameters: {params}")

# Train Nearest Centroid classifier with best hyperparameters
best_clf_nc = grid_search_nc.best_estimator_
best_clf_nc.fit(X_train_scaled, y_train)

# Prediction variables for Nearest Centroid with best hyperparameters
y_pred_train_nc_best = best_clf_nc.predict(X_train_scaled)
y_pred_test_nc_best = best_clf_nc.predict(X_test_scaled)

# Metrics for Nearest Centroid with best hyperparameters
precision_train_nc_best, recall_train_nc_best, f1_score_train_nc_best, _ = precision_recall_fscore_support(y_train, y_pred_train_nc_best, average='weighted')
precision_test_nc_best, recall_test_nc_best, f1_score_test_nc_best, _ = precision_recall_fscore_support(y_test, y_pred_test_nc_best, average='weighted')

print("\nTraining Metrics for Nearest Centroid with best hyperparameters:")
print("Accuracy:", accuracy_score(y_train, y_pred_train_nc_best))
print("Classification Report:\n", classification_report(y_train, y_pred_train_nc_best))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train_nc_best))
print("Precision :", precision_train_nc_best)
print("Recall :", recall_train_nc_best)
print("F1-Score :", f1_score_train_nc_best)

print("\nTesting Metrics for Nearest Centroid with best hyperparameters:")
print("Accuracy:", accuracy_score(y_test, y_pred_test_nc_best))
print("Classification Report:\n", classification_report(y_test, y_pred_test_nc_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_nc_best))
print("Precision :", precision_test_nc_best)
print("Recall :", recall_test_nc_best)
print("F1-Score :", f1_score_test_nc_best)