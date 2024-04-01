import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

dataBase = pd.read_csv('Database/Academic Database.csv')
pd.set_option('display.max_columns', None)

# Adding new columns:
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

# Replace values as required
dataBase["EDU_FATHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["EDU_MOTHER"].replace({"Not sure": "None", "0": "None", "Ninguno": "None"}, inplace=True)
dataBase["OCC_FATHER"].replace({"0": "Unemployed", "Home": "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["OCC_MOTHER"].replace({"0": "Unemployed", "Home": "Unemployed", "Retired": "Unemployed"}, inplace=True)
dataBase["SISBEN"].replace({"Esta clasificada en otro Level del SISBEN": "It is classified in another SISBEN Level"}, inplace=True)
dataBase["PEOPLE_HOUSE"].replace({"Nueve": "Nine", "Once": "Eleven"}, inplace=True)
dataBase["JOB"].replace({"No": "0 hours per week", "0": "0 hours per week"}, inplace=True)

# Remove unnecessary features
dataBase.drop(columns=['COD_S11', 'COD_SPRO', 'SCHOOL_NAME', 'UNIVERSITY', 'Unnamed: 9'], inplace=True)

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
numerical_features = X_raw.select_dtypes(include=np.number).columns
categorical_features = X_raw.select_dtypes(exclude=np.number).columns

# Handle missing values for numerical features
numeric_imputer = SimpleImputer(strategy="mean")
X_train_numerical_imputed = numeric_imputer.fit_transform(X_train_raw[numerical_features])
X_test_numerical_imputed = numeric_imputer.transform(X_test_raw[numerical_features])

# Handle missing values for categorical features
categoric_imputer = SimpleImputer(strategy="most_frequent")
X_train_categorical_imputed = categoric_imputer.fit_transform(X_train_raw[categorical_features])
X_test_categorical_imputed = categoric_imputer.transform(X_test_raw[categorical_features])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore')  # Ignores unknown categories
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical_imputed)
X_test_categorical_encoded = encoder.transform(X_test_categorical_imputed)

# Concatenate numerical and encoded categorical features
X_train_encoded = np.concatenate((X_train_numerical_imputed, X_train_categorical_encoded.toarray()), axis=1)
X_test_encoded = np.concatenate((X_test_numerical_imputed, X_test_categorical_encoded.toarray()), axis=1)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

####################################################################################
# Hyperparameter Tuning for Gradient Boosting Classifier

# Define the parameter grid for Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# Instantiate the grid search for Gradient Boosting
grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=0), param_grid_gb, cv=5, scoring='accuracy')

# Perform the grid search for Gradient Boosting
grid_search_gb.fit(X_train_scaled, y_train)

# Print the best hyperparameters for Gradient Boosting
print("Best hyperparameters found for Gradient Boosting:")
print(grid_search_gb.best_params_)
print()

# Print out the results of each hyperparameter for Gradient Boosting
print("Grid search results for Gradient Boosting:")
results_gb = grid_search_gb.cv_results_
for mean_score, params in zip(results_gb['mean_test_score'], results_gb['params']):
    print(f"Mean accuracy: {mean_score:.3f} with parameters: {params}")

# Train Gradient Boosting classifier with best hyperparameters
best_clf_gb = grid_search_gb.best_estimator_
best_clf_gb.fit(X_train_scaled, y_train)

# Prediction variables for Gradient Boosting with best hyperparameters
y_pred_train_gb_best = best_clf_gb.predict(X_train_scaled)
y_pred_test_gb_best = best_clf_gb.predict(X_test_scaled)

# Metrics for Gradient Boosting with best hyperparameters
precision_train_gb_best, recall_train_gb_best, f1_score_train_gb_best, _ = precision_recall_fscore_support(y_train, y_pred_train_gb_best, average='weighted')
precision_test_gb_best, recall_test_gb_best, f1_score_test_gb_best, _ = precision_recall_fscore_support(y_test, y_pred_test_gb_best, average='weighted')

print("\nTraining Metrics for Gradient Boosting with best hyperparameters:")
print("Accuracy:", accuracy_score(y_train, y_pred_train_gb_best))
print("Classification Report:\n", classification_report(y_train, y_pred_train_gb_best))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train_gb_best))
print("Precision :", precision_train_gb_best)
print("Recall :", recall_train_gb_best)
print("F1-Score :", f1_score_train_gb_best)

print("\nTesting Metrics for Gradient Boosting with best hyperparameters:")
print("Accuracy:", accuracy_score(y_test, y_pred_test_gb_best))
print("Classification Report:\n", classification_report(y_test, y_pred_test_gb_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test_gb_best))
print("Precision :", precision_test_gb_best)
print("Recall :", recall_test_gb_best)
print("F1-Score :", f1_score_test_gb_best)