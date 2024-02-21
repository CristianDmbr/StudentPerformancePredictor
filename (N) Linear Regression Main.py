import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer 

dataBase = pd.read_csv('Database/DatabaseTest.csv')
pd.set_option('display.max_columns', None)

############################################################################################################
# Adding new columns : 

# Calculate the average of SABER PRO'S 'QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO' metrics 
dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)
dataBase["Pass_Fail_SABER_PRO"] = np.where(dataBase['Average_Score_SABER_PRO'] >= 60, 'Pass', 'Fail')

# Calculate the average of Saber 11 competences
dataBase["Average_Score_Saber11"] = dataBase[['MAT_S11','CR_S11','CC_S11','BIO_S11','ENG_S11']].mean(axis=1)
dataBase["Pass_Fail_Saber11"] = np.where(dataBase['Average_Score_Saber11'] >= 60, 'Pass', 'Fail')

pass_fail_counts_SABER_PRO = dataBase['Pass_Fail_SABER_PRO'].value_counts()
pass_fail_counts_Saber11 = dataBase['Pass_Fail_Saber11'].value_counts()

############################################################################################################
# Data Splitting : 

X_raw = dataBase["Average_Score_SABER_PRO"]
y_raw = dataBase["Average_Score_Saber11"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.20, shuffle=True, random_state=0)

############################################################################################################
# Pre processing : 

X_train_num = X_train_raw.to_frame().select_dtypes(include=np.number)
numeric_imputer = SimpleImputer(strategy='mean')
numeric_imputer.fit(X_train_num)
X_train_num_imp = numeric_imputer.transform(X_train_num)
X_test_num = X_test_raw.to_frame().select_dtypes(include=np.number)
X_test_num_imp = numeric_imputer.transform(X_test_num)


############################################################################################################
# Feature Scaling : 

scaler = MinMaxScaler()
scaler.fit(X_train_num_imp)
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)

X_train = X_train_num_sca
X_test = X_test_num_sca

############################################################################################################
# Linear regression : 

obj = sklearn.linear_model.LinearRegression()
obj.fit(X_train, y_train)
y_pred_train = obj.predict(X_train)
y_pred = obj.predict(X_test)

X_disp = X_test[:,0]

# Plot outputs
plt.scatter(X_disp, y_test,  color='black', label='y_test') # Observed y values
plt.scatter(X_disp, y_pred, color='blue', label='y_pred') # predicted y values
plt.xlabel('Feature')
plt.ylabel('Final Grade')
plt.legend()
plt.show()

############################################################################################################
# METRICS : 

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error

n = len(X_train)  # Number of samples in training set
p = X_train.shape[1]  # Number of features
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Adjusted R-squared

# Printing metrics
print('Mean Squared Error (MSE):', mse)
print('R-squared score:', r2)
print('Mean Absolute Error (MAE):', mae)
print('Adjusted R-squared:', adj_r2)
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))