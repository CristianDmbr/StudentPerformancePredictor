import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
dataBase = pd.read_csv('Database/DatabaseTest.csv')
pd.set_option('display.max_columns', None)

# Adding new columns
dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)
dataBase["Pass_Fail_SABER_PRO"] = np.where(dataBase['Average_Score_SABER_PRO'] >= 60, 'Pass', 'Fail')

dataBase["Average_Score_Saber11"] = dataBase[['MAT_S11','CR_S11','CC_S11','BIO_S11','ENG_S11']].mean(axis=1)
dataBase["Pass_Fail_Saber11"] = np.where(dataBase['Average_Score_Saber11'] >= 60, 'Pass', 'Fail')

pass_fail_counts_SABER_PRO = dataBase['Pass_Fail_SABER_PRO'].value_counts()
pass_fail_counts_Saber11 = dataBase['Pass_Fail_Saber11'].value_counts()

# Data Splitting
X_raw = dataBase[["Pass_Fail_SABER_PRO"]]
y_raw = dataBase["Pass_Fail_Saber11"]

# Encode target variable using label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_encoded, test_size=0.20, shuffle=True, random_state=0)

# Preprocessing
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train_raw)
X_test_encoded = encoder.transform(X_test_raw)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded.toarray())
X_test_scaled = scaler.transform(X_test_encoded.toarray())

# Linear Regression
obj = sklearn.linear_model.LinearRegression()
obj.fit(X_train_scaled, y_train)

y_pred_train = obj.predict(X_train_scaled)
y_pred = obj.predict(X_test_scaled)

# Plot outputs
plt.scatter(X_test_raw.index, y_test, color='black', label='y_test') # Observed y values
plt.scatter(X_test_raw.index, y_pred, color='blue', label='y_pred') # Predicted y values
plt.xlabel('Index')
plt.ylabel('Final Grade')
plt.legend()
plt.show()

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error

n = len(X_train_scaled)  # Number of samples in training set
p = X_train_scaled.shape[1]  # Number of features
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Adjusted R-squared

# Printing metrics
print('Mean Squared Error (MSE):', mse)
print('R-squared score:', r2)
print('Mean Absolute Error (MAE):', mae)
print('Adjusted R-squared:', adj_r2)
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_train, y_pred_train), r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))
