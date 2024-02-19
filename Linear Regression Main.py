import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

X = dataBase["Average_Score_SABER_PRO"]
y = dataBase["Average_Score_Saber11"]
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=0)

############################################################################################################
# Encoding categorical features : 

# Separate numerical columns 
numeric_columns = dataBase.select_dtypes(include=[np.number])
# Separate categorical columns
categorical_columns = dataBase.select_dtypes(include=['object'])

# Create an instance of OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical columns
encoded_categorical = encoder.fit_transform(categorical_columns)

# Concatenate numeric and one-hot encoded categorical features
combined_features = pd.concat([numeric_columns, pd.DataFrame(encoded_categorical.toarray())], axis=1)

# Calculate correlation matrix
correlation_matrix = combined_features.corr()

############################################################################################################
# Feature Scaling  :

scaler = MinMaxScaler()
scaler.fit(X_train_raw) 

X_train_num_sca = scaler.transform(X_train_raw)
X_test_num_sca = scaler.transform(X_test_raw)

############################################################################################################
# Creata Linear Regression object :
obj = sklearn.linear_model.LinearRegression()

# Train the model using the training sets
obj.fit(X_train_raw, y_train)

# We can make a prediction with the training data
y_pred_train = obj.predict(X_train_raw)
# Remember the predictions with the new data give a better indiction of the true model performance.
# Make predictions using the testing set
y_pred = obj.predict(X_test_raw)

# I decided that for visualisation i wanted to use mock1.
X_disp = X_test_raw[:,0] # We have to choose a single column of the feature matrix so we can plot a 2D scatter plot.

# Plot outputs
plt.scatter(X_disp, y_test,  color='black', label='y_test') # Observed y values
plt.scatter(X_disp, y_pred, color='blue', label='y_pred') # predicted y values
plt.xlabel('Independent')
plt.ylabel('Dependend')
plt.legend()
plt.show()

# The mean squared error loss and R2 for the test and train data
print('Train - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_train, y_pred_train),sklearn.metrics.r2_score(y_train, y_pred_train)))
print('Test - MSE: {:.4f} R2 score: {:.4f}'.format(sklearn.metrics.mean_squared_error(y_test, y_pred),sklearn.metrics.r2_score(y_test, y_pred)))