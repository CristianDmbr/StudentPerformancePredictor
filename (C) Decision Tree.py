import numpy as np # Used for numerical functions
import pandas as pd # Used for data manipulation and analysis
from sklearn.tree import DecisionTreeClassifier,plot_tree # Used to build and visualise a decision tree
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score # Metrics
import matplotlib.pyplot as plt # Plots the charts
from sklearn.model_selection import train_test_split # Splits the data into test and train datasets
from sklearn.preprocessing import OneHotEncoder  # Encodes categorical features into numerical


# Load data
dataBase = pd.read_csv('Database/DatabaseTest.csv')
pd.set_option('display.max_columns', None)

############################################################################################################
# ADDING THE NEW COLUMNS

dataBase["Average_Score_SABER_PRO"] = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']].mean(axis=1)
dataBase["Pass_Fail_SABER_PRO"] = np.where(dataBase['Average_Score_SABER_PRO'] >= 60, 'Pass', 'Fail')

dataBase["Average_Score_SABER11"] = dataBase[['MAT_S11','CR_S11','CC_S11','BIO_S11','ENG_S11']].mean(axis=1)
dataBase["Pass_Fail_Saber11"] = np.where(dataBase['Average_Score_SABER11'] >= 60, 'Pass', 'Fail')

pass_fail_counts_SABER_PRO = dataBase['Pass_Fail_SABER_PRO'].value_counts()
pass_fail_counts_Saber11 = dataBase['Pass_Fail_Saber11'].value_counts()

############################################################################################################
# DATA SPLITTING

# Seperate the data into dependent and independent variables where
# X is the feature and Y is the target
X_raw = dataBase[['QR_PRO', 'CR_PRO', 'CC_PRO', 'ENG_PRO', 'WC_PRO']]
y_raw = dataBase["Pass_Fail_Saber11"]


#############################################################################################################
# The decision tree classifier expects a numericla input for X
# This requires OneHotEncoding 

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_raw)

#############################################################################################################
# Split the data into four different parts  
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_raw, test_size=0.3, random_state=10)

# Train the decision tree classifier
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=6, min_samples_leaf=12,
                                     class_weight=None, max_features=None, max_leaf_nodes=None,
                                     min_samples_split=2, min_weight_fraction_leaf=0.0, splitter="best")


clf_entropy.fit(X_train, y_train)

# Make predictions 
y_pred = clf_entropy.predict(X_test)

#############################################################################################################
# METRICS : 

# Accuracy
print("Accuracy : ",accuracy_score(y_test, y_pred) * 100)


#############################################################################################################
#Visualising
plt.figure(figsize=(15,10))
plot_tree(clf_entropy, filled=True, rounded=True, feature_names=X_raw.columns, class_names=np.unique(y_raw))
plt.title('Decision Tree Visualization')
plt.show()