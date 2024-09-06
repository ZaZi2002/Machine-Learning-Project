import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import joblib

## Reading test data
data_test_raw = pd.read_csv('test_data_2.csv')

## Preprocessing
# Removing outliers
number_cols = data_test_raw.select_dtypes(include=['int64']).columns
data_prep_test = data_test_raw
for i in number_cols:
    mean_col = np.mean(data_test_raw[i])
    std_col = np.std(data_test_raw[i])
    l_bound = mean_col - 2*std_col
    h_bound = mean_col + 2*std_col
    data_prep_test = data_test_raw[(data_test_raw[i] >= l_bound) & (data_test_raw[i] <= h_bound)]
    
# Removing NANs
data_prep_test = data_prep_test.dropna()

# Changing object columns to numerical
data_prep_test['MF'] = data_prep_test['MF'].map({'M':1 ,'F':0})
data_prep_test['LoE'] = data_prep_test['LoE'].map({'Dip':0 ,'Ad. Dip':1 ,'Bach':2 ,'Mst':3 ,'Doct':4 ,'P. Doct':5})
data_prep_test['Housing'] = data_prep_test['Housing'].map({'O':2 ,'R':1 ,'N':0})
data_prep_test['Car'] = data_prep_test['Car'].map({True:1 ,False:0})
data_prep_test['Res'] = data_prep_test['Res'].map({'Accept':1 ,'Reject':0})

## Seperating data from label
data = data_prep_test.drop(columns='Res')
labels = data_prep_test['Res']

#Normalization (z-score)
scaler = MinMaxScaler()
data_test = scaler.fit_transform(data)

## Importing training model
model = joblib.load('Best_Model.pkl')

## Predicting
Y_pred_test = model.predict(data)
        
## Calculating AUPRC
auprc = average_precision_score(Y_pred_test,labels)
print('AUPRC ->',auprc*100)