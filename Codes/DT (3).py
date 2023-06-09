import pandas as pd 
import pandas as pd
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# get the start time
st = time.time()


df = pd.read_csv("H:/Study/Dipu Vai/Palmer Archipelago/penguins_size.csv")


# Apply one-hot encoding to categorical features.
categorical_columns = [c for c in df.columns if df[c].dtype != 'float']
onehot_df = pd.get_dummies(df[categorical_columns])
df = df.drop(categorical_columns, axis=1)
df = df.join(onehot_df)

# Map other column types
numeric_columns = [c for c in df.columns if df[c].dtype == 'float']
onehot_columns = df.columns.difference(numeric_columns)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Standardise all numeric columns & fill missing values with mean.
for column in numeric_columns:
    na_rows = df[column].isna()
    df.loc[~na_rows, [column]] = scaler.fit_transform(df.loc[~na_rows, [column]])
    df[column] = df[column].fillna(0)


df = df.rename(columns={'sex_.': 'sex_MISSING'})


from sklearn.model_selection import train_test_split

y_columns = [c for c in df.columns if c.startswith('species')]
y = df[y_columns]
X = df.drop(y_columns, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from time import process_time_ns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score


model_training_time = {}
model_predict_time = {}
model_auc = {}
model_precision = {}
model_recall = {}
model_f1 = {}


print("Decision Treee")

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

start_time = process_time_ns()

dt = clf.fit(X_train, y_train)

model_training_time['decision_tree'] = process_time_ns() - start_time


from sklearn.metrics import accuracy_score


dsc_pred=dt.predict(X_test)


acc_dsc=accuracy_score(y_test,dsc_pred)
print("Accuracy: ", acc_dsc)

from sklearn.metrics import classification_report
print(classification_report(y_test,dsc_pred))



print("RamdomForest")

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)

acc_dsc=accuracy_score(y_test,y_pred)
print("Accuracy: ", acc_dsc)
print(classification_report(y_test,y_pred))




# main program
# find sum to first 1 million numbers
sum_x = 0
for i in range(1000000):
    sum_x += i

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')



























