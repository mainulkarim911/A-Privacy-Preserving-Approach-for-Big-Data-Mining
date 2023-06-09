import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# get the start time
st = time.time()




df = pd.read_csv('H:/Study/Dipu Vai/Titanic/train_and_test2.csv')


# drop the columns we don't use
df.drop(["zero","zero.1","zero.2","zero.3","zero.4","zero.5","zero.6","zero.7","zero.8","zero.9",
        "zero.10","zero.11","zero.12","zero.13","zero.14","zero.15","zero.16","zero.17","zero.18"], axis=1,inplace=True)


# clean data
df = df.dropna()

# built features and label
y = df["2urvived"].values
X = df.drop(['2urvived'], axis = 1)

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)


# feature scaling
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Decision Tree")

from sklearn.tree import DecisionTreeClassifier # Decision Tree
dt = DecisionTreeClassifier() 
dt.fit(X_train, y_train) 
y_pred = dt.predict(X_test) 
dt_acc = dt.score(X_test,y_test)*100 

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



print("RamdomForest")

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(X_test)
  
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




















































