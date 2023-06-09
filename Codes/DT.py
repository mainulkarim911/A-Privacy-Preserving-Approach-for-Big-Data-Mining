import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("H:/Study/Dipu Vai/Breast Cancer/data.csv")
data



data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]


import matplotlib.pyplot as plt
hasta = data[data.diagnosis == 1]
normal = data[data.diagnosis == 0]
normal

y = data.diagnosis.values
x = data.drop(["diagnosis"],axis=1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)


print("Decision Tree")

from sklearn.tree import DecisionTreeClassifier
#making the instance
model=DecisionTreeClassifier()
#learning
model.fit(x_train,y_train)
#Prediction
prediction=model.predict(x_test)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print(classification_report(y_test,prediction))

print("RamdomForest")

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(x_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(x_test)
  
print(classification_report(y_test,y_pred))



# get the start time
st = time.time()

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