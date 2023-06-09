import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

# get the start time
st = time.time()



dataset=pd.read_csv("H:/Study/Dipu Vai/Stroke Prediction Dataset/healthcare-dataset-stroke-data.csv")

df=dataset.copy()
df.tail()

df['bmi'].fillna(df['bmi'].mean(), inplace=True)


df['stroke']=df['stroke'].replace(to_replace=1,value="Yes")
df['stroke']=df['stroke'].replace(to_replace=0,value="No")


def remove_outliers(data):
    arr=[]
    #print(max(list(data)))
    q1=np.percentile(data,25)
    q3=np.percentile(data,75)
    iqr=q3-q1
    mi=q1-(1.5*iqr)
    ma=q3+(1.5*iqr)
    #print(mi,ma)
    for i in list(data):
        if i<mi:
            i=mi
            arr.append(i)
        elif i>ma:
            i=ma
            arr.append(i)
        else:
            arr.append(i)
    #print(max(arr))
    return arr


outlier=pd.DataFrame()

q1=np.percentile(df['bmi'],25)
q3=np.percentile(df['bmi'],75)
iqr=q3-q1
mi=q1-(1.5*iqr)
ma=q3+(1.5*iqr)



outlier=df[(df['bmi']<mi) | (df['bmi']>ma)]

df['bmi']=remove_outliers(df['bmi'])

smoked=df[df['stroke']=='Yes']

df = df.drop(index=[132,245]).reset_index()


q1=np.percentile(df['avg_glucose_level'],25)
q3=np.percentile(df['avg_glucose_level'],75)
iqr=q3-q1
mi=q1-(1.5*iqr)
ma=q3+(1.5*iqr)


df['avg_glucose_level']=remove_outliers(df['avg_glucose_level'])


df['gender']=df['gender'].replace(to_replace='Other',value='Male')
df['stroke']=df['stroke'].replace(to_replace='Yes',value=1)
df['stroke']=df['stroke'].replace(to_replace='No',value=0)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
df['ever_married']=le.fit_transform(df['ever_married'])
df['Residence_type']=le.fit_transform(df['Residence_type'])



df=df.drop(['id','index'],axis=1)


stroked=df[df['stroke']==1]
not_stroked=df[df['stroke']==0]


df['smoking_status']=df['smoking_status'].replace(to_replace='never smoked',value=-1)
df['smoking_status']=df['smoking_status'].replace(to_replace='Unknown',value=0)
df['smoking_status']=df['smoking_status'].replace(to_replace='formerly smoked',value=1)
df['smoking_status']=df['smoking_status'].replace(to_replace='smokes',value=2)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('encoder',OneHotEncoder(),[5])],remainder='passthrough')
df=ct.fit_transform(df)
df=pd.DataFrame(df)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


from imblearn.over_sampling import SMOTE
smt=SMOTE(random_state=0)


x_train_smote,y_train_smote=smt.fit_resample(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score


print("Decision Tree")

from sklearn.tree import DecisionTreeClassifier
dsc=DecisionTreeClassifier()
dsc.fit(x_train_smote,y_train_smote)

dsc_pred=dsc.predict(x_test)


acc_dsc=accuracy_score(y_test,dsc_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,dsc_pred))

print("RamdomForest")

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)  
  
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(x_train, y_train)
  
# performing predictions on the test dataset
y_pred = clf.predict(x_test)
  
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

























