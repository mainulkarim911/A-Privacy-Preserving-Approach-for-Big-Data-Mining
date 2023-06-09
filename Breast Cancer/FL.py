import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import flwr as fl
from keras.layers import Dense, Dropout, Flatten


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



from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=256,activation="sigmoid",input_dim=x.shape[1]))
model.add(Dense(units=512,activation="sigmoid"))
model.add(Dropout(0.3))
model.add(Dense(units=1,activation="sigmoid"))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


class MnisClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        shallow_history = model.fit(x_train,y_train,epochs=35)
        return model.get_weights(), len(x_train),{}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_train), {'accuracy': accuracy}

""" fl.client.start_numpy_client("[::]:8080", MnisClient()) """

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnisClient())

































