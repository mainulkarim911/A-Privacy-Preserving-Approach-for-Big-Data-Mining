import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt




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

# use deep learning to predict
from tensorflow import keras 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 


model  = Sequential() # ANN(Artifical Neural Network)
model.add(Dense(units=8, input_dim=8, activation = 'relu')) 
model.add(Dense(units=16, activation = 'relu')) 
model.add(Dense(units=24, activation = 'relu')) 
model.add(Dense(units=1, activation = 'sigmoid')) 


model.compile(optimizer = 'adam',           
            loss = 'binary_crossentropy',   
            metrics = ['acc'])




""" model = tf.keras.Sequential([
    tf.keras.Input((24,)),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=1, activation='softmax'),
]) """

""" model = Sequential()
# add first hidden layer with input diamension
model.add(Dense(units = 32, activation='relu', kernel_initializer = 'he_uniform', input_dim = 24))
# add second hidden layer
model.add(Dense(units = 16, activation='relu', kernel_initializer = 'he_uniform'))
# add output layer
model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))


# now we compile the model
model.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy']) """
# train the model
#model.fit(X_train, y_train, batch_size = 128, epochs = 50, verbose = 1)

#model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


class MnisClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, 
                  epochs=30,        
                  batch_size=64,    
                  validation_data=(X_test, y_test)) 
        return model.get_weights(), len(X_train),{}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_train), {'accuracy': accuracy}

""" fl.client.start_numpy_client("[::]:8080", MnisClient()) """

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnisClient())













































