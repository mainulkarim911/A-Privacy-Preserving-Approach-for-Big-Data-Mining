import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import RSquare 



columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('H:/Study/Dipu Vai/Boston House Prices/housing.csv', header=None, delimiter=r"\s+", names=columns)
df.head()

print("The shape of the dataset is:",df.shape)




X = df.drop(columns=['MEDV']).values
y = df.MEDV.values
x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1,1))



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)



n_features = X.shape[1]
dropout_proba = 0.25
batch_size = 64
opt = SGD(learning_rate=0.01)
epochs = 10000



def DNN_model_with_dropout(layers_shape, input_dim, dropout_proba, act='relu'):
    inputs = Input(shape=(input_dim,))
    hidden = Dense(layers_shape[0], activation=act)(inputs)
    for i in range(len(layers_shape)-1):
        if dropout_proba > 0:
            hidden = Dropout(dropout_proba)(hidden, training=True)
        hidden = Dense(layers_shape[i+1], activation=act)(hidden)
    if dropout_proba > 0:
        hidden = Dropout(dropout_proba)(hidden, training=True)
    outputs = Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear')(hidden) 
    model = Model(inputs, outputs)
    return model

model = DNN_model_with_dropout([64,32,16],n_features, dropout_proba, act='relu')
model.compile(optimizer = opt,loss="mse",metrics=[RootMeanSquaredError()])
print(model.summary)
history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=0)


def DNN_uncertainity(model, x_test,T=500):
    predictions = []
    for _ in range(T):
        predictions += [model.predict(x_test,verbose=0)]    
        #return predictions
    mean, std = np.mean(np.array(predictions), axis=0), np.std(np.array(predictions), axis=0)
    return mean,std




y_pred, sigma = DNN_uncertainity(model, X_test,T=500)
print("Model score:",r2_score(y_test,y_pred))