import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import flwr as fl


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
import shap

shap.initjs()

model_training_time = {}
model_predict_time = {}
model_auc = {}
model_precision = {}
model_recall = {}
model_f1 = {}

X_train_summary = shap.sample(X_train, 10)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import swish, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

model = Sequential()
# Add hidden layer with swish activation (we have negative values)
model.add(Dense(10, input_shape=[len(X_train.columns)], activation=swish))
model.add(Dropout(.5))
# Add output layer with softmax activation (for multi-class classification)
model.add(Dense(3, activation=softmax))
# Use Adam for optimization with categorical cross-entropy loss
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['accuracy'])



class MnisClient(fl.client.NumPyClient):
    def get_parameters(self,config):
        return model.get_weights()
    
    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=20)
        return model.get_weights(), len(X_train),{}
    
    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_train), {'accuracy': accuracy}

""" fl.client.start_numpy_client("[::]:8080", MnisClient()) """

fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MnisClient())




























