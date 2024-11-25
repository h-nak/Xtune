import numpy as np
import pandas as pd
import tensorflow as tf   
import time 
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam,SGD,RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap
import sys
import os
import optuna
from datetime import datetime



Dataset_Option = 1



'''
HyperParameterTuning
OPPO:history = model.fit(train_x, train_y, verbose=0, epochs=5, batch_size=128, validation_split=0.1).history
'''

num_layers = 2
input_layer_nodes = 44 
activation_func = "tanh"
mid_nodes = 32 
Learning_Rate = 0.005
train_patience = 46
train_epochs = 160
train_batch_size = 64
train_varidation_split = 0.2
optimizer_name = "Adam"
Shuffle  = True




def objective(trial):
    
    window_size = trial.suggest_int("window_size",10,150)
    
    X_train,X_test,y_train,y_test,scaler_y=gen_dataset(dataset, window_size)

    model = train_model(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return rmse

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")




if(len(sys.argv) != 1):
    arg = int(sys.argv[1])
    Dataset_Option = arg


if(Dataset_Option == 0):#when use nyc_taxi use here
    Xtune_LearningRange = 288
    file_path = '../dataset/nyc_taxi.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.value.values 
    dataset = Origin_Dataset[5000:6500]


elif(Dataset_Option == 1):#when use peyton_manning use here
    Xtune_LearningRange = 250
    file_path = '../dataset/peyton_manning_wiki_pv.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.y.values 
    dataset = Origin_Dataset[1200:2700]#new changed 


dataset = dataset.astype('float32')    
dataset = np.reshape(dataset, (-1, 1)) 


def gen_dataset(dataset, lag_max):
    test_length = int(dataset.size*0.8)
    X, y = [], []
    for i in range(len(dataset) - lag_max):
        a = i + lag_max
        X.append(dataset[i:a, 0]) 
        y.append(dataset[a, 0])   
    X, y = np.array(X), np.array(y)
    X_train_0 = X[:-test_length,:] 
    X_test_0 = X[-test_length:,:]  
    y_train_0 = y[:-test_length] 
    y_test_0 = y[-test_length:]  
    y_train = y_train_0.reshape(-1,1)
    y_test = y_test_0.reshape(-1,1)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))#0から1に正規化
    y_train = scaler_y.fit_transform(y_train)
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_0 = scaler_X.fit_transform(X_train_0)
    X_test_0 = scaler_X.transform(X_test_0)
    
    X_train = np.reshape(X_train_0, (X_train_0.shape[0],X_train_0.shape[1],1))
    X_test = np.reshape(X_test_0, (X_test_0.shape[0],X_test_0.shape[1],1))

    return X_train, X_test, y_train, y_test, scaler_y #scaler_yを追加


def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(10,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    plot_model(model,show_shapes=True)    
    early_stopping_conv = EarlyStopping(monitor='val_loss',patience=5, verbose=0, mode='auto') 
    model.fit(X_train, y_train,
            epochs=100,
            verbose=0,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping_conv],
            shuffle=False)#追加

    return model


n=0  
LogText = f"-----optuna execution_DateTime:{timestamp}---Dataset:{Dataset_Option}-----\n-----!Data_range_Fixed!--------\n"

search_space = {"window_size": [i for i in range(10,100,1)]}

for Optuna_Trials in (10,30,50):
    for algorithm in ("Default(TPE)","Random_Search","Grid_Search"):


        if(algorithm == "Random_Search"):
            Opt_StartTime = time.perf_counter()
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
            study.optimize(objective, n_trials = Optuna_Trials)
            Opt_EndTime = time.perf_counter()
        elif(algorithm == "Grid_Search"):
            if(Optuna_Trials == 50):
                Opt_StartTime = time.perf_counter()
                study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
                Optuna_Trials =100
                study.optimize(objective)
                Opt_EndTime = time.perf_counter()
            else:continue
        else:
            Opt_StartTime = time.perf_counter()
            study = optuna.create_study()   
            study.optimize(objective, n_trials = Optuna_Trials)
            Opt_EndTime = time.perf_counter()

        LogText +=f"#study:{n}\n  algorithm:{algorithm},  number_of_attempts:{Optuna_Trials}\n   {study.best_params}    (score:{study.best_value})\n   Processing Time:{Opt_EndTime-Opt_StartTime}\n"
        n+=1


with open("./ReportLog.txt","a") as f:
    f.write(LogText)
    f.close()