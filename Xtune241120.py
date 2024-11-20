import numpy as np
import pandas as pd
import tensorflow as tf    
import sys
import os
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


def gen_dataset(dataset, lag_max, test_length):
    X, y = [], []
    for i in range(len(dataset) - lag_max):
        a = i + lag_max
        X.append(dataset[i:a, 0]) 
        y.append(dataset[a, 0])       X, y = np.array(X), np.array(y)

    X_train, X_test = X[:-test_length,:], X[-test_length:,:] 
    y_train, y_test = y[:-test_length].reshape(-1,1), y[-test_length:].reshape(-1,1) 
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(10,input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    early_stopping_conv = EarlyStopping(monitor='val_loss',patience=5, verbose=1, mode='auto') 
    model.fit(X_train, y_train,
            epochs=100,
            verbose=2,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping_conv])
    return model

def xtune(dataset_list):
    candidates_opt_window_size = []
    for dataset in dataset_list:
        X_train, X_test, y_train, y_test = gen_dataset(dataset, lag_max=120, test_length=1)
        model = train_model(X_train, y_train)
        
        explainer = shap.DeepExplainer(model=model,data=X_train)
        shap_values = explainer.shap_values(X=X_test)
        
        shap_value_list = np.abs(np.array(shap_values).reshape(-1,))
        shap_value_list_reverse = np.flipud(shap_value_list)
        cumsum = np.cumsum(shap_value_list_reverse)
        cumsum_minmax = cumsum/shap_value_list.sum()
        candidate_opt_window_size = np.where(cumsum_minmax>0.98)[0][0]
        print(candidate_opt_window_size)
        candidates_opt_window_size.append(candidate_opt_window_size)
    opt_window_size = sum(candidates_opt_window_size)/len(dataset_list)   
    return opt_window_size