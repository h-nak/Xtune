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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import shap
import sys
import os
import optuna
from datetime import datetime

'''
ウィンドウサイズチューニング部と検証部を分離

TODO:多用されている2つの学習部とgendatasetを独立
TODO:comments
X:Dataset_shaping cf)以前のコメントアウト部
X:shap to hypara,&max_window_size
'''

plt.style.use('ggplot') 
plt.rcParams['figure.figsize'] = [12, 9] 
plt.rcParams["font.size"] = 20


Tuning_Switch = False
Dataset_Option = 0    #0:nyc_taxi,1:peyton_manning_wiki_pv
Model_Option = 0
Grid_Truth = False
Grid_Switch = True
Optuna_Trials = 0 #when 0 Optuna_Switch :False

'''
Hyper Parameter Setting Zone
'''
num_layers = 2
input_layer_nodes = 128 
activation_func = "relu"
mid_nodes = 48
Learning_Rate = 0.003
train_patience = 40
train_epochs = 200
train_batch_size = 32
train_varidation_split = 0.2
optimizer_name = "Adam"
Shuffle  = True

#コマンドライン引数で使用するデータを決める
#Python3.7はcaseに相当する文が無い
if(len(sys.argv) != 1):
    arg = int(sys.argv[1])
    Dataset_Option = arg

if(Dataset_Option == 0):#when use nyc_taxi use here
    xtune_length = 500
    file_path = '../dataset/nyc_taxi.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.value.values 
    dataset = Origin_Dataset[ 2000 : 5000]
    dataset1=Origin_Dataset[ 5000 : 5000+xtune_length]
    dataset1 = np.reshape(dataset1, (-1, 1)) 
    dataset2=Origin_Dataset[ 6500 : 6500+xtune_length] 
    dataset2 = np.reshape(dataset2, (-1, 1)) 
    dataset3=Origin_Dataset[ 5500 : 5500+xtune_length]
    dataset3 = np.reshape(dataset3, (-1, 1))
    Long_Dataset = np.reshape(Origin_Dataset[5000:6500], (-1, 1))#

elif(Dataset_Option == 1):#when use peyton_manning use here
    file_path = '../dataset/peyton_manning_wiki_pv.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.y.values 
    dataset = Origin_Dataset[ 0 : 1200]

    dataset1=Origin_Dataset[ 1200 : 1700]
    dataset1 = np.reshape(dataset1, (-1, 1)) 

    dataset2=Origin_Dataset[ 1700 : 2200] 
    dataset2 = np.reshape(dataset2, (-1, 1)) 

    dataset3=Origin_Dataset[ 2200 : 2700]
    dataset3 = np.reshape(dataset3, (-1, 1))
    Long_Dataset = np.reshape(Origin_Dataset[1200:2700], (-1, 1))#

elif(Dataset_Option == 2):
    file_path = '../dataset/pageviews-NFL.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.Views.values 
    dataset = Origin_Dataset[ 1500 : 3000]

    dataset1=Origin_Dataset[ 0 : 500]
    dataset1 = np.reshape(dataset1, (-1, 1)) 

    dataset2=Origin_Dataset[ 500 : 1000] 
    dataset2 = np.reshape(dataset2, (-1, 1)) 

    dataset3=Origin_Dataset[ 1000 : 1500]
    dataset3 = np.reshape(dataset3, (-1, 1))
    Long_Dataset = np.reshape(Origin_Dataset[0 : 1500], (-1, 1))#or500-1500


#ここから本文
dataset = dataset.astype('float32')    
dataset = np.reshape(dataset, (-1, 1)) 
dataset_list = [dataset1, dataset2, dataset3]
#print(df)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LogText = f"-----execution_DateTime:{timestamp}--------\ndataset={Dataset_Option}\n"
LogText += f"Model_Option:{Model_Option}\nGrid_Truth:{Grid_Truth}\n"
if(Model_Option != 0):LogText += f"---Parameters---\nGrid_dataSize:{Long_Dataset.shape}\npatience = {train_patience}\nepochs = {train_epochs}\nbatch_size = {train_batch_size}\nvaridation_split = {train_varidation_split}\n"


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
    y_train = scaler_y.fit_transform(y_train)#スケーリングcf.)https://qiita.com/suzuki0430/items/59b411aefb66703bcd6b
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_0 = scaler_X.fit_transform(X_train_0)
    X_test_0 = scaler_X.transform(X_test_0)
    
    X_train = np.reshape(X_train_0, (X_train_0.shape[0],X_train_0.shape[1],1))
    X_test = np.reshape(X_test_0, (X_test_0.shape[0],X_test_0.shape[1],1))

    return X_train, X_test, y_train, y_test, scaler_y #scaler_yを追加


def train_model(X_train, y_train):#仮実装
    if(Model_Option == 1):
        return new_train_model(X_train,y_train)
    else:
        return origin_train_model(X_train,y_train)

def new_train_model(X_train, y_train):
    
    model = Sequential()

    model.add(LSTM(input_layer_nodes,input_shape=(X_train.shape[1], X_train.shape[2])))#入力層
    for i in range(num_layers):#中間層
        model.add(Dense(mid_nodes, activation = activation_func))
    model.add(Dense(1, activation='linear'))#出力層


    #plot_model(model,show_shapes=True)    
    if optimizer_name == "SGD":
        optimizer = SGD(learning_rate=Learning_Rate)
    elif optimizer_name == "Adam":
        optimizer = Adam(learning_rate=Learning_Rate)
    elif optimizer_name == "RMSprop":
        optimizer = RMSprop(learning_rate=Learning_Rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    early_stopping_conv = EarlyStopping(monitor='val_loss',patience = train_patience, verbose=0, mode='auto') 
    model.fit(X_train, y_train,
            epochs = train_epochs,
            verbose=0,#Log_Output default:2
            batch_size = train_batch_size,
            validation_split = train_varidation_split,
            callbacks=[early_stopping_conv],
            shuffle=Shuffle)#追加

    return model


def origin_train_model(X_train, y_train):
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


'''
notebookで利用されていたgen_dataset関数をこちらに持ってきた

'''
def tunes_gen_dataset(dataset, lag_max, test_length):
    X, y = [], []
    for i in range(len(dataset) - lag_max):
        a = i + lag_max
        X.append(dataset[i:a, 0]) 
        y.append(dataset[a, 0])   
    X, y = np.array(X), np.array(y)

    X_train, X_test = X[:-test_length,:], X[-test_length:,:] 
    y_train, y_test = y[:-test_length].reshape(-1,1), y[-test_length:].reshape(-1,1) #学習データ
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train)
    
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, X_test, y_train, y_test

"""
    Xtune:3回の試行すべてを出力することに変更
"""
def xtune(dataset_list):
    candidates_opt_window_size = []
    for dataset in dataset_list:
        X_train, X_test, y_train, y_test = tunes_gen_dataset(dataset, lag_max=120, test_length=1)
        model = train_model(X_train, y_train)
        
        explainer = shap.DeepExplainer(model=model,data=X_train)
        shap_values = explainer.shap_values(X=X_test)
        
        shap_value_list = np.abs(np.array(shap_values).reshape(-1,))
        shap_value_list_reverse = np.flipud(shap_value_list)
        cumsum = np.cumsum(shap_value_list_reverse)
        cumsum_minmax = cumsum/shap_value_list.sum()
        candidate_opt_window_size = np.where(cumsum_minmax>0.98)[0][0]#0.999
        #print(candidate_opt_window_size)
        candidates_opt_window_size.append(candidate_opt_window_size)
    #opt_window_size = sum(candidates_opt_window_size)/len(dataset_list)   
    return candidates_opt_window_size

    """
    最初に与えるデータの量で時間と精度が変化
    """



def xtuneBase(dataset_list):
    for lag_max in range(5,120,1):
        for i,dataset in enumerate(dataset_list):
            X_train, X_test, y_train, y_test = tunes_gen_dataset(dataset, lag_max=lag_max, test_length=1)#_無くても実行できた不思議
            model = train_model(X_train, y_train)
            
            explainer = shap.DeepExplainer(model=model,data=X_train)
            shap_values = explainer.shap_values(X=X_test)
            
            shap_value_list = np.abs(np.array(shap_values).reshape(-1,))
            shap_value_list_reverse = np.flipud(shap_value_list)
            cumsum = np.cumsum(shap_value_list_reverse)
            cumsum_minmax = cumsum/shap_value_list.sum()
            idx = np.where(cumsum_minmax>0.98)[0][0]
            if idx == lag_max-1:
                break
            if i==len(dataset_list)-1:
                opt_window_size = lag_max
                return opt_window_size

def GridSearch(_data4G):
    temp_best_size = 0
    best_rmse = np.inf
    if(Grid_Truth):df_Grid_RMSEs = pd.DataFrame(columns=['window_size','RMSE'])
    for window_size in range(10,100,1):
        X_train,X_test,y_train,y_test,scaler_y = gen_dataset(_data4G, window_size)
        model = train_model(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        _rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        if(Grid_Truth):
            #print(window_size,_rmse)
            df_Grid_RMSEs.loc[len(df_Grid_RMSEs)] = [window_size,_rmse]
        if(_rmse<best_rmse):
            temp_best_size=window_size
            best_rmse=_rmse
    if(Grid_Truth):return temp_best_size,df_Grid_RMSEs
    else:return temp_best_size
  
def optuner(trial):
    
    window_size = trial.suggest_int("window_size",10,150,step = 10)
    
    X_train,X_test,y_train,y_test,scaler_y=gen_dataset(dataset, window_size)

    model = train_model(X_train, y_train)
    y_test_pred = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    return rmse


LogText +=f"---------------------Tuning Phase-----------------\n"

if(Tuning_Switch):
    
    
    print("entering:xtune\n")
    Xtune_StartTime = time.perf_counter()
    Xtune_WindowSizeList = xtune(dataset_list)
    Xtune_EndTime = time.perf_counter()
    
    LogText +=f"xtune:"
    for XwinSize in Xtune_WindowSizeList:
        LogText +=f"    \n  optimal_Window_size:{XwinSize}\n"
    LogText +=f"  Processing Time:{Xtune_EndTime-Xtune_StartTime}\n"

    print("entering:xtuneBase\n")
    XtuneBase_StartTime = time.perf_counter()
    XtuneBase_WindowSize = xtuneBase(dataset_list)
    XtuneBase_EndTime = time.perf_counter()
    LogText +=f"xtuneBase:\n  optimal_Window_size:{XtuneBase_WindowSize}\n  Processing Time:{XtuneBase_EndTime-XtuneBase_StartTime}\n"

if(Grid_Truth | Grid_Switch):
    print("entering:GridSearch\n")
    
    Grid_StartTime = time.perf_counter()
    if(Grid_Truth):Grid_WindowSize,df_Grid = GridSearch(dataset)
    else: Grid_WindowSize = GridSearch(Long_Dataset)
    Grid_EndTime = time.perf_counter()
    LogText +=f"Grid Search:\n   optimal_window_size={Grid_WindowSize}\n  Processing Time:{Grid_EndTime-Grid_StartTime}\n"

if(Optuna_Trials > 0):
    print("entering:optuner\n")
    Opt_StartTime = time.perf_counter()
    study = optuna.create_study()
    study.optimize(optuner, n_trials=Optuna_Trials)
    Opt_EndTime = time.perf_counter()
    LogText +=f"optuna trials:{Optuna_Trials}\n   {study.best_params}\n   Processing Time:{Opt_EndTime-Opt_StartTime}"
    

if(Dataset_Option == 0):
    Output_FilePath = f"../_out/_Window_tuner/nyc_taxi/"
elif(Dataset_Option == 1):
    Output_FilePath = f"../_out/_Window_tuner/peyton_manning_wiki_pv/"
elif(Dataset_Option == 2):
    Output_FilePath = f"../_out/_Window_tuner/National_Football_League_wiki_pv/"
else:
    Output_FilePath = f"../_out/_Window_tuner/{timestamp}"
    os.mkdir(Ou)



if(Grid_Truth):
    df_Grid.plot(x='window_size')
    plt.savefig(f"{Output_FilePath}_Grid_RMSE_{timestamp}.pdf")
    plt.close()

#LogText+="! Xtune windowsize is 100,not 100\n"
f=open(f"{Output_FilePath}/{timestamp}_ReportLog.txt",'w')
f.write(LogText)
f.close()
