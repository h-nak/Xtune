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
from datetime import datetime

'''
ウィンドウサイズチューニング部と検証部を分離予定

TODO:多用されている2つの学習部とgendatasetを独立
TODO:comments
X:Dataset_shaping cf)以前のコメントアウト部
X:shap to hypara,&max_window_size
'''

plt.style.use('ggplot') 
plt.rcParams['figure.figsize'] = [12, 9] 
plt.rcParams["font.size"] = 20

Dataset_Option = 1    #0:nyc_taxi,1:peyton_manning_wiki_pv
ShapeTracker_Switch = False
mean_loop = 20



if(len(sys.argv) != 1):
    arg = int(sys.argv[1])
    Dataset_Option = arg

if(Dataset_Option == 0):#when use nyc_taxi use here
    file_path = '../dataset/nyc_taxi.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.value.values 
    dataset = Origin_Dataset[ 2000 : 5000]

elif(Dataset_Option == 1):#when use peyton_manning use here
    file_path = '../dataset/peyton_manning_wiki_pv.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.y.values 
    dataset = Origin_Dataset[ 0 : 1200]

elif(Dataset_Option == 2):
    file_path = '../dataset/pageviews-NFL.csv'
    df = pd.read_csv(file_path)
    Origin_Dataset = df.Views.values
    dataset = Origin_Dataset[ 1500 : 3000]

dataset = dataset.astype('float32')    
dataset = np.reshape(dataset, (-1, 1)) 


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


LogText = f"-----execution_DateTime:{timestamp}--------\n"           

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



df_PredictToData_Error = pd.DataFrame(columns=['window_size','RMSE'])


if(Dataset_Option == 0):
    Output_FilePath = f"../_out/_Evaluator/nyc_taxi/{timestamp}"
    os.makedirs(Output_FilePath)
elif(Dataset_Option == 1):
    Output_FilePath = f"../_out/_Evaluator/peyton_manning_wiki_pv/{timestamp}"
    os.makedirs(Output_FilePath)
elif(Dataset_Option == 2):
    Output_FilePath = f"../_out/_Evaluator/National_Football_League_wiki_pv/{timestamp}"
    os.makedirs(Output_FilePath)
else:
    Output_FilePath = f"../_out/_Evaluator/{timestamp}"
    os.mkdir(Ou)

LogText += f"---------Dataset:{Dataset_Option}----------\n"
time_start = time.perf_counter()

for window_size in range(10,160,10):
    print(f"Evaluating window_size:{window_size}")
    LogText+=f"on window_size:{window_size}\n"
    lag_max = window_size
    RMSE_sum = 0

    for i in range(mean_loop):
        
 
        X_train,X_test,y_train,y_test,scaler_y=gen_dataset(dataset, lag_max)
        model = train_model(X_train, y_train)
        y_test_pred = model.predict(X_test)
        y_test_pred = scaler_y.inverse_transform(y_test_pred)
        df_test = pd.DataFrame(np.hstack((y_test,y_test_pred)),
                            columns=['y','Predict'])
        _rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        LogText+=f"  attempt-{i}:rmse={_rmse}\n"
        RMSE_sum += _rmse
        print(f"    attempt-{i+1}:rmse={_rmse}")


    RMSE_mean = RMSE_sum / mean_loop
    LogText += f"RMSE_mean:{RMSE_mean}\n"
    df_PredictToData_Error.loc[len(df_PredictToData_Error)] = [window_size, RMSE_mean]

    """
    explainer = shap.DeepExplainer(model=model,data=X_train)
    X = X_test[::X_test.shape[0]//3]
    shap_values = explainer.shap_values(X=X_test[::X_test.shape[0]//3])#
    print(f"x.shape[0]={X.shape[0]}")
    for i in range(X.shape[0]):
        plt.bar(range(lag_max),shap_values[0][i,:].reshape(-1))
        plt.title(f"Window size = {window_size}")
        plt.xlabel("Time Point")
        plt.ylabel("SHAP value")
        plt.savefig(f"{Output_FilePath}/shap_values_{window_size}_{i}.pdf")
        plt.close()
    """

        
df_PredictToData_Error.plot(x='window_size')
plt.savefig(f"{Output_FilePath}/_RMSE_mean.pdf")
plt.close()
time_end = time.perf_counter()
time_grid = time_end - time_start
print(f"Evaluation_time:{time_grid}")

df_RMSE_sorted = df_PredictToData_Error.sort_values('RMSE')


LogText += df_RMSE_sorted.to_string()
f=open(f"{Output_FilePath}/_ReportLog.txt",'w')
f.write(LogText)
f.close()