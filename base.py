from tkinter import*
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from finta import TA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
import plotly.graph_objects as go
from itertools import chain



def appmain():
    firstfunc()

root = tk.Tk()
root.title('Stock Market Prediction App')
root.geometry("1510x760+0+0")
root.resizable(False, False)
frameback1=Frame(root,bg="white")
frameback1.place(x=0,y=0,height=760,width=1510)
label1=Label(root,text="Stock Market Prediction App",font=('Ariel',32,'bold'),bg="white",fg="black" )
label1.place(x=400,y=100)
label2=Label(root,text="Enter Stock Ticker to predict stock prices",font=("Ariel",10), bg="white",fg="black")
label2.place(x=400,y=200)
entry_ticker=Entry(root,font=("times new roman",15),bg="grey")
entry_ticker.place(x=400,y=230)
but1=Button(root,command=appmain,cursor="hand2",text="predict", font=("times new roman",15),fg="white",bg="orange",bd=0,width=10,height=1)
but1.place(x=400,y=270)

def appfunc(str):
    NUM_DAYS = 1700
    INTERVAL = '1d'    
    symbol = entry_ticker.get()
    INDICATORS = ['RSI', 'MACD', 'STOCH','ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']
    start = (datetime.date.today() - datetime.timedelta( NUM_DAYS ) )
    end = datetime.datetime.today()

    df = yf.download(symbol, start=start, end=end, interval=INTERVAL)
    df.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'}, inplace=True)

    if(str=='open'):
        val=df.iloc[:,0:1].values
    elif(str=='close'):
        val=df.iloc[:,3:4].values
    elif(str=='high'):
        val=df.iloc[:,1:2].values
    else:
        val=df.iloc[:,2:3].values
    
    scaler=MinMaxScaler(feature_range = (0,1))
    scaled_dataset=scaler.fit_transform(val)

    test_data=scaled_dataset[-20:]
    train_data=scaled_dataset[:-20]

    X_train = []
    Y_train = []
    for i in range (60,len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        Y_train.append(train_data[i,0])
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss ='mean_squared_error')
    regressor.fit(X_train, Y_train, epochs=100, batch_size=32)

    actual_price=val[-20:]
    inputs=val[len(val)-len(test_data)-60:]

    inputs=inputs.reshape(-1,1)
    inputs=scaler.transform(inputs)

    X_test=[]
    for i in range(60,80):
        X_test.append(inputs[i-60:i,0])
    X_test=np.array(X_test)
    X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

    predicted_price= regressor.predict(X_test)
    predicted_price=scaler.inverse_transform(predicted_price)

    return predicted_price

def firstfunc():
    predicted_close=appfunc('close')
    predicted_open=appfunc('open')
    predicted_high=appfunc('high')
    predicted_low=appfunc('low')
    datelist=pd.date_range(datetime.date.today() - datetime.timedelta(19) , datetime.datetime.today())
    pr_close = list(chain.from_iterable(predicted_close))
    pr_open = list(chain.from_iterable(predicted_open))
    pr_high = list(chain.from_iterable(predicted_high))
    pr_low = list(chain.from_iterable(predicted_low))

    df = pd.DataFrame({'open': [56.858566, 56.685326, 56.163055, 55.343414, 54.765556, 54.682037, 55.16124, 55.606586, 55.977737, 56.049416, 56.222424, 56.264362, 55.98048, 55.802864, 56.45114, 57.63729, 58.67455, 59.185955, 59.739418, 59.921757],
                             'close': [56.063663, 55.25031, 54.48389, 54.129486, 54.138077, 54.555756, 55.19798, 55.08875, 55.16508, 55.656445, 55.752007, 55.251675, 55.327988, 56.356285, 57.70953, 58.535614, 58.819435, 59.48699, 59.405746, 59.14773],
                             'high': [57.65835, 57.08457, 56.26746, 55.745094, 55.711456, 56.185658, 56.538155, 56.81987, 56.83996, 57.031303, 57.094715, 56.820316, 56.68396, 57.402473, 58.564686, 59.48767, 59.921455, 60.525932, 60.753407, 60.691822],
                             'low': [56.32778, 55.64318, 54.915905, 54.45938, 54.449463, 54.96315, 55.05132, 55.169575, 55.563747, 56.1003, 56.25874, 55.871914, 55.519295, 56.10474, 57.260387, 58.424088, 59.18915, 59.4932, 59.636803, 59.59518],
                             'date': datelist
                             })

    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
    fig.show()            
root.mainloop()    