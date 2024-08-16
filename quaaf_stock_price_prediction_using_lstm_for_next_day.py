
# enable and detect GPU

import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

import importlib.util
np.random.seed(70)
package_name = 'yfinance'
spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed")

import yfinance as yf
st.title("Stock Price Trend and Prediction Dashboard")
stock=st.text_input("Enter the ticker for which you want to predict the stock price:","AAPL")
#stock=input("Enter the ticker for which you want to predict the stock price:")
df=yf.download(stock, start='2012-01-01')

#3. Visualize

#df['Close'].tail(10)

plt.figure(figsize=(12,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend(yf.Ticker(stock).info['longName'])
st.pyplot(plt)

data=df['Close']

#4. Prepare the data

#convert to numpy aarray
#We use 80% of the data for training
train_pct=0.8
dataset=data.values
train_data_len=math.ceil(len(dataset)*train_pct)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset.reshape(-1,1))

#scaled_data.shape

# The window we use is n_period as input

n_period=60
#training dataset
train_data=scaled_data[0:train_data_len]
x_train=[]
y_train=[]

for i in range(n_period, len(train_data)):
    x_train.append(train_data[i-n_period:i])
    y_train.append(train_data[i])

#len(x_train)

#Convert numpy array
x_train, y_train=np.array(x_train), np.array(y_train)

x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
#x_train.shape

#5. Train the model
#Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
#loss function is chosen as mean_squared_error'


#Build the LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train,y_train, batch_size=1, epochs=1)

# Prediction for testing dataset
test_data=scaled_data[train_data_len-60:,:]
x_test=[]
y_test=dataset[train_data_len:]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions_scaled=model.predict(x_test)
predictions=predictions_scaled*(scaler.data_max_[0]-scaler.data_min_[0])+scaler.data_min_[0]

#predictions.shape

df["Predicted_Close"]=pd.NA

#df.shape[0]-predictions.shape[0]

df.iloc[df.shape[0]-predictions.shape[0]:,-1:]=predictions

df["Predicted_Close"]=df["Predicted_Close"].fillna(0)
df["Predicted_Close"]=df["Predicted_Close"].astype("float64")

#plotting the results
plt.figure(figsize=(12,8))
plt.title('Prediction vs Actual')
plt.plot(df['Close'])
plt.plot(df.loc[df["Predicted_Close"]>0]['Predicted_Close'])
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.legend(['Actual','Prediction'],loc='lower right')
st.pyplot(plt)
st.dataframe(df.loc[df["Predicted_Close"]>0])
