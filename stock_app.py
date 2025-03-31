import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

# Define the date range
start = '2010-01-01'
end = '2019-12-31'

# Streamlit App
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)

# Display Data Summary
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

# Closing Price vs Time Chart
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
st.pyplot(fig)

# Moving Averages
st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.plot(ma100, 'r', label='100-Day MA')
plt.plot(ma200, 'g', label='200-Day MA')
plt.legend()
st.pyplot(fig)

# Splitting Data
train_size = int(len(df) * 0.70)
data_training = df['Close'][:train_size]
data_testing = df['Close'][train_size:]

# Scaling Data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1, 1))

# Preparing Training Data
x_train, y_train = [], []
for i in range(100, len(data_training_array)):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Model Training (if model is not pre-trained)
st.subheader("Training LSTM Model (First-time use only)")
model = Sequential([
    LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(units=80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(units=120, activation='relu'),
    Dropout(0.5),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=32)

# Save Model
model.save("stock_model.h5")

# Testing Data Preparation
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], axis=0)
input_data = scaler.transform(np.array(final_df).reshape(-1, 1))

x_test, y_test = [], []
for i in range(100, len(input_data)):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Model Prediction
y_predicted = model.predict(x_test)
y_predicted = y_predicted * (1 / scaler.scale_[0])  # Reverse Scaling
y_test = y_test * (1 / scaler.scale_[0])

# Predictions vs Original Chart
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
st.pyplot(fig2)
