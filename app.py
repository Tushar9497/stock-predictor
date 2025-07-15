import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.title('📈 Stock Market Price Predictor')

stock = st.text_input("Enter stock symbol (e.g., AAPL, GOOG)", "GOOG")

if stock:
    start = '2012-01-01'
    end = '2022-12-31'
    data = yf.download(stock, start=start, end=end)

    st.subheader('Raw Data')
    st.write(data.tail())

    data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    past_100 = data_scaled[-100:]
    model_input = np.array([past_100])

    model = load_model('stock_model.keras', compile=False)
    predicted = model.predict(model_input)
    predicted_price = scaler.inverse_transform(predicted)

    st.subheader("📊 Predicted Closing Price")
    st.write(f"${predicted_price[0][0]:.2f}")
