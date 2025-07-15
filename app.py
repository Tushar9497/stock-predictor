import streamlit as st
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Real-Time Stock Price Predictor with Recent History")

@st.cache_resource
def load_model_and_scaler():
    interpreter = tf.lite.Interpreter(model_path="stock_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

interpreter, input_details, output_details = load_model_and_scaler()

# User input
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TCS.NS)", "AAPL")

if st.button("Predict Next Price"):
    try:
        # Download data
        data = yf.download(ticker, period="120d", interval="1d")
        close_prices = data['Close'].dropna()

        if len(close_prices) < 100:
            st.error("❌ Not enough data. Need at least 100 closing prices.")
        else:
            # 🔍 Show past 20 days' data
            st.subheader("📅 Last 20 Days Closing Prices")
            st.dataframe(close_prices.tail(20).reset_index().rename(columns={'Date': 'Date', 'Close': 'Closing Price'}), use_container_width=True)

            # Prepare data
            last_100 = close_prices[-100:].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            last_100_scaled = scaler.fit_transform(last_100)

            input_array = np.array(last_100_scaled).reshape(1, 100, 1).astype(np.float32)

            # Predict
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            predicted_price = scaler.inverse_transform(prediction)

            st.success(f"📊 Predicted Next Closing Price: ₹{predicted_price[0][0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
