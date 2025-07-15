import streamlit as st
import numpy as np
import yfinance as yf
import tensorflow as tf

st.title("📈 Real-Time Stock Price Predictor")

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="stock_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# User selects a stock
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TCS.NS)", "AAPL")

if st.button("Predict Next Price"):
    try:
        # Download past 120 days of data to ensure we have at least 100 valid closing prices
        data = yf.download(ticker, period="120d", interval="1d")
        close_prices = data['Close'].dropna().values

        if len(close_prices) < 100:
            st.error("❌ Not enough historical data (need at least 100 closing prices).")
        else:
            last_100 = close_prices[-100:]
            input_array = np.array(last_100).reshape(1, 100, 1).astype(np.float32)

            # Make prediction
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            st.success(f"📊 Predicted Next Closing Price: **₹{prediction[0][0]:.2f}**")

    except Exception as e:
        st.error(f"⚠Failed to fetch or predict data: {str(e)}")
