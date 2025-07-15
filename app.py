import streamlit as st
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.title("📈 Real-Time Stock Price Predictor")

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
        data = yf.download(ticker, period="120d", interval="1d")
        close_prices = data['Close'].dropna().values

        if len(close_prices) < 100:
            st.error("❌ Not enough data. Need at least 100 closing prices.")
        else:
            last_100 = close_prices[-100:].reshape(-1, 1)

            # Scale the data just like training
            scaler = MinMaxScaler(feature_range=(0, 1))
            last_100_scaled = scaler.fit_transform(last_100)

            input_array = np.array(last_100_scaled).reshape(1, 100, 1).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])

            predicted_price = scaler.inverse_transform(prediction)

            st.success(f"📊 Predicted Next Closing Price: ₹{predicted_price[0][0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
