import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="stock_model.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up Streamlit page
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Predictor")
st.write("Predict the next stock price based on the last 100 days' data.")

# User input
user_input = st.text_area("Enter 100 stock prices separated by commas", "")

# Prediction button
if st.button("Predict"):

    try:
        # Parse and validate input
        input_list = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        if len(input_list) != 100:
            st.error("Please enter exactly 100 numeric values.")
        else:
            # Preprocess input for model
            input_array = np.array(input_list).reshape(1, 100, 1).astype(np.float32)

            # Set input tensor and invoke
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()

            # Get prediction
            prediction = interpreter.get_tensor(output_details[0]['index'])
            predicted_price = prediction[0][0]

            st.success(f"📊 Predicted Next Price: **₹{predicted_price:.2f}**")

    except ValueError:
        st.error("Invalid input. Please enter only numeric values.")

# Optional footer
st.markdown("---")
st.markdown("Created with ❤️ by Tushar Nirmal")
