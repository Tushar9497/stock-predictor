import tensorflow as tf

# Load your .keras model
model = tf.keras.models.load_model("stock_model.keras", compile=False)

# Create a converter object
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ Enable compatibility with complex ops like LSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# ✅ Disable experimental lowering of tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Convert the model
tflite_model = converter.convert()

# Save the converted model
with open("stock_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model converted successfully!")
