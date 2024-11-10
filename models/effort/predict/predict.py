from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf

# Load the model
model_name='model.keras'
model = tf.keras.models.load_model(model_name)

# Flask
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    predictions = model.predict(input_data)
    return jsonify({'prediction': predictions.tolist()})

# Run the app
if __name__ == '__main__':
    # Example request
		# {
		#    "wall_21": 2,
		#    "km_per_week": 21 
		# }
    app.run(host='0.0.0.0', port=8080, debug=True)