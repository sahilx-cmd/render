from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model
with open('ensemble_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values
    eyetoeye = float(request.form['eyetoeye'])
    eyetomouth = float(request.form['eyetomouth'])

    # Make prediction and calculate probabilities
    prediction = model.predict([[eyetoeye, eyetomouth]])[0]
    probabilities = model.predict_proba([[eyetoeye, eyetomouth]])[0]
    no_down_syndrome_prob, down_syndrome_prob = probabilities

    # Generate prediction message
    if prediction == 0:
        prediction_text = "No Down Syndrome detected."
    else:
        prediction_text = "Down Syndrome likely detected."

    # Display both probabilities with the prediction
    prediction_text = (
        f"{prediction_text} "
        f"Probability of No Down Syndrome: {no_down_syndrome_prob:.2f}, "
        f"Probability of Down Syndrome: {down_syndrome_prob:.2f}"
    )

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
     serve(app, host='0.0.0.0', port=10000)
