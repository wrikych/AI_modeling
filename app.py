from flask import Flask, render_template, request
import pickle
import numpy as np

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input values from the form
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    
    # Create a list of input values
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # Convert the input values to a numpy array
    input_array = np.array(input_data)

    # Perform scaling only on the specified variables
    input_array[:, [0, 3, 4, 7, 9]] = scaler.transform(input_array[:, [0, 3, 4, 7, 9]])

    # Perform the prediction using your algorithm
    predicted_value = model.predict(input_array)  # Replace 'your_algorithm' with your actual prediction algorithm
    
    return_value = ''
    if predicted_value == 0:
        return_value = 'No Heart Disease'
    else:
        return_value = 'Heart Disease'

    # Render the prediction page with the predicted value
    return render_template('prediction.html', return_value=return_value)

if __name__ == '__main__':
    app.run()
