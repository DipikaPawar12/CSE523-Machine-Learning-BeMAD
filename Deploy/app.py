import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.decision_function(final_features)
    crop_to_code = {0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut', 5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans', 10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean', 15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas', 19: 'pomegranate', 20: 'rice', 21: 'watermelon'}
    
    N = 3
    res = sorted(range(len(prediction[0])), key = lambda sub: prediction[0][sub])[-N:]
    crop=['']*3
    crop[0]=crop_to_code[res[2]]
    crop[1]=crop_to_code[res[1]]
    crop[2]=crop_to_code[res[0]]
    return render_template('index.html', prediction_text='Crops Grown should be {}'.format(crop))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    #For direct API calls trought request
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
