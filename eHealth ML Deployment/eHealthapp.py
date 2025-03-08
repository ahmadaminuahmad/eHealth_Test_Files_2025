import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
eHealthapp = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@eHealthapp.route("/")
def Home():
    return render_template("index.html")

@eHealthapp.route("/predict", methods = ["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The Result is {}".format(prediction))

if __name__ == "__main__":
    eHealthapp.run(debug=True)