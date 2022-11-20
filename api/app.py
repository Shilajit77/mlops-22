from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "dt.joblib"
model = load(model_path)


@app.route("/predict", methods=['POST'])
def predict_digit():
    digit_image1 = request.json['image']
    digit_image2 = request.json['image']
    print("done loading")
    predicted_image1 = model.predict([digit_image1])
    predicted_image2 = model.predict([digit_image2])
    if predicted_image1==predicted_image2:
        return "Same Digits\n"

    else:
        return "Different Digits\n"    


if __name__=="__main__":
    app.run(host="0.0.0.0",port=5002)