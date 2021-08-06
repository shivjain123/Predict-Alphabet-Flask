from flask import Flask, jsonify, request
from app_prediction import get_prediction

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Alphabet Predicting App."

@app.route("/predict-alphabet", methods=["POST"])
def predict_data():
  #image = cv2.imdecode(np.fromstring(request.files.get("alphabet").read(), np.uint8), cv2.IMREAD_UNCHANGED)
  image = request.files.get("alphabet")
  prediction = get_prediction(image)
  return jsonify({
      "prediction": prediction
  }), 200

if __name__ == "__main__":
  app.run(debug=True)