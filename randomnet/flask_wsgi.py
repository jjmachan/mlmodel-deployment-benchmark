import numpy as np
from flask import Flask, request

from tf_model import build_big_model

model = build_big_model()

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    received_input = np.array(request.json['input'])
    model_output = model.predict(received_input)

    return {'predictions': model_output.tolist()}


if __name__ == '__main__':
    app.run()
