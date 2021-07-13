from flask import Flask, render_template, jsonify, request
from sklearn import svm
from sklearn import datasets
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
api_active = "I am alive"
# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target
# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return api_active
    else:
        input = request.json
        out = clf.predict(input)
        return jsonify(out.tolist())


if __name__ == '__main__':
    app.run("0.0.0.0", port=5000)
