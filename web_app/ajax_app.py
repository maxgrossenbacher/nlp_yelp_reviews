from flask import Flask, render_template, request, jsonify
import pickle
from build_model import TextClassifier

app = Flask(__name__)


with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('ajax/index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('ajax/submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified, use the model to classify, and
    then return the classification in a response as json.
    """
    data = str(request.json['article'])
    pred = str(model.predict([data])[0])
    return jsonify({'prediction': pred})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
