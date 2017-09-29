from flask import Flask, render_template, request, jsonify
import pickle
from build_model import TextClassifier

app = Flask(__name__)


with open('static/model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/index', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('submit.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['article_body'])
    pred = str(model.predict([data])[0])
    return render_template('predict.html', article=data, predicted=pred)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    #real time prediction ratings
    return render_template('predictions.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
