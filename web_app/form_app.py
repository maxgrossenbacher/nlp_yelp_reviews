from flask import Flask, render_template, request, jsonify
import pickle
from latent_topic_analysis import NlpTopicAnalysis
import textacy

app = Flask(__name__)

with open('static/usefulness_model_opt_gdc.pkl', 'rb') as u:
    usefulness_model = pickle.load(u)
# with open('static/sentiment_model_opt_gdc.pkl', 'rb') as s:
#     sentiment_model = pickle.load(s)
# with open('static/rating_model_opt_gdc.pkl', 'rb') as r:
    # rating_model = pickle.load(r)
with open('static/vectorizer.pkl', 'rb') as v:
    vectorizer = pickle.load(v)

@app.route('/', methods=['GET'])
def index():
    """Render a simple splash page."""
    return render_template('index_2.html')

@app.route('/submit', methods=['GET'])
def submit():
    """Render a page containing a textarea input where the user can paste an
    article to be classified.  """
    return render_template('submit_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Recieve the article to be classified from an input form and use the
    model to classify.
    """
    data = str(request.form['review'])
    nlp = NlpTopicAnalysis(text=[data])
    nlp.process_text()
    terms_list = [(list(doc.to_terms_list(n_grams=1, named_entities=True, \
                                            normalize='lemma', as_strings=True, \
                                            filter_stops=True, filter_punct=True, exclude_pos=['PUNCT','SPACE']))) for doc in nlp.corpus]
    vector = vectorizer.transform(terms_list)
    usepred = usefulness_model.predict_proba(vector.toarray())
    useful = 'Not Useful: {}%        Useful: {}%        Very Useful: {}%' .format(round((usepred[0][0]*100),2), round((usepred[0][1]*100),2), round((usepred[0][2]*100),2))
    #sentiment_pred = sentiment_model.predict_proba(vector.to_array())
    # sentiment = 'Negative: {}%        Neutral: {}%        Positive: {}%' .format(round((sentiment_pred[0][0]*100),2), round((sentiment_pred[0][1]*100),2), round((sentiment_pred[0][2]*100),2))
    #rating = rating_model.predict(vector.to_array())
    return render_template('predict_2.html', review=data, useful=useful)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    #real time prediction ratings
    return render_template('predictions.html')

@app.route('/contact', methods=['GET'])
def contact():
    #real time prediction ratings
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
