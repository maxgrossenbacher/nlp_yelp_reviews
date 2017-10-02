from flask import Flask, render_template, request, jsonify
import pickle
from latent_topic_analysis import NlpTopicAnalysis
import textacy

app = Flask(__name__)

with open('static/usefulness_model_opt_gdc.pkl', 'rb') as u:
    usefulness_model = pickle.load(u)
with open('static/sentiment_model_new_opt_svc.pkl', 'rb') as s:
    sentiment_model = pickle.load(s)
# with open('static/rating_model_opt_gdc.pkl', 'rb') as r:
    # rating_model = pickle.load(r)
with open('static/vectorizer.pkl', 'rb') as v:
    vectorizer = pickle.load(v)
with open('static/sentiment_vectorizer.pkl', 'rb') as sv:
    senti_vectorizer = pickle.load(sv)

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
    useful = 'Useful: {}%        Very Useful: {}%' .format(round((usepred[0][1]*100),2), round((usepred[0][2]*100),2))

    senti_vect = senti_vectorizer.transform(terms_list)
    sentiment_pred = sentiment_model.predict(senti_vect.toarray())
    sentiment =  sentiment_pred
    #rating = rating_model.predict(vector.to_array())
    return render_template('predict_2.html', review=data, useful=useful, sentiment=sentiment)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    #real time prediction ratings
    return render_template('predictions.html')

@app.route('/explore', methods=['GET'])
def explore():
    ids = [ 'YJ8ljUhLsz6CtT_2ORNFmg', '7sPNbCx7vGAaH7SbNPZ6oA','NvKNe9DnQavC9GstglcBJQ','rcaPajgKOJC2vo_l3xa42A','fL-b760btOaGa85OJ9ut3w','eoHdUeQDNgQ6WYEnP2aiRw','HhVmDybpU7L50Kb5A0jXTg','G-5kEa6E6PD5fkBRuA7k9Q','iCQpiavjjPzJ5_3gPD5Ebg','2weQS-RnoOBhb1KsHKyoSQ']
    return render_template('topic.html', ids=ids)

@app.route('/topic_anaysis', methods=['GET', 'POST'])
def topic():
    i = str(request.form['business'])
    address = 'pyLDAvis_'+i+'.html'
    print(address)
    return render_template(address)

@app.route('/contact', methods=['GET'])
def contact():
    #real time prediction ratings
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
