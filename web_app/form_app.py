from flask import Flask, render_template, request, jsonify
from sklearn.externals import joblib
from latent_topic_analysis import NlpTopicAnalysis
import textacy

app = Flask(__name__)

usefulness_model = joblib.load('static/usefulness_model_rf_word2vec.pkl')
sentiment_model = joblib.load('static/sentiment_model_gbc_word2vec.pkl')
rating_model = joblib.load('static/rating_model_gbc_word2vec.pkl')

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
    nlp.word2vec()
    docvec = nlp.doc_vectors
    usepred = usefulness_model.predict_proba(docvec)
    useful = 'Useful: {}%, Very Useful: {}%' .format(round(((usepred[0][1]*100)+(usepred[0][2]*100)),2), round((usepred[0][2]*100),2))
    sentiment_pred = sentiment_model.predict_proba(docvec)
    sentiment = 'Negative: {}%, Neutral: {}%, Postive:{}%'.format(round((sentiment_pred[0][0]*100),2), round((sentiment_pred[0][1]*100),2), round((sentiment_pred[0][2]*100),2))
    rating = rating_model.predict(docvec)
    rating = str(rating).strip('[]')
    return render_template('predict_2.html', review=data, useful=useful, sentiment=sentiment, rating=rating)

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    #real time prediction ratings
    return render_template('predictions.html')

@app.route('/explore', methods=['GET'])
def explore():
    # ids = [ 'YJ8ljUhLsz6CtT_2ORNFmg', '7sPNbCx7vGAaH7SbNPZ6oA','NvKNe9DnQavC9GstglcBJQ','rcaPajgKOJC2vo_l3xa42A','fL-b760btOaGa85OJ9ut3w','eoHdUeQDNgQ6WYEnP2aiRw','HhVmDybpU7L50Kb5A0jXTg','G-5kEa6E6PD5fkBRuA7k9Q','iCQpiavjjPzJ5_3gPD5Ebg','2weQS-RnoOBhb1KsHKyoSQ']
    return render_template('topic.html')

# @app.route('/topic_anaysis', methods=['GET', 'POST'])
# def topic():
#     i = str(request.form['business'])
#     address = 'pyLDAvis_'+i+'.html'
#     print(address)
#     return render_template(address)

@app.route('/contact', methods=['GET'])
def contact():
    #real time prediction ratings
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
