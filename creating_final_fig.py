import pandas as import pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.DataFrame({'Mutlinomial Naive Bayes':[0.445, 0.542, 0.342], 'Opt. Models TF-IDF':[0.579,0.676,0.439], 'Opt. Models Doc2Vec':[0.625,0.680,0.504], 'labels':['usefulness','sentiment','rating']})
ax = df.plot.barh(title='Weighted F1 Score', x='labels', style=['fivethirtyeight'],figsize=(11,7), fontsize=15)
plt.legend(bbox_to_anchor=(1,.8), fontsize=11)
plt.tight_layout()
plt.savefig('/Users/gmgtex/Desktop/f1_score_plt.png')
