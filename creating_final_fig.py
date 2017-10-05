import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
df = pd.DataFrame({'Mutlinomial Naive Bayes':[0.445, 0.542, 0.342], 'Opt. Models TF-IDF':[0.579,0.676,0.439], 'Opt. Models Doc2Vec':[0.581,0.724,0.493], 'Target':['usefulness','sentiment','rating']})
ax = df.plot.barh(title='Weighted F1 Score', x='Target', style=['fivethirtyeight'],figsize=(11,7), fontsize=17)
plt.legend(bbox_to_anchor=(1,.8), fontsize=15)
plt.tight_layout()
plt.savefig('/Users/gmgtex/Desktop/f1_score_plt_cv.png')
