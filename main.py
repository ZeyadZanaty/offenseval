from DataReader import DataReader
from Preprocessor import Preprocessor
from Vectorizer import Vectorizer

dr = DataReader('./datasets/training-v1/offenseval-training-v1.tsv')
data,labels = dr.get_labelled_data()
data = data[:]
prp = Preprocessor('remove_stopwords')
data = prp.clean(data)
# prp.word_cloud(labels,'OFF')
vct = Vectorizer('tfidf')
vct.vectorize(data)
# for i in range(500):
#     print('\n',data[i],labels[i])
