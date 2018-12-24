import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import numpy as np
from gensim.models import Word2Vec,FastText
from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    
    def __init__(self,type,train=False,params={}):
        self.type = type
        self.train = True
        self.params = params

    def word2vec(self):
        from os import listdir
        if 'word2vec.model' not in listdir('.') or self.train:
           model = self.train_w2v()
        else:
            model = Word2Vec.load("word2vec.model")
        vectorizer = model.wv
        vectors = [np.array([vectorizer[word] for word in tweet]).flatten() for tweet in self.data]
        max_len = np.max([len(vector) for vector in vectors])
        self.vectors = [
            np.array(vector.tolist()+[0 for _ in range(max_len-len(vector))]) for vector in vectors
            ]

    def train_w2v(self):
        from gensim.test.utils import get_tmpfile
        path = get_tmpfile("word2vec.model")
        model = Word2Vec(self.data, size=100, window=5, min_count=1, workers=5)
        model.train(self.data, total_examples=len(self.data), epochs=200)
        model.save("word2vec.model")
        return model

    def tfidf(self):
        vectorizer = TfidfVectorizer(**self.params)
        untokenized_data =[' '.join(tweet) for tweet in self.data] 
        self.vectors = vectorizer.fit_transform(untokenized_data).toarray()

    def vectorize(self,data):
        self.data = data
        vectorize_call = getattr(self, self.type, None)
        if vectorize_call:
            vectorize_call()
        else:
            raise Exception(str(self.type),'is not an available function')
        return self.vectors

